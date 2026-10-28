//! Active health checking plus a scrape of each backend's vLLM metrics.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;

use crate::backend::{Backend, BackendPool};
use crate::config::HealthCheckConfig;

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

pub fn spawn_health_checker(
    pool: Arc<BackendPool>,
    client: reqwest::Client,
    config: HealthCheckConfig,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(config.interval_secs));
        loop {
            ticker.tick().await;
            join_all(
                pool.backends
                    .iter()
                    .map(|b| check_one(b.clone(), client.clone(), &config)),
            )
            .await;
        }
    })
}

async fn check_one(backend: Arc<Backend>, client: reqwest::Client, config: &HealthCheckConfig) {
    let healthy = match client
        .get(format!("{}/health", backend.url))
        .timeout(PROBE_TIMEOUT)
        .send()
        .await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    };
    apply_probe_result(&backend, healthy, config);

    if let Ok(resp) = client
        .get(format!("{}/metrics", backend.url))
        .timeout(PROBE_TIMEOUT)
        .send()
        .await
    {
        if let Ok(text) = resp.text().await {
            parse_backend_metrics(&text, &backend);
        }
    }
}

/// Hysteresis: `unhealthy_threshold` consecutive failures mark down,
/// `healthy_threshold` consecutive successes mark up.
pub fn apply_probe_result(backend: &Backend, healthy: bool, config: &HealthCheckConfig) {
    if healthy {
        let streak = backend
            .consecutive_health_successes
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        backend
            .consecutive_health_failures
            .store(0, Ordering::Relaxed);
        if streak >= u64::from(config.healthy_threshold) && !backend.is_healthy() {
            tracing::info!(backend = %backend.url, "backend recovered, marking healthy");
            backend.mark_healthy();
        }
    } else {
        let streak = backend
            .consecutive_health_failures
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        backend
            .consecutive_health_successes
            .store(0, Ordering::Relaxed);
        if streak >= u64::from(config.unhealthy_threshold) && backend.is_healthy() {
            tracing::warn!(backend = %backend.url, streak, "backend unhealthy, marking down");
            backend.mark_unhealthy();
        }
    }
}

/// The metric name is everything before the first `{` or whitespace.
fn metric_name(line: &str) -> &str {
    line.split(|c: char| c == '{' || c.is_whitespace())
        .next()
        .unwrap_or("")
}

fn metric_value(line: &str) -> Option<f64> {
    line.split_whitespace().last()?.parse().ok()
}

/// Reads `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc`
/// (or the pre-0.6 name `vllm:gpu_cache_usage_perc`). Exact-name match only.
pub fn parse_backend_metrics(text: &str, backend: &Backend) {
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        match metric_name(line) {
            "vllm:num_requests_waiting" => {
                if let Some(v) = metric_value(line) {
                    backend
                        .queue_depth
                        .store(v.max(0.0) as u64, Ordering::Relaxed);
                }
            }
            "vllm:kv_cache_usage_perc" | "vllm:gpu_cache_usage_perc" => {
                if let Some(v) = metric_value(line) {
                    backend
                        .kv_cache_usage_bp
                        .store((v.clamp(0.0, 1.0) * 10_000.0) as u64, Ordering::Relaxed);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_breaker::CircuitBreaker;

    fn backend() -> Backend {
        Backend::new(0, "http://b0".into(), 1, CircuitBreaker::default())
    }

    #[test]
    fn parses_exact_metric_names_and_ignores_decoys() {
        let b = backend();
        let text = "# TYPE vllm:num_requests_waiting gauge\n\
                    vllm:num_requests_waiting{instance=\"x\"} 7\n\
                    vllm:num_requests_waiting_total{instance=\"x\"} 999\n\
                    vllm:kv_cache_usage_perc{instance=\"x\"} 0.4321\n";
        parse_backend_metrics(text, &b);
        assert_eq!(b.queue_depth.load(Ordering::Relaxed), 7);
        assert_eq!(b.kv_cache_usage_bp.load(Ordering::Relaxed), 4321);
    }

    #[test]
    fn accepts_old_gpu_cache_name() {
        let b = backend();
        parse_backend_metrics("vllm:gpu_cache_usage_perc 0.5\n", &b);
        assert_eq!(b.kv_cache_usage_bp.load(Ordering::Relaxed), 5000);
    }

    #[test]
    fn thresholds_add_hysteresis() {
        let b = backend();
        let cfg = HealthCheckConfig {
            interval_secs: 1,
            unhealthy_threshold: 2,
            healthy_threshold: 2,
        };
        apply_probe_result(&b, false, &cfg);
        assert!(b.is_healthy(), "one failure is not enough");
        apply_probe_result(&b, false, &cfg);
        assert!(!b.is_healthy());
        apply_probe_result(&b, true, &cfg);
        assert!(!b.is_healthy(), "one success is not enough");
        apply_probe_result(&b, true, &cfg);
        assert!(b.is_healthy());
    }
}
