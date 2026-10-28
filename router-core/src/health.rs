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
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
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
    if !record_probe(&backend, healthy, config) {
        return;
    }

    match client
        .get(format!("{}/metrics", backend.url))
        .timeout(PROBE_TIMEOUT)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.text().await {
            Ok(text) => {
                apply_scrape(&text, &backend);
            }
            Err(e) => {
                tracing::debug!(backend = %backend.url, error = %e, "metrics body unreadable");
                reset_scraped(&backend);
            }
        },
        Ok(resp) => {
            tracing::debug!(backend = %backend.url, status = %resp.status(), "metrics scrape rejected");
            reset_scraped(&backend);
        }
        Err(e) => {
            tracing::debug!(backend = %backend.url, error = %e, "metrics scrape failed");
            reset_scraped(&backend);
        }
    }
}

/// Records a probe result. Returns whether the backend should be scraped (the probe passed).
/// Scraped values are cleared only once the backend is actually marked unhealthy.
fn record_probe(backend: &Backend, healthy: bool, config: &HealthCheckConfig) -> bool {
    apply_probe_result(backend, healthy, config);
    if !healthy && !backend.is_healthy() {
        reset_scraped(backend);
    }
    healthy
}

/// A successful scrape that lacks a metric means the backend reports none: clear any old value.
fn apply_scrape(text: &str, backend: &Backend) {
    let found = parse_backend_metrics(text, backend);
    if !found.queue_depth {
        backend.queue_depth.store(0, Ordering::Relaxed);
    }
    if !found.kv_cache {
        backend.kv_cache_usage_bp.store(0, Ordering::Relaxed);
    }
}

/// Stale scraped values must not keep steering traffic once a scrape stops working.
fn reset_scraped(backend: &Backend) {
    backend.queue_depth.store(0, Ordering::Relaxed);
    backend.kv_cache_usage_bp.store(0, Ordering::Relaxed);
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

/// Takes the last whitespace-separated token as the value, so exposition timestamps are not
/// supported (vLLM emits none).
fn metric_value(line: &str) -> Option<f64> {
    line.split_whitespace().last()?.parse().ok()
}

/// Which scraped metrics a `/metrics` body contained.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ScrapeFound {
    pub queue_depth: bool,
    pub kv_cache: bool,
}

/// Reads `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc`
/// (or the pre-0.6 name `vllm:gpu_cache_usage_perc`). Exact-name match only.
/// Reports which of the two metrics were present.
pub fn parse_backend_metrics(text: &str, backend: &Backend) -> ScrapeFound {
    let mut found = ScrapeFound::default();
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
                    found.queue_depth = true;
                }
            }
            "vllm:kv_cache_usage_perc" | "vllm:gpu_cache_usage_perc" => {
                if let Some(v) = metric_value(line) {
                    backend
                        .kv_cache_usage_bp
                        .store((v.clamp(0.0, 1.0) * 10_000.0) as u64, Ordering::Relaxed);
                    found.kv_cache = true;
                }
            }
            _ => {}
        }
    }
    found
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
    fn absent_queue_metric_resets_depth() {
        let b = backend();
        b.queue_depth.store(100, Ordering::Relaxed);
        apply_scrape("vllm:kv_cache_usage_perc 0.2\n", &b);
        assert_eq!(b.queue_depth.load(Ordering::Relaxed), 0);
        assert_eq!(b.kv_cache_usage_bp.load(Ordering::Relaxed), 2000);
    }

    #[test]
    fn parse_returns_whether_queue_depth_was_found() {
        let b = backend();
        assert!(parse_backend_metrics("vllm:num_requests_waiting 3\n", &b).queue_depth);
        assert!(!parse_backend_metrics("vllm:num_requests_waiting_total 3\n", &b).queue_depth);
    }

    #[test]
    fn absent_kv_metric_resets_usage() {
        let b = backend();
        b.kv_cache_usage_bp.store(5000, Ordering::Relaxed);
        apply_scrape("vllm:num_requests_waiting 2\n", &b);
        assert_eq!(b.kv_cache_usage_bp.load(Ordering::Relaxed), 0);
        assert_eq!(b.queue_depth.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn scraped_values_reset_only_once_marked_down() {
        let b = backend();
        let cfg = HealthCheckConfig {
            interval_secs: 1,
            unhealthy_threshold: 2,
            healthy_threshold: 2,
        };
        b.queue_depth.store(9, Ordering::Relaxed);
        assert!(!record_probe(&b, false, &cfg));
        assert_eq!(b.queue_depth.load(Ordering::Relaxed), 9);
        assert!(!record_probe(&b, false, &cfg));
        assert_eq!(b.queue_depth.load(Ordering::Relaxed), 0);
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
