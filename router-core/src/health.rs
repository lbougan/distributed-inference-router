use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crate::backend::BackendPool;
use crate::circuit_breaker::CircuitBreaker;
use crate::config::HealthCheckConfig;

/// Spawns a background task that periodically health-checks all backends.
pub fn spawn_health_checker(
    pool: Arc<BackendPool>,
    _circuit_breakers: Arc<Vec<CircuitBreaker>>,
    client: reqwest::Client,
    config: HealthCheckConfig,
) -> tokio::task::JoinHandle<()> {
    let interval = Duration::from_secs(config.interval_secs);

    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            for backend in &pool.backends {
                let url = format!("{}/health", backend.url);
                let healthy = match client.get(&url).timeout(Duration::from_secs(3)).send().await {
                    Ok(resp) => resp.status().is_success(),
                    Err(_) => false,
                };

                if healthy {
                    let prev = backend.consecutive_health_successes.fetch_add(1, Ordering::Relaxed);
                    backend.consecutive_health_failures.store(0, Ordering::Relaxed);
                    if prev + 1 >= config.healthy_threshold as u64 && !backend.is_healthy() {
                        tracing::info!(backend = backend.url, "Backend recovered, marking healthy");
                        backend.mark_healthy();
                    }
                } else {
                    let prev = backend.consecutive_health_failures.fetch_add(1, Ordering::Relaxed);
                    backend.consecutive_health_successes.store(0, Ordering::Relaxed);
                    if prev + 1 >= config.unhealthy_threshold as u64 && backend.is_healthy() {
                        tracing::warn!(backend = backend.url, "Backend unhealthy after {} consecutive failures", prev + 1);
                        backend.mark_unhealthy();
                    }
                }

                // Scrape /metrics for queue depth and KV cache usage
                let metrics_url = format!("{}/metrics", backend.url);
                if let Ok(resp) = client.get(&metrics_url).timeout(Duration::from_secs(3)).send().await {
                    if let Ok(text) = resp.text().await {
                        parse_backend_metrics(&text, &backend);
                    }
                }
            }
        }
    })
}

fn parse_backend_metrics(text: &str, backend: &crate::backend::Backend) {
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        if line.starts_with("vllm:num_requests_waiting") || line.starts_with("vllm_num_requests_waiting") {
            if let Some(val) = parse_metric_value(line) {
                backend.queue_depth.store(val as u64, Ordering::Relaxed);
            }
        }
        if line.starts_with("vllm:kv_cache_usage_perc") || line.starts_with("vllm_kv_cache_usage_perc") {
            if let Some(val) = parse_metric_value(line) {
                backend.kv_cache_usage_bp.store((val * 10000.0) as u64, Ordering::Relaxed);
            }
        }
    }
}

fn parse_metric_value(line: &str) -> Option<f64> {
    line.split_whitespace().last()?.parse::<f64>().ok()
}
