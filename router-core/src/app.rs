//! Wires config into a running application: pool, breakers, strategy, metrics, health task.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use tokio::task::JoinHandle;

use crate::backend::{Backend, BackendPool};
use crate::backpressure::BackpressureController;
use crate::circuit_breaker::CircuitBreaker;
use crate::config::{RouterConfig, UpstreamConfig};
use crate::health;
use crate::metrics::RouterMetrics;
use crate::routing;
use crate::server::{self, AppState};

pub struct App {
    pub router: Router,
    /// Dropping this handle detaches the task (it keeps running); `main.rs` relies on that.
    pub health_task: JoinHandle<()>,
}

fn build_backends(config: &RouterConfig) -> Vec<Arc<Backend>> {
    config
        .backends
        .iter()
        .enumerate()
        .map(|(i, bc)| {
            Arc::new(Backend::new(
                i,
                bc.normalized_url(),
                bc.weight,
                CircuitBreaker::from_config(&config.circuit_breaker),
            ))
        })
        .collect()
}

/// No total timeout: a connect timeout plus an idle read timeout, so a long stream that
/// keeps producing bytes is never cut, while a stalled backend is.
fn build_client(upstream: &UpstreamConfig) -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(100)
        .connect_timeout(Duration::from_secs(upstream.connect_timeout_secs))
        .read_timeout(Duration::from_secs(upstream.read_timeout_secs))
        .build()
}

/// Build the axum router and background tasks from a parsed config.
/// Must be called from inside a tokio runtime (it spawns the health checker).
pub fn build_app(config: RouterConfig) -> Result<App, Box<dyn std::error::Error>> {
    let backends = build_backends(&config);
    let pool = Arc::new(BackendPool::new(
        backends,
        config.backpressure.max_backend_queue_depth,
    ));

    let strategy = routing::create_strategy(
        &config.routing.strategy,
        config.routing.prefix_cache.prefix_token_count,
        pool.len(),
    )?;

    let client = build_client(&config.upstream)?;

    let backpressure = Arc::new(BackpressureController::new(
        config.backpressure.max_in_flight,
    ));
    let metrics = RouterMetrics::new();

    let health_task =
        health::spawn_health_checker(pool.clone(), client.clone(), config.health_check.clone());

    let state = AppState {
        pool,
        strategy: Arc::from(strategy),
        strategy_name: config.routing.strategy.clone(),
        client,
        backpressure,
        metrics,
        ewma_alpha: config.routing.latency_aware.alpha,
    };

    let router = server::build_router(state, &config.metrics);
    Ok(App {
        router,
        health_task,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_urls_are_normalised_without_trailing_slash() {
        let config = RouterConfig::from_yaml(
            "routing: {}\nbackends:\n  - url: \"http://b:8000/\"\n  - url: \"http://c:8000\"\n",
        )
        .unwrap();
        let backends = build_backends(&config);
        assert_eq!(backends[0].url, "http://b:8000");
        assert_eq!(backends[1].url, "http://c:8000");
    }

    #[test]
    fn upstream_client_builds_from_config() {
        let config =
            RouterConfig::from_yaml("routing: {}\nbackends:\n  - url: \"http://b:8000\"\n")
                .unwrap();
        assert!(build_client(&config.upstream).is_ok());
    }
}
