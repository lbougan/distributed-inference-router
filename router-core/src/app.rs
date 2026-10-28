//! Wires config into a running application: pool, breakers, strategy, metrics, health task.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use tokio::task::JoinHandle;

use crate::backend::{Backend, BackendPool};
use crate::backpressure::BackpressureController;
use crate::circuit_breaker::CircuitBreaker;
use crate::config::RouterConfig;
use crate::health;
use crate::metrics::RouterMetrics;
use crate::routing;
use crate::server::{self, AppState};

/// Upstream request timeout. Applies to the whole exchange including a streamed body.
pub const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(300);

pub struct App {
    pub router: Router,
    pub state: AppState,
    /// Dropping this handle detaches the task (it keeps running); `main.rs` relies on that.
    pub health_task: JoinHandle<()>,
}

/// Build the axum router and background tasks from a parsed config.
/// Must be called from inside a tokio runtime (it spawns the health checker).
pub fn build_app(config: RouterConfig) -> Result<App, Box<dyn std::error::Error>> {
    let backends: Vec<Arc<Backend>> = config
        .backends
        .iter()
        .enumerate()
        .map(|(i, bc)| {
            Arc::new(Backend::new(
                i,
                bc.url.clone(),
                bc.weight,
                CircuitBreaker::from_config(&config.circuit_breaker),
            ))
        })
        .collect();
    let pool = Arc::new(BackendPool::new(backends));

    let strategy = routing::create_strategy(
        &config.routing.strategy,
        config.routing.prefix_cache.prefix_token_count,
        pool.len(),
    );

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(100)
        .timeout(UPSTREAM_TIMEOUT)
        .build()?;

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

    let router = server::build_router(state.clone());
    Ok(App {
        router,
        state,
        health_task,
    })
}
