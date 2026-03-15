mod backend;
mod backpressure;
mod circuit_breaker;
mod config;
mod health;
mod metrics;
mod proxy;
mod routing;
mod server;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use backend::{Backend, BackendPool};
use backpressure::BackpressureController;
use circuit_breaker::CircuitBreaker;
use config::RouterConfig;
use metrics::RouterMetrics;
use server::AppState;

#[derive(Parser)]
#[command(name = "inference-router", about = "Distributed LLM inference request router")]
struct Cli {
    /// Path to the configuration YAML file
    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("inference_router=info".parse()?),
        )
        .json()
        .init();

    let cli = Cli::parse();
    let config = RouterConfig::from_file(&cli.config)?;

    tracing::info!(
        strategy = config.routing.strategy,
        backends = config.backends.len(),
        "Starting inference router"
    );

    let backends: Vec<Arc<Backend>> = config
        .backends
        .iter()
        .enumerate()
        .map(|(i, bc)| Arc::new(Backend::new(i, bc.url.clone(), bc.weight)))
        .collect();

    let pool = Arc::new(BackendPool::new(backends));

    let circuit_breakers: Vec<CircuitBreaker> = (0..pool.len())
        .map(|_| {
            CircuitBreaker::new(
                config.circuit_breaker.failure_rate_threshold,
                config.circuit_breaker.min_requests,
                config.circuit_breaker.window_secs,
                config.circuit_breaker.open_duration_secs,
            )
        })
        .collect();
    let circuit_breakers = Arc::new(circuit_breakers);

    let strategy = routing::create_strategy(
        &config.routing.strategy,
        config.routing.prefix_cache.prefix_token_count,
    );

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(100)
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let backpressure = Arc::new(BackpressureController::new(
        config.backpressure.max_in_flight,
    ));
    let metrics = RouterMetrics::new();

    health::spawn_health_checker(
        pool.clone(),
        circuit_breakers.clone(),
        client.clone(),
        config.health_check.clone(),
    );

    let state = AppState {
        pool,
        strategy: Arc::from(strategy),
        strategy_name: config.routing.strategy.clone(),
        client,
        circuit_breakers,
        backpressure,
        metrics,
    };

    let app = server::build_router(state);
    let addr: SocketAddr =
        format!("{}:{}", config.listen.host, config.listen.port).parse()?;
    tracing::info!(%addr, "Listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
