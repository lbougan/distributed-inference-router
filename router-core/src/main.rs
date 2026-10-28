use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use inference_router::app::build_app;
use inference_router::config::RouterConfig;

#[derive(Parser)]
#[command(
    name = "inference-router",
    about = "Distributed LLM inference request router"
)]
struct Cli {
    /// Path to the configuration YAML file
    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("inference_router=info".parse()?),
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

    let addr = config.listen_addr()?;
    let app = build_app(config)?;
    tracing::info!(%addr, "Listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app.router).await?;
    Ok(())
}
