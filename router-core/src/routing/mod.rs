pub mod round_robin;
pub mod least_connections;
pub mod latency_aware;
pub mod prefix_cache;

use async_trait::async_trait;
use std::time::Duration;

use crate::backend::{BackendId, BackendPool};

/// Minimal representation of an incoming request for routing decisions.
#[allow(dead_code)]
pub struct IncomingRequest {
    pub path: String,
    pub body: Option<serde_json::Value>,
}

#[async_trait]
pub trait RoutingStrategy: Send + Sync {
    /// Select a healthy backend for the given request.
    async fn select_backend(
        &self,
        request: &IncomingRequest,
        pool: &BackendPool,
    ) -> Option<BackendId>;

    /// Notify the strategy that a request completed so it can update internal state.
    fn on_request_complete(&self, backend: BackendId, latency: Duration, success: bool);
}

pub fn create_strategy(name: &str, prefix_token_count: usize) -> Box<dyn RoutingStrategy> {
    match name {
        "round_robin" => Box::new(round_robin::RoundRobin::new()),
        "least_connections" => Box::new(least_connections::LeastConnections),
        "latency_aware" => Box::new(latency_aware::LatencyAware::new(0.3)),
        "prefix_cache" => Box::new(prefix_cache::PrefixCacheAware::new(prefix_token_count)),
        _ => {
            tracing::warn!(strategy = name, "Unknown strategy, falling back to round_robin");
            Box::new(round_robin::RoundRobin::new())
        }
    }
}
