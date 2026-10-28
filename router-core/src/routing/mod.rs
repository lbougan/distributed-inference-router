pub mod latency_aware;
pub mod least_connections;
pub mod least_load;
pub mod prefix_cache;
pub mod ring;
pub mod round_robin;

use std::sync::Arc;

use crate::backend::{Backend, BackendId};

/// What a strategy may look at when choosing a backend.
pub struct IncomingRequest {
    pub path: String,
    pub body: Option<serde_json::Value>,
}

/// Pick one of the routable backends. `routable` may be empty; return `None` then.
/// Strategies are pure selectors; accounting is the router's job.
pub trait RoutingStrategy: Send + Sync {
    fn select(&self, request: &IncomingRequest, routable: &[Arc<Backend>]) -> Option<BackendId>;
}

/// Every strategy name `create_strategy` accepts; config validation uses the same list.
pub const STRATEGY_NAMES: [&str; 5] = [
    "round_robin",
    "least_connections",
    "least_load",
    "latency_aware",
    "prefix_cache",
];

pub fn create_strategy(
    name: &str,
    prefix_token_count: usize,
    num_backends: usize,
) -> Result<Box<dyn RoutingStrategy>, String> {
    Ok(match name {
        "round_robin" => Box::new(round_robin::RoundRobin::new()),
        "least_connections" => Box::new(least_connections::LeastConnections),
        "least_load" => Box::new(least_load::LeastLoad),
        "latency_aware" => Box::new(latency_aware::LatencyAware::new()),
        "prefix_cache" => Box::new(prefix_cache::PrefixCacheAware::new(
            prefix_token_count,
            num_backends,
        )),
        other => return Err(format!("unknown routing strategy {other:?}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_strategy_name_can_be_created() {
        for name in STRATEGY_NAMES {
            assert!(create_strategy(name, 256, 3).is_ok(), "{name}");
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::circuit_breaker::CircuitBreaker;

    pub fn backends(n: usize) -> Vec<Arc<Backend>> {
        (0..n)
            .map(|i| {
                Arc::new(Backend::new(
                    i,
                    format!("http://backend-{i}:8000"),
                    1,
                    CircuitBreaker::default(),
                ))
            })
            .collect()
    }

    pub fn req() -> IncomingRequest {
        IncomingRequest {
            path: "/v1/completions".into(),
            body: None,
        }
    }
}
