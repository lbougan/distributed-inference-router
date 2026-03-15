use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;
use std::sync::Arc;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RequestLabels {
    pub strategy: String,
    pub backend: String,
    pub status: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BackendLabels {
    pub backend: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RejectLabels {
    pub reason: String,
}

#[derive(Clone)]
pub struct RouterMetrics {
    pub requests_total: Family<RequestLabels, Counter>,
    pub request_duration: Family<BackendLabels, Histogram>,
    pub active_connections: Family<BackendLabels, Gauge>,
    pub circuit_breaker_state: Family<BackendLabels, Gauge>,
    pub backend_health: Family<BackendLabels, Gauge>,
    pub rejected_requests: Family<RejectLabels, Counter>,
    pub registry: Arc<std::sync::Mutex<Registry>>,
}

impl RouterMetrics {
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let requests_total = Family::<RequestLabels, Counter>::default();
        registry.register(
            "router_requests_total",
            "Total requests routed",
            requests_total.clone(),
        );

        let request_duration =
            Family::<BackendLabels, Histogram>::new_with_constructor(|| {
                Histogram::new(exponential_buckets(0.001, 2.0, 15))
            });
        registry.register(
            "router_request_duration_seconds",
            "Request duration in seconds",
            request_duration.clone(),
        );

        let active_connections = Family::<BackendLabels, Gauge>::default();
        registry.register(
            "router_active_connections",
            "Active connections per backend",
            active_connections.clone(),
        );

        let circuit_breaker_state = Family::<BackendLabels, Gauge>::default();
        registry.register(
            "router_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half_open, 2=open)",
            circuit_breaker_state.clone(),
        );

        let backend_health = Family::<BackendLabels, Gauge>::default();
        registry.register(
            "router_backend_health",
            "Backend health (0=unhealthy, 1=healthy)",
            backend_health.clone(),
        );

        let rejected_requests = Family::<RejectLabels, Counter>::default();
        registry.register(
            "router_rejected_requests_total",
            "Rejected requests by reason",
            rejected_requests.clone(),
        );

        Self {
            requests_total,
            request_duration,
            active_connections,
            circuit_breaker_state,
            backend_health,
            rejected_requests,
            registry: Arc::new(std::sync::Mutex::new(registry)),
        }
    }

    pub fn encode(&self) -> String {
        let registry = self.registry.lock().unwrap();
        let mut buf = String::new();
        encode(&mut buf, &registry).unwrap();
        buf
    }
}
