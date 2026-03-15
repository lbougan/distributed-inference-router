use async_trait::async_trait;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{BackendId, BackendPool};

pub struct LeastConnections;

#[async_trait]
impl RoutingStrategy for LeastConnections {
    async fn select_backend(
        &self,
        _request: &IncomingRequest,
        pool: &BackendPool,
    ) -> Option<BackendId> {
        pool.healthy_backends()
            .into_iter()
            .min_by_key(|b| b.active_connections.load(Ordering::Relaxed))
            .map(|b| b.id)
    }

    fn on_request_complete(&self, _backend: BackendId, _latency: Duration, _success: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use std::sync::Arc;

    fn make_pool(n: usize) -> BackendPool {
        let backends: Vec<Arc<Backend>> = (0..n)
            .map(|i| Arc::new(Backend::new(i, format!("http://backend-{i}:8000"), 1)))
            .collect();
        BackendPool::new(backends)
    }

    #[tokio::test]
    async fn picks_backend_with_fewest_connections() {
        let strategy = LeastConnections;
        let pool = make_pool(3);

        pool.backends[0].active_connections.store(5, Ordering::Relaxed);
        pool.backends[1].active_connections.store(2, Ordering::Relaxed);
        pool.backends[2].active_connections.store(8, Ordering::Relaxed);

        let req = IncomingRequest { path: "/v1/completions".into(), body: None };
        let id = strategy.select_backend(&req, &pool).await.unwrap();
        assert_eq!(id, 1);
    }

    #[tokio::test]
    async fn skips_unhealthy() {
        let strategy = LeastConnections;
        let pool = make_pool(3);

        pool.backends[0].active_connections.store(10, Ordering::Relaxed);
        pool.backends[1].active_connections.store(1, Ordering::Relaxed);
        pool.backends[1].mark_unhealthy();
        pool.backends[2].active_connections.store(5, Ordering::Relaxed);

        let req = IncomingRequest { path: "/v1/completions".into(), body: None };
        let id = strategy.select_backend(&req, &pool).await.unwrap();
        assert_eq!(id, 2);
    }
}
