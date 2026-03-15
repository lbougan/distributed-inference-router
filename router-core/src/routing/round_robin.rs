use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{BackendId, BackendPool};

pub struct RoundRobin {
    counter: AtomicUsize,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl RoutingStrategy for RoundRobin {
    async fn select_backend(
        &self,
        _request: &IncomingRequest,
        pool: &BackendPool,
    ) -> Option<BackendId> {
        let healthy = pool.healthy_backends();
        if healthy.is_empty() {
            return None;
        }
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % healthy.len();
        Some(healthy[idx].id)
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
    async fn cycles_through_backends() {
        let strategy = RoundRobin::new();
        let pool = make_pool(3);
        let req = IncomingRequest { path: "/v1/completions".into(), body: None };

        let mut ids = Vec::new();
        for _ in 0..9 {
            ids.push(strategy.select_backend(&req, &pool).await.unwrap());
        }
        assert_eq!(ids, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }

    #[tokio::test]
    async fn skips_unhealthy() {
        let strategy = RoundRobin::new();
        let pool = make_pool(3);
        pool.backends[1].mark_unhealthy();

        let req = IncomingRequest { path: "/v1/completions".into(), body: None };
        let mut ids = Vec::new();
        for _ in 0..4 {
            ids.push(strategy.select_backend(&req, &pool).await.unwrap());
        }
        assert_eq!(ids, vec![0, 2, 0, 2]);
    }

    #[tokio::test]
    async fn returns_none_when_all_unhealthy() {
        let strategy = RoundRobin::new();
        let pool = make_pool(2);
        pool.backends[0].mark_unhealthy();
        pool.backends[1].mark_unhealthy();

        let req = IncomingRequest { path: "/v1/completions".into(), body: None };
        assert!(strategy.select_backend(&req, &pool).await.is_none());
    }
}
