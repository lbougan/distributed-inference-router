use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::circuit_breaker::CircuitBreaker;

pub type BackendId = usize;

#[derive(Debug)]
pub struct Backend {
    pub id: BackendId,
    pub url: String,
    pub weight: u32,
    pub breaker: CircuitBreaker,
    pub healthy: AtomicBool,
    pub active_connections: AtomicUsize,
    pub total_requests: AtomicU64,
    pub total_failures: AtomicU64,
    /// EWMA latency in microseconds
    pub ewma_latency_us: AtomicU64,
    /// Queue depth scraped from backend /metrics
    pub queue_depth: AtomicU64,
    /// KV cache usage percentage (0-100 scaled to 0-10000 for 2 decimal precision)
    pub kv_cache_usage_bp: AtomicU64,
    /// Consecutive health check failures
    pub consecutive_health_failures: AtomicU64,
    /// Consecutive health check successes
    pub consecutive_health_successes: AtomicU64,
}

impl Backend {
    pub fn new(id: BackendId, url: String, weight: u32, breaker: CircuitBreaker) -> Self {
        Self {
            id,
            url,
            weight,
            breaker,
            healthy: AtomicBool::new(true),
            active_connections: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            ewma_latency_us: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
            kv_cache_usage_bp: AtomicU64::new(0),
            consecutive_health_failures: AtomicU64::new(0),
            consecutive_health_successes: AtomicU64::new(0),
        }
    }

    /// Healthy per active checks and the breaker lets traffic through.
    pub fn is_routable(&self) -> bool {
        self.is_healthy() && self.breaker.is_routable()
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn mark_healthy(&self) {
        self.healthy.store(true, Ordering::Relaxed);
        self.consecutive_health_failures.store(0, Ordering::Relaxed);
    }

    pub fn mark_unhealthy(&self) {
        self.healthy.store(false, Ordering::Relaxed);
        self.consecutive_health_successes
            .store(0, Ordering::Relaxed);
    }

    pub fn record_request_start(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_end(&self, success: bool) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
        if !success {
            self.total_failures.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn update_ewma_latency(&self, latency_us: u64, alpha: f64) {
        let old = self.ewma_latency_us.load(Ordering::Relaxed);
        let new_val = if old == 0 {
            latency_us
        } else {
            let old_f = old as f64;
            let new_f = latency_us as f64;
            (alpha * new_f + (1.0 - alpha) * old_f) as u64
        };
        self.ewma_latency_us.store(new_val, Ordering::Relaxed);
    }
}

pub struct BackendPool {
    pub backends: Vec<Arc<Backend>>,
    /// Backends reporting more waiting requests than this are not routable.
    pub max_queue_depth: u64,
}

impl BackendPool {
    pub fn new(backends: Vec<Arc<Backend>>, max_queue_depth: u64) -> Self {
        Self {
            backends,
            max_queue_depth,
        }
    }

    pub fn routable(&self) -> Vec<Arc<Backend>> {
        self.backends
            .iter()
            .filter(|b| {
                b.is_routable() && b.queue_depth.load(Ordering::Relaxed) <= self.max_queue_depth
            })
            .cloned()
            .collect()
    }

    pub fn get(&self, id: BackendId) -> Option<&Arc<Backend>> {
        self.backends.get(id)
    }

    pub fn len(&self) -> usize {
        self.backends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(id: usize) -> Arc<Backend> {
        Arc::new(Backend::new(
            id,
            format!("http://b{id}"),
            1,
            CircuitBreaker::default(),
        ))
    }

    #[test]
    fn routable_excludes_unhealthy_and_open_circuit() {
        let pool = BackendPool::new(vec![backend(0), backend(1), backend(2)], 50);
        pool.backends[1].mark_unhealthy();
        let open = CircuitBreaker::new(0.5, 1, 60, 30);
        open.record_failure(crate::circuit_breaker::Admission::Normal);
        let b2 = Arc::new(Backend::new(2, "http://b2".into(), 1, open));
        let pool = BackendPool::new(
            vec![pool.backends[0].clone(), pool.backends[1].clone(), b2],
            50,
        );
        let ids: Vec<usize> = pool.routable().iter().map(|b| b.id).collect();
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn routable_excludes_backends_over_queue_depth_limit() {
        let pool = BackendPool::new(vec![backend(0), backend(1)], 50);
        pool.backends[1].queue_depth.store(51, Ordering::Relaxed);
        let ids: Vec<usize> = pool.routable().iter().map(|b| b.id).collect();
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn ewma_first_sample_is_taken_as_is() {
        let b = backend(0);
        b.update_ewma_latency(1000, 0.3);
        assert_eq!(b.ewma_latency_us.load(Ordering::Relaxed), 1000);
        b.update_ewma_latency(2000, 0.5);
        assert_eq!(b.ewma_latency_us.load(Ordering::Relaxed), 1500);
    }
}
