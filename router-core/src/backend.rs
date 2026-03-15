use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

pub type BackendId = usize;

#[derive(Debug)]
#[allow(dead_code)]
pub struct Backend {
    pub id: BackendId,
    pub url: String,
    pub weight: u32,
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
    pub fn new(id: BackendId, url: String, weight: u32) -> Self {
        Self {
            id,
            url,
            weight,
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

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn mark_healthy(&self) {
        self.healthy.store(true, Ordering::Relaxed);
        self.consecutive_health_failures.store(0, Ordering::Relaxed);
    }

    pub fn mark_unhealthy(&self) {
        self.healthy.store(false, Ordering::Relaxed);
        self.consecutive_health_successes.store(0, Ordering::Relaxed);
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
}

impl BackendPool {
    pub fn new(backends: Vec<Arc<Backend>>) -> Self {
        Self { backends }
    }

    pub fn healthy_backends(&self) -> Vec<&Arc<Backend>> {
        self.backends.iter().filter(|b| b.is_healthy()).collect()
    }

    pub fn get(&self, id: BackendId) -> Option<&Arc<Backend>> {
        self.backends.get(id)
    }

    pub fn len(&self) -> usize {
        self.backends.len()
    }
}
