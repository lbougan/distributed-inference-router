use async_trait::async_trait;
use dashmap::DashMap;
use rand::Rng;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{BackendId, BackendPool};

pub struct LatencyAware {
    alpha: f64,
    ewma_latencies: DashMap<BackendId, f64>,
}

impl LatencyAware {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ewma_latencies: DashMap::new(),
        }
    }
}

#[async_trait]
impl RoutingStrategy for LatencyAware {
    async fn select_backend(
        &self,
        _request: &IncomingRequest,
        pool: &BackendPool,
    ) -> Option<BackendId> {
        let healthy = pool.healthy_backends();
        if healthy.is_empty() {
            return None;
        }

        let latencies: Vec<(BackendId, f64)> = healthy
            .iter()
            .map(|b| {
                let lat = b.ewma_latency_us.load(Ordering::Relaxed) as f64;
                let lat = if lat == 0.0 { 1.0 } else { lat };
                (b.id, lat)
            })
            .collect();

        let min_lat = latencies.iter().map(|(_, l)| *l).fold(f64::MAX, f64::min);
        let max_lat = latencies.iter().map(|(_, l)| *l).fold(f64::MIN, f64::max);

        // If all latencies within 10%, fall back to round-robin-like pick (first)
        if max_lat <= min_lat * 1.1 {
            return Some(healthy[0].id);
        }

        // Weighted random: inverse latency -> lower latency = higher weight
        let weights: Vec<(BackendId, f64)> = latencies
            .iter()
            .map(|(id, lat)| (*id, 1.0 / lat))
            .collect();

        let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
        let mut rng = rand::thread_rng();
        let mut pick = rng.gen::<f64>() * total_weight;

        for (id, w) in &weights {
            pick -= w;
            if pick <= 0.0 {
                return Some(*id);
            }
        }

        Some(weights.last().unwrap().0)
    }

    fn on_request_complete(&self, backend: BackendId, latency: Duration, _success: bool) {
        let new_us = latency.as_micros() as f64;
        let mut entry = self.ewma_latencies.entry(backend).or_insert(new_us);
        let old = *entry.value();
        *entry.value_mut() = self.alpha * new_us + (1.0 - self.alpha) * old;
    }
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
    async fn prefers_lower_latency_backend() {
        let strategy = LatencyAware::new(0.3);
        let pool = make_pool(3);

        // Backend 0: high latency, Backend 1: low latency, Backend 2: medium
        pool.backends[0].ewma_latency_us.store(10000, Ordering::Relaxed);
        pool.backends[1].ewma_latency_us.store(100, Ordering::Relaxed);
        pool.backends[2].ewma_latency_us.store(5000, Ordering::Relaxed);

        let req = IncomingRequest { path: "/v1/completions".into(), body: None };

        let mut counts = [0u32; 3];
        for _ in 0..1000 {
            let id = strategy.select_backend(&req, &pool).await.unwrap();
            counts[id] += 1;
        }

        // Backend 1 (lowest latency) should be picked most often
        assert!(counts[1] > counts[0], "low-latency backend should be preferred");
        assert!(counts[1] > counts[2], "low-latency backend should be preferred");
    }
}
