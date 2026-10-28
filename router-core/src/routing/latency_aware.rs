use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rand::Rng;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{Backend, BackendId};

/// Latencies within this ratio of each other are treated as equal.
const FLAT_RATIO: f64 = 1.1;

/// Weighted random selection, weight = 1 / EWMA latency. Falls back to round-robin
/// while latencies are flat so the first backend is not pinned.
pub struct LatencyAware {
    fallback_counter: AtomicUsize,
}

impl LatencyAware {
    pub fn new() -> Self {
        Self {
            fallback_counter: AtomicUsize::new(0),
        }
    }
}

impl Default for LatencyAware {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingStrategy for LatencyAware {
    fn select(&self, _request: &IncomingRequest, routable: &[Arc<Backend>]) -> Option<BackendId> {
        if routable.is_empty() {
            return None;
        }
        let latencies: Vec<f64> = routable
            .iter()
            .map(|b| (b.ewma_latency_us.load(Ordering::Relaxed) as f64).max(1.0))
            .collect();
        let min = latencies.iter().cloned().fold(f64::MAX, f64::min);
        let max = latencies.iter().cloned().fold(f64::MIN, f64::max);

        if max <= min * FLAT_RATIO {
            let idx = self.fallback_counter.fetch_add(1, Ordering::Relaxed) % routable.len();
            return Some(routable[idx].id);
        }

        let weights: Vec<f64> = latencies.iter().map(|l| 1.0 / l).collect();
        let total: f64 = weights.iter().sum();
        let mut pick = rand::thread_rng().gen::<f64>() * total;
        for (b, w) in routable.iter().zip(&weights) {
            pick -= w;
            if pick <= 0.0 {
                return Some(b.id);
            }
        }
        routable.last().map(|b| b.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::test_support::{backends, req};

    #[test]
    fn shares_follow_inverse_latency() {
        let strategy = LatencyAware::new();
        let b = backends(3);
        b[0].ewma_latency_us.store(100, Ordering::Relaxed);
        b[1].ewma_latency_us.store(200, Ordering::Relaxed);
        b[2].ewma_latency_us.store(400, Ordering::Relaxed);
        let n = 20_000;
        let mut counts = [0usize; 3];
        for _ in 0..n {
            counts[strategy.select(&req(), &b).unwrap()] += 1;
        }
        // weights 4:2:1 -> expected shares 0.571, 0.286, 0.143
        let share = |i: usize| counts[i] as f64 / n as f64;
        assert!((share(0) - 4.0 / 7.0).abs() < 0.03, "{counts:?}");
        assert!((share(1) - 2.0 / 7.0).abs() < 0.03, "{counts:?}");
        assert!((share(2) - 1.0 / 7.0).abs() < 0.03, "{counts:?}");
    }

    #[test]
    fn flat_latencies_round_robin_instead_of_pinning_first() {
        let strategy = LatencyAware::new();
        let b = backends(3);
        let ids: Vec<usize> = (0..6)
            .map(|_| strategy.select(&req(), &b).unwrap())
            .collect();
        assert_eq!(ids, vec![0, 1, 2, 0, 1, 2]);
    }
}
