use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rand::Rng;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{Backend, BackendId};

/// Latencies within this ratio of each other are treated as equal.
const FLAT_RATIO: f64 = 1.1;

/// Weighted random selection, weight = configured weight / EWMA latency. Falls back to round-robin
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
        let raw: Vec<u64> = routable
            .iter()
            .map(|b| b.ewma_latency_us.load(Ordering::Relaxed))
            .collect();
        // Backends without a sample get the median sampled latency, not a tiny one.
        let mut sampled: Vec<f64> = raw.iter().filter(|l| **l > 0).map(|l| *l as f64).collect();
        sampled.sort_by(f64::total_cmp);
        let median = if sampled.is_empty() {
            1.0
        } else {
            sampled[sampled.len() / 2]
        };
        let latencies: Vec<f64> = raw
            .iter()
            .map(|l| if *l > 0 { *l as f64 } else { median })
            .collect();
        let min = latencies.iter().cloned().fold(f64::MAX, f64::min);
        let max = latencies.iter().cloned().fold(f64::MIN, f64::max);

        if max <= min * FLAT_RATIO {
            let idx = self.fallback_counter.fetch_add(1, Ordering::Relaxed) % routable.len();
            return Some(routable[idx].id);
        }

        let weights: Vec<f64> = routable
            .iter()
            .zip(&latencies)
            .map(|(b, l)| f64::from(b.weight.max(1)) / l)
            .collect();
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

    #[test]
    fn config_weight_multiplies_the_inverse_latency() {
        let strategy = LatencyAware::new();
        let slow_but_big = Arc::new(Backend::new(
            0,
            "http://b0".into(),
            4,
            crate::circuit_breaker::CircuitBreaker::default(),
        ));
        let fast = Arc::new(Backend::new(
            1,
            "http://b1".into(),
            1,
            crate::circuit_breaker::CircuitBreaker::default(),
        ));
        slow_but_big.ewma_latency_us.store(400, Ordering::Relaxed);
        fast.ewma_latency_us.store(100, Ordering::Relaxed);
        let b = vec![slow_but_big, fast];
        let n = 20_000;
        let hits0 = (0..n)
            .filter(|_| strategy.select(&req(), &b).unwrap() == 0)
            .count();
        // weights 4/400 vs 1/100 -> equal shares
        assert!(((hits0 as f64 / n as f64) - 0.5).abs() < 0.03, "{hits0}");
    }

    #[test]
    fn unsampled_backend_gets_the_median_latency_not_a_tiny_one() {
        let strategy = LatencyAware::new();
        let b = backends(3);
        b[0].ewma_latency_us.store(100_000, Ordering::Relaxed);
        b[1].ewma_latency_us.store(100_000, Ordering::Relaxed);
        // b[2] has no sample yet
        let n = 6_000;
        let hits2 = (0..n)
            .filter(|_| strategy.select(&req(), &b).unwrap() == 2)
            .count();
        let share = hits2 as f64 / n as f64;
        assert!((share - 1.0 / 3.0).abs() < 0.05, "unsampled share {share}");
    }
}
