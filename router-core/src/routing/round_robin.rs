use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{Backend, BackendId};

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

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingStrategy for RoundRobin {
    fn select(&self, _request: &IncomingRequest, routable: &[Arc<Backend>]) -> Option<BackendId> {
        let total = total_weight(routable);
        if total == 0 {
            return None;
        }
        let slot = (self.counter.fetch_add(1, Ordering::Relaxed) as u64) % total;
        weighted_pick(slot, routable)
    }
}

pub(crate) fn total_weight(routable: &[Arc<Backend>]) -> u64 {
    routable.iter().map(|b| u64::from(b.weight.max(1))).sum()
}

/// Map a slot in `0..total_weight` to the backend owning it (each gets `weight` slots).
pub(crate) fn weighted_pick(mut slot: u64, routable: &[Arc<Backend>]) -> Option<BackendId> {
    for b in routable {
        let w = u64::from(b.weight.max(1));
        if slot < w {
            return Some(b.id);
        }
        slot -= w;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::test_support::{backends, req};

    #[test]
    fn cycles_through_backends() {
        let strategy = RoundRobin::new();
        let b = backends(3);
        let ids: Vec<usize> = (0..9)
            .map(|_| strategy.select(&req(), &b).unwrap())
            .collect();
        assert_eq!(ids, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn weights_change_the_share() {
        let strategy = RoundRobin::new();
        let b = backends(2);
        let b = vec![
            Arc::new(Backend::new(
                0,
                b[0].url.clone(),
                1,
                crate::circuit_breaker::CircuitBreaker::default(),
            )),
            Arc::new(Backend::new(
                1,
                b[1].url.clone(),
                3,
                crate::circuit_breaker::CircuitBreaker::default(),
            )),
        ];
        let ids: Vec<usize> = (0..8)
            .map(|_| strategy.select(&req(), &b).unwrap())
            .collect();
        assert_eq!(ids.iter().filter(|i| **i == 0).count(), 2);
        assert_eq!(ids.iter().filter(|i| **i == 1).count(), 6);
    }

    #[test]
    fn returns_none_when_nothing_routable() {
        assert!(RoundRobin::new().select(&req(), &[]).is_none());
    }
}
