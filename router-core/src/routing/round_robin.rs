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
        if routable.is_empty() {
            return None;
        }
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % routable.len();
        Some(routable[idx].id)
    }
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
    fn returns_none_when_nothing_routable() {
        assert!(RoundRobin::new().select(&req(), &[]).is_none());
    }
}
