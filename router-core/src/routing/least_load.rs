use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{Backend, BackendId};

/// Like least-connections, but also counts the requests the backend itself reports as
/// waiting (`vllm:num_requests_waiting`), which the router cannot see from its own counters
/// when several routers or direct clients share a replica.
pub struct LeastLoad;

fn load_of(b: &Backend) -> u64 {
    b.active_connections.load(Ordering::Relaxed) as u64 + b.queue_depth.load(Ordering::Relaxed)
}

impl RoutingStrategy for LeastLoad {
    fn select(&self, _request: &IncomingRequest, routable: &[Arc<Backend>]) -> Option<BackendId> {
        routable.iter().min_by_key(|b| load_of(b)).map(|b| b.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::test_support::{backends, req};

    #[test]
    fn queue_depth_counts_as_load() {
        let b = backends(2);
        b[0].active_connections.store(2, Ordering::Relaxed);
        b[0].queue_depth.store(10, Ordering::Relaxed);
        b[1].active_connections.store(5, Ordering::Relaxed);
        assert_eq!(LeastLoad.select(&req(), &b), Some(1));
    }

    #[test]
    fn ties_go_to_the_first() {
        let b = backends(3);
        assert_eq!(LeastLoad.select(&req(), &b), Some(0));
    }
}
