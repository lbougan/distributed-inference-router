use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{Backend, BackendId};

pub struct LeastConnections;

impl RoutingStrategy for LeastConnections {
    fn select(&self, _request: &IncomingRequest, routable: &[Arc<Backend>]) -> Option<BackendId> {
        routable
            .iter()
            .min_by_key(|b| b.active_connections.load(Ordering::Relaxed))
            .map(|b| b.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::test_support::{backends, req};

    #[test]
    fn picks_backend_with_fewest_connections() {
        let b = backends(3);
        b[0].active_connections.store(5, Ordering::Relaxed);
        b[1].active_connections.store(2, Ordering::Relaxed);
        b[2].active_connections.store(8, Ordering::Relaxed);
        assert_eq!(LeastConnections.select(&req(), &b), Some(1));
    }
}
