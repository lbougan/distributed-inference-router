//! Admission control: a bounded number of in-flight requests, held as RAII permits.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct BackpressureController {
    in_flight: AtomicUsize,
    max_in_flight: usize,
}

/// A slot in the in-flight budget. Released when dropped, so every early return
/// and every finished stream releases exactly once.
pub struct Permit {
    controller: Arc<BackpressureController>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        let result =
            self.controller
                .in_flight
                .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |v| v.checked_sub(1));
        debug_assert!(result.is_ok(), "permit released below zero");
    }
}

impl BackpressureController {
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            max_in_flight,
        }
    }

    /// Try to take a slot. `None` means the router is at capacity (reply 429).
    pub fn try_acquire(self: &Arc<Self>) -> Option<Permit> {
        let acquired = self
            .in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |v| {
                (v < self.max_in_flight).then_some(v + 1)
            })
            .is_ok();
        acquired.then(|| Permit {
            controller: Arc::clone(self),
        })
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_when_full() {
        let bp = Arc::new(BackpressureController::new(2));
        let _a = bp.try_acquire().expect("slot 1");
        let _b = bp.try_acquire().expect("slot 2");
        assert!(bp.try_acquire().is_none());
    }

    #[test]
    fn dropping_permit_frees_slot() {
        let bp = Arc::new(BackpressureController::new(1));
        let a = bp.try_acquire().expect("slot");
        assert!(bp.try_acquire().is_none());
        drop(a);
        assert!(bp.try_acquire().is_some());
    }

    #[test]
    fn tracks_in_flight() {
        let bp = Arc::new(BackpressureController::new(10));
        assert_eq!(bp.in_flight(), 0);
        let a = bp.try_acquire().unwrap();
        let _b = bp.try_acquire().unwrap();
        assert_eq!(bp.in_flight(), 2);
        drop(a);
        assert_eq!(bp.in_flight(), 1);
    }
}
