use std::sync::atomic::{AtomicUsize, Ordering};

pub struct BackpressureController {
    in_flight: AtomicUsize,
    max_in_flight: usize,
}

impl BackpressureController {
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            max_in_flight,
        }
    }

    /// Try to acquire a slot. Returns false if at capacity.
    pub fn try_acquire(&self) -> bool {
        loop {
            let current = self.in_flight.load(Ordering::Relaxed);
            if current >= self.max_in_flight {
                return false;
            }
            if self
                .in_flight
                .compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    pub fn release(&self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn current(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_when_full() {
        let bp = BackpressureController::new(2);
        assert!(bp.try_acquire());
        assert!(bp.try_acquire());
        assert!(!bp.try_acquire());
    }

    #[test]
    fn release_frees_slot() {
        let bp = BackpressureController::new(1);
        assert!(bp.try_acquire());
        assert!(!bp.try_acquire());
        bp.release();
        assert!(bp.try_acquire());
    }

    #[test]
    fn tracks_current() {
        let bp = BackpressureController::new(10);
        assert_eq!(bp.current(), 0);
        bp.try_acquire();
        bp.try_acquire();
        assert_eq!(bp.current(), 2);
        bp.release();
        assert_eq!(bp.current(), 1);
    }
}
