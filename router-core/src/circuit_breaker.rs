use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitState {
    pub fn as_u64(&self) -> u64 {
        match self {
            CircuitState::Closed => 0,
            CircuitState::HalfOpen => 1,
            CircuitState::Open => 2,
        }
    }
}

pub struct CircuitBreaker {
    state: Mutex<CircuitState>,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    total_in_window: AtomicU64,
    last_failure_time: Mutex<Option<Instant>>,
    window_start: Mutex<Instant>,

    failure_rate_threshold: f64,
    min_requests: u64,
    window_duration: Duration,
    open_duration: Duration,
}

impl CircuitBreaker {
    pub fn new(
        failure_rate_threshold: f64,
        min_requests: u64,
        window_secs: u64,
        open_duration_secs: u64,
    ) -> Self {
        Self {
            state: Mutex::new(CircuitState::Closed),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            total_in_window: AtomicU64::new(0),
            last_failure_time: Mutex::new(None),
            window_start: Mutex::new(Instant::now()),
            failure_rate_threshold,
            min_requests,
            window_duration: Duration::from_secs(window_secs),
            open_duration: Duration::from_secs(open_duration_secs),
        }
    }

    pub fn state(&self) -> CircuitState {
        let mut state = self.state.lock().unwrap();
        if *state == CircuitState::Open {
            if let Some(last) = *self.last_failure_time.lock().unwrap() {
                if last.elapsed() >= self.open_duration {
                    *state = CircuitState::HalfOpen;
                }
            }
        }
        *state
    }

    pub fn allows_request(&self) -> bool {
        match self.state() {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => false,
        }
    }

    pub fn record_success(&self) {
        self.maybe_reset_window();
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.total_in_window.fetch_add(1, Ordering::Relaxed);

        let mut state = self.state.lock().unwrap();
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
            self.success_count.store(1, Ordering::Relaxed);
            self.total_in_window.store(1, Ordering::Relaxed);
        }
    }

    pub fn record_failure(&self) {
        self.maybe_reset_window();
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        self.total_in_window.fetch_add(1, Ordering::Relaxed);
        *self.last_failure_time.lock().unwrap() = Some(Instant::now());

        let mut state = self.state.lock().unwrap();
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Open;
            return;
        }

        let total = self.total_in_window.load(Ordering::Relaxed);
        let failures = self.failure_count.load(Ordering::Relaxed);
        if total >= self.min_requests {
            let rate = failures as f64 / total as f64;
            if rate >= self.failure_rate_threshold {
                *state = CircuitState::Open;
            }
        }
    }

    fn maybe_reset_window(&self) {
        let mut start = self.window_start.lock().unwrap();
        if start.elapsed() >= self.window_duration {
            *start = Instant::now();
            self.failure_count.store(0, Ordering::Relaxed);
            self.success_count.store(0, Ordering::Relaxed);
            self.total_in_window.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_closed() {
        let cb = CircuitBreaker::new(0.5, 3, 60, 1);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allows_request());
    }

    #[test]
    fn opens_after_threshold() {
        let cb = CircuitBreaker::new(0.5, 4, 60, 30);
        cb.record_success();
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allows_request());
    }

    #[test]
    fn stays_closed_below_min_requests() {
        let cb = CircuitBreaker::new(0.5, 10, 60, 30);
        for _ in 0..5 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn transitions_to_half_open_after_timeout() {
        let cb = CircuitBreaker::new(0.5, 2, 60, 1);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn half_open_success_closes() {
        let cb = CircuitBreaker::new(0.5, 2, 60, 1);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn half_open_failure_reopens() {
        let cb = CircuitBreaker::new(0.5, 2, 60, 1);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }
}
