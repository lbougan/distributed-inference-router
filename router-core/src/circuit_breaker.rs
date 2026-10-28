//! Per-backend circuit breaker.
//!
//! Failure rate is computed over a sliding window of 1-second buckets, so a burst of
//! failures right after a window boundary is judged against the previous seconds too.
//! Half-open admits exactly one probe request; its verdict decides Closed or Open.
//! Every admitted request carries an [`Admission`] token, so verdicts from requests that
//! started before the trip (`Normal`) can never close the circuit or release the probe.

use std::sync::{Mutex, MutexGuard, PoisonError};
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

/// How a request was admitted. Only a `Probe` verdict may move a half-open breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    Normal,
    Probe,
}

#[derive(Debug, Default, Clone, Copy)]
struct Bucket {
    successes: u64,
    failures: u64,
}

#[derive(Debug)]
struct Inner {
    state: CircuitState,
    opened_at: Option<Instant>,
    buckets: Vec<Bucket>,
    head: usize,
    head_start: Instant,
    /// True while the single half-open probe is out.
    probe_in_flight: bool,
}

impl Inner {
    /// Slide the window forward to `now`, zeroing buckets that fell out.
    fn rotate(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.head_start).as_secs();
        if elapsed == 0 {
            return;
        }
        let n = self.buckets.len();
        let steps = (elapsed as usize).min(n);
        for i in 1..=steps {
            self.buckets[(self.head + i) % n] = Bucket::default();
        }
        self.head = (self.head + steps) % n;
        self.head_start += Duration::from_secs(elapsed);
    }

    fn totals(&self) -> (u64, u64) {
        let failures = self.buckets.iter().map(|b| b.failures).sum();
        let successes: u64 = self.buckets.iter().map(|b| b.successes).sum();
        (failures, failures + successes)
    }

    fn clear(&mut self, now: Instant) {
        self.buckets.iter_mut().for_each(|b| *b = Bucket::default());
        self.head = 0;
        self.head_start = now;
    }

    fn refresh(&mut self, now: Instant, open_duration: Duration) {
        if self.state == CircuitState::Open {
            if let Some(opened) = self.opened_at {
                if now.saturating_duration_since(opened) >= open_duration {
                    self.state = CircuitState::HalfOpen;
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct CircuitBreaker {
    inner: Mutex<Inner>,
    failure_rate_threshold: f64,
    min_requests: u64,
    open_duration: Duration,
    /// Added to `Instant::now()`; only tests move it.
    #[cfg(test)]
    clock_offset: Mutex<Duration>,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::from_config(&crate::config::CircuitBreakerConfig::default())
    }
}

impl CircuitBreaker {
    pub fn new(
        failure_rate_threshold: f64,
        min_requests: u64,
        window_secs: u64,
        open_duration_secs: u64,
    ) -> Self {
        let now = Instant::now();
        Self {
            inner: Mutex::new(Inner {
                state: CircuitState::Closed,
                opened_at: None,
                buckets: vec![Bucket::default(); window_secs.max(1) as usize],
                head: 0,
                head_start: now,
                probe_in_flight: false,
            }),
            failure_rate_threshold,
            min_requests: min_requests.max(1),
            open_duration: Duration::from_secs(open_duration_secs),
            #[cfg(test)]
            clock_offset: Mutex::new(Duration::ZERO),
        }
    }

    pub fn from_config(cfg: &crate::config::CircuitBreakerConfig) -> Self {
        Self::new(
            cfg.failure_rate_threshold,
            cfg.min_requests,
            cfg.window_secs,
            cfg.open_duration_secs,
        )
    }

    #[cfg(test)]
    fn now(&self) -> Instant {
        Instant::now() + *self.clock_offset.lock().unwrap()
    }

    #[cfg(not(test))]
    fn now(&self) -> Instant {
        Instant::now()
    }

    /// Accounting calls into the breaker from `Drop`; a poisoned lock must not abort.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Current state. Side effect: moves Open to HalfOpen once `open_duration` has
    /// elapsed (the metrics handler calls this).
    pub fn state(&self) -> CircuitState {
        let now = self.now();
        let mut inner = self.lock();
        inner.refresh(now, self.open_duration);
        inner.state
    }

    /// May this backend be offered to the strategy right now?
    pub fn is_routable(&self) -> bool {
        let now = self.now();
        let mut inner = self.lock();
        inner.refresh(now, self.open_duration);
        match inner.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => !inner.probe_in_flight,
            CircuitState::Open => false,
        }
    }

    /// Claim the right to send one request. In HalfOpen only one caller wins the probe;
    /// the state check and the claim happen under one lock.
    pub fn try_begin(&self) -> Option<Admission> {
        let now = self.now();
        let mut inner = self.lock();
        inner.refresh(now, self.open_duration);
        match inner.state {
            CircuitState::Closed => Some(Admission::Normal),
            CircuitState::Open => None,
            CircuitState::HalfOpen => {
                if inner.probe_in_flight {
                    None
                } else {
                    inner.probe_in_flight = true;
                    Some(Admission::Probe)
                }
            }
        }
    }

    pub fn record_success(&self, adm: Admission) {
        let now = self.now();
        let mut inner = self.lock();
        inner.refresh(now, self.open_duration);
        inner.rotate(now);
        if inner.state == CircuitState::HalfOpen && adm == Admission::Probe {
            inner.state = CircuitState::Closed;
            inner.opened_at = None;
            inner.clear(now);
            inner.probe_in_flight = false;
        }
        // A stale `Normal` verdict is only counted; it never changes state.
        let head = inner.head;
        inner.buckets[head].successes += 1;
    }

    pub fn record_failure(&self, adm: Admission) {
        let now = self.now();
        let mut inner = self.lock();
        inner.refresh(now, self.open_duration);
        inner.rotate(now);
        if inner.state == CircuitState::HalfOpen && adm == Admission::Probe {
            inner.state = CircuitState::Open;
            inner.opened_at = Some(now);
            inner.probe_in_flight = false;
            return;
        }
        let head = inner.head;
        inner.buckets[head].failures += 1;
        // Late failures while Open/HalfOpen are counted but must not extend the open period.
        if inner.state != CircuitState::Closed {
            return;
        }
        let (failures, total) = inner.totals();
        if total >= self.min_requests
            && failures as f64 / total as f64 >= self.failure_rate_threshold
        {
            inner.state = CircuitState::Open;
            inner.opened_at = Some(now);
        }
    }

    /// The request ended without a verdict (client disconnected). A probe is handed
    /// back; a `Normal` request holds nothing.
    pub fn abandon(&self, adm: Admission) {
        if adm != Admission::Probe {
            return;
        }
        let mut inner = self.lock();
        if inner.state == CircuitState::HalfOpen {
            inner.probe_in_flight = false;
        }
    }

    #[cfg(test)]
    pub fn advance(&self, by: Duration) {
        *self.clock_offset.lock().unwrap() += by;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breaker() -> CircuitBreaker {
        // 50 % over >= 4 requests, 60 s window, 30 s open
        CircuitBreaker::new(0.5, 4, 60, 30)
    }

    fn trip(cb: &CircuitBreaker) {
        cb.record_success(Admission::Normal);
        cb.record_failure(Admission::Normal);
        cb.record_failure(Admission::Normal);
        cb.record_failure(Admission::Normal);
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn starts_closed_and_routable() {
        let cb = breaker();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_routable());
        assert!(cb.try_begin().is_some());
    }

    #[test]
    fn opens_at_threshold() {
        let cb = breaker();
        trip(&cb);
        assert!(!cb.is_routable());
        assert!(cb.try_begin().is_none());
    }

    #[test]
    fn stays_closed_below_min_requests() {
        let cb = CircuitBreaker::new(0.5, 10, 60, 30);
        for _ in 0..5 {
            cb.record_failure(Admission::Normal);
        }
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn half_open_after_open_duration() {
        let cb = breaker();
        trip(&cb);
        cb.advance(Duration::from_secs(29));
        assert_eq!(cb.state(), CircuitState::Open);
        cb.advance(Duration::from_secs(2));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.is_routable());
    }

    #[test]
    fn half_open_admits_exactly_one_probe() {
        let cb = breaker();
        trip(&cb);
        cb.advance(Duration::from_secs(31));
        let adm = cb.try_begin().unwrap();
        assert_eq!(adm, Admission::Probe);
        assert!(cb.try_begin().is_none(), "second caller does not");
        assert!(
            !cb.is_routable(),
            "not offered while the probe is in flight"
        );
        cb.abandon(adm);
        assert!(cb.is_routable());
        assert_eq!(cb.try_begin(), Some(Admission::Probe));
    }

    #[test]
    fn probe_success_closes_and_resets_window() {
        let cb = breaker();
        trip(&cb);
        cb.advance(Duration::from_secs(31));
        let adm = cb.try_begin().unwrap();
        assert_eq!(adm, Admission::Probe);
        cb.record_success(adm);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.try_begin(), Some(Admission::Normal));
        // Old failures are forgotten: one new failure must not re-open.
        cb.record_failure(Admission::Normal);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn probe_failure_reopens_for_another_open_duration() {
        let cb = breaker();
        trip(&cb);
        cb.advance(Duration::from_secs(31));
        let adm = cb.try_begin().unwrap();
        assert_eq!(adm, Admission::Probe);
        cb.record_failure(adm);
        assert_eq!(cb.state(), CircuitState::Open);
        cb.advance(Duration::from_secs(29));
        assert_eq!(cb.state(), CircuitState::Open);
        cb.advance(Duration::from_secs(2));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn sliding_window_forgets_old_failures() {
        let cb = breaker();
        cb.record_failure(Admission::Normal);
        cb.record_failure(Admission::Normal);
        cb.record_failure(Admission::Normal);
        cb.advance(Duration::from_secs(61));
        cb.record_failure(Admission::Normal);
        assert_eq!(
            cb.state(),
            CircuitState::Closed,
            "only 1 request in the last 60 s"
        );
    }

    #[test]
    fn sliding_window_keeps_recent_failures() {
        let cb = breaker();
        cb.record_failure(Admission::Normal);
        cb.record_failure(Admission::Normal);
        cb.advance(Duration::from_secs(30));
        cb.record_success(Admission::Normal);
        cb.record_failure(Admission::Normal);
        assert_eq!(
            cb.state(),
            CircuitState::Open,
            "3 of 4 in the last 60 s failed"
        );
    }

    #[test]
    fn abandoned_probe_is_handed_back_without_a_verdict() {
        let cb = breaker();
        trip(&cb);
        cb.advance(Duration::from_secs(31));
        let adm = cb.try_begin().unwrap();
        assert_eq!(adm, Admission::Probe);
        cb.abandon(adm);
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen,
            "no verdict: still half-open"
        );
        assert_eq!(
            cb.try_begin(),
            Some(Admission::Probe),
            "the probe is available again"
        );
    }

    #[test]
    fn stale_success_during_half_open_does_not_close() {
        let cb = breaker();
        trip(&cb);
        cb.advance(Duration::from_secs(31));
        cb.record_success(Admission::Normal);
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert_eq!(cb.try_begin(), Some(Admission::Probe));
    }

    #[test]
    fn stale_disconnect_during_half_open_keeps_the_probe() {
        let cb = breaker();
        trip(&cb);
        cb.advance(Duration::from_secs(31));
        assert_eq!(cb.try_begin(), Some(Admission::Probe));
        cb.abandon(Admission::Normal);
        assert!(!cb.is_routable());
        assert!(cb.try_begin().is_none());
        cb.abandon(Admission::Probe);
        assert_eq!(cb.try_begin(), Some(Admission::Probe));
    }

    #[test]
    fn late_failure_while_open_does_not_extend_the_open_period() {
        let cb = breaker();
        trip(&cb);
        cb.advance(Duration::from_secs(20));
        cb.record_failure(Admission::Normal);
        cb.advance(Duration::from_secs(11));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn window_keeps_oldest_bucket_at_n_minus_1_and_drops_it_at_n() {
        let cb = CircuitBreaker::new(0.5, 4, 10, 30);
        for _ in 0..3 {
            cb.record_failure(Admission::Normal);
        }
        cb.advance(Duration::from_secs(9));
        cb.record_failure(Admission::Normal);
        assert_eq!(cb.state(), CircuitState::Open);

        let cb = CircuitBreaker::new(0.5, 4, 10, 30);
        for _ in 0..3 {
            cb.record_failure(Admission::Normal);
        }
        cb.advance(Duration::from_secs(10));
        cb.record_failure(Admission::Normal);
        assert_eq!(cb.state(), CircuitState::Closed);
    }
}
