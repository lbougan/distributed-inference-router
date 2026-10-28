//! Per-request accounting that lives until the response body has been fully sent.
//!
//! The router cannot know a request is over when the backend's headers arrive: for
//! `stream: true` the body is produced for seconds afterwards. `RequestAccounting` is a
//! guard that records the outcome when dropped, and `AccountedStream` keeps that guard alive
//! until the last SSE chunk (or the client goes away).

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::Bytes;
use axum::http::StatusCode;
use futures_util::Stream;

use crate::backend::Backend;
use crate::backpressure::Permit;
use crate::circuit_breaker::Admission;
use crate::metrics::{BackendLabels, RequestLabels, RouterMetrics};
use crate::proxy::ByteStream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The backend answered and the whole body reached the client. 5xx counts as a
    /// backend failure for the breaker; 2xx/3xx/4xx do not (4xx is the client's doing).
    Completed(StatusCode),
    /// Connection failure or timeout before the response headers arrived. Backend failure.
    TransportError,
    /// The backend's stream broke after the headers were sent. Backend failure.
    StreamError,
    /// The client went away before the body finished. Neither success nor failure.
    ClientDisconnect,
}

impl Outcome {
    fn is_backend_failure(self) -> bool {
        match self {
            Outcome::Completed(s) => s.is_server_error(),
            Outcome::TransportError | Outcome::StreamError => true,
            Outcome::ClientDisconnect => false,
        }
    }

    fn status_label(self) -> String {
        match self {
            Outcome::Completed(s) => s.as_u16().to_string(),
            Outcome::TransportError => "502".to_string(),
            Outcome::StreamError => "stream_error".to_string(),
            Outcome::ClientDisconnect => "client_disconnect".to_string(),
        }
    }
}

pub struct RequestAccounting {
    backend: Arc<Backend>,
    metrics: RouterMetrics,
    strategy_name: String,
    ewma_alpha: f64,
    started: Instant,
    /// When the backend's response headers arrived (time to first byte).
    headers_at: Option<Instant>,
    outcome: Outcome,
    admission: Admission,
    _permit: Permit,
}

impl RequestAccounting {
    /// Marks the request as started on `backend`. Holds the backpressure permit.
    pub fn start(
        backend: Arc<Backend>,
        metrics: RouterMetrics,
        strategy_name: String,
        ewma_alpha: f64,
        admission: Admission,
        permit: Permit,
    ) -> Self {
        backend.record_request_start();
        Self {
            backend,
            metrics,
            strategy_name,
            ewma_alpha,
            started: Instant::now(),
            headers_at: None,
            outcome: Outcome::ClientDisconnect,
            admission,
            _permit: permit,
        }
    }

    /// Records the arrival of the backend's response headers (first byte), once.
    pub fn mark_headers_received(&mut self) {
        self.headers_at.get_or_insert_with(Instant::now);
    }

    pub fn finish(mut self, outcome: Outcome) {
        self.outcome = outcome;
    }
}

impl Drop for RequestAccounting {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        let outcome = self.outcome;
        if outcome == Outcome::ClientDisconnect {
            tracing::debug!(
                backend = %self.backend.url,
                elapsed_ms = elapsed.as_millis() as u64,
                headers_received = self.headers_at.is_some(),
                "request ended without a verdict; recorded as client_disconnect"
            );
        }

        self.backend
            .record_request_end(!outcome.is_backend_failure());

        match outcome {
            Outcome::Completed(status) => {
                // Only successful completions feed the latency estimate (a backend that
                // fails fast must not look fast), and the estimate is time to first byte:
                // full generation time scales with output length, not backend speed.
                if status.is_success() {
                    let ttfb = self.headers_at.unwrap_or_else(Instant::now) - self.started;
                    self.backend
                        .update_ewma_latency(ttfb.as_micros() as u64, self.ewma_alpha);
                }
                if outcome.is_backend_failure() {
                    self.backend.breaker.record_failure(self.admission);
                } else {
                    self.backend.breaker.record_success(self.admission);
                }
            }
            Outcome::TransportError | Outcome::StreamError => {
                self.backend.breaker.record_failure(self.admission)
            }
            Outcome::ClientDisconnect => self.backend.breaker.abandon(self.admission),
        }

        // The histogram covers the full request duration, whatever the outcome.
        self.metrics
            .request_duration
            .get_or_create(&BackendLabels {
                backend: self.backend.url.clone(),
            })
            .observe(elapsed.as_secs_f64());

        self.metrics
            .requests_total
            .get_or_create(&RequestLabels {
                strategy: self.strategy_name.clone(),
                backend: self.backend.url.clone(),
                status: outcome.status_label(),
            })
            .inc();
    }
}

/// Wraps the backend's SSE byte stream and finishes the accounting when it ends.
pub struct AccountedStream {
    inner: ByteStream,
    status: StatusCode,
    accounting: Option<RequestAccounting>,
}

impl AccountedStream {
    pub fn new(inner: ByteStream, status: StatusCode, accounting: RequestAccounting) -> Self {
        Self {
            inner,
            status,
            accounting: Some(accounting),
        }
    }
}

impl Stream for AccountedStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        if this.accounting.is_none() {
            return Poll::Ready(None);
        }
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(Some(Err(e))) => {
                if let Some(acc) = this.accounting.take() {
                    tracing::warn!(
                        backend = %acc.backend.url,
                        error = %e,
                        "backend stream failed mid-body"
                    );
                    acc.finish(Outcome::StreamError);
                }
                Poll::Ready(Some(Err(std::io::Error::other(e))))
            }
            Poll::Ready(None) => {
                if let Some(acc) = this.accounting.take() {
                    acc.finish(Outcome::Completed(this.status));
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backpressure::BackpressureController;
    use crate::circuit_breaker::{CircuitBreaker, CircuitState};
    use futures_util::StreamExt;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    struct Fixture {
        backend: Arc<Backend>,
        controller: Arc<BackpressureController>,
        metrics: RouterMetrics,
    }

    fn fixture() -> Fixture {
        Fixture {
            backend: Arc::new(Backend::new(
                0,
                "http://b0".into(),
                1,
                CircuitBreaker::new(0.5, 1, 60, 30),
            )),
            controller: Arc::new(BackpressureController::new(4)),
            metrics: RouterMetrics::new(),
        }
    }

    fn start(f: &Fixture) -> RequestAccounting {
        RequestAccounting::start(
            f.backend.clone(),
            f.metrics.clone(),
            "test".into(),
            0.3,
            Admission::Normal,
            f.controller.try_acquire().unwrap(),
        )
    }

    #[test]
    fn completed_2xx_records_success_latency_and_releases_permit() {
        let f = fixture();
        let mut acc = start(&f);
        acc.started = Instant::now() - Duration::from_millis(5);
        assert_eq!(f.backend.active_connections.load(Ordering::Relaxed), 1);
        assert_eq!(f.controller.in_flight(), 1);
        acc.finish(Outcome::Completed(StatusCode::OK));
        assert_eq!(f.backend.active_connections.load(Ordering::Relaxed), 0);
        assert_eq!(f.controller.in_flight(), 0);
        assert_eq!(f.backend.total_failures.load(Ordering::Relaxed), 0);
        assert!(f.backend.ewma_latency_us.load(Ordering::Relaxed) > 0);
        assert_eq!(f.backend.breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn completed_5xx_is_a_backend_failure() {
        let f = fixture();
        start(&f).finish(Outcome::Completed(StatusCode::BAD_GATEWAY));
        assert_eq!(f.backend.total_failures.load(Ordering::Relaxed), 1);
        assert_eq!(f.backend.breaker.state(), CircuitState::Open);
        assert_eq!(
            f.backend.ewma_latency_us.load(Ordering::Relaxed),
            0,
            "failures do not feed the EWMA"
        );
    }

    #[test]
    fn completed_4xx_is_not_a_backend_failure() {
        let f = fixture();
        start(&f).finish(Outcome::Completed(StatusCode::BAD_REQUEST));
        assert_eq!(f.backend.total_failures.load(Ordering::Relaxed), 0);
        assert_eq!(f.backend.breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn dropped_without_verdict_is_a_client_disconnect() {
        let f = fixture();
        drop(start(&f));
        assert_eq!(f.backend.active_connections.load(Ordering::Relaxed), 0);
        assert_eq!(f.backend.total_failures.load(Ordering::Relaxed), 0);
        assert_eq!(f.backend.breaker.state(), CircuitState::Closed);
        assert_eq!(f.backend.ewma_latency_us.load(Ordering::Relaxed), 0);
        assert!(f.metrics.encode().contains("status=\"client_disconnect\""));
    }

    #[tokio::test]
    async fn stream_finishes_accounting_at_end_of_body() {
        let f = fixture();
        let acc = start(&f);
        let inner: ByteStream = Box::pin(futures_util::stream::iter(vec![
            Ok(Bytes::from_static(b"data: a\n\n")),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ]));
        let mut stream = AccountedStream::new(inner, StatusCode::OK, acc);
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(&first[..], b"data: a\n\n");
        assert_eq!(
            f.backend.active_connections.load(Ordering::Relaxed),
            1,
            "still in flight"
        );
        stream.next().await.unwrap().unwrap();
        assert!(stream.next().await.is_none());
        assert_eq!(f.backend.active_connections.load(Ordering::Relaxed), 0);
        assert_eq!(f.controller.in_flight(), 0);
        assert!(f.metrics.encode().contains("status=\"200\""));
    }

    #[tokio::test]
    async fn stream_error_records_failure_and_releases_permit() {
        let f = fixture();
        let acc = start(&f);
        let err = reqwest::Client::new()
            .get("http://127.0.0.1:1")
            .send()
            .await
            .unwrap_err();
        let inner: ByteStream = Box::pin(futures_util::stream::iter(vec![
            Ok(Bytes::from_static(b"a")),
            Err(err),
        ]));
        let mut stream = AccountedStream::new(inner, StatusCode::OK, acc);
        stream.next().await.unwrap().unwrap();
        assert!(stream.next().await.unwrap().is_err());
        assert!(
            stream.next().await.is_none(),
            "stream is fused after an error"
        );
        assert_eq!(f.backend.total_failures.load(Ordering::Relaxed), 1);
        assert_eq!(f.backend.breaker.state(), CircuitState::Open);
        assert_eq!(f.controller.in_flight(), 0);
        assert!(f.metrics.encode().contains("status=\"stream_error\""));
    }

    #[tokio::test]
    async fn dropping_stream_midway_is_a_client_disconnect() {
        let f = fixture();
        let acc = start(&f);
        let inner: ByteStream = Box::pin(futures_util::stream::iter(vec![
            Ok(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ]));
        let mut stream = AccountedStream::new(inner, StatusCode::OK, acc);
        stream.next().await.unwrap().unwrap();
        drop(stream);
        assert_eq!(f.backend.active_connections.load(Ordering::Relaxed), 0);
        assert_eq!(f.backend.total_failures.load(Ordering::Relaxed), 0);
        assert_eq!(f.backend.breaker.state(), CircuitState::Closed);
        assert!(f.metrics.encode().contains("client_disconnect"));
        assert_eq!(f.controller.in_flight(), 0);
    }

    #[test]
    fn ewma_uses_time_to_first_byte() {
        let f = fixture();
        let mut acc = start(&f);
        acc.started = Instant::now() - Duration::from_millis(50);
        acc.mark_headers_received();
        acc.headers_at = Some(acc.started + Duration::from_millis(10));
        acc.finish(Outcome::Completed(StatusCode::OK));
        let ewma = f.backend.ewma_latency_us.load(Ordering::Relaxed);
        assert!((9_000..=11_000).contains(&ewma), "ewma was {ewma}");
    }

    #[test]
    fn probe_disconnect_hands_the_probe_back() {
        let f = Fixture {
            backend: Arc::new(Backend::new(
                0,
                "http://b0".into(),
                1,
                CircuitBreaker::new(0.5, 1, 60, 1),
            )),
            controller: Arc::new(BackpressureController::new(4)),
            metrics: RouterMetrics::new(),
        };
        let breaker = &f.backend.breaker;
        breaker.record_failure(Admission::Normal);
        breaker.advance(Duration::from_secs(2));
        let adm = breaker.try_begin().unwrap();
        assert_eq!(adm, Admission::Probe);
        let acc = RequestAccounting::start(
            f.backend.clone(),
            f.metrics.clone(),
            "test".into(),
            0.3,
            adm,
            f.controller.try_acquire().unwrap(),
        );
        drop(acc);
        assert_eq!(breaker.try_begin(), Some(Admission::Probe));
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
    }
}
