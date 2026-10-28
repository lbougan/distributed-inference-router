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
use crate::metrics::{BackendLabels, RequestLabels, RouterMetrics};
use crate::proxy::ByteStream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The backend answered and the whole body reached the client. 5xx counts as a
    /// backend failure for the breaker; 2xx/3xx/4xx do not (4xx is the client's doing).
    Completed(StatusCode),
    /// Connection failure, timeout, or the stream broke mid-way. Backend failure.
    TransportError,
    /// The client went away before the body finished. Neither success nor failure.
    ClientDisconnect,
}

impl Outcome {
    fn is_backend_failure(self) -> bool {
        match self {
            Outcome::Completed(s) => s.is_server_error(),
            Outcome::TransportError => true,
            Outcome::ClientDisconnect => false,
        }
    }

    fn status_label(self) -> String {
        match self {
            Outcome::Completed(s) => s.as_u16().to_string(),
            Outcome::TransportError => "502".to_string(),
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
    outcome: Outcome,
    _permit: Permit,
}

impl RequestAccounting {
    /// Marks the request as started on `backend`. Holds the backpressure permit.
    pub fn start(
        backend: Arc<Backend>,
        metrics: RouterMetrics,
        strategy_name: String,
        ewma_alpha: f64,
        permit: Permit,
    ) -> Self {
        backend.record_request_start();
        Self {
            backend,
            metrics,
            strategy_name,
            ewma_alpha,
            started: Instant::now(),
            outcome: Outcome::ClientDisconnect,
            _permit: permit,
        }
    }

    pub fn finish(mut self, outcome: Outcome) {
        self.outcome = outcome;
    }
}

impl Drop for RequestAccounting {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        let outcome = self.outcome;

        self.backend
            .record_request_end(!outcome.is_backend_failure());

        match outcome {
            Outcome::Completed(status) => {
                // Only successful completions feed the latency estimate: a backend that
                // fails fast must not look fast.
                if status.is_success() {
                    self.backend
                        .update_ewma_latency(elapsed.as_micros() as u64, self.ewma_alpha);
                }
                if outcome.is_backend_failure() {
                    self.backend.breaker.record_failure();
                } else {
                    self.backend.breaker.record_success();
                }
                self.metrics
                    .request_duration
                    .get_or_create(&BackendLabels {
                        backend: self.backend.url.clone(),
                    })
                    .observe(elapsed.as_secs_f64());
            }
            Outcome::TransportError => self.backend.breaker.record_failure(),
            Outcome::ClientDisconnect => self.backend.breaker.abandon(),
        }

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
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(Some(Err(e))) => {
                if let Some(acc) = this.accounting.take() {
                    acc.finish(Outcome::TransportError);
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
            f.controller.try_acquire().unwrap(),
        )
    }

    #[test]
    fn completed_2xx_records_success_latency_and_releases_permit() {
        let f = fixture();
        let acc = start(&f);
        assert_eq!(f.backend.active_connections.load(Ordering::Relaxed), 1);
        assert_eq!(f.controller.in_flight(), 1);
        std::thread::sleep(std::time::Duration::from_millis(2));
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
}
