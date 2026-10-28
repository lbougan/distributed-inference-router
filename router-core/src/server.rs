use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, Response, StatusCode};
use axum::routing::{any, get};
use axum::Router;
use std::sync::Arc;

use crate::accounting::{AccountedStream, Outcome, RequestAccounting};
use crate::backend::{Backend, BackendPool};
use crate::backpressure::BackpressureController;
use crate::circuit_breaker::Admission;
use crate::metrics::{BackendLabels, RejectLabels, RouterMetrics};
use crate::proxy::{forward_request, ForwardError, Forwarded, ForwardedBody, MAX_BODY_BYTES};
use crate::routing::{IncomingRequest, RoutingStrategy};

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<BackendPool>,
    pub strategy: Arc<dyn RoutingStrategy>,
    pub strategy_name: String,
    pub client: reqwest::Client,
    pub backpressure: Arc<BackpressureController>,
    pub metrics: RouterMetrics,
    pub ewma_alpha: f64,
}

pub fn build_router(state: AppState, metrics: &crate::config::MetricsConfig) -> Router {
    let mut router = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/completions", any(proxy_handler))
        .route("/v1/chat/completions", any(proxy_handler));
    if metrics.enabled {
        router = router.route(&metrics.path, get(metrics_handler));
    }
    router.with_state(state)
}

async fn health_handler() -> StatusCode {
    StatusCode::OK
}

async fn metrics_handler(State(state): State<AppState>) -> String {
    // Update gauge metrics before encoding
    for backend in &state.pool.backends {
        let labels = BackendLabels {
            backend: backend.url.clone(),
        };
        state.metrics.active_connections.get_or_create(&labels).set(
            backend
                .active_connections
                .load(std::sync::atomic::Ordering::Relaxed) as i64,
        );

        state
            .metrics
            .backend_health
            .get_or_create(&labels)
            .set(if backend.is_healthy() { 1 } else { 0 });

        state
            .metrics
            .circuit_breaker_state
            .get_or_create(&labels)
            .set(backend.breaker.state().as_u64() as i64);

        state
            .metrics
            .backend_queue_depth
            .get_or_create(&labels)
            .set(
                backend
                    .queue_depth
                    .load(std::sync::atomic::Ordering::Relaxed) as i64,
            );

        state
            .metrics
            .backend_kv_cache_usage
            .get_or_create(&labels)
            .set(
                backend
                    .kv_cache_usage_bp
                    .load(std::sync::atomic::Ordering::Relaxed) as f64
                    / 10_000.0,
            );
    }

    state
        .metrics
        .in_flight
        .set(state.backpressure.in_flight() as i64);

    state.metrics.encode()
}

fn inc_rejected(state: &AppState, reason: &str) {
    state
        .metrics
        .rejected_requests
        .get_or_create(&RejectLabels {
            reason: reason.to_string(),
        })
        .inc();
}

fn into_response(
    forwarded: Forwarded,
    backend_url: &str,
    mut accounting: RequestAccounting,
) -> Response<Body> {
    let status = forwarded.status;
    accounting.resolve_probe_on_headers(status);
    let mut builder = Response::builder().status(status);
    if let Some(h) = builder.headers_mut() {
        *h = forwarded.headers;
    }
    builder = builder.header("x-router-backend", backend_url);
    let body = match forwarded.body {
        ForwardedBody::Full(bytes) => {
            accounting.finish(Outcome::Completed(status));
            Body::from(bytes)
        }
        ForwardedBody::Stream(stream) => {
            Body::from_stream(AccountedStream::new(stream, status, accounting))
        }
    };
    builder.body(body).expect("response from valid parts")
}

/// 413 when the body exceeds `MAX_BODY_BYTES`, 400 for any other read failure, with the
/// matching `router_rejected_requests_total` reason.
fn body_error_status(err: axum::Error) -> (StatusCode, &'static str) {
    let is_too_large = std::error::Error::source(&err)
        .map(|s| s.is::<http_body_util::LengthLimitError>())
        .unwrap_or(false);
    if is_too_large {
        (StatusCode::PAYLOAD_TOO_LARGE, "body_too_large")
    } else {
        (StatusCode::BAD_REQUEST, "bad_body")
    }
}

/// Records the verdict for a failed forward and returns the client's status.
fn finish_forward_error(mut accounting: RequestAccounting, err: ForwardError) -> StatusCode {
    match err {
        ForwardError::Connect => accounting.finish(Outcome::TransportError),
        ForwardError::Body { headers_at } => {
            accounting.mark_headers_received_at(headers_at);
            accounting.finish(Outcome::StreamError);
        }
    }
    err.status()
}

/// Ask the strategy for a backend among the routable ones; if its breaker refuses the
/// request (half-open probe already taken), drop it from the candidates and ask again.
/// If every under-limit candidate refuses, the request is rejected rather than retried on
/// backends that are over the queue limit.
fn select_backend(
    state: &AppState,
    incoming: &IncomingRequest,
) -> Option<(Arc<Backend>, Admission)> {
    let mut candidates = state.pool.routable();
    while !candidates.is_empty() {
        let id = state.strategy.select(incoming, &candidates)?;
        let Some(backend) = candidates.iter().find(|b| b.id == id).cloned() else {
            tracing::error!(
                backend_id = id,
                "strategy returned an id outside the candidates"
            );
            return None;
        };
        if let Some(adm) = backend.breaker.try_begin() {
            return Some((backend, adm));
        }
        candidates.retain(|b| b.id != id);
    }
    None
}

/// admission -> selection -> forwarding -> accounting
async fn proxy_handler(
    State(state): State<AppState>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let Some(permit) = state.backpressure.try_acquire() else {
        inc_rejected(&state, "backpressure");
        return Err(StatusCode::TOO_MANY_REQUESTS);
    };

    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(err) => {
            let (status, reason) = body_error_status(err);
            inc_rejected(&state, reason);
            return Err(status);
        }
    };
    let incoming = IncomingRequest {
        path: parts.uri.path().to_string(),
        body: serde_json::from_slice(&body_bytes).ok(),
    };

    let Some((backend, adm)) = select_backend(&state, &incoming) else {
        inc_rejected(&state, "no_routable_backend");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    let mut accounting = RequestAccounting::start(
        backend.clone(),
        state.metrics.clone(),
        state.strategy_name.clone(),
        state.ewma_alpha,
        adm,
        permit,
    );
    match forward_request(
        &backend,
        &state.client,
        parts.method,
        &parts.uri,
        &parts.headers,
        body_bytes,
    )
    .await
    {
        Ok(forwarded) => {
            accounting.mark_headers_received_at(forwarded.headers_at);
            Ok(into_response(forwarded, &backend.url, accounting))
        }
        Err(err) => Err(finish_forward_error(accounting, err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_breaker::CircuitBreaker;
    use std::time::{Duration, Instant};

    fn accounting(metrics: &RouterMetrics) -> (Arc<Backend>, RequestAccounting) {
        let backend = Arc::new(Backend::new(
            0,
            "http://b0".into(),
            1,
            CircuitBreaker::default(),
        ));
        let permit = Arc::new(BackpressureController::new(1))
            .try_acquire()
            .unwrap();
        let acc = RequestAccounting::start(
            backend.clone(),
            metrics.clone(),
            "test".into(),
            0.3,
            Admission::Normal,
            permit,
        );
        (backend, acc)
    }

    #[test]
    fn connect_failure_is_a_transport_error() {
        let metrics = RouterMetrics::new();
        let (_b, acc) = accounting(&metrics);
        assert_eq!(
            finish_forward_error(acc, ForwardError::Connect),
            StatusCode::BAD_GATEWAY
        );
        let text = metrics.encode();
        assert!(text.contains("status=\"transport_error\""), "{text}");
    }

    #[test]
    fn body_failure_after_headers_is_a_stream_error() {
        let metrics = RouterMetrics::new();
        let (_b, acc) = accounting(&metrics);
        let headers_at = Instant::now() - Duration::from_millis(1);
        assert_eq!(
            finish_forward_error(acc, ForwardError::Body { headers_at }),
            StatusCode::BAD_GATEWAY
        );
        let text = metrics.encode();
        assert!(text.contains("status=\"stream_error\""), "{text}");
        assert!(!text.contains("transport_error"), "{text}");
    }
}
