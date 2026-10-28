use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, Response, StatusCode};
use axum::routing::{any, get};
use axum::Router;
use std::sync::Arc;

use crate::accounting::{AccountedStream, Outcome, RequestAccounting};
use crate::backend::{Backend, BackendPool};
use crate::backpressure::BackpressureController;
use crate::metrics::{BackendLabels, RejectLabels, RouterMetrics};
use crate::proxy::{forward_request, Forwarded, ForwardedBody, MAX_BODY_BYTES};
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

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/completions", any(proxy_handler))
        .route("/v1/chat/completions", any(proxy_handler))
        .with_state(state)
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
    accounting: RequestAccounting,
) -> Response<Body> {
    let status = forwarded.status;
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

/// 413 when the body exceeds `MAX_BODY_BYTES`, 400 for any other read failure.
fn body_error_status(err: axum::Error) -> StatusCode {
    let is_too_large = std::error::Error::source(&err)
        .map(|s| s.is::<http_body_util::LengthLimitError>())
        .unwrap_or(false);
    if is_too_large {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    }
}

/// Ask the strategy for a backend among the routable ones; if its breaker refuses the
/// request (half-open probe already taken), drop it from the candidates and ask again.
fn select_backend(state: &AppState, incoming: &IncomingRequest) -> Option<Arc<Backend>> {
    let mut candidates = state.pool.routable();
    while !candidates.is_empty() {
        let id = state.strategy.select(incoming, &candidates)?;
        let backend = candidates.iter().find(|b| b.id == id)?.clone();
        if backend.breaker.try_begin() {
            return Some(backend);
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
    let body_bytes = axum::body::to_bytes(body, MAX_BODY_BYTES)
        .await
        .map_err(body_error_status)?;
    let incoming = IncomingRequest {
        path: parts.uri.path().to_string(),
        body: serde_json::from_slice(&body_bytes).ok(),
    };

    let Some(backend) = select_backend(&state, &incoming) else {
        inc_rejected(&state, "no_routable_backend");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    let mut accounting = RequestAccounting::start(
        backend.clone(),
        state.metrics.clone(),
        state.strategy_name.clone(),
        state.ewma_alpha,
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
            accounting.mark_headers_received();
            Ok(into_response(forwarded, &backend.url, accounting))
        }
        Err(status) => {
            accounting.finish(Outcome::TransportError);
            Err(status)
        }
    }
}
