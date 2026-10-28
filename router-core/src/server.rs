use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, Response, StatusCode};
use axum::routing::{any, get};
use axum::Router;
use std::sync::Arc;
use std::time::Instant;

use crate::backend::BackendPool;
use crate::backpressure::BackpressureController;
use crate::circuit_breaker::CircuitBreaker;
use crate::metrics::{BackendLabels, RejectLabels, RequestLabels, RouterMetrics};
use crate::proxy::{forward_request, Forwarded, ForwardedBody, MAX_BODY_BYTES};
use crate::routing::{IncomingRequest, RoutingStrategy};

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<BackendPool>,
    pub strategy: Arc<dyn RoutingStrategy>,
    pub strategy_name: String,
    pub client: reqwest::Client,
    pub circuit_breakers: Arc<Vec<CircuitBreaker>>,
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

        if let Some(cb) = state.circuit_breakers.get(backend.id) {
            state
                .metrics
                .circuit_breaker_state
                .get_or_create(&labels)
                .set(cb.state().as_u64() as i64);
        }
    }

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

fn into_response(forwarded: Forwarded, backend_url: &str) -> Response<Body> {
    let mut builder = Response::builder().status(forwarded.status);
    if let Some(h) = builder.headers_mut() {
        *h = forwarded.headers;
    }
    builder = builder.header("x-router-backend", backend_url);
    let body = match forwarded.body {
        ForwardedBody::Full(bytes) => Body::from(bytes),
        ForwardedBody::Stream(stream) => Body::from_stream(stream),
    };
    builder.body(body).expect("response from valid parts")
}

async fn proxy_handler(
    State(state): State<AppState>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let Some(_permit) = state.backpressure.try_acquire() else {
        inc_rejected(&state, "backpressure");
        return Err(StatusCode::TOO_MANY_REQUESTS);
    };

    let (parts, body) = req.into_parts();
    let body_bytes = axum::body::to_bytes(body, MAX_BODY_BYTES)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let incoming = IncomingRequest {
        path: parts.uri.path().to_string(),
        body: serde_json::from_slice(&body_bytes).ok(),
    };

    let Some(backend_id) = state.strategy.select_backend(&incoming, &state.pool).await else {
        inc_rejected(&state, "no_healthy_backend");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    if let Some(cb) = state.circuit_breakers.get(backend_id) {
        if !cb.allows_request() {
            inc_rejected(&state, "circuit_open");
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    }

    let backend = state
        .pool
        .get(backend_id)
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?
        .clone();

    backend.record_request_start();
    let start = Instant::now();
    let result = forward_request(
        &backend,
        &state.client,
        parts.method,
        &parts.uri,
        &parts.headers,
        body_bytes,
    )
    .await;
    let elapsed = start.elapsed();

    let (success, status_str) = match &result {
        Ok(fw) => (fw.status.is_success(), fw.status.as_u16().to_string()),
        Err(status) => (false, status.as_u16().to_string()),
    };

    backend.record_request_end(success);
    backend.update_ewma_latency(elapsed.as_micros() as u64, state.ewma_alpha);
    state
        .strategy
        .on_request_complete(backend_id, elapsed, success);
    if let Some(cb) = state.circuit_breakers.get(backend_id) {
        if success {
            cb.record_success();
        } else {
            cb.record_failure();
        }
    }

    state
        .metrics
        .requests_total
        .get_or_create(&RequestLabels {
            strategy: state.strategy_name.clone(),
            backend: backend.url.clone(),
            status: status_str,
        })
        .inc();
    state
        .metrics
        .request_duration
        .get_or_create(&BackendLabels {
            backend: backend.url.clone(),
        })
        .observe(elapsed.as_secs_f64());

    result.map(|fw| into_response(fw, &backend.url))
}
