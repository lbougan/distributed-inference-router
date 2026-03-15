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
use crate::proxy::forward_request;
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
        state
            .metrics
            .active_connections
            .get_or_create(&labels)
            .set(backend.active_connections.load(std::sync::atomic::Ordering::Relaxed) as i64);

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

async fn proxy_handler(
    State(state): State<AppState>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    if !state.backpressure.try_acquire() {
        state
            .metrics
            .rejected_requests
            .get_or_create(&RejectLabels {
                reason: "backpressure".to_string(),
            })
            .inc();
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    let path = req.uri().path().to_string();

    let body_bytes = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let body_json: Option<serde_json::Value> = serde_json::from_slice(&body_bytes).ok();

    let incoming = IncomingRequest {
        path: path.clone(),
        body: body_json,
    };

    let backend_id = state
        .strategy
        .select_backend(&incoming, &state.pool)
        .await;

    let backend_id = match backend_id {
        Some(id) => id,
        None => {
            state.backpressure.release();
            state
                .metrics
                .rejected_requests
                .get_or_create(&RejectLabels {
                    reason: "no_healthy_backend".to_string(),
                })
                .inc();
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    };

    // Check circuit breaker
    if let Some(cb) = state.circuit_breakers.get(backend_id) {
        if !cb.allows_request() {
            state.backpressure.release();
            state
                .metrics
                .rejected_requests
                .get_or_create(&RejectLabels {
                    reason: "circuit_open".to_string(),
                })
                .inc();
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    }

    let backend = match state.pool.get(backend_id) {
        Some(b) => b.clone(),
        None => {
            state.backpressure.release();
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    backend.record_request_start();
    let start = Instant::now();

    // Rebuild the request with the buffered body
    let rebuilt = Request::builder()
        .method(http::Method::POST)
        .uri(&path)
        .header("content-type", "application/json")
        .body(Body::from(body_bytes))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let result = forward_request(&backend, &state.client, rebuilt).await;
    let elapsed = start.elapsed();

    let (response, success) = match result {
        Ok(resp) => {
            let success = resp.status().is_success();
            (Ok(resp), success)
        }
        Err(status) => (Err(status), false),
    };

    backend.record_request_end(success);
    backend.update_ewma_latency(elapsed.as_micros() as u64, 0.3);
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

    let status_str = match &response {
        Ok(r) => r.status().as_u16().to_string(),
        Err(s) => s.as_u16().to_string(),
    };

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

    state.backpressure.release();
    response
}
