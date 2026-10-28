//! An in-process stand-in for a vLLM replica with knobs for health, failures, latency,
//! streaming and the /metrics gauges the router scrapes.

use std::cmp::Ordering as Ordering2;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde_json::{json, Value};

#[derive(Clone)]
pub struct MockState {
    pub id: String,
    pub healthy: Arc<AtomicBool>,
    pub fail: Arc<AtomicBool>,
    pub latency_ms: Arc<AtomicU64>,
    pub stream_tokens: Arc<AtomicU64>,
    pub token_delay_ms: Arc<AtomicU64>,
    pub queue_depth: Arc<AtomicU64>,
    pub kv_usage_bp: Arc<AtomicU64>,
    pub requests: Arc<AtomicUsize>,
    pub in_flight: Arc<AtomicUsize>,
}

pub struct MockHandle {
    pub state: MockState,
    pub url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for MockHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl MockHandle {
    pub fn set_healthy(&self, v: bool) {
        self.state.healthy.store(v, Ordering::SeqCst);
    }
    pub fn set_fail(&self, v: bool) {
        self.state.fail.store(v, Ordering::SeqCst);
    }
    pub fn set_latency_ms(&self, v: u64) {
        self.state.latency_ms.store(v, Ordering::SeqCst);
    }
    pub fn set_stream(&self, tokens: u64, token_delay_ms: u64) {
        self.state.stream_tokens.store(tokens, Ordering::SeqCst);
        self.state
            .token_delay_ms
            .store(token_delay_ms, Ordering::SeqCst);
    }
    pub fn set_queue_depth(&self, v: u64) {
        self.state.queue_depth.store(v, Ordering::SeqCst);
    }
    pub fn requests(&self) -> usize {
        self.state.requests.load(Ordering::SeqCst)
    }
    pub fn in_flight(&self) -> usize {
        self.state.in_flight.load(Ordering::SeqCst)
    }
}

pub async fn spawn_mock(id: &str) -> MockHandle {
    let state = MockState {
        id: id.to_string(),
        healthy: Arc::new(AtomicBool::new(true)),
        fail: Arc::new(AtomicBool::new(false)),
        latency_ms: Arc::new(AtomicU64::new(5)),
        stream_tokens: Arc::new(AtomicU64::new(3)),
        token_delay_ms: Arc::new(AtomicU64::new(5)),
        queue_depth: Arc::new(AtomicU64::new(0)),
        kv_usage_bp: Arc::new(AtomicU64::new(1000)),
        requests: Arc::new(AtomicUsize::new(0)),
        in_flight: Arc::new(AtomicUsize::new(0)),
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(completion))
        .route("/v1/completions", post(completion))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    MockHandle {
        state,
        url: format!("http://{addr}"),
        task,
    }
}

struct InFlight(Arc<AtomicUsize>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn completion(State(s): State<MockState>, Json(body): Json<Value>) -> Response {
    s.requests.fetch_add(1, Ordering::SeqCst);
    s.in_flight.fetch_add(1, Ordering::SeqCst);
    let guard = InFlight(s.in_flight.clone());

    if s.fail.load(Ordering::SeqCst) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "simulated failure"})),
        )
            .into_response();
    }

    tokio::time::sleep(Duration::from_millis(s.latency_ms.load(Ordering::SeqCst))).await;

    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if !streaming {
        drop(guard);
        return (
            StatusCode::OK,
            [("x-mock-backend", s.id.clone())],
            Json(json!({
                "id": "cmpl-mock",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
            })),
        )
            .into_response();
    }

    let n = s.stream_tokens.load(Ordering::SeqCst);
    let delay = Duration::from_millis(s.token_delay_ms.load(Ordering::SeqCst));
    let chunks = stream::unfold((0u64, Some(guard)), move |(i, guard)| async move {
        match i.cmp(&n) {
            Ordering2::Less => {
                tokio::time::sleep(delay).await;
                let chunk =
                    format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"tok{i}\"}}}}]}}\n\n");
                Some((Ok::<Bytes, Infallible>(Bytes::from(chunk)), (i + 1, guard)))
            }
            Ordering2::Equal => Some((Ok(Bytes::from_static(b"data: [DONE]\n\n")), (i + 1, None))),
            Ordering2::Greater => None,
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header("x-mock-backend", s.id.clone())
        .body(Body::from_stream(chunks))
        .unwrap()
}

async fn health(State(s): State<MockState>) -> (StatusCode, Json<Value>) {
    if s.healthy.load(Ordering::SeqCst) {
        (StatusCode::OK, Json(json!({"status": "ok"})))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unhealthy"})),
        )
    }
}

/// Includes a `vllm:num_requests_waiting_total` decoy line: a prefix-based parser picks it up
/// by mistake (bug B8); Task 8 fixes the parser and the least_load e2e test relies on it.
async fn metrics(State(s): State<MockState>) -> String {
    let kv = s.kv_usage_bp.load(Ordering::SeqCst) as f64 / 10_000.0;
    let q = s.queue_depth.load(Ordering::SeqCst);
    format!(
        "# TYPE vllm:num_requests_waiting gauge\n\
         vllm:num_requests_waiting{{instance=\"{id}\"}} {q}\n\
         vllm:num_requests_waiting_total{{instance=\"{id}\"}} 999\n\
         # TYPE vllm:kv_cache_usage_perc gauge\n\
         vllm:kv_cache_usage_perc{{instance=\"{id}\"}} {kv:.4}\n",
        id = s.id
    )
}
