#![allow(dead_code)]

pub mod mock_backend;
pub mod router;

use std::future::Future;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

pub fn chat_body(prompt: &str, stream: bool) -> Value {
    json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "stream": stream
    })
}

/// Poll `cond` every 50 ms until it returns true or `timeout` elapses. Panics on timeout.
pub async fn wait_until<F, Fut>(timeout: Duration, what: &str, mut cond: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if cond().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("timed out after {timeout:?} waiting for: {what}");
}

/// Parse one sample from Prometheus text. `name` must match the full metric name and
/// `label_value` is matched as a quoted label value (`="..."`), so similar URLs cannot collide.
pub fn metric_value(text: &str, name: &str, label_value: &str) -> Option<f64> {
    let needle = format!("=\"{label_value}\"");
    text.lines()
        .filter(|l| l.starts_with(&format!("{name}{{")) || l.starts_with(&format!("{name} ")))
        .find(|l| label_value.is_empty() || l.contains(&needle))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
}
