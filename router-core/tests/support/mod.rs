#![allow(dead_code)]

pub mod mock_backend;
pub mod router;

use serde_json::{json, Value};

pub fn chat_body(prompt: &str, stream: bool) -> Value {
    json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "stream": stream
    })
}

/// Parse one gauge sample from Prometheus text, e.g. `router_active_connections{backend="x"} 1`.
pub fn metric_value(text: &str, name: &str, label: &str) -> Option<f64> {
    text.lines()
        .find(|l| l.starts_with(name) && l.contains(label))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
}
