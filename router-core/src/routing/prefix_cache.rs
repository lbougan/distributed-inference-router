use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{BackendId, BackendPool};

/// Routes requests to a consistent backend based on the prompt prefix hash,
/// maximizing vLLM's KV cache hit rate.
pub struct PrefixCacheAware {
    prefix_char_limit: usize,
}

impl PrefixCacheAware {
    pub fn new(prefix_token_count: usize) -> Self {
        // Approximate: 1 token ~= 4 chars
        Self {
            prefix_char_limit: prefix_token_count * 4,
        }
    }

    fn extract_prefix(&self, body: &Option<serde_json::Value>) -> String {
        let body = match body {
            Some(b) => b,
            None => return String::new(),
        };

        // Try chat completions format: messages[0].content
        if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
            let combined: String = messages
                .iter()
                .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
                .collect::<Vec<_>>()
                .join(" ");
            let end = combined.len().min(self.prefix_char_limit);
            return combined[..end].to_string();
        }

        // Try completions format: prompt
        if let Some(prompt) = body.get("prompt").and_then(|p| p.as_str()) {
            let end = prompt.len().min(self.prefix_char_limit);
            return prompt[..end].to_string();
        }

        String::new()
    }

    fn consistent_hash(prefix: &str, num_backends: usize) -> usize {
        if num_backends == 0 {
            return 0;
        }
        let mut hasher = Sha256::new();
        hasher.update(prefix.as_bytes());
        let hash = hasher.finalize();
        let val = u64::from_le_bytes(hash[..8].try_into().unwrap());
        (val % num_backends as u64) as usize
    }
}

#[async_trait]
impl RoutingStrategy for PrefixCacheAware {
    async fn select_backend(
        &self,
        request: &IncomingRequest,
        pool: &BackendPool,
    ) -> Option<BackendId> {
        let healthy = pool.healthy_backends();
        if healthy.is_empty() {
            return None;
        }

        let prefix = self.extract_prefix(&request.body);
        if prefix.is_empty() {
            // Fall back to least connections
            return healthy
                .iter()
                .min_by_key(|b| b.active_connections.load(Ordering::Relaxed))
                .map(|b| b.id);
        }

        let target_idx = Self::consistent_hash(&prefix, healthy.len());
        Some(healthy[target_idx].id)
    }

    fn on_request_complete(&self, _backend: BackendId, _latency: Duration, _success: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use std::sync::Arc;

    fn make_pool(n: usize) -> BackendPool {
        let backends: Vec<Arc<Backend>> = (0..n)
            .map(|i| Arc::new(Backend::new(i, format!("http://backend-{i}:8000"), 1)))
            .collect();
        BackendPool::new(backends)
    }

    #[tokio::test]
    async fn same_prefix_routes_to_same_backend() {
        let strategy = PrefixCacheAware::new(256);
        let pool = make_pool(3);

        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "Hello, how are you doing today?"}]
        });
        let req = IncomingRequest {
            path: "/v1/chat/completions".into(),
            body: Some(body.clone()),
        };

        let first = strategy.select_backend(&req, &pool).await.unwrap();
        for _ in 0..10 {
            let req = IncomingRequest {
                path: "/v1/chat/completions".into(),
                body: Some(body.clone()),
            };
            let id = strategy.select_backend(&req, &pool).await.unwrap();
            assert_eq!(id, first, "same prefix should always route to same backend");
        }
    }

    #[tokio::test]
    async fn different_prefixes_can_route_differently() {
        let strategy = PrefixCacheAware::new(256);
        let pool = make_pool(10);

        let mut seen = std::collections::HashSet::new();
        for i in 0..20 {
            let body = serde_json::json!({
                "prompt": format!("unique prompt number {} with distinct content", i)
            });
            let req = IncomingRequest {
                path: "/v1/completions".into(),
                body: Some(body),
            };
            if let Some(id) = strategy.select_backend(&req, &pool).await {
                seen.insert(id);
            }
        }
        assert!(seen.len() > 1, "different prefixes should spread across backends");
    }
}
