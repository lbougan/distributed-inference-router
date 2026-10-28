use sha2::{Digest, Sha256};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{Backend, BackendId};

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
        let Some(body) = body else {
            return String::new();
        };

        let text: String = if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
            messages
                .iter()
                .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
                .collect::<Vec<_>>()
                .join(" ")
        } else if let Some(prompt) = body.get("prompt").and_then(|p| p.as_str()) {
            prompt.to_string()
        } else {
            return String::new();
        };

        text.chars().take(self.prefix_char_limit).collect()
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

impl RoutingStrategy for PrefixCacheAware {
    fn select(&self, request: &IncomingRequest, routable: &[Arc<Backend>]) -> Option<BackendId> {
        if routable.is_empty() {
            return None;
        }
        let prefix = self.extract_prefix(&request.body);
        if prefix.is_empty() {
            return routable
                .iter()
                .min_by_key(|b| b.active_connections.load(Ordering::Relaxed))
                .map(|b| b.id);
        }
        let idx = Self::consistent_hash(&prefix, routable.len());
        Some(routable[idx].id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::test_support::backends;

    #[test]
    fn same_prefix_routes_to_same_backend() {
        let strategy = PrefixCacheAware::new(256);
        let b = backends(3);
        let body = serde_json::json!({"messages": [{"role": "user", "content": "Hello, how are you doing today?"}]});
        let req = IncomingRequest {
            path: "/v1/chat/completions".into(),
            body: Some(body),
        };
        let first = strategy.select(&req, &b).unwrap();
        for _ in 0..10 {
            assert_eq!(strategy.select(&req, &b).unwrap(), first);
        }
    }

    #[test]
    fn different_prefixes_spread_across_backends() {
        let strategy = PrefixCacheAware::new(256);
        let b = backends(10);
        let seen: std::collections::HashSet<usize> = (0..20)
            .map(|i| {
                let body = serde_json::json!({"prompt": format!("unique prompt number {i} with distinct content")});
                strategy.select(&IncomingRequest { path: "/v1/completions".into(), body: Some(body) }, &b).unwrap()
            })
            .collect();
        assert!(seen.len() > 1);
    }

    #[test]
    fn prefix_truncation_is_char_safe() {
        // 2 tokens -> 8 chars. "é" is 2 bytes; byte slicing at 8 would cut inside a char.
        let strategy = PrefixCacheAware::new(2);
        let body = serde_json::json!({"prompt": "ééééééééééé"});
        let prefix = strategy.extract_prefix(&Some(body));
        assert_eq!(prefix, "éééééééé");
    }
}
