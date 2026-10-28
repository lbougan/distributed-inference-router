use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::ring::{hash_key, HashRing, DEFAULT_VNODES};
use super::{IncomingRequest, RoutingStrategy};
use crate::backend::{Backend, BackendId};

/// Routes requests with the same prompt prefix to the same backend so vLLM's prefix cache
/// (`--enable-prefix-caching`) can reuse the KV blocks.
pub struct PrefixCacheAware {
    prefix_char_limit: usize,
    ring: HashRing,
}

impl PrefixCacheAware {
    /// `prefix_token_count` is approximated as 4 characters per token.
    pub fn new(prefix_token_count: usize, num_backends: usize) -> Self {
        Self {
            prefix_char_limit: prefix_token_count * 4,
            ring: HashRing::new(num_backends, DEFAULT_VNODES),
        }
    }

    fn extract_prefix(&self, body: &Option<serde_json::Value>) -> String {
        let Some(body) = body else {
            return String::new();
        };

        let text: String = if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
            messages
                .iter()
                .filter_map(|m| m.get("content").and_then(content_text))
                .collect::<Vec<_>>()
                .join(" ")
        } else if let Some(prompt) = body.get("prompt").and_then(|p| p.as_str()) {
            prompt.to_string()
        } else {
            return String::new();
        };

        text.chars().take(self.prefix_char_limit).collect()
    }
}

/// Text of a chat message's `content`: a string, or the `text` parts of a content-part
/// array (`[{"type": "text", "text": "..."}, ...]`) joined by spaces. Non-text parts are
/// skipped; `None` when there is no text at all.
fn content_text(content: &serde_json::Value) -> Option<String> {
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }
    let parts: Vec<&str> = content
        .as_array()?
        .iter()
        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
        .collect();
    (!parts.is_empty()).then(|| parts.join(" "))
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
        self.ring.lookup(hash_key(prefix.as_bytes()), |id| {
            routable.iter().any(|b| b.id == id)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::test_support::backends;

    #[test]
    fn same_prefix_routes_to_same_backend() {
        let strategy = PrefixCacheAware::new(256, 3);
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
        let strategy = PrefixCacheAware::new(256, 10);
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
        let strategy = PrefixCacheAware::new(2, 1);
        let body = serde_json::json!({"prompt": "ééééééééééé"});
        let prefix = strategy.extract_prefix(&Some(body));
        assert_eq!(prefix, "éééééééé");
    }

    #[test]
    fn losing_one_backend_moves_few_prefixes() {
        let strategy = PrefixCacheAware::new(256, 10);
        let all = backends(10);
        let without_3: Vec<Arc<Backend>> = all.iter().filter(|b| b.id != 3).cloned().collect();
        let mut moved = 0;
        for i in 0..2000 {
            let body = serde_json::json!({"prompt": format!("conversation {i} starts here")});
            let req = IncomingRequest {
                path: "/v1/completions".into(),
                body: Some(body),
            };
            let before = strategy.select(&req, &all).unwrap();
            let after = strategy.select(&req, &without_3).unwrap();
            if before != after {
                assert_eq!(before, 3);
                moved += 1;
            }
        }
        assert!(moved <= 300, "moved {moved} of 2000");
    }

    #[test]
    fn array_form_system_prompt_keeps_locality() {
        // 16 tokens -> 64 chars, all inside the shared system prompt.
        let strategy = PrefixCacheAware::new(16, 10);
        let b = backends(10);
        let system = "You are a careful assistant. Answer briefly and cite sources when you can.";
        let picks: std::collections::HashSet<usize> = (0..20)
            .map(|i| {
                let body = serde_json::json!({"messages": [
                    {"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "text", "text": format!("question {i}")}]}
                ]});
                let req = IncomingRequest {
                    path: "/v1/chat/completions".into(),
                    body: Some(body),
                };
                strategy.select(&req, &b).unwrap()
            })
            .collect();
        assert_eq!(picks.len(), 1, "shared system prompt must pin one backend");

        // Without reading array content every request would fall back to the same
        // least-connections pick; distinct array prompts must hash apart.
        let spread: std::collections::HashSet<usize> = (0..20)
            .map(|i| {
                let body = serde_json::json!({"messages": [
                    {"role": "system", "content": [{"type": "text", "text": format!("system prompt variant {i}")}]}
                ]});
                let req = IncomingRequest {
                    path: "/v1/chat/completions".into(),
                    body: Some(body),
                };
                strategy.select(&req, &b).unwrap()
            })
            .collect();
        assert!(spread.len() > 1, "array prompts were not hashed");
    }

    #[test]
    fn mixed_string_and_array_content_is_concatenated() {
        let strategy = PrefixCacheAware::new(256, 1);
        let body = serde_json::json!({"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "text", "text": "part one"},
                {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
                {"type": "text", "text": "part two"}
            ]},
            {"role": "assistant", "content": null}
        ]});
        assert_eq!(
            strategy.extract_prefix(&Some(body)),
            "sys part one part two"
        );
    }
}
