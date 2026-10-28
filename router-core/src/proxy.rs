//! Forwarding one request to one backend. Returns a `Forwarded` value instead of an axum
//! response so the caller can attach accounting to the body before replying.

use std::pin::Pin;
use std::time::Instant;

use axum::body::Bytes;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, Uri};
use futures_util::Stream;

use crate::backend::Backend;

/// Request bodies larger than this are rejected with 413.
pub const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;

/// RFC 7230 §6.1 hop-by-hop headers, `host` (regenerated per hop), and `x-router-backend`
/// (the router sets its own value, so an upstream one must never leak through).
const HOP_BY_HOP: [&str; 10] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "x-router-backend",
];

pub type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

pub enum ForwardedBody {
    Full(Bytes),
    Stream(ByteStream),
}

pub struct Forwarded {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: ForwardedBody,
    /// When the backend's response headers arrived, before any buffered body was read.
    pub headers_at: Instant,
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    HOP_BY_HOP.contains(&name.as_str()) || name == CONTENT_LENGTH
}

/// Copy end-to-end headers only. `content-length` is dropped because the body is re-framed.
pub fn filter_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers.iter() {
        if !is_hop_by_hop(name) {
            out.append(name.clone(), value.clone());
        }
    }
    out
}

pub fn is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.trim_start().starts_with("text/event-stream"))
        .unwrap_or(false)
}

/// Send the buffered request to `backend`. Transport failures become 502.
pub async fn forward_request(
    backend: &Backend,
    client: &reqwest::Client,
    method: Method,
    uri: &Uri,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<Forwarded, StatusCode> {
    let path = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let url = format!("{}{}", backend.url, path);

    let resp = client
        .request(method, &url)
        .headers(filter_headers(headers))
        .body(body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!(backend = %backend.url, error = %e, "backend request failed");
            StatusCode::BAD_GATEWAY
        })?;
    let headers_at = Instant::now();

    let status = resp.status();
    let headers = filter_headers(resp.headers());
    let body = if is_event_stream(&headers) {
        ForwardedBody::Stream(Box::pin(resp.bytes_stream()))
    } else {
        ForwardedBody::Full(resp.bytes().await.map_err(|e| {
            tracing::error!(backend = %backend.url, error = %e, "reading backend response body failed");
            StatusCode::BAD_GATEWAY
        })?)
    };
    Ok(Forwarded {
        status,
        headers,
        body,
        headers_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(
                HeaderName::from_static(k),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn strips_hop_by_hop_and_framing_headers() {
        let h = headers(&[
            ("host", "router:8080"),
            ("connection", "keep-alive"),
            ("transfer-encoding", "chunked"),
            ("content-length", "42"),
            ("x-router-backend", "http://upstream"),
            ("authorization", "Bearer x"),
            ("content-type", "application/json"),
        ]);
        let out = filter_headers(&h);
        assert_eq!(out.len(), 2);
        assert_eq!(out["authorization"], "Bearer x");
        assert_eq!(out["content-type"], "application/json");
        assert!(!out.contains_key("x-router-backend"));
    }

    #[test]
    fn detects_event_stream_content_type() {
        assert!(is_event_stream(&headers(&[(
            "content-type",
            "text/event-stream; charset=utf-8"
        )])));
        assert!(!is_event_stream(&headers(&[(
            "content-type",
            "application/json"
        )])));
        assert!(!is_event_stream(&HeaderMap::new()));
    }
}
