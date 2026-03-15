use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use futures_util::StreamExt;
use std::sync::Arc;

use crate::backend::Backend;

/// Forward a request to the selected backend, returning the response (streaming-aware).
pub async fn forward_request(
    backend: &Arc<Backend>,
    client: &reqwest::Client,
    original: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let method = original.method().clone();
    let path = original.uri().path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let url = format!("{}{}", backend.url, path);

    let headers = original.headers().clone();
    let body_bytes = match axum::body::to_bytes(original.into_body(), 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return Err(StatusCode::BAD_REQUEST),
    };

    let mut req_builder = client
        .request(method, &url)
        .body(body_bytes.to_vec());

    for (key, value) in headers.iter() {
        if key == "host" {
            continue;
        }
        req_builder = req_builder.header(key.clone(), value.clone());
    }

    let resp = match req_builder.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(backend = backend.url, error = %e, "Backend request failed");
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    let is_streaming = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("text/event-stream"))
        .unwrap_or(false);

    let mut response_builder = Response::builder().status(status);
    for (key, value) in resp.headers().iter() {
        response_builder = response_builder.header(key.clone(), value.clone());
    }

    if is_streaming {
        let stream = resp.bytes_stream().map(|chunk| {
            chunk
                .map(|b| axum::body::Bytes::from(b.to_vec()))
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        });
        let body = Body::from_stream(stream);
        response_builder.body(body).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    } else {
        let body_bytes = resp.bytes().await.map_err(|_| StatusCode::BAD_GATEWAY)?;
        response_builder
            .body(Body::from(body_bytes))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }
}
