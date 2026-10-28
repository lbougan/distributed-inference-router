mod support;

use std::time::Duration;

use futures_util::future::join_all;
use support::mock_backend::spawn_mock;
use support::router::{start_router, RouterOpts};
use support::{chat_body, metric_value, wait_until};

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

async fn post_chat(client: &reqwest::Client, router_url: &str, prompt: &str) -> reqwest::Response {
    client
        .post(format!("{router_url}/v1/chat/completions"))
        .json(&chat_body(prompt, false))
        .send()
        .await
        .expect("request sent")
}

async fn backend_health(c: &reqwest::Client, router_url: &str, backend_url: &str) -> Option<f64> {
    let text = c
        .get(format!("{router_url}/metrics"))
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()?;
    metric_value(&text, "router_backend_health", backend_url)
}

#[tokio::test]
async fn routes_chat_completion_and_sets_backend_header() {
    let m = spawn_mock("m1").await;
    let r = start_router(&[&m], RouterOpts::default()).await;
    let resp = post_chat(&client(), &r.url, "hello").await;
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["x-router-backend"], m.url.as_str());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "ok");
    assert_eq!(m.requests(), 1);
}

#[tokio::test]
async fn round_robin_cycles_across_backends() {
    let m1 = spawn_mock("m1").await;
    let m2 = spawn_mock("m2").await;
    let m3 = spawn_mock("m3").await;
    let r = start_router(&[&m1, &m2, &m3], RouterOpts::default()).await;
    let c = client();
    for _ in 0..12 {
        assert_eq!(post_chat(&c, &r.url, "x").await.status(), 200);
    }
    assert_eq!((m1.requests(), m2.requests(), m3.requests()), (4, 4, 4));
}

#[tokio::test]
async fn streaming_response_is_passed_through_as_sse() {
    let m = spawn_mock("m1").await;
    m.set_stream(4, 5);
    let r = start_router(&[&m], RouterOpts::default()).await;
    let resp = client()
        .post(format!("{}/v1/chat/completions", r.url))
        .json(&chat_body("stream me", true))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let text = resp.text().await.unwrap();
    assert_eq!(text.matches("data: {").count(), 4);
    assert!(text.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn unhealthy_backend_is_skipped_after_health_check() {
    let m1 = spawn_mock("m1").await;
    let m2 = spawn_mock("m2").await;
    let r = start_router(&[&m1, &m2], RouterOpts::default()).await;
    m1.set_healthy(false);
    let c = client();
    wait_until(Duration::from_secs(5), "m1 marked unhealthy", || async {
        backend_health(&c, &r.url, &m1.url).await == Some(0.0)
    })
    .await;
    let before = m1.requests();
    for _ in 0..8 {
        assert_eq!(post_chat(&c, &r.url, "x").await.status(), 200);
    }
    assert_eq!(
        m1.requests(),
        before,
        "unhealthy backend must not receive traffic"
    );
    assert_eq!(m2.requests(), 8);
}

#[tokio::test]
async fn all_backends_down_returns_503() {
    let m1 = spawn_mock("m1").await;
    let r = start_router(&[&m1], RouterOpts::default()).await;
    m1.set_healthy(false);
    let c = client();
    wait_until(Duration::from_secs(5), "m1 marked unhealthy", || async {
        backend_health(&c, &r.url, &m1.url).await == Some(0.0)
    })
    .await;
    assert_eq!(post_chat(&c, &r.url, "x").await.status(), 503);
}

// The first two arrivals always get a slot (max_in_flight = 2), so at least two 200s are guaranteed.
#[tokio::test]
async fn backpressure_returns_429_and_never_500() {
    let m = spawn_mock("m1").await;
    m.set_latency_ms(300);
    let r = start_router(
        &[&m],
        RouterOpts {
            max_in_flight: 2,
            ..Default::default()
        },
    )
    .await;
    let c = client();
    let statuses: Vec<u16> = join_all((0..8).map(|_| post_chat(&c, &r.url, "x")))
        .await
        .into_iter()
        .map(|resp| resp.status().as_u16())
        .collect();
    assert!(
        statuses.contains(&429),
        "expected some 429s, got {statuses:?}"
    );
    assert!(statuses.iter().filter(|s| **s == 200).count() >= 2);
    assert!(
        !statuses.iter().any(|s| *s >= 500),
        "no 5xx expected, got {statuses:?}"
    );
}

#[tokio::test]
async fn metrics_endpoint_exposes_router_metrics() {
    let m = spawn_mock("m1").await;
    let r = start_router(&[&m], RouterOpts::default()).await;
    let c = client();
    post_chat(&c, &r.url, "x").await;
    let text = c
        .get(format!("{}/metrics", r.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("router_requests_total"));
    assert_eq!(
        metric_value(&text, "router_backend_health", &m.url),
        Some(1.0)
    );
    assert_eq!(
        metric_value(&text, "router_active_connections", &m.url),
        Some(0.0)
    );
}

#[tokio::test]
async fn method_and_auth_header_are_forwarded_unchanged() {
    let m = spawn_mock("m1").await;
    let r = start_router(&[&m], RouterOpts::default()).await;
    // The router only mounts /v1/completions and /v1/chat/completions; send a GET there.
    let resp = client()
        .get(format!("{}/v1/completions", r.url))
        .header("authorization", "Bearer secret")
        .header("connection", "keep-alive")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(m.last_method().as_deref(), Some("GET"));
    assert_eq!(m.last_authorization().as_deref(), Some("Bearer secret"));
    assert!(!m.last_had_connection_header());
}

#[tokio::test]
async fn upstream_x_router_backend_header_is_replaced() {
    let m = spawn_mock("m1").await;
    let r = start_router(&[&m], RouterOpts::default()).await;
    let resp = post_chat(&client(), &r.url, "x").await;
    let values: Vec<_> = resp.headers().get_all("x-router-backend").iter().collect();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0], m.url.as_str());
}

#[tokio::test]
async fn open_circuit_backend_is_skipped_not_503() {
    let bad = spawn_mock("bad").await;
    let ok1 = spawn_mock("ok1").await;
    let ok2 = spawn_mock("ok2").await;
    bad.set_fail(true);
    let r = start_router(
        &[&bad, &ok1, &ok2],
        RouterOpts {
            cb_min_requests: 4,
            ..Default::default()
        },
    )
    .await;
    let c = client();
    let mut statuses = Vec::new();
    for _ in 0..60 {
        statuses.push(post_chat(&c, &r.url, "x").await.status().as_u16());
    }
    // Round-robin sends every third request to `bad` until its breaker opens after 4 failures.
    let failures_before_open = statuses.iter().filter(|s| **s == 500).count();
    assert!(
        failures_before_open <= 4,
        "got {failures_before_open} backend 500s: {statuses:?}"
    );
    assert!(
        statuses[30..].iter().all(|s| *s == 200),
        "after the breaker opens every request must succeed: {:?}",
        &statuses[30..]
    );
    assert!(
        !statuses.contains(&503),
        "an open circuit must not surface as 503"
    );
    let text = c
        .get(format!("{}/metrics", r.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        metric_value(&text, "router_circuit_breaker_state", &bad.url),
        Some(2.0)
    );
}

#[tokio::test]
async fn streaming_request_is_counted_until_the_stream_ends() {
    let m = spawn_mock("m1").await;
    m.set_stream(5, 150);
    let r = start_router(&[&m], RouterOpts::default()).await;
    let c = client();
    let url = r.url.clone();
    let c2 = c.clone();
    let streaming = tokio::spawn(async move {
        c2.post(format!("{url}/v1/chat/completions"))
            .json(&chat_body("stream", true))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mid = c
        .get(format!("{}/metrics", r.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        metric_value(&mid, "router_active_connections", &m.url),
        Some(1.0),
        "mid-stream"
    );
    assert_eq!(
        metric_value(&mid, "router_in_flight", ""),
        Some(1.0),
        "mid-stream in_flight"
    );

    let text = streaming.await.unwrap();
    assert!(text.ends_with("data: [DONE]\n\n"));
    tokio::time::sleep(Duration::from_millis(50)).await;
    let after = c
        .get(format!("{}/metrics", r.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        metric_value(&after, "router_active_connections", &m.url),
        Some(0.0)
    );
    assert_eq!(metric_value(&after, "router_in_flight", ""), Some(0.0));
    assert!(
        after.contains("status=\"200\""),
        "completed stream recorded as 200: {after}"
    );
}

#[tokio::test]
async fn oversized_body_returns_413() {
    let m = spawn_mock("m1").await;
    let r = start_router(&[&m], RouterOpts::default()).await;
    let resp = client()
        .post(format!("{}/v1/chat/completions", r.url))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "x".repeat(11 * 1024 * 1024)}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 413);
    assert_eq!(m.requests(), 0);
}

#[tokio::test]
async fn client_abort_mid_stream_releases_in_flight_and_is_not_a_backend_failure() {
    let m = spawn_mock("m1").await;
    m.set_stream(20, 100);
    let r = start_router(&[&m], RouterOpts::default()).await;
    let c = client();
    let mut resp = c
        .post(format!("{}/v1/chat/completions", r.url))
        .json(&chat_body("stream", true))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let first = resp.chunk().await.unwrap();
    assert!(first.is_some());
    drop(resp); // client goes away after the first chunk
    wait_until(
        Duration::from_secs(5),
        "in-flight back to zero after client abort",
        || async {
            let text = c
                .get(format!("{}/metrics", r.url))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            metric_value(&text, "router_in_flight", "") == Some(0.0)
                && metric_value(&text, "router_active_connections", &m.url) == Some(0.0)
        },
    )
    .await;
    let text = c
        .get(format!("{}/metrics", r.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("status=\"client_disconnect\""), "{text}");
    assert_eq!(
        metric_value(&text, "router_circuit_breaker_state", &m.url),
        Some(0.0)
    );
}

#[tokio::test]
async fn half_open_probe_is_single_and_other_backends_keep_serving() {
    let flaky = spawn_mock("flaky").await;
    let ok = spawn_mock("ok").await;
    flaky.set_fail(true);
    let r = start_router(
        &[&flaky, &ok],
        RouterOpts {
            cb_min_requests: 2,
            cb_open_secs: 1,
            ..Default::default()
        },
    )
    .await;
    let c = client();
    // Trip the breaker on `flaky` (round-robin sends every other request there).
    for _ in 0..6 {
        post_chat(&c, &r.url, "x").await;
    }
    let text = c
        .get(format!("{}/metrics", r.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        metric_value(&text, "router_circuit_breaker_state", &flaky.url),
        Some(2.0)
    );

    // After open_duration the breaker is half-open: hold the single probe open by making
    // `flaky` slow, then confirm concurrent requests all go to `ok` (none 503, none to flaky
    // beyond the one probe).
    flaky.set_fail(false);
    flaky.set_latency_ms(800);
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let before = flaky.requests();
    let statuses: Vec<u16> = join_all((0..8).map(|_| post_chat(&c, &r.url, "x")))
        .await
        .into_iter()
        .map(|resp| resp.status().as_u16())
        .collect();
    assert!(statuses.iter().all(|s| *s == 200), "{statuses:?}");
    assert!(
        flaky.requests() - before <= 1,
        "at most one probe while half-open"
    );
    // The probe succeeded, so the circuit closes and flaky re-enters rotation.
    wait_until(Duration::from_secs(3), "flaky circuit closed", || async {
        let text = c
            .get(format!("{}/metrics", r.url))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        metric_value(&text, "router_circuit_breaker_state", &flaky.url) == Some(0.0)
    })
    .await;
}
