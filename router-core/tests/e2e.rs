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
