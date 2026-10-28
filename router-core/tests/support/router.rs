use inference_router::app::build_app;
use inference_router::config::RouterConfig;

use super::mock_backend::MockHandle;

pub struct RouterHandle {
    pub url: String,
    serve_task: tokio::task::JoinHandle<()>,
    health_task: tokio::task::JoinHandle<()>,
}

impl Drop for RouterHandle {
    fn drop(&mut self) {
        self.serve_task.abort();
        self.health_task.abort();
    }
}

pub struct RouterOpts {
    pub strategy: &'static str,
    pub weights: Vec<u32>,
    pub max_in_flight: usize,
    pub health_interval_secs: u64,
    pub unhealthy_threshold: u32,
    pub healthy_threshold: u32,
    pub cb_min_requests: u64,
    pub cb_open_secs: u64,
    pub max_backend_queue_depth: u64,
    pub metrics_enabled: bool,
    pub read_timeout_secs: u64,
}

impl Default for RouterOpts {
    fn default() -> Self {
        Self {
            strategy: "round_robin",
            weights: Vec::new(),
            max_in_flight: 1000,
            health_interval_secs: 1,
            unhealthy_threshold: 1,
            healthy_threshold: 1,
            cb_min_requests: 4,
            cb_open_secs: 30,
            max_backend_queue_depth: 50,
            metrics_enabled: true,
            read_timeout_secs: 120,
        }
    }
}

pub fn test_config_yaml(backend_urls: &[&str], opts: &RouterOpts) -> String {
    let mut backends = String::new();
    for (i, url) in backend_urls.iter().enumerate() {
        let w = opts.weights.get(i).copied().unwrap_or(1);
        backends.push_str(&format!("  - url: \"{url}\"\n    weight: {w}\n"));
    }
    format!(
        "listen:\n  host: \"127.0.0.1\"\n  port: 0\n\
         routing:\n  strategy: \"{strategy}\"\n\
         backends:\n{backends}\
         health_check:\n  interval_secs: {hi}\n  unhealthy_threshold: {ut}\n  healthy_threshold: {ht}\n\
         circuit_breaker:\n  failure_rate_threshold: 0.5\n  min_requests: {minr}\n  window_secs: 60\n  open_duration_secs: {open}\n\
         backpressure:\n  max_in_flight: {mif}\n  max_backend_queue_depth: {mqd}\n\
         metrics:\n  enabled: {men}\n  path: \"/metrics\"\n\
         upstream:\n  connect_timeout_secs: 5\n  read_timeout_secs: {rt}\n",
        strategy = opts.strategy,
        hi = opts.health_interval_secs,
        ut = opts.unhealthy_threshold,
        ht = opts.healthy_threshold,
        minr = opts.cb_min_requests,
        open = opts.cb_open_secs,
        mif = opts.max_in_flight,
        mqd = opts.max_backend_queue_depth,
        men = opts.metrics_enabled,
        rt = opts.read_timeout_secs,
    )
}

pub async fn start_router(backends: &[&MockHandle], opts: RouterOpts) -> RouterHandle {
    let urls: Vec<&str> = backends.iter().map(|b| b.url.as_str()).collect();
    start_router_with_urls(&urls, opts).await
}

/// Like `start_router`, but with backend URLs written verbatim into the config.
pub async fn start_router_with_urls(urls: &[&str], opts: RouterOpts) -> RouterHandle {
    let config = RouterConfig::from_yaml(&test_config_yaml(urls, &opts)).expect("valid config");
    let app = build_app(config).expect("app builds");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let serve_task = tokio::spawn(async move {
        axum::serve(listener, app.router).await.unwrap();
    });
    RouterHandle {
        url: format!("http://{addr}"),
        serve_task,
        health_task: app.health_task,
    }
}
