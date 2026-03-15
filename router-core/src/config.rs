use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct RouterConfig {
    pub listen: ListenConfig,
    pub routing: RoutingConfig,
    pub backends: Vec<BackendConfig>,
    #[serde(default)]
    pub health_check: HealthCheckConfig,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub backpressure: BackpressureConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListenConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    #[serde(default = "default_strategy")]
    pub strategy: String,
    #[serde(default)]
    pub prefix_cache: PrefixCacheConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PrefixCacheConfig {
    #[serde(default = "default_prefix_token_count")]
    pub prefix_token_count: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            prefix_token_count: default_prefix_token_count(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackendConfig {
    pub url: String,
    #[serde(default = "default_weight")]
    pub weight: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealthCheckConfig {
    #[serde(default = "default_interval_secs")]
    pub interval_secs: u64,
    #[serde(default = "default_unhealthy_threshold")]
    pub unhealthy_threshold: u32,
    #[serde(default = "default_healthy_threshold")]
    pub healthy_threshold: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval_secs: default_interval_secs(),
            unhealthy_threshold: default_unhealthy_threshold(),
            healthy_threshold: default_healthy_threshold(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CircuitBreakerConfig {
    #[serde(default = "default_failure_rate")]
    pub failure_rate_threshold: f64,
    #[serde(default = "default_min_requests")]
    pub min_requests: u64,
    #[serde(default = "default_window_secs")]
    pub window_secs: u64,
    #[serde(default = "default_open_duration_secs")]
    pub open_duration_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_rate_threshold: default_failure_rate(),
            min_requests: default_min_requests(),
            window_secs: default_window_secs(),
            open_duration_secs: default_open_duration_secs(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct BackpressureConfig {
    #[serde(default = "default_max_in_flight")]
    pub max_in_flight: usize,
    #[serde(default = "default_max_backend_queue")]
    pub max_backend_queue_depth: u64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_in_flight: default_max_in_flight(),
            max_backend_queue_depth: default_max_backend_queue(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct MetricsConfig {
    #[serde(default = "default_metrics_enabled")]
    pub enabled: bool,
    #[serde(default = "default_metrics_path")]
    pub path: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: default_metrics_enabled(),
            path: default_metrics_path(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_strategy() -> String {
    "round_robin".to_string()
}
fn default_prefix_token_count() -> usize {
    256
}
fn default_weight() -> u32 {
    1
}
fn default_interval_secs() -> u64 {
    5
}
fn default_unhealthy_threshold() -> u32 {
    3
}
fn default_healthy_threshold() -> u32 {
    2
}
fn default_failure_rate() -> f64 {
    0.5
}
fn default_min_requests() -> u64 {
    5
}
fn default_window_secs() -> u64 {
    60
}
fn default_open_duration_secs() -> u64 {
    30
}
fn default_max_in_flight() -> usize {
    1000
}
fn default_max_backend_queue() -> u64 {
    50
}
fn default_metrics_enabled() -> bool {
    true
}
fn default_metrics_path() -> String {
    "/metrics".to_string()
}

impl RouterConfig {
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
}
