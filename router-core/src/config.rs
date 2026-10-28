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
    #[serde(default)]
    pub latency_aware: LatencyAwareConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LatencyAwareConfig {
    /// EWMA smoothing factor in (0, 1]. Higher reacts faster to the latest sample.
    #[serde(default = "default_ewma_alpha")]
    pub alpha: f64,
}

impl Default for LatencyAwareConfig {
    fn default() -> Self {
        Self {
            alpha: default_ewma_alpha(),
        }
    }
}

fn default_ewma_alpha() -> f64 {
    0.3
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

#[derive(Debug)]
pub enum ConfigError {
    Parse(serde_yaml::Error),
    Invalid(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid config syntax: {e}"),
            Self::Invalid(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl RouterConfig {
    pub fn from_yaml(contents: &str) -> Result<Self, ConfigError> {
        let cfg: Self = serde_yaml::from_str(contents).map_err(ConfigError::Parse)?;
        cfg.validate().map_err(ConfigError::Invalid)?;
        Ok(cfg)
    }

    /// Reject configs that would misbehave at runtime instead of failing at startup.
    pub fn validate(&self) -> Result<(), String> {
        let alpha = self.routing.latency_aware.alpha;
        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return Err(format!(
                "routing.latency_aware.alpha must be in (0, 1], got {alpha}"
            ));
        }
        let rate = self.circuit_breaker.failure_rate_threshold;
        if !rate.is_finite() || rate <= 0.0 || rate > 1.0 {
            return Err(format!(
                "circuit_breaker.failure_rate_threshold must be in (0, 1], got {rate}"
            ));
        }
        if self.backends.is_empty() {
            return Err("backends must not be empty".to_string());
        }
        for b in &self.backends {
            let scheme_ok = b.url.starts_with("http://") || b.url.starts_with("https://");
            if !scheme_ok || http::HeaderValue::from_str(&b.url).is_err() {
                return Err(format!(
                    "backend url {:?} must start with http:// or https:// and contain no control characters",
                    b.url
                ));
            }
        }
        Ok(())
    }

    pub fn listen_addr(&self) -> Result<std::net::SocketAddr, std::net::AddrParseError> {
        format!("{}:{}", self.listen.host, self.listen.port).parse()
    }

    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        Ok(Self::from_yaml(&contents)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_yaml_uses_defaults() {
        let cfg = RouterConfig::from_yaml(
            "listen: {}\nrouting: {}\nbackends:\n  - url: \"http://a:8000\"\n",
        )
        .unwrap();
        assert_eq!(cfg.listen.port, 8080);
        assert_eq!(cfg.routing.strategy, "round_robin");
        assert_eq!(cfg.routing.latency_aware.alpha, 0.3);
        assert_eq!(cfg.backends[0].weight, 1);
        assert_eq!(cfg.circuit_breaker.min_requests, 5);
    }

    #[test]
    fn latency_aware_alpha_is_configurable() {
        let cfg = RouterConfig::from_yaml(
            "listen: {}\nrouting:\n  strategy: latency_aware\n  latency_aware:\n    alpha: 0.5\nbackends:\n  - url: \"http://a:8000\"\n",
        )
        .unwrap();
        assert_eq!(cfg.routing.latency_aware.alpha, 0.5);
    }

    fn yaml_with(alpha: &str, backends: &str) -> String {
        format!(
            "listen: {{}}\nrouting:\n  latency_aware:\n    alpha: {alpha}\nbackends: {backends}\n"
        )
    }

    const ONE_BACKEND: &str = "\n  - url: \"http://a:8000\"";

    #[test]
    fn rejects_out_of_range_alpha() {
        for alpha in ["0", "1.5", ".nan"] {
            let err = RouterConfig::from_yaml(&yaml_with(alpha, ONE_BACKEND)).unwrap_err();
            assert!(err.to_string().contains("alpha"), "{alpha}: {err}");
        }
    }

    #[test]
    fn rejects_empty_backends() {
        let err = RouterConfig::from_yaml(&yaml_with("0.3", "[]")).unwrap_err();
        assert!(err.to_string().contains("backends"), "{err}");
    }

    #[test]
    fn rejects_backend_url_with_newline() {
        let err = RouterConfig::from_yaml(&yaml_with("0.3", "\n  - url: \"http://a:8000\\nx\""))
            .unwrap_err();
        assert!(err.to_string().contains("url"), "{err}");
    }

    #[test]
    fn rejects_out_of_range_failure_rate_threshold() {
        for rate in ["0", "1.5"] {
            let yaml = format!(
                "listen: {{}}\nrouting: {{}}\ncircuit_breaker:\n  failure_rate_threshold: {rate}\nbackends:{ONE_BACKEND}\n"
            );
            let err = RouterConfig::from_yaml(&yaml).unwrap_err();
            assert!(
                err.to_string().contains("failure_rate_threshold"),
                "{rate}: {err}"
            );
        }
    }
}
