//! Distributed LLM inference router: routes OpenAI-compatible requests across vLLM replicas.

pub mod accounting;
pub mod app;
pub mod backend;
pub mod backpressure;
pub mod circuit_breaker;
pub mod config;
pub mod health;
pub mod metrics;
pub mod proxy;
pub mod routing;
pub mod server;
