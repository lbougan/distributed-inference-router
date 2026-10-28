//! A raw TCP "backend" for failures axum cannot produce: every connection gets 200 headers
//! announcing a 100-byte body, a few bytes of it, and then a closed socket.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

pub struct RawHandle {
    pub url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for RawHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

const TRUNCATED: &[u8] = b"HTTP/1.1 200 OK\r\n\
content-type: application/json\r\n\
content-length: 100\r\n\
\r\n\
{\"id\":";

pub async fn spawn_truncating_backend() -> RawHandle {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                // Read the request head (and whatever body arrived with it), then answer.
                let mut buf = vec![0u8; 64 * 1024];
                let _ = sock.read(&mut buf).await;
                let _ = sock.write_all(TRUNCATED).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    RawHandle {
        url: format!("http://{addr}"),
        task,
    }
}
