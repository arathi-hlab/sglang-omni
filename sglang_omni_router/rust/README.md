# SGLang-Omni Rust Router

`sgl-omni-router` is the standalone Rust router process for SGLang-Omni. This
foundation provides strict configuration, a bounded HTTP/1 listener,
`GET /live`, structured logging, and joined shutdown. Worker routing and
inference APIs are added by later router changes.

## Build and run

Install [Rustup](https://rustup.rs/), enter this directory, and build the
optimized binary:

```console
cargo build --release --locked
```

Create `router.toml`:

```toml
schema_version = 1

[server]
listen = "127.0.0.1:30000"
max_connections = 1024
header_read_timeout_ms = 30000

[shutdown]
drain_timeout_ms = 30000

[logging]
format = "json"
filter = "info"
```

Validate the file without creating a runtime or listener, then start the
router:

```console
./target/release/sgl-omni-router --config router.toml --check-config
./target/release/sgl-omni-router --config router.toml
```

Check process liveness:

```console
curl --fail http://127.0.0.1:30000/live
```

At this foundation layer, `/live` is the only installed route. Readiness,
worker health, inference, media, WebSocket, voice, and operations routes are
not installed yet.

## Runtime contracts

- `server.max_connections` bounds accepted client sockets. Capacity is
  acquired before `accept`, and each accepted transport owns its permit until
  the transport closes, including after an HTTP upgrade.
- `server.header_read_timeout_ms` bounds each initial or keep-alive HTTP/1
  request head. It does not limit request bodies, handlers, or responses.
- Accepted sockets enable `TCP_NODELAY`. Connection-level accept failures are
  retried immediately; all other accept failures are logged and retried with
  backoff.
- On Unix, startup raises the process `RLIMIT_NOFILE` soft limit toward its
  operator-controlled hard limit. Startup fails if the resulting limit cannot
  fit `server.max_connections` plus the listener. `--check-config` does not
  inspect or change process limits.
- `logging.format` accepts `json` or `compact`. `logging.filter` is a tracing
  filter expression.
- The first `SIGINT` or `SIGTERM` closes the listener and drains established
  connections. A second signal or the configured drain deadline forces a
  failed shutdown.

The listener accepts any configured numeric socket address. This layer does
not provide authentication or TLS, so expose it only on a trusted network or
behind an authenticated TLS proxy.

For the complete field reference and operating behavior, see the
[Omni Router guide](../../docs/basic_usage/omni_router.md). For pinned
toolchains and local quality gates, see [DEVELOPMENT.md](DEVELOPMENT.md).

The existing Python router remains available as `sgl-omni-router-py`; see the
[Python router guide](../../docs/basic_usage/python_router.md).
