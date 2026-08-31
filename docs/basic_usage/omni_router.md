# Omni Router

The SGLang-Omni Rust Router is a standalone process. This foundation change
provides the process, listener, configuration, liveness, logging, and shutdown
contracts on which worker routing is built. It does not yet install worker or
inference routes.

## Build and start

Install [Rustup](https://rustup.rs/), then build the optimized binary from a
source checkout:

```bash
cd sglang_omni_router/rust
cargo build --release --locked
```

Create `router.toml` in that directory:

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

Validate the file without creating a runtime or listener:

```bash
./target/release/sgl-omni-router --config router.toml --check-config
```

Start the router and check liveness:

```bash
./target/release/sgl-omni-router --config router.toml
curl --fail http://127.0.0.1:30000/live
```

`GET /live` returns `200` with `live`. Other methods on the exact path return
`405`; unknown paths return `404`. `/ready`, worker health, inference, media,
WebSocket, voice, and operations routes are not installed at this layer.

## Configuration

Configuration is UTF-8 TOML. Unknown fields, duplicate fields, missing
required sections, unsupported schema versions, invalid tracing filters, zero
durations, and invalid connection limits fail validation.

| Field | Default | Meaning |
| --- | ---: | --- |
| `schema_version` | required | Configuration schema; this layer accepts `1` |
| `server.listen` | required | Numeric listener address |
| `server.max_connections` | `1024` | Maximum accepted client sockets |
| `server.header_read_timeout_ms` | `30000` | Time allowed for each initial or keep-alive HTTP/1 request head |
| `shutdown.drain_timeout_ms` | required | Graceful shutdown deadline |
| `logging.format` | required | `json` or `compact` |
| `logging.filter` | required | Tracing filter expression |

`--check-config` validates only the file. It does not create the Tokio runtime,
bind the listener, initialize tracing, inspect process file limits, or change
the invoking shell.

## Listener behavior

The router acquires one permit before accepting a socket. If all permits are
owned, new connections remain in the kernel accept queue until capacity is
released. Each accepted transport retains its permit until the transport
closes, including after an HTTP upgrade.

Accepted sockets enable `TCP_NODELAY`. Connection-level accept failures are
retried immediately. Other accept failures are logged and retried after one
second so a transient Linux network or descriptor failure does not terminate
existing requests.

`server.header_read_timeout_ms` applies while Hyper waits for an initial or
keep-alive request head, including a partial head. It does not limit a request
body, active handler, response, stream, or upgraded transport.

On Unix, runtime startup raises the process `RLIMIT_NOFILE` soft limit toward
the hard limit selected by the operator. Startup fails if the resulting soft
limit cannot fit `server.max_connections` plus the listener. Additional file
descriptors used by the runtime and later router features remain part of the
deployment budget.

## Network boundary

`server.listen` accepts a numeric socket address such as `127.0.0.1:30000` or
`0.0.0.0:30000`. The router does not provide client authentication or TLS. Run
it on a trusted network or behind an authenticated TLS proxy.

## Shutdown

The first `SIGINT` or `SIGTERM` closes the listener, stops accepting new
connections, and asks established HTTP connections to drain. Idle connections
close immediately; active work can finish within
`shutdown.drain_timeout_ms`. The process joins every owned connection task
before a clean exit.

A distinct second signal or the drain deadline aborts remaining connection
tasks and exits with failure. A connection-task panic also fails the server
task instead of being silently detached.

## Related guides

- [Rust development](../../sglang_omni_router/rust/DEVELOPMENT.md)
- [Python router](python_router.md), available as `sgl-omni-router-py`
