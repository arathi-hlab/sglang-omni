# Omni Router

The SGLang-Omni Rust Router is a standalone process that routes chat,
multimodal, speech, transcription, translation, and realtime requests across
static SGLang-Omni worker replicas. It selects one compatible healthy worker
for each request or session and relays the original request and response with
bounded admission.

The router does not launch or supervise model workers. Start the workers in
the selected manifest before starting the router.

## Build and start

Install [Rustup](https://rustup.rs/), then build the optimized binary from a
source checkout:

```bash
cd sglang_omni_router/rust
cargo build --release --locked
```

The checked Omni manifest expects two compatible workers at
`127.0.0.1:8000` and `127.0.0.1:8001`. Start those workers, validate the
manifest, and start the router:

```bash
./target/release/sgl-omni-router \
  --config examples/omni.toml \
  --check-config

./target/release/sgl-omni-router \
  --config examples/omni.toml
```

Wait for readiness:

```bash
curl --fail http://127.0.0.1:30000/ready
```

Then send a request through the router:

```bash
curl --http1.1 http://127.0.0.1:30000/v1/chat/completions \
  --header 'content-type: application/json' \
  --data-binary \
  '{"model":"omni-model","messages":[{"role":"user","content":"hello"}]}'
```

## Checked manifests

Choose the manifest that matches the worker service:

| Manifest | Worker contract |
| --- | --- |
| `examples/omni.toml` | Two replicas serving multimodal chat and audio output |
| `examples/tts.toml` | Two replicas serving encoded speech and PCM streaming |
| `examples/asr.toml` | Two replicas serving transcription and translation |

Replace the example worker URLs, model IDs, service profiles, admission limits,
and worker capacities with values for the deployment. The included capacities
are starting values, not performance recommendations.

Configuration is strict. Unknown fields, duplicate identities, invalid limits,
missing admission classes, or incompatible capability rows fail
`--check-config` and startup.

## Listener and network boundary

`server.listen` accepts a numeric socket address. The checked manifests listen
on `0.0.0.0:30000` so clients can reach the router from the deployment network.

The router does not provide client authentication or TLS. Run it on a trusted
network or behind an authenticated TLS proxy. The data-plane, health, metrics,
and diagnostics routes share this listener.

## Workers and service profiles

Each worker entry defines:

- one stable worker ID and base URL;
- one trust domain;
- exact per-service capacity;
- an optional default model ID;
- correlated service-profile rows describing combinations the worker actually
  supports.

The router does not combine fields from different profile rows. A model,
modality, response format, task, stream mode, and voice mode are eligible only
when one row advertises that complete combination.

A worker URL with a DNS authority must also provide `resolved_ip`. The router
connects to that pinned address while preserving the configured authority for
HTTP `Host` and TLS SNI.

Worker membership is static for the lifetime of the process. Update the
manifest and restart the router to change the pool.

## Supported APIs

Only configured data-plane routes are installed:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat, multimodal generation, and chat audio output |
| `POST` | `/v1/audio/speech` | Encoded speech or streaming PCM |
| `POST` | `/v1/audio/speech/batch` | One ordered, unsplit speech batch |
| `POST` | `/v1/audio/transcriptions` | Multipart transcription |
| `POST` | `/v1/audio/translations` | Multipart speech translation |
| `GET` | `/v1/audio/speech/stream` | Terminating speech WebSocket |
| `GET` | `/v1/realtime?model=<id>` | Terminating OpenAI-compatible realtime WebSocket |
| `GET` | `/v1/audio/voices` | List voices on the configured voice owner |
| `POST` | `/v1/audio/voices` | Upload one voice to the configured voice owner |
| `DELETE` | `/v1/audio/voices/{name}` | Delete one voice from the configured voice owner |

Voice routes exist only when `router.voice_owner_worker_id` names a worker
with control capacity and a `voice_control` profile. The router does not store,
replicate, or reconcile worker-local voice data.

Transcription and translation are separate profile tasks. Advertising
transcription does not implicitly enable translation. Speech profiles likewise
keep encoded response formats separate from PCM streaming support.

## Routing

`round_robin` distributes requests across eligible equal replicas.
`least_requests` selects the eligible worker with the lowest exact in-flight
capacity occupancy and uses deterministic ties. Select the policy using the
deployment's model and workload measurements.

For a homogeneous cohort with a concrete shared default model and matching
service profiles, eligible fixed-length requests take the direct path without
body classification. Requests with body-owned routing facts reserve bounded
aggregate byte capacity, buffer once, classify once, and forward the original
bytes unchanged.

Every request uses one upstream worker. The router does not retry or reselect a
worker after dispatch. Admission and worker-capacity leases remain owned until
response EOF, error, or downstream cancellation.

Speech batches remain ordered and are never split. The router atomically
reserves the complete item count from one worker before dispatch.

## Streaming and WebSockets

HTTP request and response bodies are relayed with direct backpressure. SSE and
PCM responses are not accumulated before being sent downstream.

The speech and realtime WebSocket routes terminate both handshakes and pin one
worker for the session. Frames retain their type and order under destination
backpressure. `websocket.setup_timeout_ms` bounds setup only; it is not a
session idle timeout.

## Health and readiness

- `GET /live` reports process liveness.
- `GET /ready` returns `200` only while the router is serving and every enabled
  route has a compatible healthy worker.
- Workers begin `Unknown` and become healthy after the configured consecutive
  probe successes.
- Workers become unhealthy after the configured consecutive probe failures.
- Transport and upstream protocol failures request an immediate probe.
- Worker application errors and capacity exhaustion do not directly change
  worker health.

If readiness remains unavailable, inspect `/diagnostics` and confirm that each
enabled route has a matching worker profile, capacity entry, and healthy
worker.

## Operations

The router exposes read-only local state without contacting workers:

| Path | Response |
| --- | --- |
| `/v1/models` | Sorted model inventory computed at startup |
| `/metrics` | Prometheus lifecycle, readiness, worker, admission, and capacity gauges |
| `/diagnostics` | Bounded worker, lifecycle, admission, and capacity state |

A process-wide `x-request-id` covers data, WebSocket, health, operations,
`404`, and `405` responses. A valid caller value is preserved; otherwise the
router generates one and forwards it to the selected worker.

## Shutdown

The first `SIGINT` or `SIGTERM` fails readiness, closes admission, stops new
dispatch, asks tracked WebSockets to close, stops health work, and joins owned
work within `shutdown.drain_timeout_ms`. A second signal forces shutdown.

`server.header_read_timeout_ms` reclaims idle HTTP/1 connections while waiting
for an initial or keep-alive request head. It does not bound request bodies,
responses, streams, or WebSocket sessions. On POSIX systems,
`server.max_connections` must also fit below the process `RLIMIT_NOFILE` soft
limit with room for the listener; size the remaining descriptor capacity for
the enabled upstream and operational connections.

## Current scope

The Rust router uses one static manifest and one multi-threaded process. It does
not provide worker launching or supervision, YAML launcher configuration,
dynamic worker CRUD or discovery, queues, circuit breakers, cache-aware or PD
routing, CP/DP shared state, Python bindings, client authentication, or TLS.

The Python router remains available for RL and Python-specific control-plane
workflows. See the [Python router guide](python_router.md) for its complete
configuration and operating model.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Configuration validation fails | Read the reported field, correct the manifest, and rerun `--check-config` |
| `/ready` returns `503` | Check worker health, trust domains, capacities, and correlated profiles in `/diagnostics` |
| A request returns `503` | Check global, service-class, and selected-worker capacity |
| A request returns `502` or `504` | Check worker reachability, protocol behavior, and configured timeouts |
| The router is unreachable remotely | Check `server.listen`, host firewall rules, and the deployment network |

For the pinned Rust toolchain and local quality gates, see
the [Rust development guide](../../sglang_omni_router/rust/DEVELOPMENT.md).
