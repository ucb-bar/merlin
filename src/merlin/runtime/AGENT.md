# AGENT.md — merlin/python/merlin/runtime

## Purpose

The Python reference runtimes that execute Merlin command buffers / dispatch programs, plus
the per-backend adapters under `backends/`.

## What belongs here

- `simulator.py` / `reference.py` / `tensor.py` / `metrics.py` / `commandbuffer.py` — the
  synthetic-workload command-buffer engine (integer tensor math + metrics) and an
  independent reference recomputation it is gated against.
- `dispatch_runtime.py` — the **whole-model** host executor: outlines a captured model,
  compiles each kernel in isolation (`llvmlower.kernel_backend`, deduplicated by kernel
  body), evaluates the driver's view ops (`expand/collapse_shape`, `extract_slice`,
  `concat`, `splat`, `constant`) in numpy, invokes the compiled kernel symbols in order,
  and gates the output against the torch golden. Forward args (inputs + safetensors weights
  + extra buffers) are bound exactly as the C runtime binds them. Verified: whole
  small_llama (cos 0.9999999) and TinyLlama-1.1B (cos 1.0, next-token argmax exact on all
  tokens) == torch through the dispatch table. Kernels with a by-value scalar arg (e.g. a
  `cumsum` accumulator-init `i64`) are passed by value via `abi.ScalarArg`, NOT as a memref
  descriptor — `emit_c_interface` only wraps memrefs.
- `backends/` — host / spike adapters.
- `host_math.py` — explicit portable libm evaluation policies and their compiled runtime objects.
  The default emits nothing. A selected policy changes linked-byte identity and must pass the
  caller's original numerical contract; it does not promise errno or exception-flag equivalence.
- `host_arithmetic.py` and `host_outward.py` share explicitly selected CPU
  arithmetic implementations independently of accelerators. ISA/ABI, source
  arithmetic and effect permissions remain caller obligations. Namespaces bind
  symbols without selecting a workload or numeric policy. The available host
  profile refuses unsupported CPUs; it grants no hardware/runtime authority.
  Target ABI bindings delegate here rather than duplicate host instructions.
- `out_b64.py` — lossless, opt-in chunked container-word console decoding. Complete
  frames reconstruct every value before ordinary numerical checks; malformed,
  missing, duplicated or interrupted frames refuse. The shared console parser
  still requires terminal `DONE`. Transport changes grant no numerical support,
  output sampling, digest-only qualification or simulator certification.
- `direct_kernel_harness.py` — pure pointer-call storage and full-value publishing
  from an explicit software ABI and exact input bytes. Contains no target
  instruction or reference compiler. Calling a declared completion symbol does
  not independently prove synchronization, ownership or hardware correspondence.
- `direct_kernel_invocation.py` — explicitly selected repeated calls with the
  original ordered ABI, complete raw output snapshots for each call and observed
  completion count. Reuses source input storage and separate output histories;
  alias interfaces refuse. Histories and counts do not establish effect, source,
  platform or timing authority.
- `out_bin.py` — separate opt-in byte-oriented full-value framing. It consumes
  exact length-delimited raw payloads without text-decoding their NUL/non-UTF-8
  contents, checks the transport checksum, and requires END/DONE and a closed
  output roster before the unchanged numerical checker sees actual values.
- `out_packet.py` reuses the binary codec for closed in-memory full-value packets,
  bounded by a caller-owned typed output roster. Packet DONE is wire closure only;
  native completion and numerical comparison remain independent obligations.

## What does not belong here

- The deployable C runtime (that is `merlin/runtime/`, outside the Python tree).
- Generated artifacts (write those to `runs/` or `artifacts/`).

## Invariants

- Real implementations only; every result is gated (simulator == reference; dispatch
  runtime == torch golden). `dispatch_runtime` raises on any view op it cannot evaluate —
  no silent skips.
- Every subdirectory must also contain an AGENT.md.
