# AGENT.md — merlin/python/merlin/runtime

## Purpose

The Python reference runtimes that execute Merlin command buffers / dispatch programs, plus
the per-backend adapters under `backends/`.

## What belongs here

- `bound_caller_layout` reopens the exact core data-bound runtime, selected
  capability contract and facts before exposing the original neutral renderer's
  dense extents and ordinal buffer symbols. Its versioned format-only query
  grants no OOT hooks, target effects, readback execution or numerical/timing
  authority. Padding, transformed storage, unsupported byte order and changed
  source/selection refuse; the original allocation ceiling remains fixed.

- `tensor_payload` transports exact lazy scalar input bytes to the existing
  logical-pointer renderer. Original shape/dtype/endian/extent and readonly
  argument checks precede binary sidecars; data sections, mutable scratch,
  output poisoning and complete readback retain their selected semantics.
  This supplies no packing, memref bridge, numerical or device authority.

- `host_provider` keeps its original object/ABI/pin/linked-image checks. Its
  sole empty-builder exception recognizes the exact original
  `PointwiseSourceRouting.build` owner and reopens a completed zero-island,
  unchanged-source preparation before returning the normal no-provider route.
  Foreign, incomplete or stale owners refuse; arbitrary builders cannot return
  None to bypass provider admission. No numerical/effect witness is fabricated.

- `complete_call_timing` selects an optional reviewed normal contract boundary:
  `logical_harness.complete_call_timing` names an existing `counter_semantics`
  record and an explicit positive delta bound below 2**63. The record must bind
  actual accessor/ISA/clock source files, cycles, 64-bit width and modulo wrap;
  the existing selected counter assembly is reused. No default counter or clock
  exists. Both original callers include main preparation, submitted whole-program
  work, completion and every output publication. Static immutable staging,
  loader/startup, final metric formatting/exit and external parent readback are
  excluded explicitly. Cold/warm qualification remains UNKNOWN. Actual source
  and engine certificates remain necessary; this reviewed data is no new role.
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
- `native_runtime.py`, `_native_files.py` and `_native_launch.py` optionally
  freeze one bounded shared filesystem view for the original normal Chipyard
  backend. Selected native calls adopt that readonly view, one requested ELF
  and fresh closed writable scratch. Real kernel process/mount observations
  precede exec, and actual process records retain raw streams. Saved JSON cannot
  issue this live owner. Host release follows the original guardian's cleanup;
  unknown cleanup retains ownership. Filesystem closure does not establish the
  engine's unspecified reset/seed: current selections always execute freshly.
- `build_runtime.py` optionally adopts that same launcher for the original
  translation, object, harness and link subprocesses. Explicit tools, immutable
  lookup roots, fresh input/product rosters and file/byte budgets precede any
  selection. Per-job storage relocation changes the actual private child paths;
  the executed mapping owns its command key. Unsupported choices remain fresh.
  Signed completed observations require real kernel adoption and preserved
  original process/product files. Artifact reuse is a separate receipt and
  establishes no numerical, simulator or timing authority.
- `dispatch_placement.py` joins the original ordered outline table to actual
  typed functions/calls and placement attributes/properties before dispatch.
  Explicit nonhost intent refuses native/XNNPACK substitution and a declined
  device route before host execution or a successful host ledger entry.
  Unspecified placement retains fallback behavior. This structural requirement
  grants no device capability, command-event, ownership or synchronization role.
- `host_math.py` — explicit portable libm evaluation policies and their compiled runtime objects.
  The default emits nothing. A selected policy changes linked-byte identity and must pass the
  caller's original numerical contract; it does not promise errno or exception-flag equivalence.
- `host_math_reference.py` — the explicit `portable_exp_trig` array oracle compiles the same
  pinned Arm exp/sin/cos C sources as the selected runtime recipe. Its task-local context defaults
  to absent; binary32 inputs and round-to-nearest are required. Source, compiler, object and shared
  library bytes remain bound and checked. Arithmetic correspondence alone grants no model,
  accelerator or timing qualification; changed arithmetic requires new reference evidence.
  `portable_neural_f32` additionally selects shared Arm powf and OpenLibm tanhf/private expm1f
  sources; binary pow operands broadcast in binary32. The earlier exp/trig policy retains its
  original three-operation scope. C11 precision semantics preserve the strict no-excess-precision
  guard on RISC-V GCC configurations that otherwise report TS18661's binary16 evaluation mode.
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
- `direct_kernel_counter.py` — opt-in raw uint64 counter/control-state samples
  around every selected entry/completion boundary and empty calibration bracket.
  Complete sample objects and counts require the coherent reader; explicit
  budgets apply before storage rendering. Accessor implementations remain in
  selected runtime support. No counter units, integrity, cold/warm status,
  calibrated overhead or complete-stage cost authority follows from these data.
- `direct_kernel_phases.py` — opt-in raw samples at fixed generated console,
  calibration, output-history/count and DONE boundaries, plus inclusive main.
  It requires the same explicit counter plan and original repeated ABI/history
  roster; combined buffer/history/sample budgets precede rendering. Entry and
  selected completion retain their original counter bracket. Main includes
  inner instrumentation but ends before its own sample storage/count, exit and
  parent readback. Static initialization/loading and these excluded operations
  remain unknown, never free. Raw values do not grant completion, counter units,
  cold/warm, full eleven-stage costs or runtime/held qualification.
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

- `dispatch_arguments.py` owns captured weight/input plan construction. The
  dispatch runtime retains `_forward_plan` and `ForwardArgsCache` as its binding
  and cache entrypoints, passing the selected numeric converters explicitly.

- `host_fma_packet.py` and `host_rounding.py` own explicitly selected scalar CPU
  implementations independently of the accelerator; adapters retain symbol bindings.
  CPU selection never grants numerical or effect permissions.

An `acc_scale` stage selects either one scalar multiplier or an explicit
`acc_scale_by_column` vector. Column vectors have one finite float32 value per
final-axis column and retain scalar float32 conversion/multiply and ties-even
rounding; saturation remains separate. Reference arithmetic grants no target
admission, command schedule, device lowering or RTL qualification.

- Real implementations only; every result is gated (simulator == reference; dispatch
  runtime == torch golden). `dispatch_runtime` raises on any view op it cannot evaluate —
  no silent skips.
- `linalg_numpy.numeric_contract()` versions reference arithmetic. Named and non-contraction
  generic floating reductions preserve source index order, the original init and per-operation
  rounding; f32/f64 combiners may explicitly permit reassociation. New evaluator bytes/policy require
  newly derived reference evidence; never rewrite or relabel a historical sealed bundle.
- Every subdirectory must also contain an AGENT.md.
