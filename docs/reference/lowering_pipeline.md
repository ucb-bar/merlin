---
title: Lowering pipeline
kind: reference
status: current
owner: ir
last_verified: 2026-10-08
related: [core_dialects, llvm_integration]
code_refs: [src/merlin/xdsl_dialects/lowering, src/merlin/llvmlower]
---

# The Merlin lowering pipeline, phase by phase

Merlin's dialects (`contract`, `schedule`, `interface`, `runtime`, `dse`, and target
dialects like `toynpu`/`saturn`) **coexist with and hand off to** the upstream MLIR
dialects (`builtin`, `func`, `linalg`, `tensor`, `arith`, `scf`, `math` → `memref`,
`vector`, `cf`, `bufferization` → `llvm`). Our dialects carry *decisions and
abstractions* that must survive passes and be verified; the upstream dialects carry the
*computation* that actually lowers to machine code. The payload descends one layer at a
time while our annotations are attached, consumed, and dropped at the right boundaries.

This doc lists every phase: the dialects present, what the phase does, the **ordered
passes** (marked `[upstream]` = reuse as-is, `[merlin]` = we author it), the dialects it
produces, and the **debug checkpoint** for that boundary. Reasoning and debugging happen
per boundary: each phase's output should `verify()`, and the per-kernel phases gate on a
torch reference.

Status legend: ✅ built & verified · ◐ partially built · ⏳ planned (approved).

## Explicit generated entry projection

The normal model builder accepts `entry_weight_projection`, an optional typed
`GeneratedDispatchABI` contract. After shared preparation, the compiler derives
a complete argument table from the original capture, quant-inner lift and
quantization hoist. It can remove unused whole immutable stored or generated-zero
parameters after conservative pure tensor DCE. Model inputs, captured buffers,
outputs and prepared trailing arguments retain their ownership and semantics.

Selection requires explicit generated-caller, unobserved-address, readonly-weight
and arithmetic-effect permissions. Every argument/result type and direct caller
is checked. Unknown symbolic callers and ABI attributes refuse; unknown effects
and physical memory allocations remain live. Session state or stream declarations
cannot grant immutable ownership to a captured weight. Source files and the
presence or absence of optional storage/session/hoist files are checked before
planning, application and C runtime generation.

One original-to-projected mapping rewrites the entry, direct callers, generated
C interface, descriptors, state/stream indices and retained packed weight spans.
The original bundle and module remain unchanged. Default `None` retains the
existing route. The mechanism is target independent; it grants no device layout
or DMA alignment fact and predicts no runtime speedup.

Public API: `merlin.llvmlower.entry_weight_projection`.

---

## Phase 0 — Frontend ingest & normalization  ✅
**In:** `builtin func linalg tensor arith scf math` (+ `quant_ext` custom op from m2m)
**Does:** make the incoming module pure, valid upstream linalg-on-tensors.
**Passes**
- `[merlin]` `lower_quant_ext` — `quant_ext.dequantize_per_channel` → `linalg.generic`
  (`(w−zp)·scale`, axis-broadcast, output dtype).
- `[merlin]` `_zero_fill_contraction_accumulators` (in m2m) — every matmul `outs` gets a
  `linalg.fill 0` (matmul is `out += A·B`; an unfilled `tensor.empty` is undefined).
- `[merlin]` `add_c_interface` — `llvm.emit_c_interface` on public funcs.
- `[merlin]` textual normalization — printer-quirk fixes (rank-reduced `extract_slice`,
  `f0x` float hex, `inf`/`nan`).
**Out:** clean linalg-on-tensors. **Debug:** `module.verify()`; per-op finiteness via
`truncate.py` (catches uninitialized-memory bugs here).
Files: `llvmlower/passes_xdsl.py`, `model2MLIR/m2m/ir/{import_fx,decompositions}.py`.

## Phase 1 — Contract inference  ◐
**In:** linalg-on-tensors → **+`contract`**
**Does:** infer facts and attach them; never rewrites the payload.
**Passes**
- `[merlin]` `merlin-infer-contract-facts` — reuse count, immutability, lifetime, layout.
- `[merlin]` `merlin-attach-target-capabilities` — `contract.capability` from the target contract.
- `[merlin]` `merlin-verify-contracts` — all requirements dischargeable.
**Out:** linalg + `contract.*`. **Debug:** `analyses.check_contract_discharged`.
Files: `xdsl_dialects/lowering/contract_facts.py`, `contract.py`.

## Phase 2 — Schedule decisions  ◐
**In:** linalg + `contract` → **+`schedule`**
**Does:** record the chosen decisions (residency, pack hoist, dispatch grouping,
`vector_strategy<scalable_vl>` — the "go RVV" decision the backend reads).
**Passes**
- `[merlin]` `merlin-apply-kernel-policies` — facts → `schedule.*`.
- `[merlin]` `merlin-schedule-cost-check` — reject illegal schedules vs the target contract.
**Out:** linalg + `contract` + `schedule`. **Debug:** `analyses.check_place_legality`.
Files: `xdsl_dialects/lowering/schedule_decisions.py`, `schedule.py`.

## Phase 3 — Global-opt cleanup + interface materialization + outlining  ◐ (outliner ✅)
**In:** linalg + `contract` + `schedule` → **+`interface`**, then **`func @kernel_N`**
**Does:** canonicalize/fuse the linalg, wrap it in `interface.*` abstractions, then
**outline** each compute op / `group_dispatch` set into its own kernel function.
**Passes**
- `[upstream]` global-opt order (from IREE GlobalOptimization): `linalg-generalize-named-ops`
  → `linalg-fuse-elementwise-ops` (×2, with reshape bubble/sink between) →
  fuse-dequant-into-matmul → propagate-linalg-transpose → `fold-unit-extent-dims` → global-LICM.
- `[merlin]` `merlin-materialize-interface` — linalg payload → `interface.resident_pack/
  matmul/commit/...` wrapping the kernels (visibility variant from `schedule`). *(synthetic
  workload path today; real-model interface wrapping is future.)*
- `[merlin]` `merlin-outline-dispatches` ✅ **built** (`lowering/outline.py`) — one
  `func @forward$kernel_N` per compute dispatch; clones fills/constants/`tensor.empty` in for
  isolation; lifts region-captured free values (gather bodies) to explicit operands. Verified
  **value-preserving** (host output bit-identical to the monolithic compile) and scales to the
  real models (small_llama 183 kernels, tiny_llama 1402, 155 matmuls).
- `[merlin]` `merlin-verify-interface-legality` + `analyses.check_no_use_after_evict`.
**Out:** driver `@forward` + N kernel funcs. **Debug:** each kernel verifies and — because the
glue (`extract_slice`) stays in the driver — each clean-linalg kernel round-trips the printer
and compiles standalone; `llvmlower/kernel_backend.py` is the per-kernel bisection harness
(every real matmul kernel gated against the numpy reference).
Files: `xdsl_dialects/lowering/{interface_lowering,outline}.py`, `llvmlower/kernel_backend.py`.

## Phase 4 — Target lowering (optional)  ◐
**In:** `interface` + kernels → **+`toynpu`/`saturn`** (only for HW-primitive ops)
**Does:** map an interface op to a target-dialect op when the target has a primitive
(e.g. `saturn.matmul`); everything else stays general linalg → LLVM.
**Passes:** `[merlin]` `merlin-interface-to-target` (driven by `dialect_plan.yaml`).
**Out:** mixed `interface`/`target`/linalg kernels. **Debug:** dialect-plan lowering table coverage.
Files: `xdsl_dialects/lowering/target_lowering.py`, `xdsl_dialects/targets/`.

## Phase 5 — Runtime lowering → command buffer  ◐ (whole-model dispatch program ✅)
**In:** target/interface + kernels → **+`runtime`**, then a **dispatch table**
**Does:** emit the Merlin-owned command buffer: an ordered table of
`{kernel_symbol, in_buffers, out_buffer, grid, schedule_tags}` (+ RES_PACK/EVICT residency
events). Merlin owns this ABI; targets only adapt to it.
**Passes:**
- `[merlin]` synthetic path: `merlin-target-to-runtime` → `merlin-runtime-legalize` →
  `merlin-runtime-to-command-buffer` (`runtime_lowering.py`, `emit_command_buffer.py`).
- `[merlin]` `merlin-emit-dispatch-program` ✅ **built** (`lowering/dispatch_program.py`) —
  flattens the outlined driver into a serializable DAG of `dispatch` + `view` nodes over
  SSA-identified buffers; `prune_dead_nodes` drops the dead cloned-accumulator copies;
  `verify_program` proves it is a DAG (every input is an earlier output or a model arg).
  Runs on the real models (small_llama 443 nodes, tiny_llama 3616), JSON-serializable.
**Out:** `runtime.*` module (synthetic) **or** a `DispatchProgram` dict (whole model).
**Debug:** `analyses.check_command_buffer_consistency` / `verify_program`; the Python
simulator executes it. Authored-pass catalog + `run_dialect_plane` in `lowering/passes.py`.
Files: `xdsl_dialects/lowering/{runtime_lowering,emit_command_buffer,dispatch_program,passes}.py`.

---

### The backend descent (Phases 6–8) runs **per outlined kernel** — MLIR all the way, `llvm` dialect at the edge.

## Phase 6 — Per-kernel MLIR→LLVM-dialect lowering  ✅ (whole-module today; per-kernel ⏳)
**In:** one kernel func (linalg/tensor/arith/scf) → **`llvm` dialect only**
Sub-phases (the actual `pipeline.py` pass list):

- **Optional exact tensor destination reuse** (before bufferization):
  `reuse_tensor_destination` is an independent, default-off preparation rewrite:
  a sole-use pure pointwise producer of a static unit-stride insert slice writes
  into the matching view of a fresh filled destination. The scalar arithmetic
  is cloned unchanged; reading the prior output, index-sensitive bodies, dynamic
  slices and encoded tensor layouts refuse. Upstream `eliminate-empty-tensors`
  runs before one-shot bufferization, and existing self-copy cleanup is implied.
  Upstream alias analysis may still require an additional destination copy across
  external calls; this feature supplies no new alias or ownership assertion.
  `initialize_tensor_border_only` is a separate default-off refinement and
  implies destination reuse. It fills only disjoint exterior slabs of the fresh
  destination, then clones the unchanged producer into the completely overwritten
  interior. Each exterior element belongs to the slab for its first dimension
  outside the interior, proving full coverage without overlap. The interior's old
  values must remain unread. A live original splat is preserved independently;
  upstream bufferization still decides storage reuse. Shape-derived initialization
  savings are not a cycle prediction, and the complete model gate remains required.
- **6a Optional activation arithmetic** (before bufferization):
  `approximate_transcendental_activation` selects the provenance-scoped f32
  exp/erf/tanh polynomial independently of the host vector schedule. Normalization
  and unsupported float dtypes retain their existing lowering. The option is
  default off, changes numerical arithmetic, and requires the caller's original
  model accuracy gate. `vectorized_transcendental_activation` implies this option
  and additionally selects activation vectorization.
  `fuse_activation_polynomial_fma` additionally selects fused polynomial
  evaluation and the narrow FMA intrinsic lowering. This changes approximation
  arithmetic and requires the same original accuracy gate; normalization keeps
  its existing libm lowering.
  Explicit `cpu_bf16_conv`, `cpu_layer_norm`, and `cpu_flash_sdpa` builders reproduce
  selected, pinned PyTorch CPU arithmetic schedules without installing them in
  the default pipeline. Callers bind the source/backend evidence and finite/RNE
  obligations; unsupported shapes, masks or arithmetic policies refuse. These
  numerical compatibility builders still require the original whole-model gate.
- **6a Optional exact scalar contractions** (before bufferization):
  `scalar_contraction_accumulator` replaces recognized static f32 tensor generics
  with scalar loop arguments for their final reduction dimension after ordinary
  tensor fusion and generalization, immediately before bufferization. Named
  materializations retain the ordinary fusion boundary; only remaining pure
  contractions are eligible. Pure dimension
  maps, parallel output dimensions, and an exact separate multiply/add body are
  required. It preserves the initial destination and increasing reduction order,
  including transpose and batch maps; fastmath permissions and unsupported bodies
  remain unchanged. Each output is inserted once into a carried tensor, so upstream
  bufferization retains responsibility for input/destination aliases. The option
  is default off and makes no physical noalias claim.
  The alternatives `scalar_contraction_accumulator_2_outputs` and
  `scalar_contraction_accumulator_4_outputs`, plus
  `scalar_contraction_accumulator_8_outputs`, carry independent scalar
  results through the same increasing reduction loop and
  share input extracts that do not depend on the last parallel dimension. Each
  output retains its own separate multiply/add order. They require that parallel
  extent to be divisible by the output count; partial tiles remain in linalg. The
  accumulator schedules cannot be selected together.
  `unroll_scalar_contraction_reduction_by_2` and
  `unroll_scalar_contraction_reduction_by_4` compose with one accumulator
  schedule and attach upstream LLVM partial-unroll metadata only to its
  reduction loops. They retain separate arithmetic and increasing reduction
  order. The LLVM branch property `loop_annotation` carries the metadata through
  SCF, CF and LLVM lowering; tests require the final LLVM unroll-count metadata.
  Both are default off, are alternatives, and add no numeric permission.
  The registry declares separate alternative groups for output count and
  reduction unroll count. Each unroll choice requires exactly one output
  schedule after implication closure; pipeline marker guards also remain.
  See [compiler feature selection](compiler_feature_selection.md).
- **6a Bufferize** (tensor → memref): **+`memref bufferization`**
  - `[upstream]` `one-shot-bufferize{bufferize-function-boundaries}`
  - `[upstream]` `buffer-results-to-out-params{modify-public-functions hoist-static-allocs}`
    *(must include `modify-public-functions` or the public entry keeps heap-returned results)*
  - `[upstream]` `buffer-hoisting`, `buffer-loop-hoisting`
- **6b Vectorize for RVV** (linalg → vector): **+`vector`**
  - `[upstream]` `linalg-tile{tile-sizes=[vscale]}`, `linalg-vectorize{vectorize-nd-extract scalable}`,
    vector-transfer lowerings → scalable `vector<[n]xf32>`.
  - *(baseline today: `[upstream]` `convert-linalg-to-loops` — scalar loops that clang
    auto-vectorizes to RVV; the scalable-vector path is the optimization upgrade.)*
- **6c Control flow + index** : **+`cf`**
  - `[upstream]` `convert-scf-to-cf`, `expand-strided-metadata`, `lower-affine`
- **6d Custom ISA insertion** ✅ **built** (`llvmlower/custom_isa.py`; only if the kernel maps
  to a custom instruction)
  - `[merlin]` `merlin-lower-inline-asm`: Merlin op → `llvm.inline_asm` / `llvm.call_intrinsic`
    (incl. `llvm.intr.vp.*`, `llvm.intr.vscale`), 1:1, no LLVM fork. Demonstrated: a CUSTOM-0
    instruction (`.insn`, `0x00b5050b`) the toolchain can't name is compiled into an rv64gcv
    object and confirmed in the disassembly.
- **6e Convert to LLVM dialect** : **→ `llvm` only**
  - `[merlin, opt-in]` `lower_fma_to_intrinsic` converts scalar/vector f32/f64
    `math.fma` to `llvm.intr.fma` after loop formation. Fused arithmetic remains
    fused; other math operations retain the established lowering. The separate
    `lower_roundeven_to_intrinsic` feature selects the exact LLVM rounding
    intrinsic at this same stage and can compose with the FMA feature.
  - `[upstream]` `convert-math-to-libm` (erf/exp/tanh → libm; newlib on riscv),
    `convert-vector-to-llvm`, `convert-math-to-llvm`, `convert-index-to-llvm`,
    `convert-arith-to-llvm`, `finalize-memref-to-llvm`, `convert-func-to-llvm`,
    `convert-cf-to-llvm`, `reconcile-unrealized-casts`, `canonicalize`, `cse`, `symbol-dce`.
**Out:** pure `llvm`-dialect MLIR (still inspectable/rewritable). **Debug:** print the
`llvm`-dialect module; it must contain no `unrealized_conversion_cast`.
Files: `llvmlower/pipeline.py` (+ `kernel_backend.py` for the per-kernel version).

## Phase 7 — Edge: leave MLIR  ✅
**In:** `llvm` dialect → **LLVM IR text**
**Does:** the *one* crossing out of MLIR.
- `[upstream]` `translate_module_to_llvmir` (torch-mlir wheel, LLVM 23).
- `[merlin]` `_fix_float_literals` (the printer emits `f0x..`/bare inf the IR parser rejects).
**Out:** `.ll`. **Debug:** `clang -fsyntax-only` the `.ll`.
Files: `llvmlower/pipeline.py` (`translate_module_to_llvmir`).

Ordinary upstream lowering writes `lowering_recipe.json` alongside its artifacts.
It records the actual command, resolved pass pipeline and positional gates, selected
features, prepared input/runner/generated schedule hashes, executable identity and
redacted `MERLIN_*` environment. A successfully returned LLVM text is hashed after
literal normalization and any selected loop outlining. Every new invocation drops
the prior receipt. A failure before pipeline resolution leaves no receipt; a later
failure has no successful-return hash.
This scope ends before later host transforms, native codegen and linking. It does
not close imported compiler modules, the entire environment or toolchain libraries,
and it does not change emitted IR or existing build identities.

## Phase 8 — Native codegen & link  ✅
**In:** `.ll` → object → linked image
- `[upstream]` `clang-23 --target=riscv64-unknown-elf -march=rv64gcv` (or x86 for the oracle)
  — instruction selection, regalloc, RVV vectorization of scalar loops.
- `[merlin]` link with `mlir_runtime.c` (memrefCopy/rsqrtf), the weights blob, and the
  harness/C runtime.
**Out:** `.o` / ELF. **Debug:** `objdump -d | grep vsetvli` (RVV present); host `.so` runs.
Files: `llvmlower/codegen.py`, `runtime/abi/mlir_runtime.c`, `runtime/baremetal/spike/`.

## Phase 9 — Execute  ✅ (host monolithic + per-dispatch + whole-model spike; multicore schedule) · ⏳ (multicore C/spike exec, Zephyr)
**Does:** a runtime walks the dispatch table and invokes the compiled kernel symbols.
- **Multicore schedule** (`merlin-partition-dispatches`, `lowering/schedule_dispatch.py`):
  level-synchronous, dependency-validated partition of the dispatch DAG across harts (the
  substrate for `-p4` whole-model spike). Reports the real parallelism (small_llama 1.66× on
  4 harts; tiny_llama depth 1841 — critical-path-bound). The C/spike multicore *executor* of
  this schedule is the remaining follow-on.
- **Dispatch-table runtime** (`merlin.runtime.dispatch_runtime`) — the per-dispatch host
  executor: compiles each outlined kernel in isolation (deduplicated by body), evaluates the
  driver's view ops in numpy, invokes the kernels in order. **Verified:** whole small_llama
  **cos 0.9999999** and **TinyLlama-1.1B cos 1.0000000 (argmax exact on all 8 tokens)**
  through the dispatch table == torch — same fidelity as the monolithic compile, via the
  unified per-kernel path (`test_dispatch_runtime.py`). A by-value scalar kernel arg (a
  `cumsum` accumulator-init `i64`) must be passed by value, not as a descriptor
  (`abi.ScalarArg`) — a scalar ABI subtlety the per-kernel split exposes.
- **Host buffer layout** (`merlin.llvmlower.abi.HostModel`) — ordinary
  `(pointer, shape)` arguments remain dense row-major. For a buffer whose
  physical pitch differs, pass `StridedMemRefArg(pointer, shape, strides,
  storage_elements, dtype, access, offset=0)`, with strides, offset and capacity
  in *elements* and `access` equal to `input`, `output` or `inout`. Merlin checks
  the declared footprint, packs inputs into dense scratch before the native
  call and scatters outputs back afterward. This is necessary because the
  current statically shaped lowering can assume dense layout even when its
  MLIR C-interface descriptor carries other strides. It is not zero-copy, and
  a raw pointer's true allocation size and dtype cannot be verified here;
  callers must own that memory and match the compiled signature. Writable
  layouts must have provably distinct logical addresses; overlapping staged
  argument footprints and obvious overlap with a dense argument are rejected.
  Legacy dense tuples have no capacity or dtype, so less-obvious overlap with
  them cannot be detected; callers must not alias those buffers with a staged
  output.
- Python simulator (`merlin.runtime`) — the fast correctness oracle for the synthetic path.
- **Merlin C runtime** (`src/merlin/runtime/native/c/`) — generic descriptor builder + arg table +
  weights blob + bump allocator (`baremetal/spike/merlin_malloc.c`). Drives the whole
  compiled `forward()` on host and on **spike bare-metal RVV**.
- `src/merlin/runtime/backends/spike_model.py` — build (lower → cgen → rv64gcv
  → link weights blob + harness) + run (`spike --isa=rv64gcv_zfh_zvfh`) + verify.
**Verified:** a complete small LLaMA (RMSNorm/RoPE/attention/softmax/SwiGLU/lm_head) runs
end-to-end on spike with RVV — **spike == host == torch, cos 0.9999999** (`test_spike_model.py`).
The ~1e-7 spike-vs-host delta is RVV-vs-x86 FP reassociation, not a bug.
**Debug:** gate **spike == host** (same MLIR, ~f32 tol) and **host == torch** (model
faithfulness). Two separate checks: codegen correctness vs model precision. For large LM
outputs the bare-metal driver emits a per-token-argmax + checksum digest so the gate stays
fast.

---

## How to reason about a change
1. Identify the **phase** it belongs to (annotation? payload lowering? backend?).
2. A change to a Merlin annotation phase (1–5) must not alter the payload's numerics —
   verify by re-running Phase 9 host == torch.
3. A change to a backend phase (6–8) must keep `module.verify()` green and **spike == host**.
4. For a numeric regression, bisect with `truncate.py` (per-op finiteness / per-op golden);
   for a structural regression, diff the `llvm`-dialect module at the Phase 6 boundary.

## Upstream vs Merlin passes (at a glance)
- **Upstream (reused, never forked):** all of Phase 6/7/8 — bufferize, tile/vectorize,
  convert-*-to-llvm, translate, clang.
- **Merlin (authored):** Phase 0 normalization, Phases 1–5 (all our dialects + outlining),
  the custom-ISA 1:1 insertion (6d), and the float-literal/printer fixes.

### Private uniform-buffer copies

`fold_uniform_fill_copy` is an independent, default-off post-bufferization feature.
It requires one fresh allocation, one identity-map uniform writer with an unread
output argument, and only direct same-block source copies and an optional final
free. It refuses escaping aliases, unknown users, intervening writes, nested
copies and unproved lifetime order. Each accepted copy becomes the original
store-only writer on that copy's destination, including strided subviews. The
private allocation and fill can then be removed. Scalar values, floating-point
bit patterns, destination placement and untouched regions remain unchanged.

This can compose with `initialize_tensor_border_only`, whose tensor-level slab
fills otherwise may become temporary buffers and strided runtime copies. The
border option alone did not reduce instructions in the controlled whole-model
experiment; isolated capsule gains used a different memset implementation and
are not a whole-model performance prediction. Both options remain explicit.

### Copies with a contiguous inner region

`specialize_contiguous_copy` is an independent, default-off feature after
bufferization. It follows only known memref views to distinct fresh allocation
roots, proving that the source and destination do not overlap. Ranked positive
static shapes and strides determine their longest common contiguous suffix.
The transform retains source-lexicographic outer loops and copies each complete
suffix through rank-reduced subviews. Upstream can then lower the inner copy to
`memcpy`; the selected runtime still owns alignment and byte-tail handling.
No element conversion, word cast, or target-specific layout is introduced.

Unknown or shared allocation roots, dynamic or nonpositive domains, missing
contiguous suffixes, and already fully contiguous copies are left unchanged.
`contiguous_suffix_copy.json` records the selected rewrite count, and missing or
ambiguous runner reports fail the selected feature rather than silently skipping
it. Native tests cover multiple outer dimensions, odd byte tails, misalignment,
floating-point payload bits, overlap refusal, and untouched destination guards.
Performance requires measurement: expanding every copy to scalar element loops
was slower in the controlled whole-model experiment.

### Bound integer producers and exact readout

`IntegerSumProductsRange` proves a conservative signed-i32 accumulator interval
from integer operand intervals, term count and initial accumulator bounds. It
checks product and every-prefix accumulation overflow, including signed `-128`
inputs and nonzero integer seeds. `emit_readout(producer_range=...)` requires
this typed certificate and containment in the independently re-derived original
source readout domain. It then omits only the per-element domain trap. Saturation,
rounding thresholds, correction, lane memory order and tails remain unchanged.

This option is an explicit input precondition, not a runtime validator. Its owner
must bind the actual typed producer, immutable bias facts, generated kernel and
exclusive output storage to the consumer call. Samples or a matching shape alone
cannot establish that binding. Public readouts accepting arbitrary i32 inputs
retain the checked default; its generated C remains byte-identical.
The producer's input-to-readout storage must remain unchanged until its last
read; an overlapping output buffer cannot silently invalidate this precondition.
The bound convolution adapter supplies distinct, exclusively owned scratch and
output allocations. The ordinary checked API retains its existing overlap order.

### Explicit scalar carrier representations

`ApproximateScalarCarrierPolicy` grants a distinct absolute and relative error
budget for a scalar expression before its original finishing operations. It
requires independent validation of the original public outputs; carrier error
permission alone does not establish integer-code or whole-model equivalence.
The policy also bounds runtime coefficient payload and compile-time proof-table
payload. Code, temporary proof work, stack and aggregate runtime storage require
separate admission.

`prepare_source_scalar_carrier` validates current typed closed scalar observers,
immutable three-coefficient choices and explicit floating effects. Every fine
raw-binary32 interval must meet the budget for both rounded carrier FMAs. Fine
proof partitions can refine admission without increasing the coarse runtime
table. Rejected cells evaluate the original source expression. The caller
supplies a runtime rounding predicate; a false predicate retains the source
expression under the incoming rounding mode.

`reify_source_scalar_carrier_helper_family` constructs one readonly typed table
and independently binds each helper to its current source proof. It validates
the complete storage/helper namespace and retains each original finishing
multiplier and integer observer. Shared storage is explicit before upstream
lowering; compiler or linker deduplication is not a premise. Binding, input and
output ownership, final linked placement, runtime predicates, complete costs
and original output gates remain caller obligations. These APIs perform no
automatic discovery, pipeline selection or profitability decision.

### Current source scalar insertion

`SourceScalarCarrierSelection` is an explicit invocation-local option for
`lower_model`, `lower_model_file`, `lower_to_llvm_ir` and the ordinary model
builders. The selected native pipeline pauses after its ordinary tensor fusion
and generalization, immediately before bufferization. The parent discovers
closed integer observations in those current bytes, validates every source
witness and prepares one immutable table per expression family. Retained
helpers or historical call-site proofs are not inputs to this selection.

The response contains new private helper fragments and typed SSA edit locators
bound to the current source digest. The native child verifies the full module
and authenticates all edits before inserting them. It keeps original producer
operations, globals, resource handles and module ownership in the original
context. Only proved private scalar operations are erased. Replacement calls
retain their exact source provenance dictionaries and fused native locations.
Unsupported namespaces, resources in selected scalar expressions, live escapes,
stale packets and incomplete effects refuse the selected invocation.

`IncomingRNECapability` names a provider-owned `() -> i1` runtime predicate.
Each scalar point reads the actual incoming rounding mode. The contract requires
preserved rounding mode and flags, no writes and nontrapping behavior; it does
not authorize hoisting the predicate across a region. A false result evaluates
the original expression. Separately, `CompilerHostNumericAdmission` admits the
actual compiler host to RNE before source construction and serialization and
must restore its complete incoming environment on success or error.

The private request and response have explicit byte quotas and one invocation
owner. Callback code runs synchronously on the calling thread and must return
promptly. A deadline bounds native waiting and is checked after the callback;
it cannot interrupt arbitrary callback code. Failure terminates only the owned
child. The unselected route retains the original runner and pipeline. Numerical
policy, aggregate storage, linked predicate qualification, complete producer and
consumer cost, and original whole output validation remain explicit obligations.
