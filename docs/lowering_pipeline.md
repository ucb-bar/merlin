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
Files: `llvmlower/{translate,pipeline}.py`.

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
  (`abi.ScalarArg`) — the one ABI subtlety the per-kernel split exposes.
- Python simulator (`merlin.runtime`) — the fast correctness oracle for the synthetic path.
- **Merlin C runtime** (`merlin/runtime/c/`) — generic descriptor builder + arg table +
  weights blob + bump allocator (`baremetal/spike/merlin_malloc.c`). Drives the whole
  compiled `forward()` on host and on **spike bare-metal RVV**.
- `merlin/python/merlin/runtime/backends/spike_model.py` — build (lower → cgen → rv64gcv
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
