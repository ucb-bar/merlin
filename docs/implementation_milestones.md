# Implementation milestones

## Milestone 1 — ToyNPU target generation (done)

Generate the reference target repo and validate it end-to-end.

```bash
python -m merlin.targetgen.cli build \
  --target-name toy_npu \
  --source-dir merlin/targets/toy_npu/docs \
  --examples-dir merlin/targets/toy_npu/examples \
  --out build/generated/merlin-target-toy-npu \
  --emit xdsl,mlir,zephyr,llvm-plan,runtime

python build_tools/scripts/check_generated_target.py build/generated/merlin-target-toy-npu

python -m merlin.targetgen.cli inspect --target build/generated/merlin-target-toy-npu
```

Pass conditions: the five plans validate; the generated repo passes structural checks; the
generated runtime adapter's `run_simulator` reads the example command buffer and produces
metrics including `cycles`, `bytes_moved`, `command_count`, `pack_count`, `resident_hits`,
`evictions`, `accumulator_commits`.

## Milestone 2 — Gemmini contract skeleton (from local source path)

```bash
python -m merlin.targetgen.cli build \
  --target-name gemmini \
  --source-dir $MERLIN_GEMMINI_REPO \
  --out build/generated/merlin-target-gemmini \
  --emit contract-only
```

Emits conservative, keyword-detected plans flagged `requires_human_review: true`. No external
repo is required for the test suite; the source path is optional and only enriches the
evidence report when present.

## Milestone 3 — Saturn and Radiance contract skeletons

```bash
python -m merlin.targetgen.cli build \
  --target-name saturn \
  --source-dir $MERLIN_SATURN_REPO \
  --out build/generated/merlin-target-saturn \
  --emit contract-only

python -m merlin.targetgen.cli build \
  --target-name radiance \
  --source-dir $MERLIN_RADIANCE_REPO \
  --out build/generated/merlin-target-radiance \
  --emit contract-only
```

Saturn's `llvm_extension_plan` defaults to `requires_llvm_fork: maybe` (RVV); Radiance's
defaults to command-processor packets / external SIMT toolchain.

## Whole models on spike (RVV), verified

The whole-model path runs entire models end to end on **spike as a multicore RVV CPU**,
gated against PyTorch:

- **small LLaMA** (complete transformer: RMSNorm/RoPE/attention/softmax/SwiGLU/lm_head):
  spike == host == torch, **cos 0.9999999** (`test_spike_model.py`).
- **TinyLlama-1.1B** (real 1.1B-parameter model): runs end to end on spike; **all 8
  next-token argmax predictions identical to PyTorch**, prefix logits **cos 1.0000000,
  rel 1.05e-5** (241.7 G cycles, ~104 min on the functional ISS;
  `output/tiny_spike_result.txt`).

Pipeline: `model.mlir` → LLVM IR (`llvmlower`) → rv64gcv object (clang-23) + the **Merlin
C runtime** (`merlin/runtime/c/`: generic descriptor builder + generated arg table +
weights blob + bump allocator `baremetal/spike/merlin_malloc.c`) → ELF → spike. Driven by
`merlin/python/merlin/runtime/backends/spike_model.py`. Scalable to multi-GB weights via a
fixed-high-address weights/arena memory map (`model_link.ld`), so `-mcmodel=medany`'s ±2GB
PC-relative limit is never hit (TinyLlama's 4.2 GB weights link and load cleanly).

## Merlin-authored passes: per-dispatch outlining + dispatch program

The whole-model paths above reuse upstream MLIR passes for the mechanical linalg→LLVM
descent. The **research-plane** transforms — the ones with no upstream equivalent — are
Merlin-authored and now bring real models into the dialect plane:

- **`merlin-outline-dispatches`** (`xdsl_dialects/lowering/outline.py`): splits the
  monolithic `func @forward` into one `func @forward$kernel_N` per compute dispatch (each
  owning its linalg payload + cloned accumulator init) and a thin driver that calls them in
  order — the Merlin-owned analogue of IREE's two-phase dispatch formation, on
  linalg-on-tensors, no `flow`/`stream`/`hal` dependency. It lifts region-captured free
  values (model2MLIR gather bodies) to kernel operands. Proven **value-preserving** (host
  output bit-identical to the monolithic compile, f32 and quantized) and scales to the real
  models: **small_llama → 183 kernels, tiny_llama → 1402 kernels (155 matmuls)**, parsed +
  outlined + verified in seconds.
- **`merlin-emit-dispatch-program`** (`lowering/dispatch_program.py`): flattens the outlined
  driver into a serializable runtime **dispatch DAG** (`dispatch` + `view` nodes over
  SSA-identified buffers), with dead-node pruning and DAG verification. The target-agnostic
  command buffer the Python simulator and the C runtime both consume.
- **Whole-model dispatch runtime** (`runtime/dispatch_runtime.py`): executes an entire model
  through the dispatch table — per-kernel compiled + numpy view ops — gated against torch.
  Verified: **small_llama cos 0.9999999** and **TinyLlama-1.1B cos 1.0000000, rel 1.0e-5,
  next-token argmax exact on all 8 tokens** (1402 kernels, 1060 unique). This is the same
  fidelity as the monolithic compile, reached through the unified per-kernel route. Surfaced
  and fixed a real ABI bug: kernels with a **by-value scalar arg** (the `cumsum` causal
  position-id accumulator-init `i64`) must be passed by value, not wrapped in a memref
  descriptor (`llvmlower.abi.ScalarArg`) — caught because small_llama has no scalar-arg
  kernel and tiny does.
- **Per-kernel backend + checker** (`llvmlower/kernel_backend.py`): because the outliner keeps
  the `extract_slice` glue in the *driver*, each kernel func is clean linalg that round-trips
  the xDSL printer (the whole model does not) and compiles standalone. Every matmul dispatch of
  the real small_llama is compiled in isolation and gated against the numpy reference — the
  per-kernel bisection harness, where the historical whole-model NaN would localize instantly.
- **`merlin-partition-dispatches`** (`lowering/schedule_dispatch.py`): a level-synchronous
  multicore schedule of the dispatch DAG. Because the program is single-writer dataflow, any
  dependency-respecting order is equivalent; the pass groups nodes into ASAP levels (barrier
  between), load-balances each level across harts (LPT, matmul cost M·N·K), and `validate`
  proves every edge crosses a barrier upward — making whole-level parallel execution provably
  correct. Reports realistic parallelism: small_llama depth 171 / max-width 64 → 1.66× on
  4 harts, tiny_llama depth 1841 / max-width 438 (critical-path-bound by the sequential
  residual stream and the single large kernels, as expected).
- **`merlin-bf16-matmul-f32acc`** (`llvmlower/passes_xdsl.py`): rewrites a bf16 `linalg.matmul`
  (which accumulates in lossy bf16) into a `linalg.generic` that `extf`s operands to f32,
  accumulates in f32, and `truncf`s the result back to bf16 — matching hardware/torch. Verified
  on host: **21× lower error** than bf16-accumulation (down to the bf16-output ULP floor) on a
  K=512 contraction. This is the smolVLA precision fix (its cos≈0.94 was bf16 accumulation).
- **`merlin-lower-inline-asm`** (`llvmlower/custom_isa.py`): custom ISA with **no LLVM fork**.
  A Merlin op lowers 1:1 to `llvm.inline_asm`; a truly novel encoding (e.g. a Saturn vcix
  instruction) uses the assembler `.insn` directive inside the asm. Demonstrated end to end:
  a CUSTOM-0 instruction (`0x00b5050b`, opcode `0x0b`) the toolchain has no mnemonic for is
  compiled into an rv64gcv object and confirmed in the disassembly (`objdump` shows
  `.insn 4, 0x00b5050b`). Standard rv64gcv needs none of this — it is the accelerator on-ramp.
- **Authored-pass catalog** (`lowering/passes.py`): enumerates every Merlin-written pass (vs the
  orchestrated upstream ones, 12 entries spanning normalize → outline → runtime → edge) and
  exposes `run_dialect_plane` as the whole-model entry.

Tests: `test_outline.py`, `test_dispatch_program.py`, `test_kernel_backend.py`,
`test_passes_catalog.py` (structural checks always; host-equivalence + real-model checks gate
on the toolchain / captures).

## Toward all model2MLIR models on RVV (in progress)

Expanding from the LLaMA family to the full model2MLIR set (VLAs: smolvla, openvla, pi05,
rdt, rdt2, bitvla, groot, molmoact, xr0) across fp32/int8/fp8. Verified progress:

- **Whole-model dispatch runtime is now a full driver interpreter** (`runtime/dispatch_runtime.py`):
  handles conv (as `linalg.generic`), `scf.for`/`scf.if`, dynamic `?` dims, `tensor.pad`/
  `extract_slice`/`insert_slice` (rank-reduced)/`extract`/`insert`/`from_elements`, scalar
  `arith`, bf16 (stored as uint16, widened at the numpy boundary), and int8 weights. A real
  500M VLA (smolvla: SmolVLM2 backbone + action expert, int8 weights + bf16 compute + conv)
  **executes end to end** (4367 kernels). Per-model arg binding via an emitted
  `input_order.json`; a multi-output guard fails loudly.
- **Real int8 compute on RVV** (`tests/test_int8_compute.py`): an `i8×i8→i32` matmul lowers
  through the pipeline to rv64gcv emitting **`vwmacc.vv`** (widening integer multiply-
  accumulate — true 8-bit SIMD, not dequantized to f32), bit-exact vs numpy; a W8A8
  dynamic-quant GEMM built on it tracks the f32 matmul to **cos > 0.999**. The model2MLIR
  int8 export is weight-only (i8 weights stay i8 in memory, dequantized per-element), so
  int8 storage matches the golden exactly; W8A8 int8-*compute* is the approximate path.
- **bf16 f32-accumulation** (`llvmlower/passes_xdsl.lower_bf16_matmul_f32acc`) now covers both
  named `linalg.matmul` and `linalg.generic` batch-matmuls (attention), matching how torch
  accumulates bf16 in f32.
- **Consistent-capture harness** (`model2MLIR/workloads/capture_consistent.py`): one seeded
  process emits a self-contained bundle (inputs+golden+MLIR+weights+manifest+extra+
  input_order) per model/dtype — the prerequisite for per-model host==torch verification.

Status: small_llama + TinyLlama-1.1B verified end to end on RVV/spike. smolvla runs fully
through the dispatch runtime; a ~2% numerical residual (cos 0.978) on the int8 bundle is
under investigation (bf16-reduction vs int8-dequant fidelity). The remaining VLAs each need
their capture venv rebuilt; whole-model spike runs of the 7B/3.6B/1B models are impractical
on the functional ISS (host==torch is the gate there).

## What is real now

- **Core dialects** (`merlin/python/merlin/xdsl_dialects/`): all five (`contract`,
  `schedule`, `interface`, `runtime`, `dse`) are fully implemented in xDSL — real IRDL
  ops/types/enum attrs with local verifiers, plus cross-op analyses (use-after-evict,
  placement legality, discharged checks, command-buffer consistency).
- **Staged lowering** (`xdsl_dialects/lowering/`): linalg (`linalg.quantized_matmul`)
  → contract facts → schedule decisions → interface → `toynpu` → runtime IR → an
  executable command buffer. Every intermediate verifies; the lowered buffer executes
  on the engine with simulator output == independent reference
  (`test_xdsl_lowering_e2e.py`).
- **smolVLA frontend** (`merlin/python/merlin/frontends/`): parses the real
  model2MLIR artifact (25k-line linalg-on-tensors MLIR) in ~5 s, inventories all 302
  matmuls with weights resolved through the safetensors manifest, lifts
  reuse-across-denoise-steps facts, and drives the saturn pipeline with real layer
  shapes (action_out_proj 50x720x32 ran on spike, outputs == reference). `dse` IR
  records measured residency variants: marginal at reuse=1, exploitable at reuse>=2
  (2x cycles, 3.4x bytes at reuse=10). Setup: `build_tools/scripts/setup_model2mlir.sh`,
  docs in `docs/model2mlir.md`.
- **Saturn RVV on spike** (`merlin/python/merlin/runtime/backends/` +
  `merlin/runtime/baremetal/spike/`): the same pipeline lowered through the in-tree
  `saturn` reference target compiles to a bare-metal driver around a hand-written RVV
  assembly matmul kernel and executes on `spike --isa=rv64gcv_zfh_zvfh -p4` as a
  multicore RVV CPU. Three-way output agreement (spike == independent reference ==
  Python simulator), real mcycle counts, single-vs-multi-hart equality
  (`test_rvv_spike.py`; auto-skips without `MERLIN_CHIPYARD`). VCS replay of the same
  ELF is gated on `MERLIN_SATURN_SIMV`.
- **Runtime simulator** (`merlin.runtime`): real pure-Python integer tensor math
  (matmul + bias/requant/relu epilogue), real metrics + trace, and an independent reference
  recomputation with a correctness assertion. The generated adapter executes through it and
  writes `simulator_output.json` / `reference_output.json` / `metrics.json` / `trace.json`.
- **xDSL dialect**: real IRDL ops/types + verifier; the generated test builds, verifies, and
  round-trips a module through the xDSL parser/printer.
- **MLIR/C++ dialect**: complete, idiomatic ODS + C++ + `add_mlir_dialect` wiring and a real
  `CommitOp::verify()`. Compiling requires an MLIR/LLVM build (a documented build dependency).
- **Zephyr driver**: a real blocking driver (command-FIFO over MMIO, doorbell, status poll,
  counter readout), real devicetree binding/overlay, and a real ztest. Compiling requires the
  Zephyr SDK (a documented build dependency).

## Beyond the MVP

- Wire the MLIR/C++ scaffold against an MLIR install and build `merlin-<target>-opt`.
- Build the Zephyr sample/test against a board with the Zephyr SDK.
- Wire the contract -> schedule -> interface -> target -> runtime lowering pipeline.
