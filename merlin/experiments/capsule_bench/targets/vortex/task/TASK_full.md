# Task: generate a Vortex MLIR out-of-tree target backend (capsule_bench_v0 — FULL SUITE)

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
Vortex RISC-V SIMT GPGPU under `submission/`. Your package is graded — through its CLI entrypoints only,
never imported — by compiling workload **capsules** (linalg-on-tensors + arith MLIR, exported from
PyTorch models) and matching reference behavior on the Vortex simulators. This is a
**compiler/backend** task, not a kernel-writing task.

Your backend's job, end to end: take the linalg/arith slice of a real model, optimize it, map it onto
the Vortex thread grid through a target dialect **you design**, and emit an LLVM-dialect module that
builds to an rv64 kernel binary and runs on the hardware.

## Scope: ALL public/dev capsules

Make **every** public/dev capsule under `merlin/contract/capsules/vortex/` pass. **That directory is the
authority on scope** — enumerate it and handle what you find; a family named below with no capsules
present is simply not exercised this run. Capsules are drawn from three families, in both
quantized-integer and floating-point precision:

- **ISA (`isa/`):** the primitives — elementwise map, reduction, single-tile matmul, K-accumulated
  matmul, transposed operands, edge shapes that do not divide the thread grid, and a barrier-carrying
  two-stage kernel.
- **Layers (`layers/`):** quantized linear (i8 in / i32 acc), linear+relu, linear+requant→i8,
  conv2d via im2col, plus their f32 counterparts.
- **Model slices (`model_slices/`):** MLP linear1, MLP activation+linear2, attention Q/K/V
  projections, attention QK^T, attention PV, and a softmax slice (f32).

Do not tune to the specific shapes you see: a **hidden** capsule set perturbs shapes, element counts and
reduction lengths, and a shape-specialised kernel fails it.

Read each capsule's `capsule.yaml` + `capsule.interface.mlir` for its exact ops, shapes, dtypes, and
`numeric_policy`; read `merlin/contract/interface_grammar.md` for the input grammar and
`merlin/contract/mlir_oot_backend_contract.yaml` for the package ABI. Derive everything (tiling,
grid mapping, layouts, rounding) from the contract + the RTL + `VORTEX_ISA_SPEC.md` — nothing is
restated here, and there are no vendor headers to read (see below). Each capsule dir gives you
`capsule.yaml`, `capsule.interface.mlir`, and `expected_simt_coverage.yaml`. The numeric `golden.yaml` is intentionally withheld — you do NOT get the
answers; see the QA gate below. Build one **general** backend that handles every family — do not
special-case individual capsules.

## The input you compile

Capsules are **linalg-on-tensors + arith** (the dialects a PyTorch model lowers to via torch export),
inside a `func.func @forward`. There is no bespoke interface dialect to learn — the input is stock
upstream MLIR. Tensors carry `merlin.role` attributes (`input` / `weight` / `bias` / `output`)
identifying which operands the harness binds.

The complete set of dialects you will meet, all upstream: **`func`, `tensor`, `linalg`, `arith`**, plus
**`math`** (`math.exp` in softmax, `math.sqrt` in the RMS-norm layer, `math.tanh` in the GELU ones) and
**`scf`** (`scf.while` in the divergence capsules — a `linalg.generic` body containing a data-dependent
loop — and `scf.if` **nested inside** one of those loops). Nothing else appears.

Three things that implies, each of which some capsule depends on:

- A loop's trip count may be **not bounded at compile time**, so you cannot assume every
  `linalg.generic` body is straight-line code — and since one of them nests a branch inside such a
  loop, you cannot assume the divergence is only one level deep either.
- `exp` and `tanh` have **no RISC-V instruction**; you must supply an approximation. `sqrt` does have
  one (`fsqrt.s`), and the tolerance on the capsule that uses it is tight enough to assume you use it.
- Operand **dtypes are not all i8**: one capsule takes i32 inputs and the bias operands are i32/f32,
  so a load path written around `lb` + `extsi` will read the wrong bytes.

An operand's shape, dtype and role are all declared in `capsule.yaml`; read them rather than inferring
from a family name.

## The target machine (fixed for every capsule)

Vortex at the frozen geometry in the spec sheet: **2 clusters / 2 cores per cluster (4 cores total) /
8 warps / 8 threads = 256 threads**, L2 enabled, `XLEN=64`, `FPU_TYPE=STD`, compiled `rv64imafd` /
`lp64d` on **stock LLVM — there is no compiler fork, and you may not build one**. Vortex's SIMT control
ops are `.insn r CUSTOM0` instructions (thread-mask, warp spawn, split, join, barrier, predicate) and
its thread/warp identity and geometry are CSR reads. Both are expressible as `llvm.inline_asm` on an
upstream RISC-V target, so no custom LLVM backend is needed.

This is a genuinely **multi-core** machine and the four cores do not share an L1 data cache — the L2 is
the coherence point. A `barrier` orders warps within a core and does **not** by itself publish a write
to another core. See §5 of the spec sheet before you place synchronisation.

## What you are given about the hardware — and what you are not

You are bringing up a compiler for this target. You get **the hardware and a spec sheet, not a software
stack**:

- `rtl/` — the Vortex SystemVerilog RTL, plus a **HW-dialect MLIR import** of the elaborated design.
  This is the ground truth.
- `VORTEX_ISA_SPEC.md` — the architecture/ISA spec sheet: the CUSTOM0 opcode and funct3 table with
  operand semantics, the identity/geometry CSR map, the split/join reconvergence contract, the
  memory/fence model, the kernel entry convention, and the frozen geometry.

You do **not** get Vortex's software stack, and it is denied to every arm: no `vx_intrinsics.h`, no
`vx_spawn.h` or its NDRange work-distribution runtime, no Vortex LLVM fork, no PoCL/OpenCL, no
HIP/chipStar, and none of the bundled compute kernels. **Mapping a linalg iteration space onto warps and
threads is the compiler problem this benchmark measures** — a ready-made spawn runtime would hand you
the answer. Emit your own `.insn` sequences and your own grid mapping, from the spec and the RTL.

## Deliverable (write into `submission/`)

```
submission/
  manifest.yaml          # artifact_type: mlir_oot_target_backend; target: vortex; language: cpp|python;
                         # integrity_exempt: false; (cpp) a build block; the 4 command argv templates
  mlir_oot/              # your OOT sources: optimization passes + vortex target dialect + lowerings
                         # + the runtime-annotation pass + a `vortex-opt` tool
  REPORT.md              # what you built + honest scope/limitations + final status line (see end)
  docs/public_facts_used.md   # every Vortex-specific fact you used, with its source
  docs/iteration_notes.md     # what failed, what you changed, which capsule, failure plane, better/worse
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)

The pass pipelines compose — each entrypoint is a prefix of the next.

- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify the linalg/arith interface MLIR
- `optimize_interface`: `{tool} --vortex-optimize-linalg {input_mlir}` — **global optimization, still in
  linalg/arith**: fusion, tiling, layout/packing choices, loop reordering, whatever you can justify. Output
  must still parse + `verify()` and remain semantically equivalent linalg/arith.
- `lower_interface_to_target`: `{tool} --vortex-optimize-linalg --convert-linalg-to-vortex {input_mlir}` —
  emit **your** vortex-dialect MLIR (must parse + `verify()`)
- `lower_target_to_llvm`: `{tool} --vortex-optimize-linalg --convert-linalg-to-vortex
  --convert-vortex-to-llvm --annotate-merlin-runtime {input_mlir}` — an LLVM-dialect module defining
  `merlin_kernel_body` and carrying the `merlin.grid` / `merlin.arg_table` annotations (below),
  translatable to LLVM IR

Declare these in `manifest.yaml` exactly as the runner expects (see
`merlin/contract/mlir_oot_backend_contract.yaml` and `schemas/manifest.schema.json`). Note there is **no
`emit_command_buffer`** entrypoint for this target: Vortex is a programmable core driven by a compiled
kernel, not a command stream.

## The target dialect is yours to design

You design a Vortex target dialect that models the SIMT execution the hardware actually offers — a
kernel grid over warps and threads, thread/warp identity, divergence and reconvergence, barriers, and
the global/shared memory spaces — and the passes that lower linalg/arith into it and it into the LLVM
dialect. Op and type names are **not** graded (the runner records your target MLIR as evidence and
checks it parses and verifies); what is graded is what comes out the far end. A tensor-resident
`pack/matmul/commit/evict` dialect shape is the wrong model for this target — Vortex has no fixed-function
mesh, no accumulator file, and no residency ISA.

## The device entry point and the Merlin runtime annotation

Your final LLVM-dialect module is linked against the runner's curated bare-metal harness, which owns
startup, the linker script, the KMU entry stub, and the host-side launch. Read
`merlin_vortex_abi.h` (granted with the harness) — it is the whole contract.

Your `--annotate-merlin-runtime` pass must:

- **Define exactly one device symbol**, the kernel body the KMU calls once per launched
  `(block, thread)` coordinate:

  ```c
  void merlin_kernel_body(const merlin_vx_kernel_arg_t* arg);
  ```

  `arg->args[i]` is the 64-bit device address of the *i*-th operand **in the order the capsule's
  `inputs[]` list declares them** (outputs included; they carry `role: output`). This is *not* the MLIR
  C-interface convention — there are no memref descriptors and no `_mlir_ciface_forward`. Thread and
  block identity are **not** passed in; read them from the CTA CSRs (spec sheet §4).

- **Attach a module-level `merlin.grid` attribute** — the integer number of grid coordinates your
  compiled mapping expects to be launched over, e.g. `merlin.grid = 64 : i64`. The runner launches
  exactly this many and will **not** guess a default: how work is spread over coordinates is your
  mapping decision, and a default would hand every backend the same one. A module without it is a
  package error.

- **Attach a module-level `merlin.arg_table` attribute** describing every operand in the same order —
  its `kind` (`weight` / `input` / `output`), `rank`, `dims`, and `elem_size`. This is graded evidence
  and a self-check you can run locally; the harness itself binds by capsule order, so a table that
  disagrees with the capsule is a bug in your package even though the buffers still bind.

The harness fills the argument block, launches over `merlin.grid`, reads the outputs back, and prints
the `OUT` / `METRIC` / `DONE` console protocol. **The harness, the link step, and the binary build are
the runner's** — you emit the kernel module and its annotations, nothing else.

Because this is a bare-metal device kernel, your lowering must not leave calls to host-runtime helpers
(`memrefCopy`, `malloc`, `printf`, …) in the module, and must not require libc init or TLS: the link
checks for exactly that and fails loudly rather than launching a kernel whose prologue did not run.

## How you are graded, and your QA signal

For each capsule the runner does: parse → optimized linalg → vortex dialect → LLVM dialect → translate to
LLVM IR → build an rv64 ELF against the curated harness → run on the capsule's required oracle tiers, and
compare numerics against the withheld golden per the capsule's `numeric_policy`:

- `compare: exact_int` — **bit-exact**, no tolerance (the integer families).
- `compare: tolerance_float` — within the capsule's `rtol`/`atol` (the float families).

Oracle tiers: **L2 = simx** (functionally complete, cycle-approximate) is the default numeric oracle and
runs for every capsule; **L3 = rtlsim** (Verilator, cycle-exact on the real RTL) is required only for the
capsules that declare it. Both run the same ELF.

The runner also checks **SIMT coverage** (`expected_simt_coverage.yaml`): your emitted kernel must
actually use the machine, and the identity CSRs must appear in the built binary. A kernel that collapses
to a single-threaded scalar loop is a **fail**, even if its numerics are perfect. Each capsule's
coverage file has up to three parts:

- `simt_classes` — **all** of these must appear. Thread-mask control and warp spawn, i.e. the kernel
  uses more than one lane and more than one warp.
- `simt_classes_any_of` — **at least one group, in full**. Used where the ISA offers several legal
  mechanisms and gating on a specific one would dictate your mapping. The divergence capsules require
  either `SPLIT`+`JOIN` or `PRED`: their loops have no compile-time bound, so lanes with different trip
  counts must be masked apart somehow — but which mechanism is yours to pick.
- `simt_classes_advisory` — **never gates**. Reported only. Shared-memory classes live here because
  global memory is always semantically sufficient; the scratchpad is a performance choice, and it shows
  up in the cycle count rather than in pass/fail.

Cycles, IPC and FLOP utilization are recorded per capsule and reported, but are **diagnostic only** —
they never gate pass/fail. (They are, however, the point of the comparison: write the fastest correct
backend you can.)

**You cannot run the oracle yourself** (no simx/rtlsim/reference here). After each round a **QA gate**
grades your current `submission/` and writes a **redacted verdict** to `qa/verdict.json` in your working
directory, per capsule: `status`, `failure_plane`, `coverage_violations`, `numeric_status`,
`mismatch_count`, `tiers` (L0–L3), and `all_pass`. **It contains NO expected/golden values.** **Read
`qa/verdict.json` at the start of each round** and fix the failing capsules by their `failure_plane` +
`coverage_violations`. You are relaunched across rounds; your `submission/` and `qa/verdict.json` persist.
Iterate until `all_pass: true` across all public/dev capsules.

Useful self-checks you CAN run locally (no oracle needed): build your tool, run the 4 entrypoints on each
`capsule.interface.mlir`, confirm each stage parses and verifies, confirm your LLVM-dialect module
translates to LLVM IR and assembles for `rv64imafd`, confirm it defines `merlin_kernel_body` and declares
`merlin.grid`, and confirm your `merlin.arg_table` matches the capsule's declared inputs/outputs in
order.

## Hard rules (integrity — your submission must satisfy these)

- `integrity_exempt: false`; no `import merlin` / `from merlin` / `merlin.runtime.reference` /
  `merlin.runtime.simulator` / `reference_outputs` in your sources.
- The device implementation must be **MLIR-lowered**. No hand-written C/assembly compute kernels, no
  copying or calling Vortex's bundled kernels (`tests/regression`, `tests/opencl`, `tests/hip`), no
  OpenCL/PoCL or HIP/chipStar route, no vendor BLAS. The compute must come out of your passes.
- **No Vortex software stack.** Do not read, copy, reconstruct from memory, or link any part of Vortex's
  own software: `vx_intrinsics.h`, `vx_spawn.h` and its work-distribution runtime, the Vortex LLVM fork,
  PoCL, or chipStar. Your grid mapping and your `.insn` emission must come from the spec sheet and the
  RTL. (You may of course write your own helpers — the rule is about *inheriting theirs*.)
- **No forked toolchain.** Stock LLVM only; SIMT ops via `llvm.inline_asm` `.insn` on CUSTOM0.
- **Never hardcode or embed outputs.** The grader runs hidden capsules (different deterministic data,
  same op families) after you freeze; a backend that memorizes public answers will fail them. Compute
  genuinely with one general backend.
- Do not attempt to read withheld goldens, hidden capsules, prior backends, or Merlin internals
  (they are masked/denied — do not work around the sandbox).

## Final status line (end of `submission/REPORT.md`) — write exactly one of:

1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures are listed by capsule and failure plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."

Iterate against the public/dev capsules until they all pass, then stop.
