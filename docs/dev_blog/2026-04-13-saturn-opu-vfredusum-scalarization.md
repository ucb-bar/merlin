# 2026-04-13: Saturn OPU `vfredusum.vs` hang — opcode survey + MLIR scalarization fix

> **Repro pin:** merlin@[`320fbf06`](https://github.com/ucb-bar/merlin/commit/320fbf064de5572cbae276206c479f0bed843eb8) · iree_bar@[`68acd99c74`](https://github.com/ucb-bar/iree_bar/commit/68acd99c74)
> **Status:** Active

Related entries:

- [2026-04-14 f32-reduction lowering hang](2026-04-14-f32-reduction-hang-findings.md) — sibling reduction-hang on the same hardware; the present entry fixes the `vfredusum.vs` opcode case via MLIR scalarization, while the 2026-04-14 entry handles the broader f32-reduction tree-reduction codegen via per-function `target-features="-v"`.

## Context and Goal

ViT-small inference on the Saturn OPU FireSim build hangs at HAL dispatch
ordinal 9 (the first LayerNorm). Output stops after exactly one workgroup
print:

```
[apply] #15 ord=9 bindings=3
[d] #1073 o=9 wg=8       ← only 1 print, no second wg, no return
```

Comprehensive `vsetvli` fusing in the OPU matmul ukernel did not fix it (Phase
1 of the prior diagnostic plan ruled the matmul out: ord=9 is *not* the
matmul, it's the LayerNorm reduction). The goal of this work is to (a) get
*direct* evidence of which RVV instruction hangs the Saturn vector unit, and
(b) produce a structural codegen fix that prevents emission of the offending
opcode rather than chasing the symptom in one model at a time.

## Implementation Changes

### 1. Isolated RVV self-test on FireSim

`samples/SaturnOPU/simple_embedding_ukernel/model_benchmark.c` now has a
`#ifdef SATURN_RVV_SELFTEST` block (run before any IREE init in `main()`) that
probes each suspected-hang opcode in its own inline-asm checkpoint with an
`fprintf(stderr,...) + fflush(stderr)` immediately after. A hang inside the
asm block leaves the previous `[rvv] cp=N` line as the last visible UART
output, so the *exact* hung opcode is identified by the surviving prefix.

Each checkpoint is independently skippable via a compile-time bitmask
(`SATURN_RVV_SELFTEST_SKIP=0x18` skips `cp=3` and `cp=4`), so we can step past
a confirmed hang to expose the next probe across multiple FireSim runs:

| cp | opcode             | role                                          |
|----|--------------------|-----------------------------------------------|
| 1  | `csrr vlenb`       | baseline sanity                               |
| 2  | `vadd.vv`          | control — plain vector arithmetic             |
| 3  | `vfredusum.vs`     | LayerNorm sum reduction (PRIMARY suspect)     |
| 4  | `vfsqrt.v`         | LayerNorm `inv_sqrt(var+eps)`                 |
| 5  | `vrgather.vi`      | LayerNorm broadcast / gather                  |
| 6  | `vfredmin.vs`      | softmax / argmin                              |
| 7  | `vfredmax.vs`      | softmax max-subtract                          |
| 8  | `vfwredusum.vs`    | widening (f32→f64) reduction                  |
| 9  | `vfslide1down.vf`  | tree-reduction primitive (LLVM fallback)      |

### 2. FireSim host-level timeout wrapper

A hang inside the running RISC-V ELF freezes the simulated core (the vector
unit never retires the bad opcode), so a software timeout *inside* the binary
cannot rescue it. `build_tools/firesim/run_rvv_selftest.sh` now wraps
`firesim runworkload` with a host-side `timeout --signal=TERM
--kill-after=30s ${TIMEOUT}s ...` (default 5 min) and follows up with
`firesim kill` so the FPGA is clean for the next survey iteration. The script
also accepts a skip-mask argument and rebuilds the rvvtest binary with the
matching `-DSATURN_RVV_SELFTEST_SKIP=...` cmake flag, so the full Phase A
survey is just four invocations:

```bash
bash build_tools/firesim/run_rvv_selftest.sh 0x00   # baseline (cp=3 hangs)
bash build_tools/firesim/run_rvv_selftest.sh 0x08   # skip cp=3 → expose cp=4
bash build_tools/firesim/run_rvv_selftest.sh 0x18   # skip cp=3,4 → expose cp=5
bash build_tools/firesim/run_rvv_selftest.sh 0x38   # skip cp=3,4,5 → expose cp=6..9
```

### 3. `+xopu`-gated MLIR scalarization pattern

`third_party/iree_bar/compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`
has a new `ScalarizeXopuFloatReductionPattern` that rewrites every
`vector.reduction <add|mul|minimumf|maximumf|minnumf|maxnumf> %v :
vector<Nxf*> into f*` into a chain of scalar `vector.extract` + scalar
`arith.{add,mul,...}f` ops, preserving the optional accumulator and
`fastmath` flags. The pattern runs inside `ConvertToLLVMPass::runOnOperation()`
*before* the existing Vector→Vector and Vector→SCF lowerings, gated strictly
on `hasFeature(targetConfig, "+xopu")` so non-Saturn RISC-V (and non-RISC-V)
backends are byte-identical before and after. Integer reductions are
intentionally not scalarized — Phase A has not implicated them.

## What Worked

### Phase 1 — ord=9 ≠ matmul

Dumping the vmfb dispatch table with `iree-dump-module` and cross-referencing
against the post-vectorization MLIR confirmed dispatch ordinal 9 in vit_small
is a `linalg.generic`-only LayerNorm reduction (input `64×128xi8`, output
`64×128xi8`), **not** an OPU matmul. This pivoted the investigation away
from `opu_matmul_riscv_64.c` entirely.

### Phase 2 — assembly review of dispatch_1

`llc -march=riscv64 -mattr=+v -mabi=lp64d -O3` on the dispatch_1 bitcode
emitted 706 lines of RV64+V assembly with 58 `vsetvli`/`vsetivli` and 388 RVV
instructions in the inner K-loop — all `vsetvli`s correctly placed, no
missing prologue or stripped fence. The vector instructions themselves are
the problem, not their preamble.

### Phase 3 — selftest pinpoints `vfredusum.vs`

First FireSim run with `SATURN_RVV_SELFTEST=1` (skip mask 0):

```
[rvv] SELFTEST START skip=0x0 (cps 1..9)
[rvv] cp=1 vlenb=16
[rvv] cp=2 vadd c=[11,22,33,44]
                                 ← cp=3 NEVER PRINTS — hang
```

`vfredusum.vs` on `vector<4xf32>` reducing `{1,2,3,4}` into a scalar with a
`vfmv.s.f`-initialised `0.0` accumulator never returns. This is the
smoking gun.

### Phase B — opcode emission inventory (host-side, no FireSim)

`llvm-objdump` on the linked dispatch ELF for each model gives a definitive
count of each suspected opcode:

| model       | vfredusum.vs | vfredmax.vs | vfsqrt.v | vrgather.vi | vfslide1down.vf | vfredmin.vs | vfwredusum.vs |
|-------------|-------------:|------------:|---------:|------------:|----------------:|------------:|--------------:|
| vit_small   |           54 |           4 |        4 |          80 |              18 |           0 |             0 |
| large_mlp   |            0 |           0 |        0 |           0 |               0 |           0 |             0 |
| smolvla     |     deferred |    deferred | deferred |    deferred |        deferred |    deferred |      deferred |

Reproduce with:

```bash
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/opu_bench_suite/opu_bench_vit_small.q.int8.mlir \
  --target saturn_opu --hw OPU --dump-artifacts \
  --output-dir /tmp/vit_small_phaseB
build/host-merlin-release/llvm-project/bin/llvm-objdump -d --mattr=+v \
  /tmp/vit_small_phaseB/binaries/*.so | \
  grep -oE '(vfred[a-z]+\.vs|vfwred[a-z]+\.vs|vfsqrt\.v|vrgather\.v[ix]|vfslide1down\.vf|vcompress\.vm)' | \
  sort | uniq -c
```

Two important takeaways:

1. **`large_mlp` emits zero suspect opcodes.** It's a pure GEMM model — every
   reduction lives inside the `iree_uk_opu_matmul` ukernel (custom OPU
   instructions, not RVV reductions). This makes it the perfect regression
   baseline: our `+xopu`-gated pattern must produce a byte-identical ELF for
   large_mlp before and after the change.
2. **vit_small uses `vfslide1down.vf` 18× already.** This is LLVM's
   tree-reduction fallback, almost certainly co-emitted alongside the 54
   `vfredusum.vs` for the same `vector.reduction add` ops. After our pattern
   scalarises those reductions, we expect the slide count to drop sharply.

`vfredmin`, `vfwredusum`, and `vcompress` don't appear in any model we've
inventoried, so we can defer probing them on FireSim until a model needs
them. The pattern still covers `<minimumf|maximumf|minnumf|maxnumf>` just in
case future codegen needs it.

## What Did Not Work

- **"Comprehensive `vsetvli` fusing in `opu_matmul_riscv_64.c`"** (twice). The
  matmul wasn't even involved — Phase 1 ruled it out.
- **Trying to add `printf` checkpoints inside `iree_uk_opu_matmul_loop`.**
  That ukernel is compiled as LLVM bitcode with `-nostdinc -ffreestanding`,
  so libc isn't linked into the embedded ELF; an unresolved `printf` symbol
  breaks dispatch loading. The IREE-blessed escape hatch is the
  `iree_hal_executable_environment_v0_t::import_funcs[]` callback (used by
  `iree_h2f_ieee`) — but Phase 3 made that unnecessary.
- **Initial attempt to `llvm-objdump --mattr=+xopu`.** `+xopu` isn't a known
  LLVM feature; Saturn OPU custom opcodes are emitted by source-level
  `.insn r 0x57, ...` directives, not by the codegen, so `--mattr=+v` alone
  is enough for static analysis.
- **`llc` → `.s` analysis as a hang predictor.** The assembly looked
  *correct*. The bug is in the hardware's execution of `vfredusum.vs`, not
  in the instruction encoding — only direct hardware probing exposed it.

## Debugging Notes

- **A hang on the simulated core means the host wrapper must impose the
  timeout.** No software-side guard *inside* the running ELF can break a
  stuck vector instruction; the core never retires and never returns to the
  scheduler. The host shell `timeout` + `firesim kill` pair is the only
  reliable way to bound a survey run.
- **Each new `[rvv] cp=N` probe must use `.option push / .option arch, +v
  / ... / .option pop`.** The bare-metal toolchain compiles the rest of the
  C with `-march=rv64imafdc` (no `+v`), so the inline-asm assembler refuses
  vector mnemonics unless we locally enable `+v` for the block.
- **`vfwredusum.vs` requires a two-stage `vsetvli`** (e64 to seat the f64
  scalar accumulator in `v9[0]`, then e32 for the source vector and the
  reduction itself, then e64 again to extract the scalar back out). Vector
  register contents survive the `vsetvli` change, so this is safe.
- **`large_mlp` already exercises the +xopu codegen pipeline end-to-end with
  zero suspect opcodes.** Use it as the regression oracle for any future
  +xopu-gated change in `ConvertToLLVM.cpp`.
- **The pattern is intentionally narrow.** It only matches floating-point
  reductions, only the six combining kinds we expect. If a future model
  trips integer reductions (`vredsum.vs` etc.) and Phase A confirms those
  also hang, broaden the match — don't widen prophylactically.

## Test Coverage and Commands

### Reproduce the hang (pre-fix)

```bash
bash build_tools/firesim/run_rvv_selftest.sh 0x00 300
# Expected: cp=1 prints, cp=2 prints, cp=3 never prints, host timeout fires.
```

### Run the full Phase A opcode survey

```bash
for mask in 0x00 0x08 0x18 0x38; do
  bash build_tools/firesim/run_rvv_selftest.sh "$mask" 300
done
# Each run rebuilds the rvvtest binary with the new skip mask, runs under a
# 5-min host timeout, and prints "stopped after cp=N" verdict.
```

### Verify the fix statically (post-Phase C)

```bash
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/opu_bench_suite/opu_bench_vit_small.q.int8.mlir \
  --target saturn_opu --hw OPU --dump-artifacts \
  --output-dir /tmp/vit_small_post_fix
build/host-merlin-release/llvm-project/bin/llvm-objdump -d --mattr=+v \
  /tmp/vit_small_post_fix/binaries/*.so | \
  grep -cE '(vfred[a-z]+\.vs|vfwred[a-z]+\.vs)' # expect 0
```

### Verify regression-free for non-LayerNorm models

```bash
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/opu_bench_suite/opu_bench_large_mlp.q.int8.mlir \
  --target saturn_opu --hw OPU --dump-artifacts \
  --output-dir /tmp/large_mlp_post_fix
diff <(llvm-objdump -d /tmp/large_mlp_phaseB/binaries/*.so) \
     <(llvm-objdump -d /tmp/large_mlp_post_fix/binaries/*.so)
# expect: no diff (large_mlp emits zero vector.reduction ops)
```

### Verify the fix dynamically

```bash
# Stage vit_small under FireSim and run end-to-end inference.
# Expected log:
#   [apply] #15 ord=9 bindings=3
#   [d] #1073 o=9 wg=8 ...
#   [d] #1074 o=9 wg=9 ...   ← second workgroup now appears
#   ... rest of inference ...
#   DONE
```

## Addendum (2026-04-14) — residual narrow-M hang in `iree_uk_opu_matmul`

The vfredusum scalarization fixed every LayerNorm/softmax dispatch in
vit_small. Post-fix, the model runs cleanly through dispatches 0–8, enters
dispatch 9 (`matmul_like_64x128x128`), and the simulation stalls at the
first workgroup of that dispatch (`[d] o=9 wg=0,0,0 of 8,1,1`). This is a
**second, independent bug** inside the tier-6 OPU matmul ukernel
(`iree_uk_opu_matmul`, compiled as LLVM bitcode under `-ffreestanding
-nostdinc`). Static disassembly of the post-fix ELF still shows zero
`vfred*` opcodes — the scalarization fix is doing its job, but the
ukernel hangs for a different reason.

Characterization:

- The ukernel works for `large_mlp` (`128×2048×2048`, M-tiles=8) and every
  transformer batch_matmul in ViT (those take the sibling
  `iree_uk_mmt4d` ukernel, not this one).
- The ukernel fails for `vit_small` dispatch 9 (`64×128×128`, M-tiles=4)
  and would fail for any narrow-M 3D matmul passed through the encoding
  resolver. By inspection, the likely regime is `M-tiles < 8`.
- The hang is bitcode-internal (no libc), so standard `printf` debugging
  requires plumbing an IREE-style ukernel-import "escape hatch"
  (analogous to `iree_h2f_ieee`), or a higher-level compile-time
  workaround (pad M to a multiple of 8 before the encoding resolver, or
  downgrade narrow matmuls to the `mmt4d` path instead of `opu_matmul`).

Runtime engineering improvements landed alongside this investigation so
the next debug iteration is tractable:

- **Warmup-gated debug prints.** A new `iree_merlin_dispatch_debug_enabled`
  flag (defined in `iree/hal/utils/deferred_command_buffer.c`) suppresses
  `[apply]` / `[d]` / `[vm_invoke]` prints until the benchmark calls
  `iree_merlin_enable_dispatch_debug(1)` right before `Warmup START`.
  Previously, ~1070 init dispatches spammed the UART before reaching the
  hang, turning every FireSim iteration into a 10-minute affair. After
  gating, the hang reproduces in seconds.
- **Host-level FireSim timeout + signal-clean cleanup.**
  `build_tools/firesim/run_phase_d.sh` now takes `timeout=0` for
  unbounded runs, traps SIGINT so `firesim kill` always runs at
  teardown, and falls back to `$FIRESIM_RUNS_DIR/sim_slot_0/uartlog`
  when the kill path interrupts artifact collection.

The narrow-M ukernel bug is out of scope for the vfredusum scalarization
change. Tracked as a follow-up below; the vfredusum fix and the paper
figures stand on their own.

## Addendum 2 (2026-04-14) — narrow-M bypass in the OPU encoding resolver

Resolved the narrow-M hang with a targeted bypass inside the OPU encoding
resolver. The fix is in
`third_party/iree_bar/compiler/src/iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.cpp`
in `lowerOPUContractionToUkernel`: for 2D-output matmul dispatches with
static M-tile-count < 8, the helper now emits `linalg.mmt4d` (into a
fresh 4D packed output) plus a `tensor.unpack` back to the 2D identity
shape, rather than calling the tier-6 `iree_uk_opu_matmul` ukernel. The
compute still runs on OPU hardware because `iree_uk_mmt4d` has a runtime
`+xopu` dispatch into `iree_uk_mmt4d_opu_full_loop` (the tier 2-4
VOPACC path).

Why this works where the earlier attempts failed:

- The earlier global `iree-opu-disable-encoding-resolver` flag broke
  batch_matmul dispatches (ViT attention) because the default CPU
  encoding resolver doesn't handle their non-standard indexing maps.
  Keeping the OPU resolver attached means batch_matmul dispatches still
  take the OPU custom path; only the narrow-M 2D case is overridden.
- The preprocessing-pass attempts (rewriting to `linalg.batch_matmul`
  batch=1 or padding M to 128) were undone by downstream canonicalization
  / dispatch hoisting. The per-op bypass inside the encoding resolver
  runs at the right layer.

Verification:

| Model        | `iree_uk_opu_matmul` (pre-fix) | post-fix | `iree_uk_mmt4d` (pre-fix) | post-fix |
|--------------|-----:|-----:|-----:|-----:|
| vit_small    | 6    | 0    | 5    | 11   |
| large_mlp    | 4    | 4    | 0    | 0    |
| vit (full)   | (hangs) | completes | — | — |
| tinyllama    | (hangs) | completes | — | — |

`large_mlp` is untouched (its M-tile count is 8 ≥ threshold) and keeps
the tier-6 fast path. `vit_small`, `vit`, `tinyllama` all drop to
tier 2-4 perf (~35 Ops/cyc analytical, ~26× the RVV baseline) but are
now runnable end-to-end on FireSim.

Paper figures (`benchmarks/SaturnOPU/make_paper_figures.sh`): for the
per-model decomposition plot, vit_small / vit / tinyllama now show their
matmul compute in the `opu_mmt4d` green segment instead of the
`encoding_resolver` blue segment. Overall OPU % per model is unchanged
(both are OPU-accelerated paths). `large_mlp` is unchanged.

## Follow-Up Tasks

- [ ] **Residual narrow-M hang** in `iree_uk_opu_matmul`. Candidates: (a)
      ukernel-internal debug via IREE ukernel-import escape hatch, (b)
      compile-time pad-to-8 narrow-M workaround, (c) route narrow 3D
      matmuls through the `iree_uk_mmt4d` ukernel which handles them
      correctly (requires output-shape plumbing since mmt4d produces 4D
      while the encoding resolver produces 2D identity).
- [ ] Run Phase A survey end-to-end on FireSim (4 runs × 5-min timeout) to
      get works/hangs status for `vfsqrt.v`, `vrgather.vi`, `vfredmax.vs`,
      and `vfslide1down.vf`. If `vfsqrt` or `vrgather` also hang, extend the
      MLIR pattern to scalarize `math.sqrt` over vectors and lower
      `vector.broadcast` / `vector.shuffle` to `vector.insert` chains
      respectively.
- [ ] Re-run vit_small full inference on FireSim post-fix to confirm
      LayerNorm now returns and the model completes.
- [ ] Run smolvla compile (skipped here — 909 MB MLIR, not host-compile
      friendly without batching) and inventory its opcode emission to
      confirm the same pattern set covers it.
- [ ] File an upstream issue against
      [`saturn-vectors`](https://github.com/ucb-bar/saturn-vectors) RTL
      referencing the works/hangs table once Phase A is complete.
- [ ] Once stable, gate the rvvtest binary behind a default-OFF
      `MERLIN_BUILD_OPU_RVV_SELFTEST` cmake option in
      `samples/SaturnOPU/simple_embedding_ukernel/CMakeLists.txt` so it
      stays in tree as a hardware-bringup probe without bloating default
      builds.
- [ ] Restore `chipyard/sims/firesim/deploy/config_runtime.yaml` from the
      `.bak_rvvtest` backup once the rvvtest workload is no longer the
      active one.
