# Approach (d) — hand-written C++ Gemmini dialect via IREE: SOLVED (runs + PASSES + cycles on spike)

**State (2026-06-18): WIRED, RUNS, PASSES, and emits parseable cycles on spike. The 4th canonical arm
is complete for functional correctness across all matmul + attention corpus kernels (17/20; 3 giants
skipped). No cycle-accurate (verilator) column yet — see "remaining".**

## What works (the whole path)
- Kernel emitted as a tensor-domain `linalg.matmul` fixture → compiled through the **real IREE Gemmini
  plugin** (`iree-compile --iree-plugin=gemmini --iree-gemmini-enable
  --iree-gemmini-lower-back-to-iree=false`, mirroring `models/gemmini_spike.yaml`).
- The **hand-written C++ Gemmini dialect genuinely fires**: the dispatch object carries **161 `gemmini.*`
  RoCC instructions** (config/mvin/preload/compute/mvout) — not a fallback.
- The proven IREE-runtime runner (`samples/SaturnOPU/simple_embedding_ukernel/gemmini_spike_runner.c`)
  embeds the .vmfb, creates the local-sync HAL device + embedded-elf loader, `iree_vm_invoke`s the
  matmul (CORRECT bindings via the IREE runtime — no hand-rolled dispatch ABI), reads back the result,
  and verifies vs an all-ones reference (each output == K). It now also wraps the invoke with
  `read_cycles()` and prints `METRIC cycles <n>`.

## The two fixes that unblocked it
1. **rdcycle illegal-instruction** during `iree_vm_invoke` (IREE's internal timing did `rdcycle`):
   run spike with **`--isa=rv64gcv_zicntr_zihpm`** (zicntr enables the cycle CSR). Bare-metal htif ELF,
   **NO pk** (`spike --extension=gemmini --isa=rv64gcv_zicntr_zihpm <elf>`).
2. **Cycle reporting:** added a `static inline read_cycles()` (rdcycle) + `METRIC cycles` print around
   invoke+fence in `gemmini_spike_runner.c`.

The earlier "fundamental, opaque" blocker (old bare-metal hand-rolled dispatch-ABI harness, superseded)
was simply the missing `zicntr` extension — not a deep IREE-on-spike bug.

## Build recipe (firesim cross-toolchain = clang, NOT chipyard gcc)
```
export RISCV_TOOLCHAIN_ROOT=/path/to/merlin-iree/build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV
export RISCV_NEWLIB_SYSROOT=/path/to/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf
cmake -S third_party/iree_bar -B build/firesim-merlin-release -DGEMMINI_SPIKE_MATMUL_SHAPE=<MxNxK>
touch build/firesim-merlin-release/build.ninja   # ninja auto-regen strips env; pin the configure
ninja -C build/firesim-merlin-release bench_gemmini_spike_matmul
```
Per-shape: a fixture `tests/integration/gemmini_spike/fixtures/matmul_<M>x<N>x<K>_tensor.mlir` must
exist (the CMake regex binds group1=M, group2=N, group3=K; A=M×K, B=K×N). **Convention transpose:** the
corpus encodes shapes M×K×N, so the fixture filename swaps N and K. The driver
`experiments/gemmini_perf_bench/scripts/run_iree_arm.py` generates fixtures, builds, runs, and parses
cycles for every feasible matmul/attention kernel; `merge_iree_arm.py` stitches them into the run's
`perf_results.json` as the `iree_dialect` approach.

## Result + remaining
- 17/20 matmul+attention kernels PASS on spike with cycles (3 tiny_llama giants skipped: macs 184M–1B,
  spike-infeasible). All PASS.
- **No verilator/FireSim cycle column yet:** running the full IREE runtime on cycle-accurate RTL is
  heavy. spike is FUNCTIONAL — it does not model Gemmini timing (see the perf report), so the IREE
  spike cycles are NOT a valid performance number; they confirm correctness only. Cycle-accurate IREE
  numbers are deferred to the FireSim L5 backfill.
