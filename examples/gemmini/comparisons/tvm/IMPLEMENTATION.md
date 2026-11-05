# TVM–Gemmini implementation summary

The comparison compiles models and kernels for a Gemmini accelerator attached to
a Rocket core. It does so through Apache TVM, with stock TVM search, and
measures the result on FireSim. It has two parts:

- a **TVM fork** ([`grammy117/tvm` `gemmini/direct-intrinsics`](https://github.com/grammy117/tvm/tree/gemmini/direct-intrinsics),
  based on TVM v0.19.0 `c4dc0c29`), holding the Gemmini backend and its build
  tools; this branch pins the F2 revision `f3cfd1a7d` as the submodule
  `third_party/tvm`;
- **Merlin orchestration** (this directory), holding the frozen inputs,
  complete-unit ELF builders, the FireSim study runner, budgets, watchdog and
  evidence.

This page is the short version. The full design record and its limits on claims
are in [ARCHITECTURE.md](ARCHITECTURE.md). The operation inventory is in
[TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md), and the code provenance (bringup
by Nicolas Reed, curation and later work by Graham Anderson) is in
[OWNERSHIP.md](OWNERSHIP.md).

## Compilation path

```text
frozen program + inputs + reference       (Merlin: prepare_kernel_benchmarks.py, restore_resnet_inputs.py)
  -> Relax import, stock legalize / fuse (level 0-3) / optional CSE / DCE
  -> mathematical TensorIR, workload annotation, stock extract_tasks
  -> Gemmini schedule rules: default schedule, or a ReplayTrace record from a JSONDatabase
  -> TensorIntrins (matmul/conv, copies, widening, ReLU, sums, pooling, affine)
  -> scoped SRAM -> row allocation, DMA ordering and fences
  -> LLVM RV64 operator objects + C primitive adapter (matmul.c) + static graph wrapper
  -> bare-metal ELF with guest output check (valid.elf) and wrong reference (bad-oracle.elf)
  -> FireSim: warm-call target cycles                       (Merlin: run_firesim_study.py)
```

## TVM side (`gemmini/direct-intrinsics`)

| Module | Role |
| --- | --- |
| `gemmini.py` | Entry point: prepares mathematical TensorIR and inlines eligible host producers |
| `gemmini_target.py` | Capability facts of the pinned integer Gemmini: 16×16 mesh, 16,384 scratchpad / 1,024 accumulator rows, INT8 inputs, 20-bit PE, 32-bit accumulator, bound to the `gemmini_params.h` hash |
| `gemmini_tensorir.py`, `gemmini_schedule.py` | Structural scheduling of TensorIR onto the primitive C ABI; deterministic default schedule |
| `gemmini_sketch.py`, `gemmini_meta_schedule.py` | Contraction sketches as ordinary traced schedules; the design space for stock MetaSchedule |
| `gemmini_search.py` | Stock `tune_tasks` with ReplayTrace, round-robin task scheduling and an external cycle runner |
| `gemmini_intrin.py`, `gemmini_elementwise.py`, `gemmini_reduction.py`, `gemmini_pool.py`, `gemmini_affine.py`, `gemmini_bytes.py`, `gemmini_epilogue.py`, `gemmini_bias.py`, `gemmini_regions.py` | TensorIntrin descriptions/implementations per operation family, with domain and rounding guards |
| `gemmini_memory.py` | Lowers scoped scratchpad/accumulator buffers to row allocations and checks capacity and aliasing |
| `gemmini_legacy.py`, `gemmini_conv.py`, `gemmini_tuning.py` | Historical graph/template path and nine-template XGB experiment, loaded only on explicit request |
| `apps/gemmini/compile_package.py`, `baremetal.py`, `matmul.c/.h` | Static package export, bare-metal runner, C command adapter |
| `apps/gemmini/verify_matmul.py` | Builds and audits the adapter for a given toolchain and header set |
| `apps/gemmini/tune_models.py`, `tuning_library.py` | Task extraction and local tuning tools |

The design keeps ordinary TVM where it can. Graph optimization is stock Relax.
Search is stock MetaSchedule over traced TensorIR, and tensorization uses
`RewriteTensorize`. The custom pieces are the schedule rules, the TensorIntrins,
the final placement pass that applies the database (or the default) and lowers
memory, the C runtime and the measurement adapter. Automatic bias folding,
constant-weight prepacking, scale/readout rewrites, DMA command merging and
zero-init elimination were removed before the F2 campaign.

## Default schedule and search space

The default contraction schedule picks the largest exact intrinsic tile up to 16
per dimension. It ranks bounded M/N reuse groups by logical operand traffic,
then by group size and K batching, and accepts the first candidate whose real
aligned SRAM allocation fits: at most eight tries, then a one-group fallback. It
starts output-stationary, with no hoisting and no pipelining. This is a
heuristic, not a cost model.

ReplayTrace samples traces over: intrinsic rectangle, M/N/K reuse groups,
output- vs weight-stationary dataflow, loop order, hoisting, pipelining, staging
and tails. The F2 profile is `compact`: epilogues stay at root, and unroll is 0
or 16. Stock graph alternatives are fusion levels 0–3, each with CSE on or off.
Physical addresses and required fences are fixed lowering, not search axes.

## Measurement (Merlin side)

| Piece | File | What it does |
| --- | --- | --- |
| Study preparation | `prepare_active_study.py` | Extracts contraction tasks per unit and assigns each structural workload key to one owner (20 initial trials of a 25 cap) |
| Study runner | `run_firesim_study.py`, `tune_firesim_chunks.py` | Runs owner campaigns concurrently over a slot pool, under an absolute deadline and a persistent occupancy ledger |
| Slot pool | `firesim_job_pool.py` | Locks slots, writes `gemmini_firesim_probe_request_v4` requests and validates results (booted ELF hash, hardware receipt, guest exit and `DSE_PASS`, enforced deadline, remote stop) |
| Watchdog | `firesim_timeout.py` | `median_wall_v1`: floor 3× median of 3 baseline walls, then 3× latest-nine median, between bootstrap and ceiling |
| Cost unit | `firesim_runner_seconds.py` | Converts warm-call target cycles to seconds at the documented clock for TVM's Runner |
| Complete-unit builders | `prepare_kernel_benchmarks.py`, `build_shared_chunk_elf.py` | Build whole kernels, ResNet chunks and the connected parent (all heads, all outputs), with or without a database |

The cycle label is the target-cycle count of the second (warm) call of the whole
unit, read from the guest's `DSE_CYCLES` line. It includes host work, packing,
DMA and completion, and excludes boot and the first call. Search labels apply to
extracted tasks, but every reported number comes from a rebuilt complete unit.
Timed-out or incorrect candidates get no label.

The hardware-facing adapter (`tvmfs_adapter.py` plus the `f2-exec`/`f2-flash`
host scripts) belongs to the deployment, not the compiler. The F2 copies are in
the [measurement product](../../../../out/artifacts/measurements/firesim_f2_gemmini_rocket_ddr3llc_60mhz/resnet50_kernels16/tvm_replaytrace_v1_20261105T022306Z_5998ce4/tools/)
and are described in [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md).

## Known limits

- The only end-to-end result is ResNet50 on the F2 60 MHz DDR3+LLC image. The
  selected U250 30 MHz target and the full TinyLlama/SmolVLA models were not run.
- Only contractions are tuned. Secondary operations keep fixed schedules or host code.
- Search is 20 random traces per owner from an engineered default, with no
  learned model.
- The diagnostic Spike oracle is not the spec-pinned libgemmini oracle.
  Per-program software-spec admission is incomplete; see
  [ARCHITECTURE.md](ARCHITECTURE.md#constraints-on-paper-claims).
