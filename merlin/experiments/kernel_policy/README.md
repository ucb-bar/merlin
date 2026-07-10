# Experiment: kernel_policy

## Question

Which optimization decisions recur across many kernels (XNNPACK RVV, OpenBLAS RVV, Autocomp
Gemmini, Exo, Triton/triton-cpu) and therefore deserve to become compiler policies /
abstraction candidates?

## How to reproduce

```bash
# 1. Install (Exo ingest, parquet, plots are optional extras)
pip install -e ".[dev,kernels-exo,kernels-parquet,kernels-plots]"

# 2. Index each source (repos passed by path or MERLIN_<SOURCE>_REPO)
kernel-index --source xnnpack    --repo <XNNPACK>         --target rvv     --out output/kernels/xnnpack_rvv_index.json
kernel-index --source openblas   --repo <OpenBLAS>        --target rvv     --out output/kernels/openblas_rvv_index.json
kernel-index --source autocomp   --repo <gemmini_kernels> --target gemmini --out output/kernels/autocomp_gemmini_index.json
kernel-index --source exo        --repo <exo>                              --out output/kernels/exo_index.json
kernel-index --source triton     --repo <triton>                           --out output/kernels/triton_index.json
kernel-index --source triton_cpu --repo <triton-cpu>                       --out output/kernels/triton_cpu_index.json

# 3. Aggregate -> candidates + policies + L6/L8 requirements + report + plots
kernel-extract --inputs "output/kernels/*_index.json" \
  --out output/kernels/abstraction_candidates.yaml \
  --policies output/kernels/policy_rules.yaml \
  --report output/kernels/kernel_mining_report.md --parquet --plots

# 4. Audit marker precision (do the motifs mean what they claim?)
kernel-audit --inputs "output/kernels/*_index.json" --out output/kernels/audit_samples.md
```

Outputs land in `output/kernels/` (gitignored). See `docs/kernel_mining.md` — including the
"Evaluating the results" table mapping each plot/section to the question it answers.

## Profiling (Stage F/G) — `profiling_slate.yaml`

Mining attaches *frequency* to an insight; the slate attaches *magnitude* via paired
ablations (decision on vs off) on a fixed workload set. See `profiling_slate.yaml` for the
per-insight pairs, metrics, and mechanical act/park/skip criteria.

**Toolchain (Spike + Gemmini functional model)** — already present on this host via Chipyard:

```bash
export MERLIN_CHIPYARD=/path/to/chipyard   # or /path/to/chipyard
SPIKE=$MERLIN_CHIPYARD/.conda-env/riscv-tools/bin/spike      # Spike 1.1.1-dev
GCC=$MERLIN_CHIPYARD/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc
ROCC=$MERLIN_CHIPYARD/generators/gemmini/software/gemmini-rocc-tests

# Gemmini spike model = libgemmini.so (built from generators/gemmini/software/libgemmini,
# installed in the same conda env), loaded with:
$SPIKE --extension=gemmini $ROCC/build/bareMetalC/tiled_matmul_ws-baremetal
# (/path/to/chipyard has PREBUILT rocc-tests binaries; verified working, dim=16)
```

Upstream sources if a fresh checkout is ever needed: Spike = `riscv-software-src/riscv-isa-sim`;
Gemmini Spike extension = `ucb-bar/libgemmini` (vendored as a submodule of `ucb-bar/gemmini`
under `software/libgemmini`); test harness = `ucb-bar/gemmini-rocc-tests`.

Fidelity caveat: Spike+libgemmini is a **functional** model — it yields event counts
(mvin/compute/mvout, bytes, fences), not cycles. That settles every memory/dispatch insight
in the slate; cycle-level claims (e.g. `double_buffering`, parked) need Chipyard
Verilator/FireSim.

**Instruction cost model (L2.5).** `merlin/python/merlin/cost_model/` turns the Spike event
counts into *predicted cycles* without per-candidate RTL: a linear per-command model
(`gemmini.py`) whose coefficients are fit against the Verilator sim by
`calibrate.py` (isolation microbenchmarks in `calib/`) and validated against the Stage-F
slate harnesses. Run `python -m merlin.cost_model.calibrate` (needs `MERLIN_CHIPYARD` + a
built Gemmini Verilator sim); coefficients freeze in `cost_model/gemmini_cost_coeffs.json`
with their error band. This is the shared currency for ranking the full regime grid cheaply
and for scoring Autocomp candidates.

## Status

Pipeline implemented in `merlin/python/merlin/kernels/`. Reusable logic lives there, not here.
**Stage-F L2 measured** (harnesses in `stageF/`, results in `output/kernels/stageF/`):
`resident_packed_tensor` → **ACT** (RHS traffic ratio = R, exploitability 1.00);
`accumulator_commit` → **ACT** (4.0× commit bytes; CPU epilogue eliminated);
`command_buffer_batching` → **ACT** (85% of commands are config/fence at 39 tiles; batching
removes 54%); `vl_agnostic_loop` → **PARK** (no instret win at VLEN=128; portability only);
`double_buffering` parked pending cycle-level Verilator sim
(`simulator-chipyard.harness-GemminiAndOPUShuttleConfig` built and ready);
`weight_stationary_dataflow` reclassified as a target-contract fact.
