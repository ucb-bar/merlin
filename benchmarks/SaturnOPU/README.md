# Saturn OPU Analysis

This folder contains compile-only profiling and paper-figure helpers for the
Saturn OPU path. All model compilation goes through `tools/merlin.py` and uses
the in-tree `build/host-merlin-release` compiler so OPU plugin changes are
visible in the generated artifacts.

For the full methodology, metric definitions, CSV schemas, figure
construction, and interpretation caveats, see `METHODS.md`.

## Full Model Decomposition

Run:

```bash
bash benchmarks/SaturnOPU/make_paper_figures.sh --refresh
```

The workflow compiles each registered model into `build/compiled_models/` with
`--dump-artifacts`, `--dump-phases`, and `--dump-graph`, then analyzes:

- per-dispatch source MLIR under `sources/`
- phase dumps under `phases/`
- linked LLVM IR under `files/*.codegen.ll`
- final RISC-V assembly under `files/*.s`

The generated analysis files are:

- `model_dispatch_decomposition.csv`: detailed dispatch rows with op kind,
  analytical ops, OPU path, ukernel tile shape, opcode evidence, and source
  artifact path.
- `model_layer_decomposition.csv`: grouped dispatch-layer summaries.
- `per_model_summary.csv`: model-level normalized compute totals used by the
  paper plot. This includes both OPU analytical compute share and
  `opu_dispatches / dispatches` so near-100 compute numbers do not imply that
  every dispatch or layer is implemented on the OPU.
- `opu_path_opcode_summary.csv`: path-level OPU and RVV opcode audit.

The generated paper figures are written to `figures/`:

- `optimization_journey.{pdf,png}`
- `per_model_decomposition.{pdf,png}`

The decomposition plot uses dispatch share for the normalized bar so models
with many non-OPU elementwise dispatches do not look fully covered just because
their matmuls dominate compute. The right annotation remains two-part:
`<OPU compute share> compute` and `<OPU dispatches>/<all dispatches>
dispatches`. The compute share is a MAC-equivalent analytical-op metric, not a
runtime-cycle share and not a claim that all model dispatches are on OPU.
Within the OPU portion, color family identifies the OPU path (`encoding`
versus runtime `mmt4d`), while opacity identifies tile quality: full opacity is
the 16x16 path, medium opacity is 8x8 or other smaller tiles, and low opacity
is a narrow tile path.

## Focused Runs

Analyze one model without recompiling when artifacts already exist:

```bash
conda run -n merlin-dev uv run python benchmarks/SaturnOPU/analyze_model_paths.py --model yolov8_nano
```

Force regeneration for one model:

```bash
conda run -n merlin-dev uv run python benchmarks/SaturnOPU/analyze_model_paths.py --model tinyllama --refresh
```

## Matmul Microbenchmarks

- `compile_matmul_opu_i8_ukernel_all.sh`
  - Compiles an i8 matmul workload for `models/saturn_opu.yaml` (`--hw OPU`).
  - Checks the emitted assembly for required OPU opcodes.

- `compile_matmul_opu_fp8_ukernel_all.sh`
  - Compiles the FP8 mmt4d path for static codegen inspection.
