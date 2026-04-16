# ASPLOS'27 Saturn OPU — Archival Record

Frozen snapshot of the exact merlin + iree_bar state that produced
every figure, table, and binary in the paper. Never merged; ongoing
work lives on `dev/main`.

## External SHA pins

| Repo | Branch | SHA |
| --- | --- | --- |
| merlin (this repo) | `ASPLOS27-OPU` | this commit |
| iree_bar (fork) | `ASPLOS27-OPU` | via submodule |
| chipyard | `main` | `bcb612918bb964a0dad33a26a1921c161eb598d0` |
| saturn-vectors | `opu-int8` | `ea373800169ce7bbb9ca27ececf74101a7caf9d5` |
| shuttle | — | `337385f3634ad0489fa903d9152fcd29d2279f3b` |
| firesim | — | `b084672c2f8cf32e55d78f73a001074c23f8a2b8` |

FireSim hardware recipe (see `build_tools/hardware/saturn_opu_u250.yaml`):
`alveo_u250_firesim-opu-v128-d64-shuttle`, `FireSimOPUV128D64ShuttleConfig`
at 60 MHz. V128-D64 = 128-bit vector register width, 64-depth OPU
tile (128 ops/cyc peak int8).

## Oversize MLIRs (regenerate — excluded from git)

Two quantized MLIRs exceed GitHub's 100 MB limit and are not tracked
on this branch. Regenerate from the in-tree ONNX sources:

```bash
uv run tools/merlin.py compile \
    models/opu_bench_suite/opu_bench_vit.onnx \
    --target saturn_opu --hw OPU_LLM           # ViT v3  → 151 MB

uv run tools/merlin.py compile \
    models/tinyllama/tinyllama.onnx \
    --target saturn_opu --hw OPU_LLM           # TinyLlama → 2.2 GB
```

Their compiled `.vmfb` artifacts are already inside
`compiled_vmfb.tar.zst` (see next section).

## Binary archive (out-of-tree)

Binaries live at `/scratch2/agustin/artifacts/ASPLOS27-OPU/` and are
keyed by sha256 in `ARCHIVE_BINARIES.md`:

- `firesim_bench_binaries.tar.zst` — 67 `bench_model_*` ELFs (~1.48 GB)
- `compiled_vmfb.tar.zst` — 9 model `.vmfb` + IREE intermediates (~496 MB)
- `sha256sums.txt`

Restore:

```bash
ART=/scratch2/agustin/artifacts/ASPLOS27-OPU
ELFDIR=build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel
VMFBDIR=build/compiled_models/opu_bench_suite

mkdir -p "$ELFDIR" "$VMFBDIR"
(cd "$ELFDIR"  && tar --zstd -xf "$ART/firesim_bench_binaries.tar.zst")
(cd "$VMFBDIR" && tar --zstd -xf "$ART/compiled_vmfb.tar.zst")
(cd "$ART" && sha256sum -c sha256sums.txt)
```

## Paper figures

```bash
# Batch (reads pre-computed CSVs)
bash benchmarks/SaturnOPU/make_paper_figures.sh

# Individual — examples
uv run benchmarks/SaturnOPU/plot_optimization_journey.py
uv run benchmarks/SaturnOPU/plot_model_decomposition.py \
    --include-models=opu_bench_convnet,tinyllama,yolov8_nano,opu_bench_vit,dronet,mlp_fast \
    --rename=opu_bench_vit:ViT,mlp_fast:MLP \
    --figsize-w=4.5 --out-name=per_model_decomposition_over110
uv run benchmarks/SaturnOPU/plot_sankey_paper.py          # hybrid
uv run benchmarks/SaturnOPU/plot_sankey_yolov8_paper.py   # yolov8
```

Input CSVs (all under `benchmarks/SaturnOPU/`):

| CSV | Plot |
| --- | --- |
| `optimization_journey.csv` | optimization journey |
| `model_dispatch_decomposition.csv`, `per_model_summary.csv` | per-model decomposition |
| `opu_utilization_per_dispatch.csv`, `opu_utilization_per_model.csv` | utilization |
| `opu_path_opcode_summary.csv` | opcode audit |
| `firesim_v128d64_results.csv` | authoritative speedups |
| `model_layer_decomposition.csv` | layer-level breakdown |

Raw uartlogs: `benchmarks/SaturnOPU/firesim_sweep_results/uartlogs/`.
Sweep orchestrators: `build_tools/firesim/run_final_sweep.sh`,
`run_remaining.sh`, `run_opu_microbench.sh`, `run_rvv_selftest.sh`.

Dispatch classification methodology: `METHODS_dispatch_classification.md`.
Regenerate `classify_dispatches.py` outputs by dumping IREE
intermediates (`--iree-hal-dump-executable-intermediates-to`) for
each model, then running the classifier against the dump root.

## Measured speedups (FireSim V128-D64, 60 MHz)

| Model | OPU (cyc) | RVV (cyc) | Speedup |
| --- | ---: | ---: | ---: |
| ConvNet    | 183M  | 2.42B  | **13.24×** |
| TinyLlama  | 3.45B | 37.85B | **10.97×** |
| YOLOv8-n   | 1.81B | 18.80B | **10.38×** |
| ViT (v3)   | 3.14B | 28.01B | **8.91×** |
| DroNet     | 152M  | 926M   | **6.08×** |
| MLP-Fast   | 342M  | 484M   | **1.41×** |
| MLP-Wide   | —     | —      | **1.25×** (overhead-dominated) |
| Hybrid     | —     | —      | 1.08× |
| Large-MLP  | —     | —      | 0.52× (K=512 memory-bound) |

Geomean over all 8 validated pairs: **3.07×**. Geomean over models
≥1.10× (paper table): **4.91×**.

## Non-goals

This branch is a reference snapshot. Don't PR onto it. Dev/main-bound
OPU support lives on `saturn-opu/core-support`; NPU work on
`saturn-npu/core-support`. Bitstream reproduction requires the
chipyard + saturn-vectors + firesim SHAs above.
