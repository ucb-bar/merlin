# Saturn OPU Figure Methods

This document describes how the Saturn OPU benchmark figures and CSV metrics
are generated, what each metric means, and what the current limitations are.
The scripts are intentionally compile-artifact driven: we do not infer OPU
coverage from model names or from the requested compile mode alone.

## Entry Point

Regenerate all model data and paper figures with:

```bash
bash benchmarks/SaturnOPU/make_paper_figures.sh --refresh
```

The script runs three steps:

1. `analyze_model_paths.py`
   - Compiles or reuses model artifacts.
   - Reads per-dispatch MLIR, linked LLVM IR, and final assembly.
   - Emits dispatch-level, layer-level, model-level, and opcode-summary CSVs.
2. `plot_optimization_journey.py`
   - Reads `optimization_journey.csv`.
   - Emits `figures/optimization_journey.{pdf,png}`.
3. `plot_model_decomposition.py`
   - Reads `model_dispatch_decomposition.csv` and `per_model_summary.csv`.
   - Emits `figures/per_model_decomposition.{pdf,png}`.

The workflow uses the Merlin command-line entry point and the in-tree compiler
build. It does not call Python IREE compilation APIs:

```bash
conda run -n merlin-dev uv run tools/merlin.py compile <model.mlir> \
  --target saturn_opu \
  --hw <OPU mode> \
  --dump-artifacts \
  --dump-phases \
  --dump-graph \
  --output-dir build/compiled_models/<model>/<target_hw_model> \
  --build-dir host-merlin-release
```

The compiler is therefore the one under:

```text
build/host-merlin-release
```

That matters because this build contains the Saturn OPU compiler and runtime
changes used by the benchmark.

## Registered Models

`analyze_model_paths.py` owns the model registry. Each row specifies the input
MLIR, target, hardware mode, and output artifact directory.

| Key | Plot label | Input MLIR | Target | HW mode |
| --- | --- | --- | --- | --- |
| `mlp` | MLP | `models/mlp/mlp.q.int8.mlir` | `saturn_opu` | `OPU` |
| `mlp_wide` | MLP-Wide | `models/mlp_wide/mlp_wide.q.int8.mlir` | `saturn_opu` | `OPU` |
| `opu_bench_large_mlp` | Large MLP | `models/opu_bench_suite/opu_bench_large_mlp.q.int8.mlir` | `saturn_opu` | `OPU` |
| `opu_bench_vit_small` | ViT-Small | `models/opu_bench_suite/opu_bench_vit_small.q.int8.mlir` | `saturn_opu` | `OPU` |
| `opu_bench_vit` | ViT | `models/opu_bench_suite/opu_bench_vit.q.int8.mlir` | `saturn_opu` | `OPU_LLM` |
| `opu_bench_hybrid` | Hybrid CNN+Trf | `models/opu_bench_suite/opu_bench_hybrid.q.int8.mlir` | `saturn_opu` | `OPU_LLM` |
| `opu_bench_convnet` | ConvNet | `models/opu_bench_suite/opu_bench_convnet.q.int8.mlir` | `saturn_opu` | `OPU_IM2COL` |
| `dronet` | DroNet | `models/dronet/dronet.q.int8.mlir` | `saturn_opu` | `OPU_IM2COL` |
| `yolov8_nano` | YOLOv8-n | `build/compiled_models/yolov8_nano/saturn_opu_OPU_IM2COL_yolov8n.q.int8/yolov8n.q.int8.mlir` | `saturn_opu` | `OPU_IM2COL` |
| `tinyllama` | TinyLlama | `build/compiled_models/tinyllama/saturn_opu_OPU_LLM_tinyllama.q.int8/tinyllama.q.int8.mlir` | `saturn_opu` | `OPU_LLM` |

YOLOv8-n and TinyLlama reuse already imported MLIR in `build/compiled_models`
because their source MLIR is produced by earlier model import/export steps.
They are still compiled by `tools/merlin.py compile` for this analysis.

Missing models are hard failures. The analyzer does not silently skip a model
because that would make the paper metrics non-representative.

## Artifact Completeness

For each model, artifacts are considered complete only if the output directory
contains:

- `sources/*.mlir`
- `files/*.codegen.ll`
- `files/*.s`
- `phases/*.1.input.mlir`

If `--refresh` is passed, the model is recompiled even when those artifacts
exist. Without `--refresh`, complete artifacts are reused.

The important artifact directories are:

- `sources/`: one MLIR file per dispatch executable.
- `configs/`: configured dispatch MLIR.
- `phases/`: whole-module phase dumps, including the early input phase.
- `files/*.codegen.ll`: linked LLVM IR after ukernel lowering.
- `files/*.s`: final RISC-V assembly.

## Dispatch Inclusion

The unit of analysis is a compiled dispatch. The analyzer reads every
`sources/*.mlir` file and extracts the `hal.executable.export` symbol.

Only symbols matching the main model async-dispatch convention are counted as
model dispatches:

```text
...$async_dispatch_<id>_<operation_name>
```

Initializer and auxiliary encoding dispatches are kept in
`model_dispatch_decomposition.csv` for auditability, but they are marked:

```text
include_in_model = 0
```

They do not contribute to model totals, OPU compute share, or dispatch share.

## Operation Classification

Each included dispatch is assigned an `op_kind` and an analytical operation
count. The count is a MAC-equivalent static estimate, not a measured cycle or
runtime instruction count.

The rules are:

| Dispatch pattern | `op_kind` | Analytical ops |
| --- | --- | --- |
| `batch_matmul_BxMxNxK` | `batch_matmul` | `2 * B * M * N * K` |
| `matmul_MxNxK` or `matmul_like_MxNxK` | `matmul` | `2 * M * N * K` |
| `matvec_NxK` | `matvec` | `2 * N * K` |
| `softmax_<shape>` | `softmax` | `5 * outer_elements * reduction_size` |
| `reduction_<shape>` | `reduction` | `(reduction_size - 1) * outer_elements` |
| `elementwise_<shape>` | `elementwise` | element count |
| `slow_memcpy`, `memcpy`, explicit encodes | `data_movement` | largest tensor element count |
| `linalg.conv*` | `conv` | `2 * output_elements * kernel_area` when shapes are recoverable |
| generic reduction-like body | `reduction` | largest input elements minus output elements |
| generic im2col/pack-like body | `data_movement` | largest tensor element count |
| other generic body | `elementwise` | largest tensor element count |

For small per-dispatch MLIR files, the analyzer attempts to parse MLIR through
`iree.compiler.ir` so it can inspect operation names such as `linalg.softmax`.
If bindings are unavailable or parsing fails, the analyzer falls back to
structured text matching. The MLIR binding parse is used for exploration and
classification; model compilation itself always uses `tools/merlin.py compile`.

## OPU Path Classification

For each dispatch, the analyzer locates the corresponding function in:

- linked LLVM IR from `files/*.codegen.ll`
- final assembly from `files/*.s`

It then combines ukernel-call evidence from LLVM IR with opcode evidence from
assembly.

### Assembly Evidence

The current OPU opcode counters are:

| Counter | Assembly pattern |
| --- | --- |
| `vopacc` | `.insn r 87, 2, 81` |
| `opmvinbcast` | `.insn r 87, 6, 89` |
| `opu_fetch` | `.insn r 87, 6, 93` |

The analyzer also counts RVV reduction and gather evidence:

- `vred*.vs` / `vfred*.vs`
- `vfsqrt.v`
- `vrgather*`
- `vfslide1down.vf`

### Path Rules

The dispatch path is assigned using these rules, in order:

| Evidence | `opu_path` | Meaning |
| --- | --- | --- |
| `iree_uk_opu_matmul_qdq` call and VOPACC assembly | `fused_qdq` | Fused OPU matmul plus quant/dequant/requant path |
| `iree_uk_opu_matmul` call and VOPACC assembly | `encoding_resolver` | OPU encoding resolver path, direct 2D output |
| `iree_uk_mmt4d` call and VOPACC assembly | `runtime_mmt4d_opu` | Runtime mmt4d path that enters the XOPU early handler |
| VOPACC assembly without a recognized ukernel call | `inline_vopacc` | Inline or custom VOPACC lowering |
| matmul-like op with no VOPACC | `rvv_matmul` | Matmul did not use OPU opcodes |
| conv op with no VOPACC | `direct_conv` | Direct convolution path |
| reduction or softmax op | `rvv_reduction_softmax_norm` | Reduction, softmax, or norm-like RVV path |
| data movement op | `data_movement` | Pack, copy, im2col, encode, or related movement |
| fallback | `elementwise_other` | Elementwise/scalar/other non-OPU compute |

`opu_ops` equals `ops` only for:

- `encoding_resolver`
- `runtime_mmt4d_opu`
- `fused_qdq`
- `inline_vopacc`

All other paths contribute zero OPU ops.

## Tile and Tiling-Tier Classification

For recognized ukernel calls, the analyzer extracts the ukernel tile arguments
from LLVM IR. For calls of the form:

```text
call i32 @iree_uk_opu_matmul(..., i32 M0, i32 N0, i32 K0, i32 flags, ...)
call i32 @iree_uk_mmt4d(..., i32 M0, i32 N0, i32 K0, i32 flags, ...)
```

the last four `i32` constants are interpreted as:

```text
M0, N0, K0, flags
```

The first three are written to:

- `tile_m`
- `tile_n`
- `tile_k`

The plot segment further splits OPU compute into tile classes:

| Base path | Tile rule | Segment |
| --- | --- | --- |
| encoding resolver | `M0 >= 16` and `N0 >= 16` | `encoding_16x16_tile` |
| encoding resolver | model M/N is smaller than tile M/N, or `M0 == 1`, or `N0 == 1` | `encoding_narrow_tile` |
| encoding resolver | other tile | `encoding_other_tile` |
| runtime mmt4d OPU | `M0 >= 16` and `N0 >= 16` | `runtime_16x16_tile` |
| runtime mmt4d OPU | `M0 == 8` and `N0 == 8` | `runtime_8x8_tile` |
| runtime mmt4d OPU | model M/N is smaller than tile M/N, or `M0 == 1`, or `N0 == 1` | `runtime_narrow_tile` |
| runtime mmt4d OPU | other tile | `runtime_other_tile` |
| fused QDQ | any | `fused_qdq` |
| inline VOPACC | any | `inline_vopacc` |

The current compiled model artifacts do not show 32x32 or 64x64 selected as
ukernel tile arguments. The runtime has code paths that can sub-tile larger
tiles, and the query-tile-size hook has XOPU entries for larger tiles, but the
active encoding resolver used by these model compiles currently chooses the
16x16-centered tile family. The figure therefore does not include a fake
32x32 or 64x64 bucket.

## Output CSVs

### `model_dispatch_decomposition.csv`

One row per dispatch artifact. Important columns:

| Column | Meaning |
| --- | --- |
| `model_key`, `model` | Internal key and plot label |
| `idx`, `symbol` | Async dispatch id and exported symbol |
| `source_file` | Per-dispatch source MLIR used for classification |
| `include_in_model` | `1` for model dispatches, `0` for auxiliary/init dispatches |
| `layer_id` | Synthetic grouping label for layer-level summaries |
| `op_kind` | Static operation class |
| `segment` | Final plot segment after OPU path and tile classification |
| `opu_path` | Base OPU/non-OPU path before tile splitting |
| `ops` | Static analytical ops for this dispatch |
| `opu_ops` | `ops` if the dispatch uses a confirmed OPU path, otherwise `0` |
| `compute_pct` | Dispatch `ops / total model ops` |
| `shape`, `M`, `N`, `K`, `B` | Parsed dimensions when recoverable |
| `vopacc`, `opmvinbcast`, `opu_fetch` | OPU opcode counts from assembly |
| `rvv_reduction`, `rvv_sqrt`, `rvv_gather` | RVV evidence counts |
| `tile_m`, `tile_n`, `tile_k` | Extracted ukernel tile shape |
| `tiling_tier` | Tile-class label such as `16x16x1` or `narrow` |
| `evidence` | Semicolon-separated audit string |

### `model_layer_decomposition.csv`

Groups included dispatch rows by:

```text
model_key, model, layer_id, op_kind, segment
```

This file is useful for finding which dispatch/layer families contribute most
to OPU or non-OPU compute.

### `per_model_summary.csv`

One row per plotted model. Segment columns contain analytical op totals by
segment. The summary also includes:

| Column | Meaning |
| --- | --- |
| `total_ops` | Sum of analytical ops for included dispatches |
| `opu_ops` | Sum of analytical ops for OPU segments |
| `opu_pct` | `100 * opu_ops / total_ops` |
| `dispatches` | Count of included model dispatches |
| `opu_dispatches` | Count of included dispatches with `opu_ops > 0` |
| `opu_dispatch_pct` | `100 * opu_dispatches / dispatches` |
| `artifact_dir` | Compile artifact directory used for the row |

### `opu_path_opcode_summary.csv`

Groups dispatch rows by:

```text
model_key, model, segment
```

It sums dispatch counts, analytical ops, and opcode evidence. This is the
quickest audit file for checking whether a plotted OPU segment actually had
VOPACC assembly evidence.

## Per-Model Decomposition Figure

Generated files:

- `figures/per_model_decomposition.pdf`
- `figures/per_model_decomposition.png`

The normalized bar is a **dispatch-share** bar:

```text
segment_width = dispatches_in_segment / total_included_dispatches
```

This was chosen because a compute-weighted bar made models such as TinyLlama
look almost fully green even though only a minority of dispatches use OPU. For
TinyLlama, the plot should communicate both facts:

- only `135/698` dispatches use OPU paths
- those dispatches account for `99.96%` of analytical compute

The right-side annotation reports:

```text
<OPU analytical compute share> compute
<OPU dispatches>/<all included dispatches> dispatches
```

The annotation is intentionally separate from the bar. A large green compute
share does not mean every dispatch or every layer is on OPU.

### Colors and Opacity

Colors are defined in `palette.py`.

| Visual encoding | Meaning |
| --- | --- |
| Blue OPU family | Encoding resolver path |
| Green OPU family | Runtime mmt4d OPU path |
| Purple | Fused QDQ OPU path |
| Yellow | Inline VOPACC path |
| Red | RVV matmul |
| Orange | Direct convolution |
| Gray | Reduction, softmax, norm, elementwise, scalar |
| Pale blue-gray | Pack, copy, im2col, data movement |

Within a single OPU color family, opacity indicates tile quality:

| Opacity | Meaning |
| --- | --- |
| Full | 16x16 OPU tile class |
| Medium | 8x8 or other smaller tile class |
| Low | Narrow/vector-style tile class |

The opacity is not a measured efficiency number. It is a visual tier derived
from the selected ukernel tile shape.

## Optimization Journey Figure

Generated files:

- `figures/optimization_journey.pdf`
- `figures/optimization_journey.png`

This figure is built directly from `optimization_journey.csv`, which records
single-kernel ops/cycle measurements for a `1024x1024x1024` i8 workload.

Current plotted rows:

| Step | Meaning |
| --- | --- |
| RVV Baseline | Standard RVV vectorized matmul without ukernels or OPU |
| Fixed Tile Query | Correct 16x16x1 tile selection for OPU in query_tile_sizes |
| K-loop Unroll x4 | mmt4d K-loop unrolled by four with register-pair rotation |
| Const Weights | Compile-time packed RHS via global-opt data tiling |
| Encoding Resolver | Direct OPU path with identity result encoding and no unpack dispatch |

The earlier `Initial mmt4d+OPU` row was removed from the plotted journey
because it described a broken/intermediate state and did not help readers
understand the final method.

## Interpretation Guidelines

Use `opu_pct` when discussing where the model's analytical compute goes. This
is the answer to: "How much of the static compute is in dispatches that use
confirmed OPU evidence?"

Use `opu_dispatches / dispatches` when discussing coverage across the compiled
program. This is the answer to: "How many dispatches actually use an OPU path?"

Use `segment` and `tile_m/tile_n/tile_k` when discussing which OPU path was
taken and how it was tiled.

Use `opu_path_opcode_summary.csv` when checking that a path has actual assembly
evidence.

Do not interpret these metrics as:

- runtime speedup
- cycle share
- memory bandwidth share
- whole-layer semantic coverage from the original framework graph
- proof that every dispatch in a model uses OPU

## Useful Audit Commands

List all selected OPU ukernel tile shapes:

```bash
conda run -n merlin-dev uv run python - <<'PY'
import csv
from collections import Counter

counts = Counter()
with open("benchmarks/SaturnOPU/model_dispatch_decomposition.csv") as f:
    for row in csv.DictReader(f):
        if row["include_in_model"] == "1" and int(row["opu_ops"]) > 0:
            counts[(row["tile_m"], row["tile_n"], row["tile_k"])] += 1

for tile, count in counts.most_common():
    print(tile, count)
PY
```

Show the top non-OPU dispatches by model:

```bash
conda run -n merlin-dev uv run python - <<'PY'
import csv
from collections import defaultdict

rows = []
with open("benchmarks/SaturnOPU/model_dispatch_decomposition.csv") as f:
    for row in csv.DictReader(f):
        if row["include_in_model"] == "1" and int(row["opu_ops"]) == 0:
            rows.append(row)

by_model = defaultdict(list)
for row in rows:
    by_model[row["model"]].append(row)

for model, model_rows in sorted(by_model.items()):
    print(model)
    for row in sorted(model_rows, key=lambda r: int(r["ops"]), reverse=True)[:5]:
        print(" ", row["idx"], row["segment"], row["op_kind"], row["ops"], row["symbol"])
PY
```

Check path-level opcode evidence:

```bash
column -s, -t benchmarks/SaturnOPU/opu_path_opcode_summary.csv | less -S
```

## Known Limitations

The analytical op model is static and approximate. It is intended for
comparative decomposition, not for cycle prediction.

Non-matmul operations are estimated using simple tensor-shape heuristics.
Those estimates are good enough to avoid calling everything "free", but they
are less exact than the matmul formulas.

Layer ids are synthetic dispatch group labels, not original framework layer
names. They are stable enough for debugging dispatch families, but they should
not be quoted as semantic model-layer names without additional graph mapping.

The current figure uses dispatch share for the bar and compute share in the
annotation. This is deliberate: dispatch share prevents misleading full-green
bars, while compute share explains why a small number of OPU dispatches can
still dominate runtime-relevant work.

The current compiled artifacts do not select 32x32 or 64x64 OPU ukernel tiles.
If a future compile mode enables those tiles, the analyzer will record them via
`tile_m/tile_n/tile_k`, but the palette may need new visible buckets.
