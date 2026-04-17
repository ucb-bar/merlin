# SmolVLA: Export → Compile → Kernels → Run

This is the single entry point for everything SmolVLA in Merlin. If you are
integrating with our quantization scheme, porting kernels, or reproducing a
benchmark, start here. Each section gives the concrete command or pin, plus
a link to the deep-dive doc if you need more.

## Current pins

| Thing | Value | Where it is defined |
|---|---|---|
| HuggingFace model | `lerobot/smolvla_base` | default `--model-id` in [`models/smolVLA/export_smolvla.py`](https://github.com/ucb-bar/merlin/blob/dev/main/models/smolVLA/export_smolvla.py) line 74 |
| Understanding-PI0 fork | branch `mlir-smolvla` | `.gitmodules` → `third_party/Understanding-PI0` (ucb-bar fork) |
| Understanding-PI0 commit | submodule SHA | `git -C third_party/Understanding-PI0 rev-parse HEAD` |
| AutoComp fork | `third_party/autocomp` | `.gitmodules` → `third_party/autocomp` |
| AutoComp commit | submodule SHA | `git -C third_party/autocomp rev-parse HEAD` |
| Kernel catalog | 21 atomic/fused kernels | [`benchmarks/SaturnNPU/kernels/`](https://github.com/ucb-bar/merlin/tree/dev/main/benchmarks/SaturnNPU/kernels) |

To get the **exact** SHAs for your Merlin checkout, run from the repo root:

```bash
git submodule status third_party/Understanding-PI0 third_party/autocomp
```

Submodule SHAs move with the Merlin branch. Always report them alongside the
Merlin commit you are on (`git rev-parse HEAD`).

## 1. Export: SmolVLA → MLIR

Produces `models/smolVLA/smolVLA*.mlir`. Three variants: `fp32`, `int8`, `fp8`.

```bash
./merlin setup submodules --submodules-profile core --submodule-sync
# Then from repo root:
uv run models/smolVLA/export_smolvla.py --mode all --device cuda
```

Details (flags, quantization mode selection, `Understanding-PI0` venv override):
[`models/smolVLA/README.md`](../../models/smolVLA/README.md).

The exporter wraps `third_party/Understanding-PI0/understanding_pi0/smolvla_mx/`
— that's where the SmolVLA-specific PyTorch loading, calibration hooks, and
quantization scheme live. If you are replacing our data-sampling step with
your own, that's the folder you want to read first.

## 2. Compile: MLIR → target artifact

```bash
./merlin compile models/smolVLA/smolVLA.q.fp8.mlir \
  --target npu_ucb --quantized \
  --compile-to global-optimization --dump-phases
```

Targets: `npu_ucb` (Saturn NPU), `gemmini_mx`, `saturn_opu`, `spacemit_x60`.
The `--target <X>` switch picks a compile profile from `models/<X>.yaml`.

How the compiler lowers SmolVLA (dialect layers, op coverage, known gaps):
[`docs/architecture/npu_compilation_pipeline.md`](../architecture/npu_compilation_pipeline.md).
What's inside the NPU dialect plugin itself:
[`compiler/src/merlin/Dialect/NPU/README.md`](https://github.com/ucb-bar/merlin/blob/dev/main/compiler/src/merlin/Dialect/NPU/README.md).

## 3. Kernels: what runs on the NPU

The 21 kernels that back a full SmolVLA forward pass live under
[`benchmarks/SaturnNPU/kernels/`](https://github.com/ucb-bar/merlin/tree/dev/main/benchmarks/SaturnNPU/kernels).
Each kernel directory has variants (shape/dtype combinations) and an ISA body.

- **Catalog and variant counts**: [`benchmarks/SaturnNPU/kernels/README.md`](https://github.com/ucb-bar/merlin/blob/dev/main/benchmarks/SaturnNPU/kernels/README.md).
- **Golden-data workflow** (how kernels are validated against PyTorch refs, how
  to add new shapes, how the harness wires into the simulator):
  [`benchmarks/SaturnNPU/golden_data/KERNEL_GUIDE.md`](https://github.com/ucb-bar/merlin/blob/dev/main/benchmarks/SaturnNPU/golden_data/KERNEL_GUIDE.md).
- **Layer → kernel decomposition** (which SmolVLA layer lowers into which
  kernel, and why those fusions exist):
  [`benchmarks/SaturnNPU/LAYER_DECOMPOSITION_TRACE.md`](https://github.com/ucb-bar/merlin/blob/dev/main/benchmarks/SaturnNPU/LAYER_DECOMPOSITION_TRACE.md).

The kernels themselves were hand-derived with AutoComp
(`third_party/autocomp`) against the fp8/int8 schemes from the export step.
If you change the quantization scheme, expect the fp8 matmul kernels and the
fused norm/scale kernels to need regeneration.

## 4. Run and verify

- **Kernel-level golden tests**: see the workflow in `KERNEL_GUIDE.md` above.
- **Full-graph benchmarking on the Saturn NPU simulator**:
  [`benchmarks/SaturnNPU/README.md`](https://github.com/ucb-bar/merlin/blob/dev/main/benchmarks/SaturnNPU/README.md).
- **Graph manifest** (exact op/layer/kernel call order used by the benchmarks):
  `benchmarks/SaturnNPU/smolvla_graph_manifest.json`.

## Deep dive / history

For the debugging arc that produced the current compile + kernel story —
precision tuning, dispatch-async runtime, narrow-M matmul fixes, AutoComp
kernel derivation — read the workstream log
[`docs/dev_blog/2026-03-12-smolvla-fp8-int8-global-opt-workstream.md`](../dev_blog/2026-03-12-smolvla-fp8-int8-global-opt-workstream.md).
The top of that entry has an "At a glance" TOC that splits it into three
sub-workstreams; you usually don't need to read it linearly.

## Common questions

**Q: What version of SmolVLA are you running against?**
A: HF model `lerobot/smolvla_base` (see `--model-id` default in
`export_smolvla.py`). The weights travel with the HF hub; pin your HF
`revision` if you need byte-for-byte reproducibility with a specific Merlin
commit.

**Q: What Understanding-PI0 branch / commit?**
A: Branch `mlir-smolvla` on `ucb-bar/Understanding-PI0`. Exact SHA is the
submodule pin — run `git submodule status third_party/Understanding-PI0`.

**Q: Where does our quantization scheme differ from stock?**
A: See `third_party/Understanding-PI0/understanding_pi0/smolvla_mx/` — that
module implements the MX-style fp8 calibration + int8 fallback paths. The
exported `.q.fp8.mlir` files encode the chosen per-op precisions.

**Q: Are the Autocomp kernels regenerable?**
A: Yes — the harnesses under `third_party/autocomp/` drive the generation.
If you change the quant scheme, `quantized_matmul_fp8/`, `fused_matmul_bias/`,
and `fused_norm_scale/` are the most likely kernels to need re-derivation.
