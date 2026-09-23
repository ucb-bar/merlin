# AGENT.md — merlin/experiments/dataset_accuracy

Status: active — SST-2 (BERT-base) measured; ImageNet blocked on dataset access; MobileBERT-TINY
blocked on an unfetched git-lfs checkpoint (see the Status section).

## Purpose

Dataset-level accuracy for the Voyager head-to-head (plan workstream G3b). The first milestone
reproduces Voyager's Table 3 (arXiv 2509.15205, DAC 2026) with **Voyager's own public quantization
code**. It covers the ImageNet cells (ResNet-18/50) and the SST-2 cells (BERT-base, MobileBERT-TINY),
at FP32 / BF16 / E4M3 / Posit8 / INT8 / MXINT8. Quantization accuracy depends on the algorithm, not
the compiler, so "parity" means matching the paper's cells within **±0.3 top-1 points** before
anything is claimed. Later milestones evaluate on-device accuracy, which the paper does not do.

## What is here

- `paper_table3.yaml` — Table 3 transcribed from the arXiv e-print LaTeX (`6_results.tex`), all rows.
  It also carries the accelerator regression's own gold table (see below).
- `scripts/voyager_accuracy_eval.py` — the worker. It runs in `out/build/voyager-venv`, imports nothing
  from merlin, and uses Voyager's own helpers by import. Its modes:
  - `index`: ImageNet stream order;
  - `quantized`: ImageNet, batched;
  - `glue`: SST-2;
  - `test_codegen`: Voyager's `test/test_codegen.py --evaluate` run as-is.
- `scripts/run_table3.py` — the driver (`.venv/bin/python`). It runs the worker, verifies the
  `voyager_compiler` (and, for MobileBERT-TINY, `voyager_accelerator`) pins, and writes the product.
  Every cell is compared to the paper (±0.3) and to the accelerator gold (its own ±1.0).

## Which Voyager path made Table 3 (evidence)

The paper says only "using the bit-accurate C++ model ... inference on entire datasets". The pinned
checkouts show the flow. The accelerator's `run_regression.py::run_accuracy` (voyager-accelerator
e3a725db) runs `test_codegen.py --dump_dataset` from the paired compiler (cac504ef), then its C++
`AccuracyTester` over the dump. It passes when within ±1.0 of its `ACCURACY_RESULTS` gold table. That
means:

- **ImageNet eval set = the first 1000 images of the `timm/imagenet-1k-wds` validation stream.** Both
  compiler revisions dump `imagenet.retrieve_dataset(1000, "resnet")`. One image is 0.1 point, so the
  ±0.3 band is ±3 images, and parity is meaningful only on *those* images. A different 1000-image draw
  has ~1.4 points of standard error. Full-50k cells are reported with Wilson 95% intervals, never as
  the parity cell.
- **SST-2 eval set = the full validation split (872)**, so the paper protocol is a full-set number.
- **Weights:** ResNet-18 `IMAGENET1K_V1`, ResNet-50 `IMAGENET1K_V2` (named in `run_accuracy`; the same
  as torchvision `DEFAULT`), BERT `JeremiahZ/bert-base-uncased-sst2`, and MobileBERT-TINY from
  `models/mobilebert/mobilebert-tiny-sst2-bf16/` in the accelerator repo. `examples/imagenet/main.py`
  (V1 weights, train-set calibration) is a different path; its ResNet-50 fp32 is 76.1, not 80.4.
- **Preprocessing:** `get_transforms("resnet")` — Resize 256, CenterCrop 224. torchvision evaluates
  ResNet-50 V2 at resize **232**, so Voyager's FP32 cell is not torchvision's 80.858.
- **Recipe (`--recipe accelerator`, default)** comes from `run_accuracy`'s flags:
  - INT8 is per-tensor symmetric, int24 bias, calibrated on **10** samples at batch 1 (the first 10 of
    the eval stream for ImageNet; the first 10 train rows for SST-2);
  - MXINT8 is `bs = max(IC, OC)` with power-of-two scales;
  - E4M3 and Posit8 are direct casts under `--bf16`.

  `--recipe ci` uses `test/run_ci.py` instead (INT8 calibrated on 3).
- **Two references that disagree.** The accelerator gold differs from Table 3 by up to 2.0 points
  (ResNet-18 INT8 69.5 vs 71.5), so the numbers moved between revisions of the same flow.

## Arms (what the harness runs)

Neither arm is the C++ AccuracyTester, which needs the accelerator's C++ build. That is the next
milestone.

1. **Quantized graph** (`quantized` / `glue`). Voyager's own quantize-and-evaluate helpers run up to
   `convert_pt2e`. For BERT they are called as-is with only `transform`/`compile` stubbed, and are
   evaluated by Voyager's own `evaluate`/`evaluate_gm` loops. ImageNet mirrors the pre-compile half
   of `torchvision_models.quantize_and_dump_model` with a dynamic batch dimension, so 50k images can
   be batched. `--check-static` measured the difference against upstream's batch-1 graph as 0.0 max
   abs, with every top-1 identical.
2. **As-is** (`--literal`, `test_codegen` mode). Voyager's `test_codegen.py` `main()` runs via
   `runpy` with the `run_ci` compile argv plus `--evaluate --dump_dataset`, evaluating the *lowered*
   graph.

## Upstream defects found (pinned compiler f9d4c498 in `voyager-venv`), each worked around minimally

| Defect | Where | Harness treatment |
|---|---|---|
| `evaluate()` passes fp32 images to a `--bf16` ResNet and raises | `test/utils/models/torchvision_models.py` | cast inputs to the model dtype for that loop only; recorded as `input_cast` |
| `dump_dataset` builds an fp32 attention mask for a graph traced on a bf16 one; `evaluate_gm` raises | `test/utils/dataset/glue.py` + `models/{bert,mobilebert}.py` | cast the dumped floats to bf16 (the values the traced path produces); recorded |
| `load_dataset("glue", ...)` no longer parses under huggingface_hub 1.x | `test/utils/dataset/glue.py` | loaded as `nyu-mll/glue`, the repo the hub redirects `glue` to (same commit) |
| `test_codegen` sets 32 threads | `test/test_codegen.py` | capped to `--threads` |

## Invariants

- Voyager code is **imported from the pinned checkouts, never copied or patched on disk**. The
  transcribed accelerator recipe and the mirrored torchvision function are marked in the worker
  docstring.
- Recorded: weights (torchvision enum plus file sha256; HF checkpoint revision), preprocessing,
  calibration keys/rows, library versions, dataset revision (ImageNet shard sha256s).
- ImageNet: the `timm/imagenet-1k-wds` **validation split only** (64 shards, 7.0 GB), in
  `out/artifacts/cache/imagenet-1k-wds/` (purgeable). `--fetch` refuses when it would leave under
  10 GB free. Hugging Face files (SST-2, BERT) go to `out/artifacts/cache/huggingface/`.
- The HF token goes from `.env` into the download subprocess's environment only. It is never in
  argv, logs or artifacts. Workers run with the hub offline.
- Thread budget: `--threads` (default 12) plus `--workers` (default 3) loaders.
- On this host (Zen3, AVX2, no native bf16), bf16 conv runs at ~14 GFLOP/s against fp32's 390
  (oneDNN). The bf16-based ImageNet columns run ~3.5 img/s for ResNet-18 at 12 threads, so a full
  50k pass is ~4 h per column per model. Use `--eval stratified:<N>:<seed>`. channels_last is 4×
  faster but a different kernel, so it is not used without an agreement check. bf16 linear (BERT) is
  only 2× slower than fp32.
- Posit8 is not blocked by the unfetched compiler LFS files: `quantization/dtypes/posit_gold/*.txt`
  are posit16 *softmax* tables, and Posit8 is computed by `dtypes/posit.py`.
- Products: `out/artifacts/dataset-accuracy/<imagenet-1k|glue-sst2>/v1/...` via `new_product`, with
  `provenance.record()` naming the pins. Smoke runs (`--synthetic`) go to a scratch dir and are never
  compared to the paper.

## Run

```bash
.venv/bin/python merlin/experiments/dataset_accuracy/scripts/run_table3.py --task sst2 \
    --columns FP32,BF16,E4M3,Posit8,INT8,MXINT8
.venv/bin/python merlin/experiments/dataset_accuracy/scripts/run_table3.py --task imagenet \
    --fetch --literal --eval stratified:10000:0
```

## Status

- **ImageNet: blocked on access.** The HF account behind the approved token is not on the authorized
  list of `timm/imagenet-1k-wds` or `ILSVRC/imagenet-1k` (file downloads return `GatedRepoError 403`;
  metadata is readable). The account owner must accept the terms on those dataset pages. No mirror or
  re-host is used. The harness is smoke-tested end-to-end on synthetic JPEGs.
- **MobileBERT-TINY SST-2: blocked.** Its checkpoint in the accelerator checkout is an unfetched
  git-lfs pointer (`models/mobilebert/mobilebert-tiny-sst2-bf16/model.safetensors`, 33 MB object), and
  this host has no git-lfs.
- **BERT-base SST-2 is measured** on the full 872-sentence validation split, with the accelerator
  recipe and the quantized-graph arm (product `dataset-accuracy_glue-sst2_v1_20260914T221557Z_ada9e07`).
  Four of six cells are within ±0.3 of the paper; E4M3 and INT8 are not.

  | Column | Paper | Accel gold | Ours | Δ paper | ±0.3 |
  |---|---|---|---|---|---|
  | FP32 | 93.2 | 92.89 | 93.23 (813/872) | +0.03 | yes |
  | BF16 | 93.0 | - | 93.23 (813/872) | +0.23 | yes |
  | E4M3 | 93.1 | 93.0 | 92.66 (808/872) | -0.44 | no (inside gold ±1.0) |
  | Posit8 | 92.8 | 92.78 | 92.89 (810/872) | +0.09 | yes |
  | INT8 | 92.4 | 92.32 | 91.28 (796/872) | -1.12 | no (outside gold ±1.0 too) |
  | MXINT8 | 93.1 | 93.0 | 93.12 (812/872) | +0.02 | yes |

  INT8 moves with calibration alone. With the `ci` recipe (3 calibration rows instead of 10) it is
  91.86 (801/872, Δ −0.54; product `..._20260914T231225Z_4c56845`). Per-tensor scales from a few
  rows are the unstable part, and upstream calibration reuses the first row's attention mask for
  every step. The untested remaining explanations are the compiler revision (paper: paired cac504ef;
  here: f9d4c498) and the C++ AccuracyTester's lowered numerics against this pre-transform graph.
- **Harness smoke (ImageNet, synthetic JPEGs):** every mode runs end-to-end, including the as-is
  `test_codegen` for INT8 and MXINT8 (~7 min each for ResNet-18 at 200 images, including compile).
  On noise inputs, the lowered INT8 graph agrees with the batched pre-transform graph on 191/200
  top-1, and the as-is batch-1 bf16 model agrees with the batched BF16 on 197/200. Near-tie logits
  inflate that, but batched eager bf16 is not bit-identical to batch 1. So the ImageNet run reports
  the paper-subset cells from the as-is arm, with the agreement rate beside them.
- **Provenance caveat:** the `voyager_compiler` / `voyager_accelerator` pins do not declare
  `content_check`, so `provenance.record()` cites them as "read set NOT verified by content". Turning
  that on is a registry change to `merlin/contract/hardware_pins.yaml`.
