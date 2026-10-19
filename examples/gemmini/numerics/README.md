# Numerics: inputs, models and guides for full-model correctness checks

The [shared experiment protocol](../protocol/README.md) defines the selected
32-token-prefill/32-decode TinyLlama workload, one-camera SmolVLA chain, exact
grouped-16 INT8 recipes, host math and metric semantics. The stored cohort below
and the r53b quantization guide are historical inputs, not implicit selections for
every launch. Current functional capture inputs and quality cohorts differ; bind
the selected input/reference manifests explicitly.

This folder holds what `python -m merlin.model_check` needs to judge a Gemmini device submission on
complete models (ResNet50, SmolVLA and TinyLlama): which captured programs make up each model and what is
judged, the pinned evaluation and calibration inputs, and the files that bind Spike to the RTL and the
FPGA build. It also holds guides for anyone building an int8 implementation that must match Merlin's.
Contract and numerics policy: [`docs/reference/model_check.md`](../../../docs/reference/model_check.md).

| Path | What it is | How it is used |
|---|---|---|
| [`models.yaml`](models.yaml) | Programs of each model, how they chain, what is judged and its reference admission bars, how `--mode full` builds images | Default `--models` of every subcommand, e.g. `python -m merlin.model_check check --submission S --model M --split visible --out R.json` |
| [`sources.yaml`](sources.yaml) | Pinned sources (revision + sha256) and the selection rule of every split | Read by `python -m merlin.model_check inputs-prepare` (see [Reproducibility](#reproducibility)) |
| [`binding.yaml`](binding.yaml) | Which files bind Spike to the RTL and to the FPGA build, with citations the receipt re-reads | Default `--spec` of `python -m merlin.model_check binding --out R.json` |
| [`MANIFEST.json`](MANIFEST.json) | sha256 of every visible, sealed and derived input file, and each input's selection | Written by `inputs-prepare`; `merlin/tests/infra/test_model_check_inputs.py` checks the committed files against it |
| [`calibration/`](calibration/) | `<model>.json`: calibration item ids + sha256, disjoint from every evaluation input | Written by `inputs-prepare`; the calibration items themselves stay private |
| [`inputs/`](inputs/) | `<model>/visible.npz` + `<model>/visible.json`: the 8 visible inputs per model in their stored form | Written by `inputs-prepare`; read with `merlin.model_check.inputs.stored(<this folder>, model, "visible")` |
| [`QUANTIZATION.md`](QUANTIZATION.md) | Historical r53b quantization and recovery evidence; current recipes are in [`../protocol/`](../protocol/README.md) | Historical guide |
| [`OFFLOAD_REFORMULATIONS.md`](OFFLOAD_REFORMULATIONS.md) | Analysis of moving non-matmul work onto Gemmini (exact vs approximate) | Guide |

## Stored inputs

Each model's visible inputs are two files. `visible.npz` holds one array per input and field under the
key `<id>.<field>` (e.g. `v00.image`, `v03.frame`, `v03.state`); `visible.json` maps each id to its
metadata, one line per input. A model without arrays (tiny_llama) has only the `.json`. Both files are
byte-stable, so re-running `inputs-prepare` on the pinned sources reproduces them exactly.

Stored forms are small and lossless; the model-ready arrays are derived from them by each model's own
preprocessing in its pinned environment:

- **resnet50** — array `image` (uint8): 224×224 RGB after the `IMAGENET1K_V2` resize (232, bilinear,
  antialias) and center crop; normalization is `(u8/255 − mean)/std` in float32, verified bit-identical
  to torchvision's own transform when the input is prepared. The metadata names the Imagenette member,
  its synset and ImageNet label, and the source image's sha256.
- **tiny_llama** — no arrays; this historical metadata's `tokens` are 160 token ids of a WikiText-2
  article (originally 128 prefill + 32 teacher-forced decode), from the checkpoint's tokenizer, with
  the article's split and index. The selected functional chain now uses 32+32. The new quality
  protocol defines 64-token segments and an explicit next-token score mask; it requires new bound
  manifests and a PPL evaluator rather than relabeling these old files.
- **smolvla** — arrays `frame` (uint8 raw top-camera frame) and `state` (raw state); the metadata holds
  the episode, frame index, raw state, task and noise seed. LeRobot's SmolVLA processors produce the
  image, state and token tensors.

## Reproducibility

**Sealed inputs are never committed.** They are produced by the same rules into the private root
(`/work/merlin/private/<target>/numerics/stored/<model>/sealed.{npz,json}`, in the same format) and
appear here only as sha256 in `MANIFEST.json`, so the set can be verified when it is released. The
model-ready arrays of every input stay private as well (`derived/<model>/<split>/<id>/`). Regenerate with:

```sh
python -m merlin.model_check inputs-prepare --model resnet50 --data <dir with the pinned sources> \
    --private /work/merlin/private/gemmini/numerics --python <resnet50 venv>/bin/python --out report.json
```
(`tiny_llama` also takes `--set tokenizer_path=<checkpoint snapshot> --set rows_python=<venv with pyarrow>`;
`smolvla` takes `--set policy_path=... --set vlm_path=...` and a LeRobot 0.6 environment.) The public
files are written beside `--sources` (default: this folder's `sources.yaml`), and `MANIFEST.json` is
updated for the model.
