# Numerics inputs for the full-model check

Inputs that `python -m merlin.model_check` judges a device submission on, for ResNet50, SmolVLA and
TinyLlama. Contract and numerics policy: [`docs/reference/model_check.md`](../../../docs/reference/model_check.md).

| File | What it is |
|---|---|
| [`models.yaml`](models.yaml) | Programs of each model, how they chain, what is judged |
| [`sources.yaml`](sources.yaml) | Pinned sources (revision + sha256) and the selection rule of every split |
| `inputs/<model>/visible.{npz,json}` | The 8 visible inputs per model, in their small stored form |
| `calibration/<model>.json` | Calibration item ids + sha256 (disjoint from every evaluation input) |
| `MANIFEST.json` | sha256 of every visible, sealed and derived file |

`visible.npz` holds one array per input and field under the key `<id>.<field>`; `visible.json` maps each
id to its metadata (read both with `merlin.model_check.inputs.stored`). Stored forms are small and
lossless; the model-ready arrays are derived from them by each model's own preprocessing in its pinned
environment:

- **resnet50** — array `image` (uint8): 224×224 RGB after the `IMAGENET1K_V2` resize (232, bilinear, antialias) and
  center crop; normalization is `(u8/255 − mean)/std` in float32, verified bit-identical to torchvision's
  own transform when the input is prepared. The metadata names the Imagenette member and its ImageNet label.
- **tiny_llama** — no arrays; the metadata's `tokens`: 160 token ids of a WikiText-2 article (128 prefill +
  32 teacher-forced decode), from the checkpoint's tokenizer.
- **smolvla** — arrays `frame` (uint8 raw top-camera frame) and `state` (raw state); the metadata holds the
  episode, frame, raw state, task and noise seed; LeRobot's SmolVLA processors produce the image, state
  and token tensors.

**Sealed inputs are never committed.** They are produced by the same rules into the private root
(`/work/merlin/private/<target>/numerics/stored/<model>/sealed.{npz,json}`) and appear here only as sha256 in `MANIFEST.json`, so
the set can be verified when it is released. Regenerate with:

```sh
python -m merlin.model_check inputs-prepare --model resnet50 --data <dir with the pinned sources> \
    --private /work/merlin/private/gemmini/numerics --python <resnet50 venv>/bin/python --out report.json
```
(`tiny_llama` also takes `--set tokenizer_path=... --set rows_python=<venv with pyarrow>`; `smolvla` takes
`--set policy_path=... --set vlm_path=...` and a LeRobot 0.6 environment.)
