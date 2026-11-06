# TorchAO W8A8 saved capture delivery

These are **saved, materialized captures** for compiling the source graph. A
receipt verifies the listed files' bytes. It does not prove that the model
loader, checkpoint, Python environment, or framework execution can be rebuilt
from the receipt. Every source receipt states `source_closure_verified: false`.
The independent `golden.npy` files are TorchAO W8A8 framework references.
Current strict Merlin host comparisons against those references fail, so these
bundles are **compiler inputs**, not numerical or hardware acceptance evidence.

## Capture identities and scope

Each portable source archive must contain `capture_receipt.json` plus **exactly
the files named in its `artifacts` map**. This includes `model.mlir`,
`weights.safetensors`, `weights.safetensors.manifest.json`, `inputs.npz`,
`input_order.json`, `golden.npy`, and `extra.npz`. TinyLlama additionally lists
`meta.json` and `frontend-trace.json`. Use the receipt map as the final
membership authority; do not add ambient diagnostics such as
`model.generic.mlir` or `_print_generic.py` from the capture directory.

| Source capture | Materialized artifacts | Typed signed i8×i8→i32 roots | Source MLIR SHA256 | Source receipt SHA256 |
| --- | ---: | ---: | --- | --- |
| `spectformer_torchao_w8a8_20261106` | 22,056,551 bytes | 41 | `9ae3f772ed2d192899cea502e5dec0098d5abf95f56f1707c7ba3cdfcf5481ab` | `bd554a8cba6caf1b9c9c133073b7a839d3e596b70b2304d99c66aa03cd31038d` |
| `tinyllama_l2_s8_true_w8a8_int8_full` | 578,538,682 bytes | 15 | `930620b4d4b5749cc718736b6708595ab8bb06c4a44cfa37e86538fbd8020071` | `75634410bbfda3a89b7280586cd0cc49614acfd2ad5c17b9ff1582e2773683a3` |
| `tinyllama_full22_s8_true_w8a8_int8_w8a8_full` | 2,404,778,870 bytes | 155 | `0cb6ffc2bc89c5beedfdedd4afa321dd51eed81dc926b1376d2c1f82bee08428` | `f8e4d1652d606492956504551cc4b10c848a051ac59f662b457ec731a50c92ee` |

The full 22-layer capture remains required even though its weights do not fit
physical Kodiak's 512 MiB DRAM. Its simulator and compiler gate is separate
from the bounded two-layer hardware exercise. The source graph also retains
float operations that TorchAO did not quantize; `--dtype int8_w8a8` must not
quantize them again.

| Capture | Weights SHA256 | Input SHA256 | Framework golden SHA256 |
| --- | --- | --- | --- |
| SpectFormer | `77909d9c4f435d9fd438091a84a468a64bbc5023282fb14e28efd237e3f91071` | `8c0c5a7e988d6eda1053b9f51a18bd16e8bd7c1f726cc4053a9ec0176c264c1d` | `a131a6790842bad1e31007382f741536376df1b8d6bfc3252ee82ed09bb572f3` |
| TinyLlama L2 s8 | `9dd1b123bb3d7af13d6ed533750a771312b7f845a8abcf6d3f5d28db49740e49` | `4ef8c26df6c904c47fcccbe81a960f38b3859a9db042364b6f17dc3aba67c8a7` | `a405acacfcfea378c17041e2fcf43cc9365b57d3880562ac23d7fc91dfe7faf2` |
| TinyLlama full22 s8 | `3c092a63b16d179dbee3301ab082bf112a0b55333b55e17c7c50f06346c187c4` | `4ef8c26df6c904c47fcccbe81a960f38b3859a9db042364b6f17dc3aba67c8a7` | `1e2abf2aaccf0119782fca435fc5ffa40c3604af4fd68f59e8a8733bfab2f9fd` |

The source receipts record Torch `2.16.0.dev20261104+cpu` and TorchAO
`0.18.0`. SpectFormer used its trained `checkpoint_best.pth` (SHA256
`77cacfdbe53f3615666291d2a9c45aa6dd9132e48e0448bee90c3a3f1a116e3e`)
and a loader whose SHA256 is
`6b57f9682a89efbc95feddcedbdd45590f6e36986307566193038e2fde1e2aa6`.
The checkpoint is **not** in the capture archive. The local loader defaults to
an environment-specific checkpoint path and silently selects random
initialization if that file is absent. A recapture
must supply the same checkpoint bytes explicitly and confirm the loaded state.
The upstream SpectFormer implementation, image preprocessing, and full Python
environment are likewise outside this portable archive.

The TinyLlama loader SHA256 is
`cbed6414805368b6aad894b47cab3b26fa4b9081385713fbfe0b453129a9803f`.
Its declared checkpoint is `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. The L2 capture keeps its
first two pretrained decoder layers; the full capture keeps all 22. Both use
seeded synthetic token IDs. The Hugging Face checkpoint, loader source, and
dependency environment are **not** archive members. The receipt's absolute
loader paths and source digests are provenance assertions, not a sealed source
closure or a recipe that guarantees recapture.

## Relocate after extraction

The source `model.mlir` embeds an absolute `prov.weights_file` path from the
capture machine. Keep the extracted source untouched and materialize a fresh
final directory with the helper:

```sh
PYTHONPATH=src python examples/kodiak/relocate_capture.py \
  /path/to/extracted/spectformer_torchao_w8a8_20261106 \
  /path/to/final/spectformer_torchao_w8a8_20261106
```

The command verifies the source receipt, copies every receipted artifact,
changes only that module attribute to the adjacent final `weights.safetensors`,
and writes a new `capture_receipt.json`. It preserves the original bytes in
`model.source.mlir` and `capture_receipt.source.json`. The new
`relocation_receipt.json` records both SHA256 identities, the unchanged typed
root count, and `source_closure_verified: false`. It refuses a destination that
already exists, an original path still present in the derived model, changed
bytes beyond that attribute, or a copied weight hash mismatch. The derived
receipt and MLIR SHA **will differ** from the immutable source hashes above.

For the 23 MiB SpectFormer capture, task-local relocation produced derived
MLIR SHA `1cd1d2cacdcae4c7f2f28ad8f72b2bef712fdac39b0c1bfd8803b21507aeb71f`.
The derived receipt verified; all 41 typed roots remained identical; and
Merlin's `resolve_forward_args` loaded 222 forward arguments from the final
directory. The derived hash is path dependent. The two TinyLlama captures have
not yet undergone relocation on a separate host.

## Compile gate

From a Merlin checkout that includes the source-preserving
`--dtype int8_w8a8` CLI, use the derived capture as an explicit input:

```sh
merlin-compile --workload spectformer --target rvv --dtype int8_w8a8 \
  --capture-bundle /path/to/final/spectformer_torchao_w8a8_20261106 \
  --run host --verify --json
```

For bounded and full TinyLlama, substitute the corresponding derived bundle
path and a descriptive workload label. A failed `--verify` is a real numerical
gate failure, not permission to replace `golden.npy`. A `--no-verify` run may
diagnose compilation, image execution, or OPU routing but does not establish
TorchAO framework fidelity. Report framework comparison, exact Spike, matched
FireSim, and physical Kodiak as separate gates. The current strict host
framework comparison fails for SpectFormer (relative L2 `0.00718247` with
other thresholds exceeded) and the TinyLlama source graph; no saved capture
should be described as a complete passing hardware flow on that basis.
