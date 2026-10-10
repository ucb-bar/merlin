# Iteration workloads and headline validation

Use the same independent iteration models across accelerator examples. They are
small enough for capture, compiler development and numerical debugging without
loading a headline checkpoint. The capture worker seeds model construction.

| Iteration loader | Comparable operator patterns | Not established by passing it |
| --- | --- | --- |
| [Residual CNN](residual_cnn/loader.py) | Convolution, residual add, ReLU, spatial reduction, classifier | ResNet50 depth, BatchNorm folding, ImageNet accuracy or full-model coverage |
| [Causal decoder](causal_decoder/loader.py) | Embedding, RMSNorm, causal multi-head attention, gated MLP, logits | TinyLlama weights, rotary embeddings, KV-cache decoding or session coverage |
| [Multimodal policy](multimodal_policy/loader.py) | Vision patches, text/state inputs, concatenation, normalization, cross-attention, action output | SmolVLA architecture, checkpoint, denoising trajectory or policy accuracy |
| [Mixed MLP](coverage_mlp/loader.py) | Linear, activation and normalization seams | Any headline model or target support |

These models are independently authored patterns, **not extracted validation
subgraphs** and not architectural or numerical equivalents of the headline models.
Their smaller shapes are deliberate. Inspect exact frontend and MLIR signatures,
layouts, storage/compute/accumulator precision and transfer obligations before
making a property-specific comparison. A matching operator name alone is insufficient.

### Optional profiles

The residual CNN, causal decoder and multimodal policy each read an optional
adjacent `profile.json` (schema `merlin.iteration_workload_profile.v1` with the
loader's own `workload_id`). Without one, the model, parameters and inputs are
exactly the default development workload. A profile is strictly validated:
unknown keys, wrong types and inconsistent sizes are refused, and its digest is
recorded as the capture's `session_provenance.profile_sha256`. Every optional key
is off or at its default when absent:

| Loader | Optional keys |
| --- | --- |
| Residual CNN (also needs `channels`, `spatial_side`, `blocks`) | `stem` (`conv3` or a 7x7 stride-2 `conv7s2`), `stem_pool` (3x3 stride-2 max pool), `downsample` (stride-2 block with a 1x1 stride-2 projection shortcut), `bottleneck` (1x1 reduce, 3x3, 1x1 expand), `batchnorm` (eval-mode, seeded running statistics) |
| Causal decoder | `seq` (at most 16), `hidden` (at most 64), `heads`, `kv_heads` (grouped key/value heads), `ffn`, `rope`, `decode_step` with `cache_len` (one token with explicit key/value cache inputs) |
| Multimodal policy | `tokens` (any count up to 32), `queries` (query length distinct from the key/value length), `gelu_mlp` (tanh GELU block on the context), `time_embedding` (sinusoidal timestep input added to the state) |

These patterns stay small enough for Phase 1 iteration. Exercising one does not
establish the same pattern at headline scale, a session protocol or accuracy.

### Performance-scale generators

A target may declare `workload_spec.performance_applications`: independent workloads whose forms
feed only the Phase 2 form scope (`corpus derive --performance-capture LABEL=PATH`), never Phase 1
forms or source capsules. These generators sit in the multi-tile regime of a small accelerator; each
reads an optional adjacent `profile.json` that declares exactly its default keys.

| Generator | Pattern | Default widths |
| --- | --- | --- |
| [Decoder](decoder_scale/loader.py) | Grouped-query attention with rotary embeddings, RMSNorm, gated MLP, logits | hidden 832, heads 16, kv_heads 4, ffn 2176, layers 2, seq 96, vocab 4096 |
| [Decoder step](decoder_step_scale/loader.py) | One token over an explicit key/value cache; every projection a one-row contraction | hidden 896, heads 16, kv_heads 8, ffn 2432, layers 2, cache 95, vocab 4096 |
| [Vision encoder](vit_scale/loader.py) | Strided patch embedding, LayerNorm, multi-head self-attention, GELU MLP | image 136, patch 8, hidden 384, heads 8, mlp 1536, layers 2 |

The residual CNN at a larger profile serves the same purpose. Widths are deliberately off the
held-out headline models' layer widths and shapes, so a performance member is never one of their layers. A target's roster file (for example
`examples/gemmini/target/phase0-workloads.yaml`) names the loader and profile of every label;
`merlin experiment corpus stage-workloads` writes one fresh workload root per label for capture.

The [host-control/math probes](host_control_math/README.md) independently exercise
scalar arithmetic, masks, selection and trigonometry with an exact saved-capture
checker. They are diagnostic inputs, not permission for host fallback or evidence
of accelerator support.

The [per-tensor dequantization probes](quant_boundary/loader.py) are separate,
target-independent operation diagnostics. The [rank-4 companion](quant_boundary/loader_rank4.py)
checks every int8 input value at a non-power-of-two scale; each has one result because the native-host
qualifier currently accepts one exact FP32 result ABI. Both exercise int8 inputs
and the PyTorch `quantized_decomposed.dequantize_per_tensor` frontend path.
Capture it with `--dtype int8 --already-quantized`; the saved MLIR and golden can
then be checked through `merlin_experiments.model_qualification --native-host-only`.
That exact-input native-host result helps review a host fallback, but is not a
generated Phase 1 capsule, a sealed source capture, or evidence of accelerator
execution or every scale and input value.

The held-out headline roster is **TinyLlama, SmolVLA and ResNet50**, declared through
the existing claim-model policy and each target's `workload_spec.models`. Do not
feed their captures, layer frequencies or results into capsule derivation or
performance selection. Independent model-scale derivation sources remain under
`workload_spec.applications`. Changing the roster requires fresh Phase 0 evidence
and corpora; saved runs retain their original inputs.

## Capture and inspect

Run the existing worker using an isolated model2MLIR interpreter with frontend
tracing enabled by its public API. No target code is needed in these loaders:

```bash
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/causal_decoder/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out out/artifacts/workloads/iteration/causal-decoder/fp32
```

The worker generates `frontend-trace.json`, `pytorch-opset.json`, `linalg.mlir`,
and a complete `model.mlir` bundle with external `weights.safetensors`,
inputs/goldens and a byte-bound `capture_receipt.json`. It reuses the exact
conversion/model instance, rather than recapturing a second model. Python,
NumPy and Torch are seeded before loader import, and deterministic algorithms
are required for the selected framework build. Repeat with the other loaders.
Inspect original, quantized and prepared static call counts and their exact
MLIR correspondence. An unsupported capture is a reported gap, not a reason to
silently substitute another operator or precision. Quantized runs require a
reviewed operation-scoped software/quantization spec, not a blanket model cast.

`application_demand_inventory(..., application_metadata={...})` accepts
explicit `workload_id`, `workload_role: iteration` and
`coverage_scope: full_capture` for these small models. Here `full_capture` means
the **small iteration model**, never its comparable headline workload. The
accounting report preserves these labels without turning them into support claims.

## Full validation

Use the existing model2MLIR loaders for the real headline checkpoints. Record
checkpoint revision, frontend/runtime versions, representative input source and
preprocessing, precision policy and execution scope. Random-weight or truncated
TinyLlama, random-input ResNet50 and a single SmolVLA denoising step are explicitly
scoped diagnostics, not complete headline validation. SmolVLA's session workflow
and TinyLlama's prefill/decode workflow need their own interfaces and checks.

[Headline capture and lowering](headline_validation/README.md) gives reproducible
commands for all three models, names each scope, and shows where to inspect the
separate MLIR, tensor payload, session contract and lowering-stage artifacts.

Phase 0 audits declarations and capsule obligations. Phase 1 must prove complete
lowering, execution and numerical agreement; Phase 2 measures performance on a
separate cohort. Report accelerator, host and unsupported work independently and
never count host-only execution as successful accelerator coverage.

See [the Phase 0 specification guide](../../docs/guides/phase0_specification.md)
for the generated evidence layout and admission requirements.
