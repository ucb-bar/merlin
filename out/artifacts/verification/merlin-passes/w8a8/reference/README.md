# EL4 numerical reference intake

Agustin directed Jeremy to `examples/gemmini` on `el4-experiment`. The remote branch is
`feat/el4-experiment`; this intake pins commit `0cc289de52a10bed256f32c90cfbb54a21b3c81f`
without merging it or changing the shared checkout. The eight source documents/recipes are
retained with `.snapshot` suffixes under `el4-0cc289de52a1/`; `reference-intake.json` records
their original paths, Git object ids and content hashes. All three selected recipe file hashes
and canonical recipe digests match `protocol/experiment.yaml`.

The [historical quantization guide](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/numerics/QUANTIZATION.md#L77)
records torch 2.10.0, TorchAO 0.16.0 for ResNet50/TinyLlama and 0.15.0 for SmolVLA. It explicitly
labels its r53b captures historical. The selected [protocol](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/README.md)
and recipe JSONs are a newer selection. Its run manifest is a template with unfilled capture
identities; this intake does not infer selected capture versions from the historical guide.

## Which recipe belongs to which boundary

| Layer | Reference | Consequence for the W8A8 proof |
|---|---|---|
| Incoming calibrated program | Selected capture recipes: static activations, tensor scales, range -128/127; weights -127/127 with one scale shared by 16 output channels; observer epsilon 2^-12 | Preserve the captured scale bytes. Observer calibration is upstream of the Merlin pass. These settings do not change the dynamic fallback's epsilon. |
| Merlin dynamic fallback | Reduced-range -127/127, per-reduction-coordinate groups, floor 0.00001, explicit zero-row branch | TorchAO's symmetric reduced-range activation quantizer supplies a versioned primitive reference. Applying that recipe to both operands is an explicit Merlin policy, not TorchAO's default whole-Linear configuration. |
| TorchAO default dynamic Linear | Version 1 in both installed references; reduced-range activations but full-range weight metadata and a different weight epsilon | Its integer-linear dispatcher refuses reduced-range weight metadata. Configured two-operand comparisons call TorchAO quantization and `int_scaled_matmul` explicitly; captured full-range weight reuse exercises the actual accepted integer-linear dispatch. No dispatcher predicate is patched. |
| Historical r53b epilogue | Convert accumulator, multiply by activation scale, then weight scale | Matches the operation order of the current Merlin fallback policy. |
| New selected capture epilogue | `integer_epilogue_v1`, integer bias preload, grouped scales, binary64 constant formation and one f32 multiplier at narrowing exits | Already-captured arithmetic remains part of the actual input program. It must not be silently replaced by the historical two-multiply expression. Device readout is outside this W8A8 checker. |
| Implementation revision | EL4's `lower_contraction_int8` also handles promoted bf16/f16 inputs and widening-based captured reuse | This is different code from the checked branch. A reference pointer does not automatically qualify those additional paths. |
| Model selection | Selected TinyLlama prefill is 32 tokens with 32 decode invocations; the older retained proof-development input has 8 tokens | General theorems remain shape-parametric. Existing retained-model coverage is not relabeled as coverage of this newer selection. |

## Versioned API checks

`lean_torchao_oracle.py` runs independently of Merlin in isolated TorchAO 0.15.0 and 0.16.0
environments sharing the existing torch 2.10.0+cpu / NumPy 2.4.6 reference installation.
It records the actual source-file digests and retains initialization diagnostics. Optional C++/GPU
extensions are unavailable; these observations use CPU quantization and integer arithmetic only.
The environments are reference executions, not reproduced CUDA capture environments.

`test_lean_w8a8_torchao_reference.py` checks real, conditionally qualified W8A8 outputs against
configured TorchAO primitives, native MLIR, Lean and the separate ordered NumPy recipe. Captured
Q/DQ reuse is checked under the declared additive source interpretation against TorchAO's actual
integer-linear dispatcher; native frozen MLIR cannot directly execute those source operations.
Separate per-tensor quantize/dequantize cases compare TorchAO with Lean and standard native MLIR,
including i32/i64 zero points, scalar/ranked tensors, saturation and odd-zero-point rounding ties.
Invalid conversion observations and out-of-bound contractions are refused before native evidence.

Set `MERLIN_TORCHAO_0_15_0_PYTHON` and `MERLIN_TORCHAO_0_16_0_PYTHON` to those interpreters before
running `merlin/tests/ir/test_lean_w8a8_torchao_reference.py`. These are additional to the existing
Lean/native MLIR reference settings. `MERLIN_LEAN_REQUIRED=1` makes missing references a failure;
ordinary optional test runs report them as skipped rather than silently claiming qualification.

This source guidance resolves where to find the numerical references. It is not independent
approval of the current provisional Lean policy, a frozen-specification change, an IEEE-model
proof or a full-pass verification receipt. The source input-bound and numerical proofs continue
separately. Completed qualification counts and evidence are recorded in the parent progress log.
