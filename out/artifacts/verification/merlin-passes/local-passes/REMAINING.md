# Remaining obligations

## Completed

- **Proved:** f32 `fix_bool_sitofp` acceptance → complete-program execution of the declared
  false/+0, true/+1, poison-preserving repair policy. Includes SSA, nested regions, calls and
  ordered suffixes. The IEEE two-case premise is discharged, without a new axiom.
- **Qualified:** both sites in the actual SmolVLA denoise capsule; independent synthetic programs;
  compiler-valid mutations; pinned PyTorch/TorchAO CPU comparisons; replayable provisional receipts.
- **Proved component:** TorchAO's actual scalar quantizer body and complete correspondence with
  a source-derived materialized program. This is **not captured-call refinement**.

## Still required

| Obligation | Status |
| --- | --- |
| Independent admission of the new bool-to-float contract/instance | Human review; prior float-to-bool approval does not extend automatically |
| Actual selected Gemmini MLIR + capture receipt + exact version/recipe/scales | Missing artifacts; branch documents alone do not bind a capture |
| TorchAO captured-call → tensor-reference execution bridge | Engineering remains: row reduction/layout, poison, parameter addressing and declaration/call handling |
| Concrete-model reduction/elision and primitive correspondence | Unproved: `AbsReductionCorrect`, `ZeroPointElisionCorrect`, clamp/rounding/conversion fidelity; frozen uninterpreted floats do not supply these laws |
| Nonfinite conversion and empty-input contract | Review required: CPU zeros do not define Torch's platform-dependent invalid casts; MLIR can produce poison. Both pinned TorchAO versions refuse the tested empty quantization |
| Broader formats/layouts/overloads | Explicitly unsupported; bool proof is f32 only |
| `lower_quant_ext` extensions | 230 observed quantizers and 627 observed i64-zero-point dequantizers are outside the narrow existing proof |

TorchAO receipts remain `proved_materialization_provisional`, with
`source_call_refinement_proved=false` and `approved_full_verification=false`.
There is still work beyond human review. No approved full-pass/full-model claim is made.

See [README](README.md), [review proposal](TORCHAO-REVIEW.md) and
[qualification records](qualification.json) for exact boundaries, versions and evidence.
