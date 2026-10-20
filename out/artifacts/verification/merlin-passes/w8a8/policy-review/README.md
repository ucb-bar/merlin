# W8A8 policy statement for review

This is a review draft, not an approved independent specification. The current implementer has
read Merlin's pass. The repository requires independent specification review before a whole-policy
claim is pinned. No new axiom, float law, or policy lock is introduced by this document.

The target theorem is general: acceptance of complete actual before/after programs implies
`after` refines `P_W8A8(before)` for every permitted input and shape. SmolVLA/TinyLlama provide
integration evidence; they are not the theorem's input domain or a basis for hardcoded rules.

## Reference identity

The existing reference fixture records TorchAO **0.16.0**, PyTorch **2.10.0+cu128** and symmetric
per-token activation quantization with bounds -127/127, epsilon 1e-5 and zero point zero:
`merlin/tests/data/torchao_affine/PROVENANCE.json`.

Official versioned sources:

- [quant_api.py, lines 907–922](https://github.com/pytorch/ao/blob/v0.16.0/torchao/quantization/quant_api.py#L907):
  the reduced-range symmetric activation recipe.
- [quant_primitives.py](https://github.com/pytorch/ao/blob/v0.16.0/torchao/quantization/quant_primitives.py):
  `choose_qparams_affine_with_min_max`, `_quantize_affine_no_dtype_cast`, and
  `_quantize_affine_no_zero_point_no_dtype_cast` for scale, reciprocal/multiply, rounding and clamp.
- [plain_layout.py, lines 269–285](https://github.com/pytorch/ao/blob/v0.16.0/torchao/dtypes/uintx/plain_layout.py#L269):
  integer contraction, activation scale then weight scale.

The locally installed capture environment is newer (TorchAO 0.18.0, PyTorch 2.14.1). Its current
configuration default is version 2. That environment is not evidence that retained captures were
produced under those versions. No capture provenance is inferred from today's environment.

## Proposed contract and current proof obligations

| Component | Intended operation | Evidence and remaining obligation |
|---|---|---|
| Reach | The explicitly selected captured-quantization contractions; optional gather/conv/nonlinear policies are excluded | Must be reconstructed from the source, not provenance roles |
| Dynamic groups | Each operand's parallel coordinates share a scale over its reduction coordinates | Scalar abs/max is proved; tensor axes and iteration order remain |
| Dynamic scale | f32 division by 127, floor at the f32 representation of 0.00001 | Explicit zero-row branch approved by the user; see below |
| Reciprocal | One f32 `1 / scale` per group | Scalar execution is proved |
| Quantize | Multiply by reciprocal, round to nearest even, upper/lower clamp to 127/-127, signed i8 conversion | Ordered scalar execution and checked captured constants are proved; IEEE fidelity remains trusted/qualified |
| Existing quantized input | Retain captured signed bytes and their scale, only under the declared zero-point/granularity contract | Static/per-channel paths, transposes and reshapes need source checks |
| Integer contraction | Signed i8 extension, i32 product and addition, starting at zero | Scalar/repeated execution, standard matmul maps, checked zero fill and K extraction are proved; whole-site source correspondence remains |
| Overflow | Require `K * 16384 < 2^31` for full-range i8 inputs | Every-prefix bound is proved; standard matmul checker derives K from the actual type/map layout in its adapter; whole-site typing connection remains |
| Requantize | Convert i32 once, multiply by the left operand's scale then the right operand's scale when present | Ordered scalar execution is proved for zero, one or two scales; tensor maps remain |
| Source shape | Canonical multiply-add contractions, checked static geometry and initialization | Must not treat the mere presence of mul/add operations as proof of the source meaning |
| Unsupported | Refuse or mark abstracted with a reason; never issue a full verified receipt | Whole-site/whole-boundary integration remains |

Scalar proofs quantify over raw bit patterns and poison, under every current FloatModel. They do
not establish clamp-before-convert definedness for a concrete IEEE instance. In particular, the
abstract model permits `toInt` to return poison even for an apparently clamped value. Any stronger
numerical/definedness result needs an explicit reviewed domain/model contract, not a hidden premise.

## Concrete zero-row choice

The formula in design §5.5 and the TorchAO reference is `max(amax / 127, 0.00001)`. Merlin adds:
**if amax equals zero, use 0.00001 directly; otherwise use that formula**.

`MerlinVerify.Rules.W8A8Scale.guarded_eq_iff` proves the exact condition relating the two formulas.
`guard_counterexample` proves that unconditional agreement is false for the current abstract float
model. That witness is intentionally non-IEEE and is not a runtime bug report.

**User decision (2026-10-20): include the explicit zero-row rule.** The declared recipe therefore
uses the floor directly when the zero comparison succeeds, and the division/maximum expression
otherwise. The scalar implementation theorem proves this recipe without an extra float law.
This approves that concrete policy choice; it does not constitute review of the entire policy
transformer or establish numerical agreement with TorchAO. `GuardAgreement` remains an unassumed
comparison obligation for the older unguarded expression.

## Actual model records

- SmolVLA: `merlin/contract/capsules/model/SY_model_smolvla/capsule.interface.mlir`, dynamic scheme,
  816 function arguments, approximately 11 MB of IR.
- TinyLlama: `merlin/contract/capsules/model/SY_model_tiny_llama/capsule.interface.mlir`, static scheme,
  203 function arguments, approximately 2.1 MB of IR.

The review must distinguish dynamic quantization from reuse of calibrated static inputs. Neither
bundle is declared verified by this draft. Numerical agreement with the reference remains a separate
qualification result from the checker soundness theorem.

Actual preparation captures under the default paper configuration are retained locally in
`../model-boundaries/20261020T215617Z/`. TinyLlama's `prepare-quantize-int8` boundary changes;
SmolVLA's does not under captured-scope selection. A diagnostic probe accepts all 155 hinted
integer bodies and 155 hinted requantization bodies in TinyLlama's output. It explicitly reports
`whole_pass_verified: false`: hints, body counts and unchanged boundaries establish neither
complete changed-site coverage nor source-to-output correspondence.
