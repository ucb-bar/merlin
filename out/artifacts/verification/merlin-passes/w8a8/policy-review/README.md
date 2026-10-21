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
| Dynamic groups | Each operand's parallel coordinates share a scale over its reduction coordinates | Ordered reduction, checked axes and zero initialization are proved; the draft derives groups for static canonical matrices |
| Dynamic scale | f32 division by 127, floor at the f32 representation of 0.00001 | Explicit zero-row branch approved by the user; see below |
| Reciprocal | One f32 `1 / scale` per group | Scalar/tensor execution and canonical source-to-scale wiring are checked |
| Quantize | Multiply by reciprocal, round to nearest even, upper/lower clamp to 127/-127, signed i8 conversion | Ordered scalar/tensor execution and checked captured constants are proved; IEEE fidelity remains trusted/qualified |
| Existing quantized input | Retain captured signed bytes and their scale, only under the declared zero-point/granularity contract | The draft checks zero points, scale shape and channel/layout correspondence for supported transposes and per-tensor reshapes |
| Integer contraction | Signed i8 extension, i32 product and addition, starting at zero | Scalar/tensor execution, standard matmul maps, checked zero fill and canonical source wiring are implemented |
| Overflow | Require `K * 16384 < 2^31` for full-range i8 inputs | Every-prefix bound is proved; the component checker derives K from actual geometry and the draft requires K < 131072 |
| Requantize | Convert i32 once, multiply by the left operand's scale then the right operand's scale when present | Ordered execution, projection maps and canonical source scale correspondence are checked for zero, one or two scales |
| Source shape | Canonical multiply-add contractions, checked static geometry and initialization | Must not treat the mere presence of mul/add operations as proof of the source meaning |
| Unsupported | Refuse or mark abstracted with a reason; never issue a full verified receipt | Whole-boundary and suffix integration are implemented; source semantics and independent policy/domain review still block qualification |

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

Tensor-stage diagnostics additionally accept all 155 TinyLlama requantization stages with
per-tensor scales. Native tests compare complete emitted dynamic W8A8 contractions against a
separate ordered NumPy recipe on finite inputs, including zero rows, scale floors, rounding
ties and empty dimensions. These comparisons complement the Lean component proofs; they neither
establish a whole-policy theorem nor make the recipe independently authored or reviewed.

## Concrete reference implementation available for review

The draft transformer is now executable in `MerlinVerify/Rules/W8A8Reference.lean`, using
`Rules/W8A8Emission.lean`. Its policy name is deliberately `w8a8-reference-draft`. It receives
only the actual source module and explicit reach (`all` or captured-only); the target cannot
supply the reference. The current supported domain is static canonical matrix multiply/add
contractions with zero initialization, K < 131072, f32 or signed-i8 operands, and at least one
f32 operand. Dynamic operands use per-parallel-coordinate groups. Captured reuse requires
source-derived signed storage, zero points, scale shapes and channel/layout correspondence.
Selected unsupported cases refuse expansion. Broader generic geometries remain unsupported.

The whole TinyLlama source/output function matches this reference, including all 155 changed
contractions. Source definedness and reference support remain separate, failing obligations for
its captured quantization operations under the current frozen semantics. The draft cannot be
promoted by changing a receipt's review bit: the driver and byte-bound API retain the review
refusal. Required review covers the recipe, reach, source eligibility, calibrated-reuse rules,
unsupported/NaN behavior and statement/domain pins. This implementer-read draft remains evidence
for that review; it is not labeled an independently authored specification.

## Captured-source semantics change request

The retained TinyLlama boundary contains **310 per-tensor dequantizations and 89 per-tensor
quantizations**. Every scale/zero-point parameter is a scalar constant through `tensor.splat`:
all 399 scales are finite positive f32 values and all zero points are literal i64 zero. The
observed scales range from approximately 0.0004902 to 1.4986. Dequantization metadata includes
the reduced range -127/127; quantization uses -128/127 and the string output dtype `int8`.
This is byte-bound capture inspection, not a Lean source-definedness proof or producer-version
qualification. The detailed records are `../tinyllama-quant-source-inventory.json` and
`../tinyllama-quant-parameter-inventory.json`.

The current frozen source contract accepts only signed-i8/i32/f32 dequantization with its
declared metadata. It has no quantize semantics. The complete TinyLlama function therefore
cannot pass source-definedness or reference-support qualification, even though all 155 W8A8
sites match the source-derived reference.

`Rules/W8A8DequantWidth.lean` is a concrete additive review candidate for i64 zero points. It
proves exact i32 compatibility and output typing/nonpoison for initialized, valid tensor inputs.
Forty finite samples agree with pinned PyTorch and executed standard MLIR. Installing it and
expanding the accepted source metadata require source-contract review; the frozen dispatcher
and specification files have not been changed. F12 separately shows why supporting a source
contract must not silently bless a discrepant production lowering.

The remaining per-tensor quantize contract needs particular care. The
[versioned PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/ao/quantization/fx/_decomposed.py#L42-L128)
extracts scalar tensor parameters, computes a Python-scalar reciprocal, then multiplies, rounds,
adds the zero point, clamps and converts dtype. Its scalar precision/order must be represented
explicitly; an unproved f32/f64 reciprocal identity cannot be introduced as an implementation
shortcut. A CPU observation for an exceptional cast cannot define a portable fallback: the
[versioned conversion documentation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_tensor_docs.py#L4991-L4996)
states that nonrepresentable float-to-integer conversions have undefined, platform-dependent
behavior.

`Rules/W8A8QuantizeDomain.lean` makes the existing conversion obligation precise. `defined_iff`
characterizes nonpoison quantization by the exact ordered float-to-byte conversion succeeding;
`conversion_sufficient` gives the resulting byte. `not_uniformly_defined` and
`region_counterexample` prove that initialized data and named clamps cannot establish a
nonpoison result under every currently permitted FloatModel. The countermodel is deliberately
non-IEEE. This is a limitation of the stated model/domain claim, not a newly observed CPU bug.
Support (`NotStuck`) still permits an `.ok` outcome containing poison.

Required independent review therefore covers both the policy and its claimed numerical domain:
the selected floating interpretation, valid input/intermediate conditions, exceptional values,
captured-source quantization semantics and version binding. A concrete IEEE interpretation or
an explicit checked semantic-domain restriction needs its own justified contract. We have not
silently narrowed the input domain, added a floating law, equated support with definedness, or
substituted passing model samples for that review.

An additive concrete candidate now discharges clamp-before-convert definedness without a
floating axiom. `Rules/W8A8ByteFloat.lean` implements the relevant binary32 operations through
integer fields, and `W8A8ByteModel`, `W8A8ByteTensor` and `Check/PolicyW8A8ByteDefined.lean`
connect its theorem to the actual quantization checker. The remaining numerical-domain premise
is explicit: each rounded input-times-inverse intermediate must be non-NaN. Initialization or
non-NaN original operands alone do not prove that condition (infinity times zero is a
counterexample). This candidate has finite native qualification but is not an approved global
floating interpretation or a captured-source quantization contract.

The concrete captured-source proposal is now `Rules/W8A8SourceQuantize.lean`, with full-range
conversion proofs in `W8A8SourceByte.lean`. It follows the linked PyTorch 2.10 source and
[clamp definition](https://docs.pytorch.org/docs/2.10/generated/torch.clamp.html): binary64 scalar
reciprocal, f32 scalar conversion, multiply, roundeven, zero-point addition, lower-then-upper
clipping to -128/127 and signed-byte conversion. The eligibility domain is finite positive f32
scalar calibration scales and scalar i32/i64 zero points. Other captured contracts are refused.

Review must distinguish this candidate's `run_defined` theorem from universal source
definedness. Its post-addition non-NaN condition is an explicit semantic premise, not yet an
automatically established property of all model inputs. Zero times an overflowing reciprocal
and NaN data remain poison observations, never platform-dependent expected native outputs.
The candidate is tested against pinned PyTorch and standard MLIR, including all 89 actual
TinyLlama quantization parameter sets, but is not installed in `Sem/Ops` or approved by these
experiments. Integrating a reviewed contract and checked numerical domain is still required.
