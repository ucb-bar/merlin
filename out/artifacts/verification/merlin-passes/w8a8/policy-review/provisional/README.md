# Provisional W8A8 contract for continued integration

On 2026-10-21 Jeremy authorized continuing with an assumed recipe while the external
TorchAO reference is unresolved. The existing draft below is the development specification.
This does not establish TorchAO fidelity, independent review, or a full verified-pass receipt.
No approved specification, frozen dispatcher, statement pin or axiom is changed.

## Recipe used by the current proofs

- Reach and eligibility come from the actual source IR. The supported contraction domain is
  static canonical matrices with zero initialization and reduction extent K < 131072.
  Unsupported geometry, body wiring or calibrated-reuse conditions refuse.
- Dynamically quantized operands use one scale per parallel-coordinate group: ordered
  absolute maximum over reduction coordinates; the explicit zero-row scale 0.00001;
  otherwise f32 `max(amax / 127, 0.00001)`; f32 reciprocal; multiply, nearest-even round,
  clipping to -127/127 and signed-byte conversion.
- Calibrated reuse retains captured signed bytes and their source scale only when the
  source-derived zero-point, granularity and layout conditions hold.
- Integer accumulation starts at zero, sign-extends bytes, and uses i32 multiplication and
  addition. The checked full-range bound makes every accumulator prefix exact.
- Requantization converts the accumulated integer once, then multiplies by the left scale
  followed by the right scale when those scales are present.
- Captured per-tensor source quantization requires scalar finite positive f32 scale and
  scalar i32/i64 zero point. Its provisional sequence uses a binary64 reciprocal, conversion
  to f32, multiply, nearest-even round, converted zero-point addition, lower-then-upper
  clipping to -128/127, and signed-byte conversion.
- Captured per-tensor dequantization uses the existing additive width candidate, including
  scalar i32/i64 zero points and the actual operation's checked metadata. This is distinct
  from repairing the separately documented per-channel production discrepancy.

## Floating interpretation and domains

Ordered execution proofs remain parametric in the floating model. The concrete byte candidate
`W8A8ByteModel.extend base` supplies binary32 clipping and signed-byte conversion using integer
fields. Other floating operations retain the explicit base interpretation. The test adapter's
host Float/Float32 model remains test infrastructure, not a proved IEEE instance.

Initialized quantizer output still requires the stated non-NaN intermediate condition.
Calibration facts establish actual parameter bindings and source support; they do not establish
that condition for arbitrary activations. We do not assume that initialization, a finite
original input, or a positive scale alone implies it under every abstract floating model.

Whole-function qualification must carry a justified input/context domain and prove source
definedness and policy-reference support on that same domain. Finite tests and exact AST
matching cannot supply those proofs.

## Where later corrections belong

| Decision to revisit | Current implementation boundary |
|---|---|
| W8A8 recipe, selection and reuse | `Rules/W8A8Reference.lean`, `Rules/W8A8Emission.lean` and component rules |
| Captured source Q/DQ order and metadata | `Rules/W8A8SourceQuantize.lean`, `W8A8DequantWidth.lean`, `W8A8SourceContract.lean` |
| Program-level source dispatch | `Rules/W8A8SourceSemantics.lean`, an explicit provisional extension |
| Floating behavior and numerical domain | `W8A8ByteFloat.lean`, `W8A8ByteModel.lean` and source-domain checks |
| Strict whole-function claim | `Check/PolicyExact.lean` and `PolicyParametric.lean`; source-definedness and reference-support proofs remain separate |

`PolicyExact.check_sound` is generic in the semantics, so changing the source interpretation
does not require a new equality theorem. It does require replaying qualification under that
interpretation. Existing frozen-semantics suffix proofs cannot silently be reused under a
different source interpretation.

`PolicyParametric.check_sound` also accepts checked SSA renaming. Its generic program theorem
requires operation/call laws, proved for the provisional source interpretation by
`Rules/W8A8SourceAlpha.lean`. It preserves whole-module metadata and covers internal calls;
it does not establish the separately required numerical domain or source definedness.

`PolicyW8A8Supported.check_sound` now adds independently checked source/reference support
and strict typed refinement under the provisional interpretation. Its current support fragment
requires calibration facts from supported producers and handles source operations at the current
block level; nested source operations and internal-call support refuse. The interpretation can
still execute internal calls, and the SSA matcher can compare them. Support permits UB and
poison, so this result does not replace the initialized-execution/numerical-domain obligation.

Agustin's eventual review should correct or approve the intended contract from independent
references. The public draft commands retain their review refusal in the meantime.
