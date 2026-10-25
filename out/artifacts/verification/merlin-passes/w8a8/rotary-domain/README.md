# W8A8 rotary input bounds and source byte-weight bounds

Worker continuation, 2026-10-22. Scope: numerical source execution needed by Merlin's
`lower_contraction_int8`. Production transforms, frozen/shared interfaces, frontend and
`QuantExtLowering*` work remain unchanged. Publication remains held.

**This is a conditional prefix-bound candidate. Complete W8A8 still refuses quantizer 1533.**
It strengthens the separately named finite/nonzero-f32 candidate with explicit per-argument
exponent caps. The earlier finite-only and finite/nonzero domains and their evidence remain
preserved. No archive is claimed to meet this domain, and this is the retained eight-token
development boundary, not the selected EL4 paper capture.

## Domain and exact claim

The old candidate still requires every f32 input element finite and nonzero. Other input
types retain typed/initialized requirements and their declared integer intervals. The new
contract optionally bounds each f32 argument's eight-bit exponent field, ignoring its sign.
Both signs and subnormals are covered. Zero f32 inputs remain outside the inherited domain.
Caps on other types, malformed arity and caps above 254 refuse. The input witness uses a
checked nonzero subnormal when necessary, so cap zero also has an inhabited domain.

`PolicyW8A8RotaryPrefix.entry_bound` proves: if the actual entry/prefix and a requested live
SSA bound check accept, then entry binding and execution of those actual ordered operations
succeed, the requested value is initialized, and every in-bounds f32 element satisfies the
checked finite exponent bound. It does not assert execution of the operations after that
prefix, whole-model success, or acceptance of the complete policy boundary.

The context retains the unchanged seven source and three dynamic numerical laws, plus the
previous explicit arithmetic agreement with Lean's executable rounded model. Joint concrete
model existence, native primitive fidelity and native rsqrt agreement remain unproved. No
new axiom, frozen specification, assumption admission or transcendental law was introduced.
Domain inhabitation proves that inputs exist; it does not construct this numerical context.

## Implemented components

| Files, under `MerlinVerify` | Result |
| --- | --- |
| `Rules/W8A8IEEEBounds.lean` | Binary32 product bounds account for both rounding shifts, signed zero and underflow. A checked exponent sum guarantees a finite product. Addition of a signed zero preserves a conservative bound. |
| `Rules/W8A8IntegerFloatBounds.lean`, `Check/PolicyW8A8RotaryCast.lean` | Actual integer intervals and signed conversion widths determine checked f32 conversion bounds; interval tables above 4096 refuse. |
| `Rules/W8A8SingleWrite.lean`, `Check/PolicyW8A8SingleWrite.lean` | Actual ordered output-map coordinates are checked against the output iteration space. Each cell reads its original initializer once, then receives its proved postcondition. Repeated writes refuse this rule; trace enumeration is limited to 65,536 points. |
| `Check/PolicyW8A8Rotary{Facts,Domain,Scalar,Transpose,Defined,Prefix}.lean` | Per-argument bounds, witness checking, scalar transfers, actual transpose maps/body, views, source analysis and demanded SSA bounds. Ordinary reduction invariants forget bounds that have not been proved preserved. |
| `Rules/W8A8ByteBounds.lean`, `Check/PolicyW8A8RotaryDequant.lean` | All 256 signed-byte patterns determine bounds from the actual scale and zero bits, preserving conversion/subtraction/multiplication order. This proves source numerical facts, not dequantization-lowering correspondence. |
| `Driver/W8A8Rotary.lean`, separate `W8A8RotaryMain.lean` | Separate source/reference prefix results, requested-bound checks, explicit premises and refusals. |
| `src/merlin/verify/lean/rotary_{evidence,policy}.py` | Actual exports, strict contract/evidence validation, fresh theorem audits, byte-bound receipts and regenerated replay. Every result remains `abstracted`. |

The new rules are general over the checked widths, arguments, maps and shapes. Model names
and captured SSA numbers do not select production behavior.

## Actual retained boundary

Positions are proved in `[0,8)` through the actual `linalg.index`, signed index cast,
multiply-by-one, add-zero and view chain. The actual signed-i64-to-f32 conversion yields
exponent cap 129. The proved conservative multiplication condition is
`max(leftCap,1) + max(rightCap,1) <= 379`; therefore the proposed frequency-argument cap is
`379 - 129 = 250`. This is an operationally derived proposed domain, not a guessed frequency
value or a fact inferred from a model name. The actual multiplication rechecks the condition.

| Actual source value | Checked fact |
| --- | --- |
| Input 201, frequency tensor | Proposed exponent cap 250, inherited finite/nonzero input condition |
| 986 / 997 / 1152, generated positions and views | Integer interval `[0,8)` |
| 1154 / 1163, converted positions and broadcast | Finite, exponent cap 129 |
| 1168, rotary contraction | Finite, exponent cap 254; its actual reduction extent is one |
| 1175 / 1178, transpose and concat | Finite, exponent cap 254 |
| 158 source dequantizers | Individual scale/zero-derived exponent caps; all byte patterns checked |

Both the 1,060-operation source prefix and 1,087-operation reference prefix satisfy all
**163 requested bounds**. Source quantizer 1256 remains initialized. Both next refuse
quantizer **1533**, consuming result 1528 after attention. Sine/cosine output bounds and the
subsequent floating contractions/softmax have not been inferred by this candidate.

The authoritative before/after JSON hashes remain respectively
`651a1568cb65ee40c4fff5eb00bd9067a1f505954b8e37754cbfbb100808a908` and
`c17fd3a6366441a62bf98cf31f16703fcfed60232f2d8fba809e4b36c5c2fafc`.
`retained-bound-proposal-v1.json` binds the actual source, position facts and every dequantizer.
`retained-exponent-contracts-v1.json` contains the explicit proposed entry caps and demands.

The native fixture retains the 38 original rotary producer operations. Controls include
both signs of the largest allowed exponent and nonzero subnormals, with observations at
positions two and seven. The old maximum-finite-frequency witness still yields infinity
and NaN and is excluded by the new proposed cap. These are fragment executions, not a
full-model execution or universal transcendental fidelity proof. No poison conversion is
executed natively. Earlier rotary and RMSnorm counterexamples remain preserved.

## Qualification status

Fresh Lean guard: **9,974 theorems, zero violations**, existing frozen lock passes.
Final regression: **334 passed**, zero failures/errors/skips, in 688.099 seconds. This includes
42 new rotary cases and 292 prior affected cases; historical totals overlap. All eight actual
changed geometry/index-width cases independently replay, including two empty-row cases. The
retained TinyLlama receipt also independently replays against the exact original exports,
checking all 163 demands and preserving the result-1533 refusal.

The current private build is `out/build/lean-w8a8-rotary-domain/repo`; previous private builds
remain untouched. Runtime identity is
`83cb81d716feb8e6ef4cac0bfa1b3b75566feaf73e8763a36933fd2c9d9c3195`.
All **1,263 frozen records**, the older numerical capsule's 1,230 records, the historical
752-file runtime and six originally qualified sources retain their recorded hashes.
`qualification-v2.json` indexes the exact tests, audits, receipts and limitations.
`implementation-v2.patch` contains only the 23 owned additive source/test files;
`source-v2/` retains their bytes. No source was staged, committed, integrated or pushed.

The development batch exposed an opaque general type-equality reduction in a concrete
kernel example. Direct matching of the actual type now makes domain inhabitation kernel
checkable; both native/kernel development tests pass. Review also identified Python's
boolean/integer equality as too weak for exact JSON replay comparison. Canonical typed JSON
comparison and duplicate-key rejection now have real-receipt forgery regressions. Earlier
runs are retained as history, including an extra replay attempted with a resolved base Python
instead of the original virtualenv path; that attempt honestly refused the missing MLIR
environment. The eight qualified replays use the recorded compiler's original virtualenv.

## Active scoped-paper continuation

The user-approved [eight-day goal](../../progress/planner/eight-day-paper/README.md) targets
October 30. The subsequent [modular milestone](../modular-policy/README.md) now exposes complete
changed-boundary correspondence separately from successful whole-model execution. Its
W8A8-specific entry/receipt preserves that distinction and the policy's draft status, with
33 tests and an independently replayed complete retained boundary. Attention/1533 remains
a longer-term proof obligation and the stretch track, rather than a prerequisite for every
local preservation theorem.

The frontend's explicit numerical model and W8A8's byte extension also need compatibility
review: their default min/max NaN-payload behavior differs. Separately qualified specialized
models cannot be assumed to form one common numerical context. Frontend owns the compatibility
investigation; this worker has not changed either model or admitted a connecting law.

## Remaining work

1. Prove finite/magnitude propagation through the actual ordered Q/K/V contractions, using
   the checked byte/scale bounds and actual reduction cardinalities. General multi-write
   floating accumulation needs a counted invariant; the single-write rule does not supply it.
2. Justify sine/cosine interpretation and range for finite angles. They remain unsupported
   numerical facts here, rather than an assumed non-NaN output.
3. Continue through rotary products, QK contraction, masking, max/subtract/exp/sum/divide and
   the final V contraction. Account for overflow, attained maxima, nonempty reduction axes
   and nonzero softmax denominators in the source's exact order. The actual mask and maximum
   initializer are **negative infinity** (`0xff800000`); do not assume all masked logits finite
   or every exponential strictly positive. `attention-dependencies-v1.json` retains the exact
   ordered source operations and operand types for the next proof batch, without promoting
   that inventory into additional proved facts.
4. Discharge the joint concrete primitive interpretation, restore/bind the intended input
   domain where justified, obtain independent policy/domain review and bind the selected
   capture/version/recipe. Review is not the only remaining obligation.
5. Complete remaining pass variants and centrally qualify admitted integration. Shared
   Main/CLI, locks, frontend and quant-ext ownership remain separate.

`whole_pass_verified`, `policy_reviewed`, `primitive_model_laws_verified` and
`concrete_context_inhabited` remain false. F11/F12/F13 and the historical JSON decode failure
are not marked fixed. The unrelated structure failure remains `zephyr_model.py`, 3557 > 3537.
