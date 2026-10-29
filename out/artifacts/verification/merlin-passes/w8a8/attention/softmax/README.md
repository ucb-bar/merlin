# Softmax numerical components and remaining source binding

## Latest continuation: denominator lower bound and bounded division now compile

The new `W8A8UnitRounding`/`W8A8UnitAdd` modules compiled strictly in builds v37/v38.
`W8A8SoftmaxRow.denominator_unit` compiles and its counted
`denominator_finite_lower` proves both the finite sum cap and real denominator≥1.
NEW `W8A8DivisionBounds.divide_by_unit_bounded` follows actual unpacked `divCore`
and both rounding shifts: a finite divisor with raw exponent≥127 and numerator
cap≤253 gives result cap=max(numeratorCap+1,3), allowing either sign and underflow.
No reciprocal identity or general division-monotonicity premise is added.

The same row module now proves finite/NN actual probability cap128 and a complete
ordered weighted-value trace bound, including equal list lengths to exclude zip
truncation. Eight terms and Vcap138 give output cap146. These are numerical row/
trace results, not source SSA binding, actual initialized model success or1533 acceptance.

Strict builds v39-v42 passed. Expanded row fixture v4 passes13printed standard3
roots, including exact singleton/masked probabilities, signed zero/subnormal input,
overflow from a nonzero tiny divisor, and explicit all-masked probabilities=NaN.
Failed v3 remains retained. A NEW full audit is recorded in worker-status; preceding
10972audit predates these additions and cannot qualify them retroactively. Actual
mask/row correspondence and reference QKV remain open. Historical next-step paragraphs
below are superseded by this newer numerical result, not by a relaxed assumption.

### Actual reduction execution bridge

New ReductionBody/checked body/ReductionCells/ReductionTensor/ReductionOrder modules
compile. They prove the actual optional-poison scalar region, filtered per-cell
trace in loop order, generic tensor execution and initialized projected results.
Byte maximum operand commutation is proved only for non-NaNs; a concrete two-NaN
payload counterexample remains. The actual captured max1470/sum1489 bodies are
recognized from the pinned authoritative source export. The nine-root reduction
fixture and focused5025-theorem/zero audit pass; counts overlap earlier closures.

Native development compares seven actual MLIR programs/38samples, including empty
rows/outputs and division. Zero tested non-NaN bit differences; four sum NaN payload/
sign differences are retained explicitly. Three compiler-verified wrong body/key/SSA
mutations refuse. These are empirical observations, not universal native fidelity.
Latest ReductionNamed.execute also compiles and links the actual named dispatch to
generic execution; it postdates that focused audit. Row/cross-SSA and diagonal mask
admission remain unfinished. Source1533stillrefuses; no receipt or complete flag changed.

This is active W8A8-owned work in the separate unqualified rounded-real profile:
`out/build/lean-w8a8-rounded-real/repo`. It does not change or rebuild any sealed capsule,
production pass, shared interface, source specification, other worker's checkout or ledger.
The planner remains thread01a1650a-b70b-7cb3-adb6-6f2f50bff524. Worker-status TOP is the
current job record; all earlier pauses and the old planner-briefing prompt are historical.

## What is proved

All statements below concern actual pinned unpacked binary32 arithmetic and the explicitly
declared, noncomputable rounded-real transcendental candidate. No native primitive fidelity,
independent policy approval, captured source admission or complete pass correctness follows.

| Source module under merlin/lean/MerlinVerify/Rules | Result and essential premises |
| --- | --- |
| W8A8MaskedMaximum | The actual ordered byte-maximum fold over finite scores or genuine -Inf masks stays in that domain. If at least one actual member is finite, the result is finite, retains its exponent cap, and equals an actual finite row member. All-masked rows retain -Inf. |
| W8A8RealOrder | Integer-unit magnitudes are monotone in finite magnitude words. Actual byte ordering is sound for finite real values; maximum dominates both finite operands. |
| W8A8MaskedOrder | Extends that order to -Inf masks and proves that the ordered row maximum dominates every finite row member. |
| W8A8RealSign | Exact signed integer-unit/unpack correspondence, exponent alignment, and sign preservation through the actual normalization, rounding and packing. |
| W8A8SubtractionOrder | Actual subtraction of ordered finite operands produces a nonpositive value; it follows from the aligned signed mantissa, with no assumed subtraction-order law. |
| W8A8SubtractionBounds | Actual subtraction is finite with cap=max(left,right)+2 when both input caps are at most252. Uses shared W8A8PairBounds unit/cap conversion; unsafe large operands may overflow. |
| W8A8AttentionDifference | Actual finite self-subtraction is +0, so its exponential is exactly1. A masked -Inf minus a finite maximum stays -Inf, so its exponential is +0. An all-masked row instead produces NaN. |
| W8A8PositiveRounding | Positive mantissas at or above the minimum exponent survive both actual rounding shifts and packing. It includes subnormals and does not assume flushing to zero. |
| W8A8PositiveAdd | Actual NN addition preserves a positive operand; an ordered NN fold containing a positive member stays positive/nonzero. Infinity remains allowed here, until the separate finite sum bound applies. |
| W8A8SoftmaxRow | A row with caps≤252 and an actual finite member has finite, NN exponentials in[0,1], including an actual1. With count≤2^22, its actual ordered denominator is finite and positive/nonzero. |

The positive denominator result uses the attained actual member and the actual rounded
addition theorem. Exponent caps or sampled outputs alone do not establish it. Other
exponentials may underflow to zero; the theorem does not claim that every term is positive.

## Evidence and its limits

The trig packet at`../trig/` contains exact executed sources, helpers, argv, environments,
compiler/auditor references and logs for every attempt. Failed elaborations and resource
attempts remain preserved. None has been converted into a qualification success.

- module-build-v36 compiled all22new library/driver modules under explicit autoImplicit=false
  and relaxedAutoImplicit=false. The standalone Main is checked by strict --run in the
  retained strict development probe. Old project objects and pinned external caches are
  reused as documented; this is not an independent rebuild of the entire compiler.
- The trig kernel fixture passes14printed roots; the masked-maximum fixture passes7; the
  row fixture passes6. Cases include exact singleton/masked denominators, subnormal behavior,
  exceptional domains, invalid keys/types/arity/SSA and unsafe subtraction overflow.
- The earlier new-model audit found10813theorems/zero violations. The later row-import audit
  found3737/zero. These have separate exact closures and identities; their counts are not
  additive. The full post-strict audit is tracked in worker-status and must finish before
  it is cited as evidence for that complete candidate.
- root-statements-development-v1 preserved17new root statements and confirmed all217original
  pinned statements and SPEC_LOCK still match. This direct auditor/lock check did not run
  the default Lake guard build and did not approve a new specification.
- Actual retained TinyLlama source/ref demands in contractv3 pass179SOURCE/165REFERENCE,
  with prefixes1060/1087 and first refusal1533. New caps include cos1180/sin1193=127,
  rotary Q1372=148,K1398=149, logits1437=180,scaled1448=179. The strict replay is a separate
  development observation, not a receipt or HOST release qualification.

## Next arithmetic obligation

The denominator is now proved finite/nonzero. To derive a useful finite quotient cap, prove
a suitable lower bound using its actual1member and the actual ordered additions, or prove
the corresponding division bound directly. Nonzero alone admits tiny subnormal divisors
and does not establish finite bounded division. Preserve that distinction.

Use an additive rounded-power lower-bound lemma. The current primitive proof already
exposes both shifts, their exponents and positive mantissas. A representable power-of-two
threshold should survive NN addition. This must follow from those computations; do not add
an assumed denominator≥1, an arbitrary reciprocal law or a native floating approximation.
Then derive the actual binary32 quotient cap/NN result and the final ordered V contraction.

## Source mask and SSA obligations

The numerical row contract is not yet connected to the actual max/sub/exp/sum/div sequence.
The retained source has these carriers:

```text
1448 finite scaled logits ->1462 select(genuine -Inf mask, finite score)
1462 ->1470 actual maximum reduction, initialized with -Inf1469
1462, reshaped1470 ->1477 subtraction ->1483 exponential
1483 ->1489 ordered addition reduction, initialized with +01488
1483, reshaped1489 ->1496 division ->1515 final V contraction ->1528
```

There is conservative source evidence for a diagonal unmasked witness: mask1141 is copied
from AND1135; one component is causal comparison1101, built from index grids, and another
is equality1129 between gathers1113/1121 from the same captured tensor1031. At a diagonal
coordinate the gather indexes should coincide and the causal comparison should hold.
The inverse mask1454 is then used by select1462. This is a structural observation, **not**
a completed theorem about execution, actual affine maps, casts, capture values or layouts.

Prove that witness from the actual checked source structure and initialized tensor/index
facts. Keep true -Inf values and the all-masked counterexample; do not replace masks with a
large finite constant, assume a nonempty row, or turn an intermediate condition into a
declared original-input contract without proving/reviewing the association.

The current scalar cap facts do not retain the relation between a score tensor, its row
maximum, shifted exponentials and the corresponding denominator. Numerical caps cannot
reconstruct it. A checked segment/whole-pattern theorem or a suitable additive relation
environment is needed to link these separate SSA values. Coordinate-dependent/row-existence
facts also cannot be called elementwise facts without a proof of that interface. Preserve
the old Elementwise/frozen definitions; agree ownership before any shared-interface change.

Reference reconstructed Q/K/V caps remain missing independently of these source proofs.
Actual source/ref1533 remains a refusal until all required conditions are checked. Whole-pass,
native, capture admission, reviewed policy and composed pipeline flags remain false.

## HOST qualification remains provisional

The older ordered capsule has the preserved original/snapshot contract association defect.
The new four-file binding wrapper passes24development tests and an independent adversarial
probe, and retained creation/replay pass, but two separate integrity-read mismatches stopped
curated controls. A bounded five-call forensic run did not reproduce the mismatch or capture
a bad buffer. Its cause remains unresolved. No failed attempt is validated, skipped or
retried into qualification, and this numerical work does not repair that environmental issue.
