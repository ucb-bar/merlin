# Merlin W8A8 verification progress

Scope: Merlin-authored shared passes between model2MLIR and the OOT compiler, beginning with
`lower_contraction_int8`. Upstream LLVM/MLIR transformations are trusted. The production pass
and the frozen semantics are unchanged.

## Implemented integer component

- `Rules/W8A8Accumulation.lean` connects signed byte extension, modular i32 products and a left
  accumulation fold to an integer dot product. Every prefix is exact from zero under the
  explicit premise `K * 16384 < 2^31`. The bound covers -128 as well as symmetric quantized bytes.
- `Rules/W8A8IntegerBody.lean` proves execution of the actual scalar instruction sequence,
  including every initialized/poison combination and arbitrary external environments.
- `Check/PolicyW8A8Body.lean` provides an executable Boolean checker and proves that acceptance
  establishes those semantics. Repeated execution of an accepted actual region satisfies the
  modular and bounded integer results. Alpha matching checks actual types, attributes, wiring,
  operation order and SSA freshness; provenance hints do not supply proof facts.

This follows the executable-checker plus acceptance-to-semantics theorem construction in
[Axon, sections 2.1 and 5.4](https://arxiv.org/html/2605.01660v3). It is a component of Merlin's
policy checker, not a proof that the entire W8A8 pass is correct.

## Validation

`merlin/tests/ir/test_lean_w8a8_body.py`: **19 passed, zero skipped** with required tools.
Three independently sized float matmuls are run through Merlin's real pass. Eight mutated
integer bodies are valid under xDSL and the compiler's MLIR verifier, then refused by Lean:
unsigned extension, addition replacing multiplication, subtraction replacing accumulation,
squaring one input, dropping the accumulator, returning the accumulator, and nsw/nuw additions.
Another test covers consistent SSA renaming and malformed alias refusal. Seven numerical
cases compare actual Lean execution, actual MLIR integer folding, and an independent integer
reference, including -128 and both signed wraparound boundaries. MLIR folding is an oracle here;
these tests do not attempt to verify an upstream pass.

Targeted Lean builds pass. The imported theorem audit reports **1117 theorems, zero violations**;
claims use only the standard allowed axioms. Ruff passes, and the frozen-file lock is unchanged.
The repository structure check has one pre-existing failure: `zephyr_model.py` is 3557 lines
against its recorded ceiling of 3537; this file is identical at the base commit and was not edited.

Local generated evidence: `body-tests.xml`, `body-tests/`, `body-audit.json`, and
`body-qualification.json`. The qualification record binds the tested sources and tool fingerprints;
these regenerable files are not committed. New component statements are not yet promoted to the
reviewed top-level statement pins, and the full integrated guard was not run for this component.

## Still required for the W8A8 pass claim

The remaining obligations are complete source-to-quantization group/axis correspondence,
captured-weight policy, source definedness and correspondence between complete before/after programs.
Standard integer matmul coordinates, K bounds and actual matrix initialization now have component
proofs, described below. No C5 whole-pass receipt is issued. Independently reviewed Lean policy semantics, site/chain integration, all policy
mutations and qualification on the chosen model remain required. The paper model has not been
finally selected; the user identified SmolVLA and TinyLlama as candidate integration cases, both
now captured under the default paper configuration. The implementer has read the pass, so a recipe
reconstructed from it cannot honestly be labeled an independently authored specification. No new assumption or float law was added.

## Ordered scalar components and captured models

`Rules/W8A8Scalar.lean` and `Check/PolicyW8A8Scalar.lean` now prove execution of the scalar
abs/max, guarded scale, reciprocal, roundeven/clamp/convert and ordered requantization bodies.
The constant-capture checker derives bounds and denominator bits from the actual prefix and
invalidates facts at SSA rebindings. These statements quantify over poison and all raw bit patterns
under every current float model; they add no float law and do not prove IEEE conversion definedness.

The user approved the explicit zero-row branch on 2026-10-20. See the
[policy review draft](policy-review/README.md) and finding F9 for the precise distinction from the
unguarded expression. This decision does not freeze or independently review the entire policy.

The combined integer/scalar qualification has **36 passing tests, zero skipped**. New scalar
coverage exercises three real-pass shapes, eleven compiler-verified MLIR mutations, dishonest
capture hints, SSA fact invalidation, and an explicitly scoped component probe. Local retained
sources and JUnit results are in `scalar-tests-20261020/` and `scalar-tests-20261020.xml`.
The scalar imported-theorem audit checks 2089 theorems with zero violations. Frozen-file locks
and Ruff pass; the same unrelated structure ceiling violation remains.

Actual default-configuration preparation boundaries are retained in
`model-boundaries/20261020T215617Z/`. TinyLlama's W8A8 stage changes; a diagnostic accepts all
155 hinted integer bodies and 155 hinted requantization bodies in its output. All 155 integer
contractions have the standard `(m,k), (k,n), (m,n)` maps with parallel/parallel/reduction iterators.
SmolVLA's W8A8 boundary is unchanged under captured-scope selection. The probe explicitly reports
`whole_pass_verified: false`; these counts establish neither changed-site correspondence nor
whole-boundary correctness. No C5 policy receipt is issued.

## Tensor contraction and actual K bound

The tensor contraction component is now proved end to end from an accepted region through mapped
reads, repeated writes and canonical tensor output. `PolicyW8A8Matmul.check_zero_sound` proves that
initialized signed bytes and a zero-initialized destination produce exact integer dot products.
The layout checker recomputes access bounds and requires K < 131072; the proof establishes this
implies K * 16384 < 2^31. The selected trace has exactly K points per output cell, independently of
M and N. General execution also preserves poison and arbitrary modular i32 initialization.

The zero-filled destination is explicit in this component theorem; the initializer connection
below discharges it from a checked actual producer. Source operand/scale correspondence and
complete policy integration remain open; no whole-pass receipt is issued. The adapter checks actual SSA operand/result types and passes actual maps,
iterators and region syntax to the component checker. Provenance selects diagnostic candidates
only, and the report retains its explicit component scope.

**14 tensor tests pass, zero skipped**, in addition to the unchanged 36 integer/scalar tests.
These cover real-pass shapes and zero extents; both sides of the strict K threshold; four valid
MLIR mutations (left/right maps, reduction axis and i16 accumulation); and three native MLIR
comparisons, with four initialized tensor samples each. Actual Lean loop execution, native MLIR
and a separate integer reference agree, including -128, nonzero initialization and both modular
wraparound directions. The new imported-theorem audit checks 1905 theorems, zero violations.
Evidence is retained in `tensor-qualified-20261020/`, `tensor-qualified-20261020.xml` and
`tensor-audit-20261020.json`. A full tensor diagnostic also accepts all 155 TinyLlama integer
contractions; `tinyllama-tensor-probe-20261020.json` explicitly says `whole_pass_verified: false`.

## Checked initialization and its connection to the contraction

`PolicyW8A8Zero.check_binding_sound` proves that a matched actual empty/constant/fill prefix
binds the selected SSA destination to a canonical zero matrix. `PolicyW8A8InitializedMatmul`
connects that value, including out-of-bounds poison, to the checked tensor contraction. A caller
cannot certify initialization by declaring a tensor type or an expected value. The rule currently
recognizes the closed three-operation matrix initializer; other producer arrangements abstain.

All 155 hinted TinyLlama integer sites have accepted actual initializers in
`tinyllama-zero-probe-20261020.json`. This closes the initializer component obligation, while
actual source-to-quantized operand correspondence and the full C5 claim remain unimplemented.
The initializer imported-theorem audit checks 2165 theorems with zero violations.

Initializer mutation tests retain generic MLIR assembly when changing a named fill body: custom
assembly otherwise reconstructs its default region and erases that mutation. Nonzero fill,
returning the dirty destination element and bypassing the fill are valid under both xDSL and
the compiler's MLIR verifier, then refused. Consistent SSA renaming is accepted; malformed alias
input is refused separately. No production pass, frozen semantics or reviewed lock was changed.

Final combined regression for these components: **58 passed, zero skipped** (90.95 seconds).
The retained snapshots, native inputs/results, source digests and limits are recorded in
`qualified-20261020/`, `qualified-20261020.xml` and `qualification-20261020.json`.

## Floating tensor stage components

`Rules/W8A8TensorOperands.lean` and `Rules/W8A8Pointwise.lean` connect checked mapped reads to
complete tensor overwrites. `Rules/W8A8FloatTensor.lean` lifts the ordered scalar scale,
reciprocal, quantize and zero/one/two-scale requantize contracts through these loops. Old
destination cells may be arbitrary, including poison. `Rules/W8A8AbsMax.lean` preserves the
lexicographic reduction trace and repeated output writes without assuming a floating identity,
associativity or commutativity. Its initial accumulator remains explicit; the actual floating
zero initializer is still a separate obligation.

`PolicyW8A8Geometry`, `PolicyW8A8Pointwise` and `PolicyW8A8AbsMax` recompute maps, iterators,
shapes, bounds and bodies. Expected parameter dimensions are explicit contract inputs; the
future source/site rule must derive them from the source, rather than accepting the candidate's
chosen axis. Streaming constant facts justify literal captures in the already executed environment
even as the tensor loop changes its index. The previous scalar prefix APIs remain available.

The tests exercise all nine floating stages from three actual Merlin contractions, including
zero extents. Compiler-verified mutations cover reduction and scale axes, transposed output maps,
scale order and scalar bodies. Rank-zero through rank-three requantization component fixtures
exercise zero, one and two scales; the zero/one-scale fixtures are constructed templates, not
claimed production captures. Six native differential cases compare Lean execution, native MLIR
and an independent ordered NumPy f32 reference, including signed i32 endpoints and values above
the exact f32 integer range. Native numerical comparison here covers requantization; it does not
qualify abs/max, scale or quantize floating execution.

The imported-theorem audit checks **2406 theorems with zero violations**. A diagnostic accepts
all **155** TinyLlama tensor requantization stages with scalar (per-tensor) scales. Its report
retains `whole_pass_verified: false`; these are stage checks, not source-to-output correspondence
or C5 receipts. Evidence is in `tensor-stages-qualification.xml`,
`tensor-stages-qualification.json` and `tinyllama-float-tensor-probe.json`.

Targeted qualification and scalar-capture regression: **47 passed, zero skipped** (124.89 seconds).
Ruff, artifact layout, whitespace and frozen-file hashes pass; the same unrelated module-size
ceiling violation remains. The earlier 58-test record is historical and was not relabeled as a
fresh complete regression.

## Shared literal initialization and initialized abs/max

`Rules/W8A8Fill.lean` proves the closed empty/literal/fill sequence for arbitrary static shapes
and scalar literals decoded by the existing semantics. `Check/PolicyW8A8Fill.lean` checks that
sequence and proves its actual destination binding after execution. The existing integer matrix
initializer delegates to this common implementation with its public contracts preserved.

`PolicyW8A8InitializedAbsMax.lean` specializes it to the exact positive-f32-zero bits and connects
the resulting canonical tensor to the checked abs/max reduction. This discharges the floating
initializer obligation for this recognized producer arrangement, including rank zero and zero
extents. The proof preserves the ordered reduction; it never treats zero as a floating identity.
Source operand/group correspondence and complete C5 integration remain separate obligations.

The new qualification has **22 passed, zero skipped** (32.49 seconds): actual left/right
reductions from three Merlin contractions, eight compiler-verified initializer mutations,
rank-zero through rank-three fills, SSA renaming/alias refusal and four native fill comparisons.
The native cases compare actual Lean execution and MLIR bit for bit, including positive-zero
bits and empty output. This does not add native abs/max arithmetic coverage. The imported-theorem
audit checks **2660 theorems, zero violations**. Retained evidence is in
`absmax-initialization-qualification.xml`, `absmax-initialization-audit.json` and
`absmax-initialization-qualification.json`; the integer adapter regression is recorded separately.
The integer initializer regression also passes: **8 tests, zero skipped** (9.38 seconds).
Both initialized-reduction and initialized-matmul modules build with only the allowed standard
axioms. Ruff, artifact layout, whitespace and frozen-file hashes pass; no production pass or
frozen semantics was edited.

## Native comparison of complete dynamic contractions

`test_lean_w8a8_native_recipe.py` runs complete real-pass output against a separate ordered NumPy
recipe, without using Merlin's quantization helpers for its expected answers. Twenty finite
samples span four shapes, zero rows, floor-limited scales, rounding ties, differently scaled
rows/columns, zero K and empty output. All output bits agree. The tests also detect four
compiler-verified MLIR corruptions: roundeven replaced by round, wrong requantization scale
axes, nonzero integer initialization and a changed clamp bound. A known-answer reference check
exercises ties-to-even and per-group scales.

This is native numerical qualification of the emitted sequence against the declared recipe.
It does not run Lean's unsupported host max/min evaluator, compare against eager TorchAO, or
replace the missing source-to-policy theorem and independent specification review. The component
theorems remain parametric. **Nine targeted tests pass, zero skipped** (4.24 seconds); evidence
is in `native-recipe-qualified.xml`. No production pass or frozen semantics was edited.

Combined W8A8 regression after the floating-stage and initializer changes: **119 passed, zero
skipped** (211.66 seconds), in `qualified-floating-initialization.xml`. The combined audit checks
**2981 imported theorems, zero violations**. Nineteen native/reference tests were also rerun with
passed-test retention enabled: **19 passed** (27.13 seconds), preserving 18 native execution
reports and eight full-recipe reference reports in `retained-floating-native/`. The default
temporary-directory policy removes passing cases from the combined regression; retained native
evidence is therefore kept separately. Source digests, report digests and remaining obligations
are recorded in `floating-initialization-qualification.json`. The same unrelated structure
ceiling failure remains. No C5 whole-pass receipt is issued.

## Whole-module policy checking foundation

`Check/PolicyModule.lean` now proves the generic module obligation against a fixed transformer
that receives only the actual source. Candidates cannot provide their own reference, turn an
unsupported expansion into identity, or replace it with the target. Conditional acceptance
composes with a checked downstream suffix, including typed domains. Strict refinement explicitly
requires support of the policy reference; source support cannot substitute for this premise.
The W8A8 source-derived draft and its source/domain checks are being integrated separately.
This foundation introduces no new numerical law, reviewed policy or whole-pass receipt.

## Source-derived W8A8 reference and domain qualification

`Rules/W8A8Emission.lean` and `Rules/W8A8Reference.lean` construct a complete reference program
from the actual source, without consulting the target or role hints. The current draft supports
static canonical matrix contractions with K < 131072, checked multiply/add bodies and positive-zero
source initialization. It derives dynamic grouping from the source convention, or retains
calibrated signed bytes and their actual scale through supported reshapes and transposes after
checking constant-zero zero points and channel correspondence. Unhandled selected sites refuse
the expansion; their original source is not substituted as a fallback.

`Check/PolicyW8A8Module.lean` proves whole-function matching and the stronger qualification
obligation: defined original source and strict target refinement of the supported policy
reference on typed, nonpoison inputs. The reference remains an explicitly **unreviewed draft**,
because its implementer has read the pass. No independently authored specification or reviewed
statement pin is implied by these proofs.

The compiled `check-w8a8-draft` command reports matching, original-source definedness and
reference support separately. Dynamic examples pass these three checks. Actual TinyLlama
before/after programs match the complete reference across all 155 contractions, including
calibrated reshape/transpose reuse. Its source/reference support still refuses the captured
quantization operations, including i64 zero-point dequantization. The command therefore reports
`abstracted`, `whole_pass_verified: false`, and the exact independent obligations. The retained
record is `tinyllama-source-reference-draft-report.json`; the emitted reference is
`tinyllama-reference-draft.mlir`.

Eighteen module tests pass, including valid MLIR wrong source operands, wrong scale
sources, swapped integer operands, changed floor, omitted lowering, scope widening and
unsupported source-body/init cases. Four calibrated fixtures cover direct, reshaped and
transposed reuse. An unknown captured dequantization kind refuses instead of narrowing reach.
The imported module audit checks **7233 theorems with zero violations**.
F11 records the real pass's overly broad generic-body recognition; the reference refuses that
source, and the production pass is unchanged. The byte-bound draft receipt adapter and regression
qualification are being completed separately.
