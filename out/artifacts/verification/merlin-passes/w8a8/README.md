# Merlin W8A8 verification progress

Work resumed on 2026-10-21. The [laptop handoff](laptop-handoff/README.md) records the previous
pause checkpoint; the latest progress is below. The full pass is not yet qualified. Agustin
has been nominated for the pending independent review.

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

## Byte-bound draft policy receipts

`merlin.verify.lean.policy.check_w8a8_draft` snapshots both actual boundary files, validates their
authoritative exports and round trips, invokes the compiled command, audits its theorem, and
records the checker/lock/exporter identities. A bound report retains all independent obligations.
The API refuses a driver that attempts to assert review or return `verified` for the draft.
Pending review does not hide output mismatches. Ordinary refinement-chain replay rejects even a
forged green policy receipt rather than composing a numerical policy as equivalence.

Final targeted integration regression: **58 passed, zero skipped** (138.90 seconds), including
the 18 source-derived module cases, the integer-body/export-helper regression, five draft receipt
cases and existing receipt/preprocessing-chain regressions. The checker build was finalized
before these receipt tests. Evidence is retained in `module-integration-qualified-v2/`,
`module-integration-qualified-v2.xml`, `w8a8-module-qualified-audit.json`,
`tinyllama-module-qualified-report.json` and `module-integration-qualification.json`.
Ruff, whitespace, artifact layout and frozen-file hashes pass; the same unrelated structure
ceiling violation remains. No full W8A8 correctness claim or independently reviewed policy pin
is introduced by this integration.

## Captured dequantization width contract candidate

`Rules/W8A8DequantWidth.lean` adds an unwired i32/i64 source-contract candidate. Lean proves
compatibility with the current i32 contract and output typing/nonpoison results on valid,
initialized tensor inputs. Per-tensor conversion/subtraction and per-channel integer
subtraction remain distinct. Frozen source dispatch, specification files and statement pins
are unchanged; TinyLlama therefore retains its source-support refusal.

Twelve targeted tests pass, zero skipped (25.17 seconds). Forty candidate samples agree bit
for bit with pinned PyTorch 2.10.0+cpu / NumPy 2.4.6 and executed standard MLIR. They span both
integer widths, per-tensor/per-channel parameters, integer endpoints, the f32 exact-integer
boundary, signed zero, scalar tensors and empty outputs. Three unsupported-contract probes
refuse explicitly. The test adapter labels its scope and cannot issue a verified verdict.
The imported-theorem audit checks **2646 theorems, zero violations**.

F12 separately retains the production i64 per-channel discrepancy: byte 1, zero point 16777217,
scale 1 produces -16777215, while PyTorch and the candidate produce -16777216. The production
pass is unchanged. Evidence is in `dequant-width-development/`, `dequant-width-development.xml`,
`dequant-width-audit.json` and `dequant-width-qualification.json`. These results prepare source
semantics review; they do not close captured quantization support or whole-policy review.

Four existing PyTorch/float-adapter regression tests also pass, zero skipped (10.94 seconds),
in `dequant-width-regression.xml`. The full Lean guard passes with **7464 theorems, zero
violations**, including frozen file and existing statement pins. Ruff, whitespace and artifact
layout pass. The repository structure check still reports only the pre-existing runtime module
ceiling failure (3557 > 3537 lines); no unrelated runtime file was changed.

## Mixed operands and calibrated channel layouts

`test_lean_w8a8_operand_paths.py` adds **16 passing tests, zero skipped** (17.26 seconds).
Actual mixed signed-byte/f32 contractions qualify for both operand positions and both canonical
add operand orders. An unsigned source conversion refuses expansion. Four calibrated per-channel
cases match the full source-derived reference through independent left/right transposes, while
retaining the separate source-semantics refusal. Nonzero zero points and reduction-channel
scales are refused, even when the real pass rewrites them and their shapes remain well-typed.

Four native companions verify mixed one-scale requantization and calibrated two-scale order,
including transposed byte storage, full-range signed bytes and distinct channel scales.
F13 records the production reuse path's missing eligibility checks. No compiler fix or new
source semantics is implied. Evidence is retained in `operand-paths-development/` and
`operand-paths-development.xml`.

## Policy composition through actual downstream boundaries

`Check/PolicyW8A8Chain.lean` proves strict refinement of `P_W8A8(original)` by the last program
in a checked downstream suffix, preserving the original-source definedness obligation. Each
function retains its own typed, nonpoison input domain. A step may use universal C1 or existing
typed normalization; the typed path checks source membership and the exact common signature.
This does not move a numerical policy through preceding rewrites or equate the final program
with the original floating computation.

`check_w8a8_chain_draft` snapshots and round-trips every successive actual program and rechecks
the entire suffix. It requires byte-identical joins and checks all live inputs, intermediate
exports and the bound report before publishing. Stored receipt verdicts cannot replace these
checks. Ordinary refinement chains continue to refuse numerical-policy receipts. The generic
receipt freshness check now includes the certificate/report file too.

The public commands are available through the existing `merlin-verify lean` entry point:

```sh
merlin-verify lean w8a8-draft "$W8_STAGE" \
  --compiler-python "$MERLIN_M2M_PYTHON"
merlin-verify lean w8a8-chain-draft "$W8_STAGE" "$NEXT_STAGE" \
  --work out/artifacts/verification/merlin-passes/w8a8/my-chain \
  --compiler-python "$MERLIN_M2M_PYTHON"
```

The default reach is captured-only; `--scope all` explicitly requests broader source reach.
`--index-bits` states the input domain. Both commands report `abstracted` and exit 2 while the
reference remains unreviewed, even if every operational obligation passes. A checked suffix
alone does not bind an external compiler handoff or approve source quantization semantics.

Final integration regression: **59 passed, zero skipped** (166.30 seconds), including module,
policy receipt, typed/universal policy suffix and ordinary receipt/chain tests. A multi-function
case requires the typed splat proof while universal C1 refuses, and still preserves each
function's original input domain. Mutation/race tests refuse changed scale floors, disconnected
joins, forged receipt verdicts, changed live intermediate files and changed bound reports.
The public CLI regression adds **11 passed, zero skipped** (17.46 seconds). The full Lean guard
passes with **7481 theorems, zero violations**; frozen files and existing statement pins pass.
The checker build was finalized before these receipt tests.

Evidence is in `policy-chain-qualified/`, `policy-chain-qualified.xml`,
`policy-chain-qualified-guard.log`, `policy-cli-qualified/`, `policy-cli-qualified.xml` and
`policy-chain-qualification.json`. Ruff, whitespace and artifact layout pass; the same unrelated
runtime module ceiling is the only structure failure. Complete actual-model qualification is
still blocked by source quantization semantics and independent policy review, rather than
being inferred from the passing synthetic and real-pass integration cases.

## Handoff qualification and bounded diagnostics

Policy-chain checks now accept `--handoff path/to/handoff.json`, binding the last checked program
to the actual compiler input bytes and rechecking both the handoff record and file before
publication. `--checker-timeout` bounds the checker process (300 seconds by default). A timeout
produces an unmeasured attempt and cannot leave an old successful receipt as the current result.
When policy-prefix qualification fails, suffix diagnostics are explicitly unmeasured with a
null match result; an unrun check is not classified as passing or as a disproved rewrite.

`Check/ExactSyntax.lean` proves equality of complete ASTs, including every nested region, SSA id,
type and attribute. Identical source-derived references use this linear structural check rather
than SSA-alignment synthesis. Independent source support/definedness checks still run. A valid
unmodelled source operation can therefore match its unchanged reference but cannot qualify.

The combined module, handoff, timeout, chain, receipt and CLI regression passes **54 tests,
zero skipped** (55.30 seconds). The full Lean guard passes with **7548 audited theorems, zero
violations**, including the new conversion-domain theorems; frozen files and existing statement
pins are unchanged. Ruff, whitespace and artifact layout pass. The unrelated runtime module
ceiling remains the only structure failure.

Both retained actual model chains now have handoff-bound draft receipts in
`model-policy-identity-qualification/`. TinyLlama matches all 155 W8A8 contractions but refuses
source quantization support. SmolVLA's unchanged W8A8 boundary matches, but qualification refuses
an unmodelled tensor type. Both receipts remain abstracted; their seven downstream boundaries
are explicitly unmeasured because the policy prefix has not qualified. Earlier interrupted and
timed-out diagnostics are retained separately and establish no claim.

`Rules/W8A8QuantizeDomain.lean` additionally proves the exact conversion-success condition and
an unrestricted-float-model counterexample to unconditional nonpoison results. This is an
explicit numerical-domain obligation, not a new CPU defect or added assumption. The expanded
[review packet](policy-review/README.md) records captured-source contracts, versioned PyTorch
references and the remaining independent policy/domain review requirement.

Final evidence is in `policy-identity-qualified/`, `policy-identity-qualified.xml`,
`policy-identity-qualified-guard.log` and `policy-identity-qualification.json`. No production
compiler pass, frozen semantics, source quantization dispatcher or approved policy lock changed.

## Concrete byte conversion and nonpoison quantization

`Rules/W8A8ByteFloat.lean` defines binary32 min/max, symmetric clipping and signed-byte
conversion through integer fields. Lean proves that clipping a non-NaN input to -127/127
makes conversion defined, including when the unclipped value is infinite. The conversion
checks the truncated integer: LLVM's defined -128.9 to -128 case is retained.

`W8A8ByteModel` replaces only the default f32 min/max and signed-byte conversion of an explicit
base model. `W8A8ByteTensor` and `Check/PolicyW8A8ByteDefined.lean` lift the result through actual
region execution and the existing tensor checker. Initialized operands and a non-NaN rounded
product at every accessed coordinate are explicit premises. No arithmetic law is assumed and
the candidate is not installed as an approved global IEEE model.

The test adapter's explicit `w8a8_byte_model` mode compares the actual Lean denotation with
native MLIR. Twenty complete dynamic-contraction samples across four shapes also agree with
the independent NumPy recipe. Scalar conversion, min/max and quantizer cases bring the total
to **801 native sample executions**. Arithmetic NaNs are compared by classification; three
poison quantizer cases run only in Lean. Default adapter behavior still refuses min/max and
the candidate refuses unsupported f64 min/max.

Final qualification: **31 tests passed, zero skipped** (59.56 seconds); **7601 theorems audited,
zero violations**. Frozen files, Lake configuration and existing statement pins are unchanged.
The oracle fixture builds the existing verified library target before execution, including
candidate imports without rebuilding the receipt checker. Ruff, whitespace and artifact layout
pass; the unrelated runtime module size ceiling remains the only structure failure.

Evidence: `byte-model-qualified-v3/`, `byte-model-qualified-v3.xml`,
`byte-model-qualified-v3-guard.log` and `byte-model-qualification.json`. The earlier v2 guard
record includes a rejected build-configuration experiment and is not final qualification.
The existing actual-model receipts remain abstracted: these component results do not approve
the policy, supply captured-source quantization semantics or qualify their unmeasured suffixes.

## Captured per-tensor quantization candidate

`Rules/W8A8SourceQuantize.lean` supplies the missing per-tensor source operation as an additive
review candidate. Following the versioned PyTorch implementation, it extracts scalar parameters,
computes the reciprocal in binary64, converts that scalar to f32, multiplies and rounds, adds
the converted integer zero point, then clips lower-first to -128/127 and converts to i8. The
zero-point addition remains present for literal zero; no reciprocal-precision identity is
assumed. `W8A8SourceByte.lean` proves conversion safety for this asymmetric range.

The runtime candidate checks finite positive calibration scales, scalar f32 scales and scalar
i32/i64 zero points. Lean proves output typing, an exact conversion-domain characterization and
successful nonpoison execution under the explicit post-addition non-NaN condition. This does
not prove the condition for arbitrary model inputs or every unrestricted FloatModel.

Qualification compares Lean, pinned PyTorch 2.10.0+cpu/NumPy 2.4.6 and a separately constructed
standard-MLIR reference on **64 samples** across both integer widths and four shapes. It also
refuses invalid scales, unsupported metadata and undefined conversions. All **89 retained
TinyLlama quantization parameter sets** agree on **1424 scalar observations** with synthetic
inputs. That second experiment binds the actual capture and parameter inventory; it does not
execute or verify the complete TinyLlama function.

Combined regression: **50 passed, zero skipped** (93.39 seconds). Full guard: **7646 theorems,
zero violations**. Frozen semantics, Lake configuration, statement pins and the receipt checker
binary are unchanged. Evidence is in `source-quantize-qualified/`,
`source-quantize-qualified.xml`, `source-quantize-qualified-guard.log`,
`source-quantize-captured-parameters/qualification.json` and `source-quantize-qualification.json`.

Installing this candidate requires source-contract review. Full qualification also needs a
justified numerical domain and checks establishing it for the claimed inputs, SmolVLA's missing
tensor semantics, independent W8A8 policy review and actual-model suffix qualification. The
candidate test adapter cannot issue a verified receipt or enable frozen source dispatch.

## Source-derived accumulation bound and model reach

`Rules/W8A8ReferenceBounds.lean` connects success of the actual source recognizer to the
full-range signed-byte accumulation bound. It proves exactness of every accumulator prefix,
then derives the trace length from the checked tensor coordinates. The theorem does not need
a provenance assertion or an assumed symmetric byte range.

The proof-backed source diagnostic accepts all **155 selected TinyLlama contractions**:
133 have K=2048 and 22 have K=5632. The complete 22-test module regression passes (31.28 seconds),
including actual pass output with K=131071 and refused expansion with K=131072 or 131073.
The full guard passes with **7666 theorems and zero violations**; locks and existing statements
are unchanged. This is the conservative full-i8 bound, not a claim that every larger dot
product overflows.

The same captured-scope diagnostic finds **zero selected W8A8 contractions in retained SmolVLA**.
Inspection of its byte-bound source identifies one dynamic `tensor.empty` with result
`tensor<?xi64>` (SSA result 1033, dynamic extent 1029). It is the first support refusal;
the capture's unchanged W8A8 boundary does not demonstrate active policy coverage.

Evidence: `reference-bounds-qualified-v2/`, `reference-bounds-qualified-v2.xml`,
`reference-bounds-qualified-guard.log`, `reference-bounds-{tinyllama,smolvla}.json`,
`smolvla-dynamic-source-inventory.json` and `reference-bounds-qualification.json`.
Independent review is still pending; Jeremy has nominated Agustin as reviewer.

## Source-operation adapter and checked numerical domains

`Rules/W8A8SourceContract.lean` recognizes actual per-tensor Q/DQ operations from their SSA
declarations, operand/result shapes and types, zero-point widths and exact metadata. Unknown
semantic attributes and duplicate keys refuse; provenance cannot choose the contract. Scalar
f32 parameters and scalar i32/i64 zero points are explicit. Dequantization accepts full or
reduced signed metadata without inferring a narrower range for its stored bytes.

`Check/PolicyW8A8SourceDomain.lean` checks actual initialized observations and the precise
post-zero-point-addition non-NaN condition. `Check/PolicyW8A8Source.lean` combines the static
adapter with Q/DQ runtime checks. Lean proves that acceptance implies successful execution
returning typed, initialized candidate output. This is a theorem about each accepted invocation
under the explicit floating model, not an all-input theorem for a complete model. Initialized
dequantizer float outputs may contain NaNs; initialization is not a finiteness assertion.

The test adapter binds samples to actual function-argument SSA values before fetching the
operation operands. It cannot issue verified receipts. Tests cover both zero-point widths,
rank-zero and empty tensors, unsupported metadata, unknown/duplicate attributes, missing SSA
bindings, malformed runtime tensors, poison, invalid scales and exceptional intermediates.
Compiler-verified source cases and direct malformed-JSON tests are labeled separately.

The retained TinyLlama source adapter recognizes **399 operations: 310 DQ and 89 Q**, with no
static-contract refusals. All 89 retained Q parameter sets additionally pass the new runtime
domain checker on the existing synthetic samples; their candidate outputs equal the retained
observations. This replays domain checks, not a fresh native run or whole-model execution.

Qualification: **97 tests passed, zero skipped** (182.31 seconds); full guard **7709 theorems,
zero violations**, with frozen files and statement pins unchanged. Ruff, whitespace and
artifact layout pass; structure retains only the unrelated runtime size-ceiling failure.
Evidence: `source-adapter-qualified/`, `source-adapter-qualified.xml`,
`source-adapter-qualified-guard.log`, `source-contract-tinyllama.json`,
`source-domain-captured-parameters/` and `source-adapter-qualification.json`.

These candidates remain uninstalled pending independent review. Automatic whole-function
domain justification, reviewed source-semantic integration and actual suffix qualification
remain required for full W8A8 receipts.

## Calibration producer facts and quantified source support

`Check/PolicyW8A8SourceFacts.lean` learns exact scalar parameters from executed constants and
rank-zero splats. Facts have explicit SSA dependencies and are invalidated at every write;
region-entry diagnostics also forget facts shadowed by block arguments. Unsupported producers
and declared parameter types do not supply value facts. Learning is restricted to scalar
calibration types, avoiding evaluation of unrelated dense tensor constants.

The checker proves that accepted parameter values are the actual currently bound tensors with
initialized scalar parameters and a finite positive scale. Fact propagation is generic over
semantics with an explicit producer-agreement proof, which the frozen semantics supplies.
This prepares composition with reviewed source extensions without assuming their correctness.

`calibrated_quantize_supported` then proves non-stuck, typed quantization for every float model
and every typed input tensor of the recognized shape. It deliberately permits poison results:
the activation/intermediate domain is still required for initialized outputs. This theorem
does not install source dispatch or establish whole-function definedness.

All **399 retained TinyLlama Q/DQ parameter pairs** pass the calibration check. The combined
source regression passes **44 tests, zero skipped** (79.53 seconds). The full guard audits
**7759 theorems, zero violations**, with frozen files and existing statement pins unchanged.
Tests cover invalid scales, unknown producers/arguments, scalar/tensor SSA overwrites and
changed literals; initialized nonzero zero points are not confused with symmetric reuse.

Evidence: `source-facts-qualified/`, `source-facts-qualified.xml`,
`source-facts-qualified-guard.log`, `source-facts-tinyllama.json` and
`source-facts-qualification.json`. Independent policy/source-contract/domain review and complete
source/suffix qualification remain open.

## Provisional source-program integration and prefix calibration

Jeremy authorized continuing against the current draft recipe while its TorchAO reference is
unresolved. The [provisional contract](policy-review/provisional/README.md) records the exact
development choices and where a later correction belongs. No policy approval, new axiom or
frozen specification change follows from this instruction.

`Check/PolicyW8A8SourcePrefix.lean` lifts calibration-fact soundness from individual operations
to arbitrary complete successfully executed prefixes. Its segment check traverses once and
proves that each selected quant_ext site receives the corresponding parameter check. Unknown
source contracts refuse. All **399 retained TinyLlama sites** pass this scan; a scan does not
establish that the complete model executes or satisfies the activation/intermediate domain.

`Rules/W8A8SourceSemantics.lean` supplies an explicit provisional program interpretation for
per-tensor Q/DQ. Actual tensor values determine runtime types; the source adapter checks the
operation syntax and metadata. Other operations retain frozen dispatch, and regions/functions
execute under the same extension. Its adapter-agreement and calibration-producer theorems are
kernel checked. The new `w8a8_source_program` differential mode executes complete functions,
including internal calls, without changing the receipt binary or frozen dispatcher.

`Check/PolicyExact.lean` makes exact source-derived policy matching generic in the semantics,
with independent source-definedness and reference-support premises for strict qualification.
The downstream theorem keeps the same interpretation and input/context domain. This route
requires complete AST equality, including SSA ids and module attributes. The new regression
exposes a remaining bridge: raw actual pass exports renumber SSA values, while existing C1
renaming proofs are tied to frozen semantics. Their successful conditional match cannot be
reused for the provisional interpretation. Identity-path acceptance is tested and is not
reported as coverage of changed W8A8 sites.

Qualification runs in an isolated source/build snapshot containing tracked Lean sources and
these explicit additions. A separate BF16 task was writing unfinished library files in the
shared workspace; those files and its `Main.lean` changes were left untouched. The isolated
full guard audits **7854 theorems with zero violations**, preserving frozen locks and existing
statement pins. Its receipt binary has the same SHA-256 as the preexisting shared binary.
This is not a claim that the concurrently edited shared tree passes the full guard.

The combined regression passes **111 tests, zero skipped** (206.91 seconds), including 48
native source-quantization sample executions across both zero-point widths and scalar, matrix
and empty tensors. Complete source execution, internal calls, Q-to-DQ byte flow, invalid
calibration, malformed SSA/type mutations, exceptional conversions and module metadata are
covered. Ruff, whitespace and artifact layout pass; the unrelated runtime module size ceiling
remains the structure failure.

Evidence: `source-program-qualified-v3/`, `source-program-qualified-v3.xml`,
`source-program-isolated-guard.log`, `source-program-isolation.json`,
`source-prefix-tinyllama.json` and `source-program-qualification.json`.

The remaining engineering includes the generic SSA bridge, whole-function source/domain
qualification and actual suffix qualification. The provisional specification remains explicit.

## SSA bridge for the provisional source interpretation

`Check/ParametricAlpha.lean` proves SSA equivalence under explicit operation/call laws, retaining
operand arity and region emptiness. `Rules/W8A8SourceAlpha.lean` proves those laws for the
provisional source interpretation, including the actual source-contract recognizer. No float
identity or certificate-supplied value is used.

`Check/ParametricModule.lean` lifts this result to complete modules through a common call-fuel
induction. It covers internal calls, declarations and missing symbols rather than assuming
that calls name external declarations. Complete outer metadata and non-function operations
are compared separately; every recognized function body participates. A structural guard
refuses extra function regions and unrecognized function views before body erasure.

`Check/PolicyParametric.lean` compares raw compiler output to the fixed source-derived policy
under this interpretation. Its equality theorem quantifies over every context/input; its
strict qualification theorem still requires independent source definedness and policy-reference
support on the same domain. Matching an unsupported function cannot issue a verified receipt.

The retained **TinyLlama boundary matches**, covering all **155 changed contractions**, under
the new theorem and provisional semantic laws. This closes the SSA bridge recorded above.
The source and policy numerical domains, full source qualification and actual suffix checks
remain independent requirements. The reference remains provisional, and no frozen dispatcher,
public receipt command or checker binary was changed.

Qualification: **73 tests passed, zero skipped** (131.73 seconds), including 26 new SSA/module
cases plus the complete source-program and existing module suites. Real-pass cases cover
dynamic and calibrated inputs, supported layouts, empty geometry, nested captures and internal
calls. Mutations refuse changed arithmetic/wiring, stale ids, module/function metadata, extra
top-level operations and extra function regions. Matching an unsupported unchanged function is
explicitly distinguished from source qualification.

The isolated full guard audits **7908 theorems, zero violations**, with frozen files and
statement pins unchanged. Ruff, whitespace and artifact layout pass; structure retains the
same unrelated runtime size-ceiling failure. The concurrent BF16 files and Main changes remain
excluded from this batch's source snapshot and untouched.

Evidence: `parametric-policy-qualified/`, `parametric-policy-qualified.xml`,
`parametric-policy-qualified-guard.log`, `parametric-policy-isolation.json`,
`parametric-tinyllama-qualified.json` and `parametric-policy-qualification.json`.

## Whole-function source support and strict typed policy refinement

`Check/PolicyW8A8SourceProgress.lean` connects checked source contracts, exact calibration facts
and actual SSA operand values to typed, successful Q/DQ steps under every floating model.
`Check/PolicyW8A8Support.lean` lifts that result through complete block execution, preserving
the typing environment and invalidating calibration facts on every write. Ordinary operations
and regions reuse the existing typing rules under the explicit provisional interpretation.

The function theorem proves `FuncNotStuck` for every input of the checked declared types.
It does not require or infer initialized activation data. Source Q/DQ inside nested regions
and support of internal calls currently refuse; their separate matching/execution results do
not supply missing support proofs. Supported calibration producers remain actual scalar
constants and rank-zero splats, with all source contract conditions recomputed.

`Check/PolicyW8A8Supported.lean` checks source and expanded-reference support independently,
then composes those proofs with the complete raw-module policy match. Acceptance establishes
strict typed refinement of the actual output to the draft policy reference, plus source
modelling, on the same declared input domain. Empty function rosters refuse. Unknown source
semantics cannot acquire non-vacuity merely by matching unchanged output.

Both retained **TinyLlama before/after functions pass typed support**, and the **combined
policy refinement check passes**, covering all **155 changed contractions**. The source-derived
reference's support is checked directly by the combined checker. This strengthens the prior
conditional match; it is not an initialized-execution proof or independent policy approval.

Support permits UB and poison. One regression executes the admissible-scale `0 * overflowing
reciprocal` case and observes poison despite passing support; another observes the frozen
false-assertion UB behavior despite passing support. Both remain explicitly unverified for
initialized execution. No undefined native conversion or failing native assertion is executed.

The public receipt binary, frozen source dispatcher, specification locks and statement pins
remain unchanged. Numerical-domain/initialized-execution qualification, semantic promotion and
actual suffix qualification remain open.

Qualification: **113 tests passed, zero skipped** (177.04 seconds), plus the separately run
UB distinction case (**1 passed**, 3.30 seconds). There are 24 new support cases; the regression
also replays source-program, parametric-policy, module and calibration-fact checks. The isolated
full guard audits **7948 theorems, zero violations**, preserving frozen files and statement pins.
Ruff, whitespace and artifact layout pass; structure retains the unrelated runtime size ceiling.
Concurrent BF16 work and Main changes were left untouched and excluded from this snapshot.

Evidence: `source-support-qualified/`, `source-support-qualified.xml`,
`source-support-ub-qualified/`, `source-support-ub-qualified.xml`,
`source-support-qualified-guard.log`, `source-support-isolation.json`,
`source-support-tinyllama-{before,after}.json`, `supported-policy-tinyllama-qualified.json`
and `source-support-qualification.json`.

## Universal initialized source dequantization and stronger qualification

`Check/PolicyW8A8Dequant.lean` now proves successful typed execution with actual runtime
scale/zero-point tensors, including i64 zero points. With initialized operands, outputs are
initialized under every floating model. This removes the unnecessary constant/positive-scale
restriction from dequantization support without changing quantization's calibration checks.
NaN/Inf floating bits count as initialized values; this result does not assert finiteness.

`Check/PolicyW8A8Defined.lean` composes source DQ with established ordinary leaf definedness
rules. `Check/PolicyW8A8Qualified.lean` checks universal initialized execution of both source
and source-derived reference on the same declared typed, initialized input domain, then
combines those obligations with the full semantics-generic SSA policy match. Acceptance also
proves initialized execution of the actual output. Support or matching cannot supply a missing
proof, and an empty function roster refuses. The driver remains explicitly provisional.

**155 tests passed, zero skipped** (257.30 seconds), including 29 new cases and 30 whole-function
DQ samples against independently constructed native MLIR. Tests cover both widths, scalar,
matrix and empty geometry, exceptional scale values, uninitialized temporaries, actual
contract/wiring mutations, unsupported calls and the separation between typed and initialized
qualification. The isolated guard audits **7976 theorems, zero violations**, with unchanged
frozen files and statement pins. Ruff, whitespace and artifact layout pass. Structure retains
the unrelated `zephyr_model.py` size-ceiling failure; generated package docs have no changes.

Retained TinyLlama still passes typed policy refinement for its **155 selected contractions**
with the recorded **captured** selection. It refuses initialized qualification. Requesting the
broader `all` selection does not match that captured output; it must not silently change the
policy denominator. Whole-model initialized execution is still unproved.

The new initialized route currently refuses quantizers, internal calls and operations with
regions. No new float law, approved recipe or full verified-pass receipt was introduced.
Concurrent BF16 work, Main, public CLI, findings and shared ledger were left untouched.

Evidence: `defined-policy-qualified/`, `defined-policy-qualified.{xml,log}`,
`defined-policy-guard.log`, `defined-policy-isolation.json`, `defined-policy-qualification.json`,
`defined-policy-tinyllama-captured-support.json` and `defined-policy-tinyllama-initialized.json`.

## Fact-aware nested source support

`Check/CalibratedProgress.lean` factors block/function progress into a reusable rulebook over
actual typing and calibration environments. Its region checker forgets every calibration fact
that depends on a newly bound block argument, preserving the evaluator's capture/shadowing
behavior. Successful SSA writes continue to invalidate facts before learning new producers.

`Check/PolicyW8A8Support.lean` now checks source Q/DQ within nested ordinary structured regions.
Each enclosing level uses the preceding proved book; a structural AST depth supplies the budget.
Captured constants and local constant/splat producers use the same proofs. Exhaustion and
unsupported operations refuse. The original duplicate block/function proof was replaced with
wrappers over the generic fact-aware progress theorem, preserving the public support claim.

**81 regression tests passed, zero skipped** (150.63 seconds), including 28 new cases. Those
cover nested depths 1–3, i32/i64 zero points, captured/local calibration, runtime DQ parameters,
shadowed constants, overwritten tensor parameters, missing captures, metadata and region
mutations. Eight independent native MLIR comparisons cover nested Q/DQ execution. These tests
continue to distinguish typed support from initialized execution. The isolated guard audits
**7994 theorems, zero violations**; frozen files and statement pins remain unchanged. Ruff,
whitespace and artifact layout pass. Structure's unrelated runtime size-ceiling failure remains.

Retained TinyLlama again passes typed policy refinement for all **155 captured selections**.
Internal function calls remain unsupported by this progress checker. Numerical-domain and
initialized quantizer proofs remain separate; no full-pass receipt or policy approval is added.
Concurrent BF16 edits, Main, public CLI, shared findings and ledger were left untouched.

Evidence: `nested-support-qualified/`, `nested-support-qualified.{xml,log}`,
`nested-support-guard.log`, `nested-support-isolation.json`, `nested-support-qualification.json`
and `nested-support-tinyllama.json`.

## Proved interpretation bridge and initialized mixed functions

`Rules/W8A8SourceFree.lean` proves exact agreement of the provisional and frozen interpreters
when the checked operation regions or every module function body contain no source Q/DQ
overrides. The module theorem includes internal calls at every fuel, external declarations,
missing symbols and stuck/UB/poison outcomes, with the identical floating model and context.
It supplies explicit support/definedness/refinement transfer theorems. Equality of executions
is kept separate from evidence of successful initialized execution.

`Check/DefinedProjection.lean` provides a reusable adapter from refined definedness books:
unknown extra input facts start empty, and returned extra facts are erased. Region checking
still derives its own index bounds. `PolicyW8A8Defined` uses that adapter and the local bridge
for ordinary operations, permitting source DQ followed by initialized nested arithmetic.
It also checks the whole-module bridge before reusing the existing full definedness analysis.
No source calibration, bound or numerical-domain hypothesis is supplied by an output hint.

The ordinary floating matmul baseline now passes universal initialized qualification; its
actual W8A8 output still refuses initialized qualification because the policy quantizer's
numerical domain has not been proved. This separates a defined baseline from an unsupported
quantized reference. Mixed DQ/ordinary-region cases pass at depths 1 and 3, while out-of-bounds
reads refuse. Source Q/DQ within regions remains typed-support-only; initialized execution of
those source-containing regions and general internal-call support are still open.

**101 regression tests passed, zero skipped** (221.08 seconds), including 20 new bridge cases.
The isolated guard audits **8045 theorems, zero violations**, with unchanged frozen files and
statement pins. The shared tree's lock-only check also passes. Ruff, whitespace and artifact
layout pass; structure retains the unrelated runtime size-ceiling failure, and regenerated
package docs have no tracked changes. Across these three checkpoints, 203 distinct regression
cases have passed; overlapping runs are not additional coverage.

The retained TinyLlama final handoff export passes the interpretation bridge; its earlier
source-Q/DQ boundary correctly refuses it. This is interpretation agreement only, not a
suffix-refinement, initialized-handoff or full-pass result. The prior 155-site typed policy
refinement remains recorded separately. No frozen dispatcher, public receipt driver, compiler
pass, other agent's active files, shared ledger or findings were edited.

A development Lean interpreter process exited 139 once. The exact request, isolated test and
final 101-case run passed afterward; no retry or exception suppression was added. The original
failure and replay logs are retained. Earlier interrupted qualification attempts are retained
but are not used as passing evidence.

Evidence: `source-free-qualified-v3/`, `source-free-qualified-v3.{xml,log}`,
`source-free-guard-v3.log`, `source-free-isolation-v3.json`, `source-free-qualification.json`,
`source-free-tinyllama-{source,handoff}.json` and `source-free-{development,poison-repro}.log`.

Next priorities remain universal initialized source/dynamic quantization on a justified
numerical domain; source-containing regions and internal-call contracts; independent recipe
review and semantic promotion; then public receipt/pipeline integration and actual suffix
qualification. Existing MLIR/LLVM passes remain outside the Merlin-owned verification scope.

## Initialized source dequantization inside generic regions

`Check/DefinedGeneric.lean` generalizes the proved structured initialization rules over an
explicit operation interpretation and arbitrary refinement-fact model. Reusing the frozen
generic executor requires a proved agreement law on the operation head for arbitrary body
closures. Bodies execute under the chosen interpretation. Initialized destinations and complete
overwrites retain separate mapping, shape, body-initialization and output-coverage obligations.

`PolicyW8A8Defined` now keeps refinement facts through whole-function execution and recursively
checks generic bodies with source DQ. Captured runtime parameters, block-argument shadowing and
local index bounds use the existing proved refined walker. This closes the previous initialized
source-DQ region refusal and preserves bounds across top-level operations in mixed functions.
Source quantization and internal function calls still refuse initialized qualification.

**127 regression tests passed, zero skipped** (270.06 seconds), including 26 new cases. They
cover both zero-point widths, depths 1–3, initialized source DQ against native MLIR, retained
bounds, poison in each operand, uninitialized yielded destinations, metadata/region mutations
and out-of-bounds reads. The isolated audit checks **8065 theorems, zero violations**, with
unchanged frozen files and statement pins. Ruff, whitespace and artifact layout pass. Structure
retains the unrelated runtime size-ceiling failure. Other agents' active files remain untouched.

The theorems quantify over every floating model; this checkpoint adds no numerical laws or
new policy assumption. Full W8A8 initialized execution remains unproved. Work on explicit,
conditional quantizer numerical preconditions is being kept separate from this qualification.

Evidence: `nested-defined-qualified/`, `nested-defined-qualified.{xml,log}`,
`nested-defined-guard.log`, `nested-defined-isolation.json`, `nested-defined-qualification.json`.

## Conditional numerical domain for source quantization

`Rules/W8A8FiniteLaws.lean` states primitive floating-operation premises explicitly and derives
the source quantizer's exact intermediate non-NaN obligation. Finite data and a positive normal
f32 scale suffice; overflow of a multiplication to infinity is permitted. No axiom or model
instance is introduced. A proved counterexample shows the current arbitrary floating-model
interface does not imply these premises, so they cannot be silently treated as established.

`PolicyW8A8FiniteDomain.check_sound` proves that a guard on actual bits suffices for initialized
source quantization **conditional on those premises**. It performs no floating arithmetic and
is conservative: some refused infinite-data/subnormal-scale instances still clip successfully.
`PolicyW8A8FiniteParameters.check_sound` separately binds the scale class to actual calibration
SSA producers. All **399 retained TinyLlama Q/DQ scales**, including **89 source quantizers**,
pass this scale guard. Activation finiteness is a separate, unproved whole-function obligation.
The [provisional packet](policy-review/provisional/README.md) records the premises and review
boundary in detail. Neither driver metadata nor passing samples discharge these proof premises.

**83 regression tests passed, zero skipped** (160.21 seconds), including 20 new cases and 48
new native MLIR source-quantizer comparisons, also compared with pinned PyTorch. Cases include
i32/i64 zero points, scalar/matrix/empty tensors, normal/subnormal scale boundaries, NaN/Inf,
the tiny-scale/zero-input poison counterexample and actual/unknown calibration producers.
The isolated guard audits **8100 theorems, zero violations**, with unchanged frozen files and
statement pins. Ruff, whitespace and artifact layout pass; regenerated package docs are
unchanged. Structure retains the unrelated runtime module size-ceiling failure.

A development interpreter process again exited 139. The exact request and 20 debugger runs
passed, followed by the clean 83-case qualification above. Its cause remains unresolved; the
original failure and diagnostic logs are retained. Tests contain no automatic retry or failure
suppression, and the clean run does not establish that the runtime defect has been fixed.

The stronger whole-function checker still refuses source quantization. Remaining work includes
propagating justified numerical preconditions through actual source/reference functions,
qualifying the floating interpretation, internal-call contracts, independent policy review,
and public receipt/pipeline integration. This checkpoint does not change the production pass,
frozen semantics, public receipt binary or other agents' active files.

Evidence: `finite-domain-qualified-v1/`, `finite-domain-qualified-v1.{xml,log}`,
`finite-domain-guard-v1.log`, `finite-domain-isolation-v1.json`,
`finite-domain-qualification-v1.json`, `finite-domain-tinyllama-parameters.json`,
`finite-domain-runtime-development.log` and `finite-domain-gdb-qualified-repro.log`.

## Conditional initialized functions and actual reused-byte W8A8 rewrites

`Check/DefinedContext.lean` supports explicit context/call conditions and per-value refinements.
The function theorem requires a proof of those conditions for the actual module evaluator and
call fuel. `FiniteFacts` and `FiniteProducers` derive numerical classes from actual constant bits,
preserve them through splats, and replace them on SSA writes. Ordinary initialized arithmetic
does not establish a finite or positive-normal result: delegated rules erase those classes.

`PolicyW8A8FiniteQuantize` proves initialization for actual recognized source quantizers from
those operand facts. `PolicyW8A8FiniteDefined` composes it through complete functions under the
explicit finite-f32 input domain and the primitive model laws. Both zero-point widths, normal
scale endpoints, scalar/matrix/empty tensors and Q→DQ functions pass. The model-domain witness
is synthetic; no concrete IEEE or host-model law instance is claimed. NaN input remains an
excluded counterexample, tested separately rather than admitted by a misleading all-input claim.

`PolicyW8A8FiniteQualified` now combines source/reference initialized execution on that same
domain with the existing source-derived policy expansion and whole-module SSA validation.
Actual Merlin contraction rewrites reusing both calibrated byte operands pass this route,
including source quantization prefixes. Selected dynamic policy quantizers still refuse:
their numerical proof is not supplied by the source quantizer theorem. The policy and floating
interpretation remain provisional, and public verified receipts remain unavailable.

The retained TinyLlama program still refuses conditional initialized qualification for both
source and reference. Its first source refusal is an embedding `linalg.generic` after 776 prefix
operations. The actual entry token operand is `tensor<1x8xi64>` and the gathered table is
`tensor<32000x2048xf32>`; the type alone does not establish valid token indices. Its token-range
contract, structured loop bounds and subsequent numerical obligations remain explicit. This
diagnostic is not a compiler counterexample or an expansion into proving frontend/upstream passes.

The [provisional packet](policy-review/provisional/README.md) records the exact context/input
domains and boundaries. Source/source-reference proof composition now works for accepted real
rewrites; general dynamic quantization, numerical propagation, calls, independent review and
public receipt/suffix integration remain unfinished.

**218 regression tests passed, zero skipped** (354.24 seconds), including 44 new source-function
cases and 12 new policy cases. The function tests include 48 native MLIR comparisons. Two
actual transformed-program executions also match an independent ordered NumPy quantize/int32
MAC/scale recipe. Positive policy cases all run the real Merlin pass and assert one selected
rewrite; no unchanged function is presented as changed-site coverage. Mutation tests retain
valid matrix shapes when checking swapped contraction operands. The isolated guard audits
**8201 theorems, zero violations**, with unchanged frozen files, pins and public receipt binary.
Ruff, formatting, whitespace, artifact layout and the shared-tree lock check pass. Package-doc
regeneration is unchanged; structure retains the unrelated runtime module size-ceiling failure.
The earlier intermittent interpreter crash remains unresolved; this qualification added no retry
and observed no process crash.

Evidence: `finite-functions-qualified-v1/`, `finite-functions-qualified-v1.{xml,log}`,
`finite-functions-guard-v1.log`, `finite-functions-isolation-v1.json`,
`finite-functions-qualification-v1.json`, `finite-functions-tinyllama-policy.json` and
`finite-functions-tinyllama-first-refusal-v3.json`. The accompanying diagnostic source is
`finite-functions-first-refusal-v3.lean`; it reports checker refusals and is not a proof or receipt.

## Checked internal calls and inferred numerical postconditions

`DefinedContext` now has one inference engine for operation/region result facts; its existing
Boolean checks project the inferred types. Soundness preserves the actual result refinements,
successful execution and initialization. Existing public claim statements retain their domains.
`FiniteImplication` checks the caller's input facts against the callee's requirements, including
types, initialization and the direction of numerical weakening. A finite fact cannot prove a
normal scale; a normal fact can satisfy a finite-input requirement.

`DefinedCallContracts` checks actual flat callees, metadata, regions, result types and runtime
result arity. Its contract premise is closed by `PolicyW8A8Called`: stage zero has no contracts;
each later stage checks actual function bodies using only the preceding stage's contracts.
The soundness theorem matches stage fuel to `evalFunc`, with final fuel taken from the actual
module's function count. No declaration, recursion cycle or provenance annotation supplies a
contract. Inferred finite/normal results cross the call boundary with their proofs intact.

`PolicyW8A8FiniteQualified` now tries that route alongside direct initialized checking, using
the same explicit finite-f32 input and primitive-model-law domain. This closes the previous
top-level internal-call refusal for admitted callees. The direct checker remains a no-call
fragment. Source Q and calls inside structured regions still require further conditional
region proofs; the new route does not silently assume those bodies valid.

**197 regression tests passed, zero skipped** (334.98 seconds), including **48 new call cases**.
They cover both zero-point widths, scalar/matrix/empty tensors, quantize/Q→DQ call chains,
definition order, inferred numerical returns, missing/malformed callees, invalid signatures,
recursion, external declarations, poison, UB, nonfinite arguments and forged `prov.verified`
annotations. Four call-chain cases include **32 native MLIR comparisons**. Two real W8A8
rewrites inside callees, reached through three call levels, also match the independent ordered
NumPy recipe. The isolated guard audits **8260 theorems, zero violations**. Frozen files, pins
and the public receipt binary remain unchanged. Ruff, formatting, whitespace, artifact layout
and the lock check pass; regenerated package docs are unchanged. Structure retains the
unrelated runtime module size-ceiling failure. No retry or crash suppression was introduced.

The retained TinyLlama source/reference pair still refuses initialized qualification. Its
previous embedding-input contract and later numerical obligations remain unresolved; call
support cannot establish missing index ranges. Independent recipe/context/domain review,
dynamic quantizer proofs, conditional structured regions and public receipt/suffix integration
remain required. No full-pass receipt, new float axiom or production-pass change is added.

Evidence: `called-qualified-v1/`, `called-qualified-v1.{xml,log}`, `called-guard-v1.log`,
`called-isolation-v1.json`, `called-qualification-v1.json` and `called-tinyllama-policy.json`.

## Conditional numerical facts through actual structured regions

`DefinedElementwise` proves that bounded tensor reads, writes and canonicalization preserve
elementwise refinements. `DefinedContextLoops` carries explicit context/call premises through
the actual loop-coordinate updates and ordered reductions. A full identity overwrite proves
its initialized numerical postcondition from actual body execution, with an explicitly
uninitialized old destination. `DefinedContextFill` checks the actual fill region as well.

`DefinedContextGeneric` infers overwrite result facts and checks initialized reduction
invariants through a proved weakening relation. `FiniteStructured` instantiates this for
finite/normal facts, conservatively widening normal reduction initializers to finite. The
direct and checked-call W8A8 books now use those conditional rules recursively. Quantizers,
captured calibration values and closed internal-call contracts can appear in nested regions;
SSA rebinding still replaces facts, and ordinary initialized arithmetic cannot imply finiteness.
This closes the preceding structured-region gap without changing either existing theorem
domain or adding verification of any upstream optimization pass.

The separate `W8A8DynamicLaws` module proves abs/max finiteness, guarded-scale normality,
reciprocal finiteness and initialized ordered byte quantization under three additional explicit
primitive premises. Its synthetic witness proves that those premises are jointly satisfiable;
no concrete IEEE/host model is established. These lemmas remain separate from the existing
source-quantizer context. Connecting them to generated dynamic-policy SSA producers and complete
rewrites is the next implementation obligation, not a completed coverage claim.

**246 relevant regression tests passed, zero skipped** (427.51 seconds), including **49 new
structured-region cases**. Eight cases contribute **64 native MLIR comparisons**, covering
quantization after a copy, inside a parallel region, inside a nested reduction and through
an internal call in a nested body. A real `lower_contraction_int8` rewrite after a proved finite
copy also qualifies conditionally and matches the independent ordered NumPy recipe. Refusal
tests cover uninitialized destination use, arithmetic that lacks finiteness evidence, bad
calibration, maps, body types, metadata, captured NaN and actual SSA overwrites.

The isolated guard audits **8357 theorems, zero violations**, over **333 snapshot source files**.
Frozen files and statement pins are unchanged. Ruff, formatting, whitespace, artifact layout
and the shared-tree lock check pass; package-doc regeneration is unchanged. Structure retains
the unrelated runtime module size-ceiling failure. The isolated public binary remains the
earlier qualified binary; concurrent shared-tree driver/binary work is excluded and its
observed digest is recorded separately. No public receipt integration or production-pass edit
is included here. The earlier interpreter crash remains unresolved; this qualification used
no retry/failure suppression and observed no process crash.

The retained TinyLlama source/reference pair still refuses initialized qualification; input
token ranges and later numerical obligations remain unproved. Remaining work is generated
dynamic SSA safety, the concrete floating interpretation and input contracts, independent
recipe/domain review, public receipt/pipeline composition and qualification of every selected
site. Recorded incorrect/unsupported pass cases must be resolved or refused. The policy is
still provisional and `whole_pass_verified` remains false.

Evidence: `structured-qualified-v1/`, `structured-qualified-v1.{xml,log}`,
`structured-guard-v1.log`, `structured-isolation-v1.json`, `structured-qualification-v1.json`
and `structured-tinyllama-policy.json`. The [provisional packet](policy-review/provisional/README.md)
records the unchanged source domain and separate draft dynamic premises.

## Initialized dynamic W8A8 policy refinement

The dynamic numerical lemmas are now integrated through actual generated SSA and complete
source/reference functions. `FloatFacts` retains finite/normal classes and exact f32 literal
evidence, with proved projection/embedding between refinement models. Constants are learned
from actual execution; splats preserve their elementwise facts. Capture lookup reads the
current binding for ids used in the small scalar region, so earlier overwritten constants
cannot supply a denominator or clamp bound.

`DefinedRegions` allows a proved region contract alongside ordinary operation inference.
`PolicyW8A8DynamicBodies` and `PolicyW8A8DynamicCaptured` apply the existing exact scalar-body
checks and prove finite abs/max results, positive-normal scales, finite reciprocals and
initialized byte conversions. The structured executor carries these postconditions through
actual checked mappings, reductions and overwrites. A source-free body bridge establishes
agreement of interpretations before reusing the frozen-semantics component proofs.

`PolicyW8A8DynamicDefined.check_sound` now proves initialized complete functions containing
the emitted dynamic sequence. `DefinedClosedCalls` generalizes the existing staged call
proof without changing its domain; both source-only and dynamic books use it, with actual
body-derived contracts and the evaluator's module fuel. `PolicyInitialized` provides shared
whole-module policy composition. `PolicyW8A8DynamicQualified.check_sound` requires initialized
source/reference execution on one domain plus the complete source-derived policy match, and
proves strict policy refinement and initialized actual output for every admitted function.

The **new route** explicitly requires the seven source laws plus the three separate dynamic
laws for the base of `W8A8ByteModel.extend`. Its input domain remains typed, initialized
arguments with finite f32 elements. Earlier source-only theorem domains are unchanged; those
weaker-context checkers still refuse dynamic quantizers. No concrete IEEE/host law instance
has been established and the synthetic witness is only a non-vacuity proof. The new route is
conditional qualification of the draft policy, not independent policy approval or promotion
of frozen semantics/public receipts.

**298 relevant regression tests passed, zero skipped** (477.19 seconds), including **52 new
dynamic cases**. Positive checks run the real pass and assert one selected rewrite. Cases cover
canonical matrices and empty M/K/N, signed-byte/f32 operands in both positions and source add
orders, per-tensor Q/DQ reuse with one dynamic operand and i32/i64 zero points, and internal
callees through three call levels. **50 new qualified observations** compare the exact actual
output artifact under Lean, native MLIR and the independent ordered NumPy recipe. Another
**20** compare captured-source-plus-dynamic execution with the ordered recipe. These are
sampled semantic-fidelity checks, not proofs of a concrete floating-model instance.

Negative checks reject wrong denominators, floors, reciprocals, rounding, clamp values,
uninitialized destination use, stale/NaN literals and malformed region/type/metadata data.
Numerically initialized outputs with incorrect contraction wiring, scale order, reduction axes
or accumulator values still fail policy matching. The recorded F11 square-as-product source
discrepancy remains refused, and dynamic initialization cannot widen captured-scope selection.

The isolated guard audits **8518 theorems, zero violations**, over **347 snapshot source files**.
Frozen files and statement pins remain unchanged. Ruff, formatting, whitespace, artifact
layout and shared-tree lock checks pass; regenerated package docs are unchanged. Structure
retains the unrelated runtime module size-ceiling failure. Concurrent BF16/frontend and public
driver work is excluded from this qualification; the isolated and observed shared-tree binary
digests are recorded separately. No production pass is edited. The previous intermittent
interpreter crash remains unresolved; this clean qualification used no retry or suppression.

The retained TinyLlama source/reference pair still refuses the stronger route. Its token-index
domain and later numerical obligations remain unproved; supporting dynamic stages cannot
establish those facts. Remaining work includes concrete semantic/domain admission and
independent recipe review, source numerical/index contracts, qualification of every selected
retained-model site, and public receipt/pipeline/suffix integration. Recorded discrepancies
must be resolved or refused without shrinking the selected denominator. Full-pass verification
and public verified receipts remain unavailable.

Evidence: `dynamic-qualified-v1/`, `dynamic-qualified-v1.{xml,log}`, `dynamic-guard-v1.log`,
`dynamic-isolation-v1.json`, `dynamic-qualification-v1.json` and `dynamic-tinyllama-policy.json`.
The [provisional packet](policy-review/provisional/README.md) states the new stronger context
and its precise relation to the earlier claims.

## Conditional dynamic qualification through pipeline boundaries

`PolicyW8A8DynamicChain.check_sound` now composes the initialized W8A8 policy result through
every actual downstream boundary, retaining the original source, reference, numerical context
and finite-input domain. It proves initialized final output as well as strict refinement of
the policy reference. Complete-module SSA equivalence uses the additive source semantics
directly. Other existing suffix rules are reused only after both complete modules pass the
proved source-free interpretation check; unchanged Q/DQ operations cannot hide that obligation.

The existing public draft commands now include a separate `conditional_numerical` report.
It names the new theorem and its stronger numerical/input context, records source/reference
initialization and each downstream result, and keeps policy/model approval false. The Python
adapter validates the exact theorem, scope, width, domain, selected function roster and suffix
coverage, and records the actual theorem's axiom audit inside the report bound to the exact
program exports and checker binary. Earlier draft claim identities and domains are preserved;
no full-pass verified receipt is issued.

**193 relevant regression tests passed, zero skipped** (286.86 seconds), including **36 new
integration cases**. Real pass outputs cover ordinary/empty matrices, mixed signed-byte/f32
operands, captured reuse with dynamic operands and internal calls. Tests compose actual Merlin
preprocessing boundaries, accept consistent SSA renaming with/without source quantization,
refuse missing interpretation bridges, detect first/middle/last suffix mutations, leave suffixes
unmeasured after failed policy qualification, and reject malformed or promoted numerical
reports. Existing timeout, byte continuity, handoff and freshness checks also pass. The run
repeats the **50** qualified Lean/native/ordered-NumPy observations and **20** captured-source
observations from the preceding batch; these are sampled fidelity evidence, not new IEEE proofs.

The isolated guard audits **8543 theorems, zero violations**, over **349 snapshot source files**.
The isolated public checker was rebuilt before qualification and stayed fixed throughout the
tests. Concurrent BF16/frontend and public Main/CLI work remains excluded; the shared-tree binary
is recorded separately. Frozen files and statement pins are unchanged. Ruff, formatting,
whitespace, artifact layout, source access/name gates and regenerated package docs pass.
Structure retains the unrelated `zephyr_model.py` size-ceiling failure. The earlier intermittent
interpreter crash remains unresolved; this qualification used no retry or suppression.

Still outstanding are independent policy/domain review, concrete numerical-model admission,
source index/numerical contracts and qualification of every selected retained-model site.
The retained TinyLlama input-bound/numerical refusal and recorded production discrepancies are
not resolved by pipeline integration. Typed suffixes retaining Q/DQ also require an additive
semantic proof beyond the current SSA route. Full-pass verification and frozen receipt promotion
remain unavailable.

Evidence: `dynamic-public-qualified-v1.{xml,log}`, `dynamic-public-guard-v1.log`,
`dynamic-public-isolation-v1.json`, `dynamic-public-freeze-v1.json` and
`dynamic-public-qualification-v1.json`. The provisional review packet records the exact new
reporting route and its unchanged review requirements.
