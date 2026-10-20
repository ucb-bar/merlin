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

The scalar-body checker does not establish enclosing tensor coordinates, actual reduction length,
zero initialization, complete quantization groups/axes, captured-weight policy, tensor requantization,
source definedness, or the correspondence between complete before/after programs. No C5 whole-pass
receipt is issued. Independently reviewed Lean policy semantics, site/chain integration, all policy
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

The zero-filled destination remains an explicit theorem input. Actual initializer recognition,
source operand/scale correspondence and complete policy integration remain open; no whole-pass
receipt is issued. The adapter checks actual SSA operand/result types and passes actual maps,
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
