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
zero initialization, quantization scales, rounding/clamping, captured-weight policy, requantization,
source definedness, or the correspondence between complete before/after programs. No C5 whole-pass
receipt is issued. Independently reviewed Lean policy semantics, site/chain integration, all policy
mutations and qualification on the chosen model remain required. The model/configuration has not
been selected. The implementer has read the pass, so a recipe reconstructed from it cannot honestly
be labeled an independently authored specification. No new assumption or float law was added.
