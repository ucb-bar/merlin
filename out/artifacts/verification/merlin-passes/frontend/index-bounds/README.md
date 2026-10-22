# Actual attention-mask bounds: additive candidate

This extends the inactive runtime checkpoint. It does not change the approved f32 policy,
compiler passes, W8A8, protected semantics, production lakefile, A2 or SPEC_LOCK. The older
`full-model/` qualification is historical; this packet records the new interpretation.

## New universal proofs

- `FrontendRangeDefined.check_sound` connects the bounds-aware checker to the existing MLIR
  evaluator, lifted conservatively to the inactive runtime interpretation. Declared inputs are
  typed and initialized. They receive **no input intervals**. The checker derives each interval
  from the actual SSA producer, and requires actual signed nonnegative index bounds at reads.
- `FrontendRangeProjection.check_sound` consumes a full authoritative module and derives an
  actual contiguous operation slice, its free captures, their types and the observation return.
  Acceptance proves successful execution and initialized typed outputs **from that slice's
  declared initialized inputs**. It does not assert execution of any omitted operations.
- SmolVLA's actual operations **122 through 156 inclusive**, zero-based, are accepted. Both
  index tensors at the attention-mask gather are bounded by their actual coordinate generators,
  casts, constants, integer arithmetic, splats, broadcasts and reshapes. The slice contains 35
  top-level operations, including all nested operations, and takes an initialized boolean mask.
- `FrontendMaskRange.check_sound` strengthens complete actual selection-cut acceptance with an
  interval postcondition: a shared value/seed element bound survives the compact gather,
  scatter and final reshape. Inputs may have arbitrary inaccessible storage. Empty selections
  and false-mask seed positions are covered.

The range model uses actual signed bit-vector decoding. Addition and multiplication retain an
interval only when the entire signed result range fits. Equal-width signed index casts preserve
bits. `linalg.index` receives the actual checked iteration extent, rather than only a nesting
depth. Tensor-element intervals are passed to scalar body arguments through proved in-bounds
indexing maps. A zero-sized structured output is initialized because there are provably no body
calls; its element bounds are not invented. Unknown tensor indices, unsafe integer ranges,
unsupported flags and uninitialized output reads refuse acceptance.

All theorem names are under `MerlinVerify.Rules`. The new range checker is independent of a
floating-point numerical law. The specialized selection theorem uses the approved direct f32
instance to discharge its comparison premise. Execution samples remain distinct from these
universal theorems.

## Receipt integration

`verify_candidate(..., bounds_slices=((122, 35),))` checks the original complete f32 repair,
actual initialized initial segment, ordered suffixes, and the requested actual slice. The receipt
records the slice's separate cut-input contract. A refused requested slice refuses the combined
receipt; it cannot be omitted silently. Replay re-exports and verifies the actual MLIR, repeats
round trips and theorem audits, and reruns the actual Lean checkers. Sources, compiled objects,
checker binary, specification/approval and ordered artifacts remain bound to the receipt.

The source-policy claim is retained exactly. No cut claim is promoted to whole-model execution,
and no frozen-model receipt is used to discharge a candidate suffix.

## Qualification and reproduction

Use the tools and environment in the preceding `full-model/README.md`. Run from this checkout:

```sh
cd merlin/lean
lake build MerlinVerify merlin-lean merlin-lean-audit
lake -f FrontendRuntime.lakefile.toml build merlin-lean-frontend-runtime
cd ../..
python -m pytest merlin/tests/ir/test_lean_frontend_ranges.py
python out/artifacts/verification/merlin-passes/frontend/index-bounds/qualify.py
```

The tests run the authoritative compiler verifier **before** Lean for every mutation counted as
MLIR-valid: positive/negative row errors, positive/negative column offsets, excessive scaling,
unsupported overflow flags, uninitialized output uses and a changed actual row generator.
Unknown tensor-valued input indices are refused explicitly. Actual full-module projection and
receipt replay are exercised, including refusal to attach the still-unproved position read.

Native comparisons cover the actual attention slice plus independent sizes 0, 1 and 8, each
with alternating, all-true and all-false masks. An independent 4-by-4 integer position grid
also exercises nontrivial multiplication, broadcast addition, views and a bounded gather.
These tests check interpretation fidelity. The
existing f32 NaN/infinity/signed-zero/fraction and pinned PyTorch **2.10.0+cpu** suites are retained.
Exact measured versions, counts and skips are recorded in `validation.json` and `qualification.json`.

## Remaining technical work

The proved initial segment still stops after **116** top-level operations. The independent
attention slice does not bridge the position-embedding read at operation 117. The following
producer proof can proceed without a new floating comparison law:

1. Operations 74 and 78 reduce 31 comparison results per output, selecting integer 0 or 1.
   Prove their actual per-output reduction counts lie in `[0,32)`. A comparison may produce
   either Boolean; no ordering, monotonicity or finite-input premise is needed for this bound.
2. The existing new integer/view rules can then transport these ranges through operation 84
   (multiply by 32), operation 88 (add), and operations 89/90 (views), obtaining `[0,1024)`.
3. Use the selection interval postcondition, including its zero seed, to bound output 1064.
   The actual embedding read then needs this row bound and the checked column extent 768.
4. Compose the resulting execution through the real intermediate SSA state to the already
   checked attention slice. Do not skip the embedding operations or assume its output.

Other independent technical work remains: exact non-alpha candidate suffix rules, complete
remaining-operation coverage, and initialized full-model returns. Token input bounds and the
302 non-boolean float-to-i8 conversions have distinct domain/interpretation obligations, detailed
in `../full-model/remaining-obligations.md`. They are not discharged by the f32 bool theorem.
Runtime admission and proposed additional contracts await human review; they are not new axioms.
The list is an inventory of known obligations, not a completeness claim for the full model.

## Measured result

Implementation: `1820d2038520660d1bfa49fc361a9a2508b51466`, based on
`4d48281abc2d183d363e0841024855651e582ca8`. The additive `integration.patch` applies
against that base using an isolated Git index. It changes no protected/compiler/W8A8 file.

- **90 tests passed**, zero failures/errors/skips, 971.12 seconds; 16 new cases.
- **9,826 theorems audited**, zero violations; the new claims use only the established standard logical axioms.
- Library/checker/auditor build: 1,483 jobs; separate candidate build: 1,422 jobs.
- Fresh actual-boundary receipt, independent attention-slice claim and replay: passed.
- **15 new native samples**, zero skips. Existing 20 selection-native samples, 16 pinned
  PyTorch selection samples and f32 IEEE edge suites also passed in the regression run.
- Oracle: PyTorch **2.10.0+cpu**, NumPy 2.5.4, Python 3.12.14. Historical newer-version
  results remain supplemental. Native LLVM: 23.0.0git at
  `a47bddccec30255619bb8c37fa59700e661d4e66`; authoritative torch-mlir: 20261001.
- Full lock/statement guard: the same three prior admission differences; no new failure.
  Structure gate: the same unrelated `zephyr_model.py` ceiling failure. Neither is green.

Runtime/claim admission remains pending human review. The f32 policy/direct-instance approval
is retained unchanged. Native tests qualify the proof engine's interpretation of upstream
operations; they do not prove LLVM implementation correctness.
