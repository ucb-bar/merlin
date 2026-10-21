# Dynamic mask-selection runtime candidate

Status: additive candidate, **not admitted or activated**. The user authorized implementing
this candidate while keeping protected definitions and locks unchanged. Human review of the
runtime interpretation and integration admission remain separate. A2's f32 bool policy and
its direct IEEE comparison instance already have the recorded, scoped user approval.

## What this changes

This extends the Lean proof engine's interpretation of upstream MLIR operations. It does
not require or claim verification of LLVM's implementations or lowering of those operations.
Native MLIR execution is a fidelity oracle, not a formal implementation proof.

The frozen model has confirmed gaps for dynamic `tensor.empty`, `scf.if` and `scf.for` in
this capture. Those observations are **not an exhaustive list of full-model obligations**.
The candidate only establishes the explicitly stated cut contract below. It does not prove
that the surrounding model supplies that contract or otherwise executes successfully.

The candidate lives in `Rules/FrontendMaskRuntime.lean` and is an explicit `OpSem` parameter.
The existing dispatcher, typing definitions, compiler pass, W8A8 code, real SPEC_LOCK and
real guard are unchanged. Existing pipeline receipts still name their original interpretation.
A frozen-model suffix receipt cannot discharge the new interpretation's suffix premise.

## Proposed operation interpretation

- **Runtime tensor shapes:** a shape pattern retains every static dimension and consumes
  exactly one initialized, nonnegative signed index word for each dynamic dimension, in
  order. Wrong width, missing/extra size operands and unsupported result encodings refuse.
- **Dynamic `tensor.empty`:** construct a functional tensor with the resolved shape and
  poison contents. Allocation never establishes initialized elements. No physical allocation
  or resource-exhaustion behavior is modeled, consistently with existing static tensors.
- **`scf.if`:** an initialized scalar i1 chooses exactly one branch. The existing evaluator
  performs region arguments, SSA reads/bindings and yields. Uninitialized or incorrectly
  typed controls return `stuck`; this candidate does not propose a new poison-to-UB rule.
- **`scf.for`:** decode controls at the context's index width as nonnegative signed words;
  require a positive stride. Compute the trip count with mathematical natural arithmetic,
  check the explicit iteration budget, and check that even the final counter advancement
  fits the signed word. Each iteration passes IV followed by carried values and checks
  yield arity. Structural recursion consumes one iteration, proving termination. The scoped
  program checker requires the captured index64 forms and matching tensor/index results.
- **Other operations:** delegate to the existing frozen dispatcher. Unknown attributes and
  unsupported SCF forms remain refusals. `unsignedCmp` is explicitly unsupported.

The supported subset is deliberately conservative: negative loop controls, unsupported
widths, excessive budgets and signed advancement overflow refuse. They are not reclassified
as undefined behavior. Upstream operation specifications are in the pinned LLVM source
`a47bddccec30255619bb8c37fa59700e661d4e66` (`SCFOps.td`, `TensorOps.td`, `ArithOps.td`).
[SCF documentation](https://mlir.llvm.org/docs/Dialects/SCFDialect/) and
[tensor documentation](https://mlir.llvm.org/docs/Dialects/TensorOps/) are useful references.
Current online SCF wording makes the final advancement constraint explicit; the pinned
source lacks that newer wording. The nonnegative, bounded, no-overflow subset used here
avoids relying on that textual difference.

## Exact cut and input contract

The captured selection has 26 top-level operations including its return, and 20 nested
operations: 46 operations plus the function wrapper. The function exposes three cut inputs:

| Input | Captured type | Required meaning |
|---|---|---|
| Mask | `tensor<1x32x32xi1>` | Every element is an initialized boolean |
| Values | `tensor<1x1024xi64>` | Every element is an initialized i64 word |
| Seed | `tensor<1x1024xi64>` | Every element is an initialized i64 word |

These are cut assumptions, not conclusions about upstream SmolVLA execution. The two
flattening chains read the same mask root. The first loop compacts selected values into
an exact-count dynamic tensor. The second scatters them back at selected original indices;
unselected positions retain the seed. It is not a second gather.

The input representation uses canonical row-major tensors and arbitrary boolean/i64 word
functions. It covers every initialized bit pattern at these shapes. There is no assumption
that selected values or the seed contain zeros. The count is obtained by actual MLIR
`extui`, `linalg.reduce`, scalar extraction and `index_cast` execution, not an input hint.

## Formal obligations

The proof artifacts cover:

1. Shape resolution proves initialized sizes, correct widths, nonnegativity and exact arity.
2. The actual count-provider AST computes the initialized mathematical number of true mask
   entries. Its provenance attributes are accounted for through checked semantic fields.
3. Actual gather/scatter regions are checked for operand identities, static types, flags,
   nested branches, SSA freshness, result definitions and every use.
4. Loop invariants establish input reads in bounds, compact reads/writes below capacity,
   initialized branch conditions, initialized compact output, and initialized scatter output.
   The all-false case has a zero-sized compact tensor and makes no compact reads.
5. Finite iteration and representable final advancement establish termination without
   counter overflow. Selected-position counters are bounded by the exact count, at most 1024.
6. The complete captured cut is checked against the concrete AST and connected to the existing
   Lean evaluator, including the count prefix, both reshape chains and final returned values.
7. `FrontendBoolSimulationWith.check_sound_candidate` preserves the accepted f32 A2 repair
   theorem over actual before/after programs under the extension. `FrontendBoolCandidateChain.compose`
   composes an explicitly proved suffix under the same interpretation with that repair policy.
   It does not assert preservation of the original potentially poison-producing cast program.

General component theorems cover finite masks of arbitrary length within their stated bounds.
The whole-cut theorem/checker is scoped to this captured 1024-element program and its exact
signature. Independent synthetic sizes qualify the general interpreter and components; they
do not silently broaden the whole-cut theorem's coverage.

## Floating point and review boundaries

No floating format is added. The approved f32 direct comparison instance proves UNE-to-positive-zero
for all f32 encodings using the existing IEEE unpack/comparison definitions, including NaNs.
Poison is handled by the operation evaluator. No axiom or native sample supplies that law.
The historical host-conversion bridge remains unused and unproved. The repair covers the
eligible signed and unsigned f32-to-i1 cases; f16, bf16 and f64 remain outside this checkpoint.

Runtime semantics and lock/claim admission await review. Formal theorems are statements about
the declared interpretation, not human approval of it. The separate A2 approval is not extended
to other A2 contracts or to this runtime candidate. Full-model input-definedness, support,
other numerical interpretations and downstream verification remain separate obligations;
this document does not claim that list is exhaustive.

Before activation, admission must bind the **bodies** of the runtime interpretation and source
policy, the approved f32 instance, the iteration-budget/index-width configuration, and the
specification to the receipt/chain interpretation identity. Pinning only elaborated theorem
statements is insufficient when they refer to definitions in the currently unprotected `Rules/`
directory. The candidate's qualification hashes make those bodies reviewable; they do not
silently activate a new lock policy. Frozen-interpretation receipts remain inapplicable to a
suffix proof under this candidate.

## Qualification and concrete patch

Implementation commit: **`0843cf2eb`**, based on `626b97dd43ed055ff2f9dbaf02b371837ae25cd2`.
[`runtime-candidate/integration.patch`](runtime-candidate/integration.patch) contains the exact
additive implementation and tests. Its application was checked against that base using an
isolated Git index. It does not activate the interpretation or apply a lock admission.
The branch can be integrated by reviewing/cherry-picking its coherent commits; do not merge
into the user's main checkout during this task.

- Full library, checker and audit build: **1457 jobs passed**.
- Relevant frontend, runtime-gap, chain, guard, pinned-oracle and native suites: **62 passed,
  0 failed, 0 skipped**. Nineteen cases belong to the new runtime suite.
- Whole-library audit: **9005 theorems, zero violations**. Public claims use only `propext`,
  `Classical.choice`, `Quot.sound`; no comparison axiom or native decision is introduced.
- Raw AST regenerated independently from the authoritative isolated cut: Lean whole-cut
  acceptance proved by kernel reduction.
- Ruff and diff whitespace checks passed.
- The full guard is **not green**: the existing A2 addition and two receipt/chain schema
  changes still await their prior admission. No new protected-file or pinned-statement
  mismatch was introduced by this additive candidate.
- The structure check is **not green**: baseline `zephyr_model.py` has 3557 lines against
  its recorded 3537 ceiling. This branch does not change it.

The new tests verify mutated MLIR with the authoritative compiler and then ask the actual
Lean checker to reject it. They cover wrong input roots, gather/scatter indices, increments,
yields, count providers, allocation inputs, mask identity and extra unrelated operations.
Runtime refusal cases include poison controls/sizes, negative sizes/controls, zero stride,
budget exhaustion, final signed advancement overflow and unsigned-loop attributes. The
native observer checks compact elements as well as the final scatter and count, including
empty/all-false selections, all-true selections and both branch outcomes.

Machine-readable statements, test results, guard/build logs and exact oracle/source bindings
are in [`runtime-candidate/`](runtime-candidate/README.md). Native execution tests qualify
fidelity; they are separate from the universal Lean proofs. Existing newer-version PyTorch
results remain supplemental; this run uses the pinned **2.10.0+cpu** oracle.

## Review disposition

| Obligation | Disposition |
|---|---|
| Actual-AST acceptance to complete cut execution | Proved in Lean |
| Initialized sizes/conditions/elements, bounds and termination on the cut domain | Proved in Lean |
| Bool-repair preservation under this explicit interpretation | Proved in Lean |
| Repair-policy suffix composition under that same interpretation | Proved with an explicit suffix equality premise |
| Universal f32 UNE-to-positive-zero, including NaNs; poison lifting | Previously proved, retained; scoped policy/instance approval recorded |
| Native and pinned PyTorch behavior | Execution-tested; not universal proofs |
| Runtime semantic and integration admission | Awaiting human review |
| Full-model establishment of cut assumptions and all other obligations | Unproved; remaining inventory is not exhaustive |
| Active suffix receipts under this candidate interpretation | Not implemented/admitted; frozen receipts cannot substitute |
| f16/bf16/f64 frontend proofs | Unsupported/deferred; production denominator retained |

### Recorded execution oracles

The artifact-bound replay completed with **20 native selection samples** (actual SmolVLA cut
and independent lengths 0, 1, 8 and 1024, four masks each), plus **16 pinned PyTorch selection
samples**. The regression suites additionally reran the retained f32 IEEE edge cases.

| Oracle/tool | Exact recorded version |
|---|---|
| CPU intent oracle | PyTorch `2.10.0+cpu`, NumPy `2.5.4`, CPython `3.12.14` |
| Authoritative MLIR verifier/JIT | torch-mlir `20261001`, compiler CPython `3.12.14`; xDSL `0.73.0` |
| Native runner library | LLVM `23.0.0git`, source `a47bddccec30255619bb8c37fa59700e661d4e66`; `.so` SHA-256 recorded in `qualification.json` |
| Retained bool-cast native tests | LLVM 23 `llc`, GCC `16.2.1 20260810` |
| Lean | `4.35.0`, commit `eb6cafb7290e8bd5f2937522d8d804f58f02bafc` |
| Mathlib | `56a005855ea17e0b0958f73ec801ae140b0a1679` |

There were **zero skips**. The full source/export digests match the earlier authoritative
boundary, and the unchanged production checker binary retains SHA-256
`fb236e0c0a7a97d2022131d9597cfca57ad0c7222ed9f42d2c5b4fc8d3e59c30`.
The candidate adapter separately accepts the same full actual repair boundary and both
checked projections. The original receipt's frozen-model support refusal remains valid;
no receipt was silently relabeled as candidate progress.
