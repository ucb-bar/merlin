# Controller and source admission review

Reviewed `591687da8bf6a7f99807dbe0978ff297669867d3` against
`8f30c77acbd92e4c4637b9a75286a917b1a30a49`. No production, proof, specification,
lock, ledger or approval changes were made in this review. No heavy proof or
native qualification was launched here. Exact commands, source hashes and test
counts are in [controller-test-results.json](controller-test-results.json).

## Finding requiring a fix proposal

Unexpected projection errors can become nonblocking source exclusions.

| Location | Error converted to an exclusion | Reproduced behavior |
| --- | --- | --- |
| `src/merlin/verify/compiler_loop_generic_profile.py:111` | Any `ValueError`, `KeyError`, `TypeError` or `IndexError` from parsing/projection | An otherwise admitted generic source runs the actual ordinary unroll through `compiler_transforms.apply`, returns its transformed output, submits no job, and records `compiler_completed=true`, one unverified boundary |
| `src/merlin/verify/compiler_loop_source_profile.py:63` | Any `ValueError` from scalar source projection | An otherwise admitted word source becomes `SourceOutsideProfile` after an injected projector error |
| `src/merlin/verify/compiler_ir_source_profile.py:165` | Any `ValueError` from memory decoding/projection | An otherwise admitted memory-DCE source becomes `SourceOutsideProfile` after an injected decoder error |

The tests first establish that the unchanged source is admitted. They then
replace only the projector with a function raising an unexpected implementation
error. These are small host fault injections, not faulty-source examples.
The generic test runs the real transform controller and ordinary producer;
it does not execute `lower_model`, Lean or native code. The word and memory
tests establish source-admission classification only.

Retained regressions are

- [test_source_admission_faults.py](test_source_admission_faults.py), four expected
  failures and one passing `RuntimeError` blocking control.
- [test_source_admission_other_faults.py](test_source_admission_other_faults.py),
  two expected failures and four passing `TypeError`/`RuntimeError` blocking controls.

The six failures retain their JUnit records, exact source/controller results,
coverage documents and traceback output beside this report. All test temporary
trees use `tmp_path_retention_policy=all`. These failures contradict the intended
rule that unexpected admission errors block. They do not demonstrate a false
Lean theorem or an incorrectly verified target. The fallback is labeled
unverified and grants no proof credit, but compilation is allowed to continue.

Suggested fix for separate review is to reserve nonblocking exclusion for a
dedicated, explicitly raised domain exception at known source checks. Let
unexpected `KeyError`, `TypeError`, `IndexError` and generic implementation
`ValueError` reach `source_admission`, which turns them into blocking
`GateFailure(unavailable)`. The word and memory recognizers need the same
distinction. Simply removing every `ValueError` catch would also disable some
intentional word-to-generic fallback, so migrate documented domain failures to
the typed exception and retain the existing known-exclusion tests. No fix is
applied here.

## What the controller does enforce

Source decisions are made from the actual source before ordinary production
in the observed tensor, Q/DQ, host fusion and loop paths. The source-only
predicates do not use a proposed target to decide eligibility. The storage
controller admits the original full storage bundle. The generic loop path
selects each adjacent supported profile independently. A later source exclusion
does not cancel an earlier queued proof obligation.

For an admitted source, changed outputs go to the canonical worker. The worker
has no path that converts a failed target proof into a source exclusion.
Generic target preparation failures become blocking `GateRejected` or
`GateUnsupported`, while missing tools remain unavailable. Existing
`GateFailure` objects preserve their original verdict and evidence rather than
being relabeled by admission. New exact input snapshots and final rereads bind
loop/buffer parameters, intermediate buffers and host-fusion endpoints.

Tensor source recognition checks both subprocess status and the recognizer's
schema, profile, index width, support boolean and explicitly false proof-credit
and policy-approval fields. Exported data, input bytes, checker and selected
exporter/interpreter files are checked again before a decision is used. Only
the dedicated native source parsing exception becomes a grammar exclusion;
context, import, export and process failures remain blocking. Dense decoded
storage resource failures also remain blocking in the inspected paths.

Coverage records bind the exact queued request, endpoints, parameters and
intermediate text, then require one matching completed verdict. Changed jobs,
foreign jobs, duplicate verdicts and missing final outputs cannot earn proof
credit. Accepted means a verified boundary; excluded means unverified; byte
identity means unchanged with no proof credit. Coverage is descriptive and does
not make saved reports or fabricated completions authoritative.

This is a qualified conclusion. The six projector-error counterexamples above
prevent an unconditional claim that every unexpected admission failure blocks.
The observed target and missing-tool controls do remain blocking.

## Changed interpretation of compilation success

Known source exclusions deliberately continue ordinary compilation and are
reported at the end with zero proof credit. A corrupt transformation of an
excluded source is outside the selected formal guarantee. Thus the current
policy cannot be described as emitting outputs only after every selected pass
has a successful proof.

In particular, authentic changed memory CSE is now explicitly unverified in
`src/merlin/llvmlower/leading_cse.py:235`. The original live memory capture still
authenticates the observation, but it supplies no changed-CSE equivalence proof.
Unchanged memory CSE remains an identity with zero proof credit. Corpus tooling
must read `coverage.json`; zero jobs alone no longer implies an identity or
formal success.

At this exact cut, `DEFAULT_CHECKS` still contains five selectors. Q/DQ, draft
MACC, host graph fusion and both configuration checks require explicit selection.
The separately discussed expansion of `all` is not present in this reviewed
commit. Numerical and ISA approval flags remain independent of source admission.

## Worker result guards

The new generic loop handler checks its result schema/status/domain/relation,
kind, canonical complete and adjacent roots, `completeData`, standard-three
axioms, nonempty unique entries and explicitly false native fidelity. Pending
or empty generic returns cannot pass that guard.

The earlier incomplete-result gap remains for `tensor_refinement`,
`compiler_loop_refinement` and `compiler_buffer_refinement`. The fallthrough at
`src/merlin/verify/gate_worker.py:236` still does not validate those three result
shapes. The earlier guard proposal remains unapplied and was prepared against
the older cut; it needs review/rebasing before any application. No ordinary
canonical handler was observed to return an incomplete result. The new coverage
journal consumes worker verdicts and does not repair this separate worker gap.

Fresh protocol fault probes on this exact `591687da8` cut reproduce nine expected
failures, one for each combination of the three handlers and `None`, empty
dictionary or pending return. Every injected incomplete result becomes accepted.
These tests replace only the canonical handler and invoke the actual worker
protocol; they run no compiler, Lean or native proof. Their failures are separate
from the 161 passing host controls above. Exact commands, worker/check hashes,
JUnit and retained verdicts are in
[incomplete-handler-expected-failures.json](incomplete-handler-expected-failures.json).

## Test results and focused follow-up

| Cohort | Result | Scope |
| --- | --- | --- |
| Ten existing controller/profile/custody test files | 156 passed, no skips, 14.83 seconds | Pure host and synthetic protocol controls |
| Generic bad-target/schema/completion controls and missing worker SDK controls | 5 passed, no skips, 1.35 seconds | Two real workers refuse absent SDKs; no Lean proof executes |
| New unexpected-projector-error probes | 6 expected failures, 5 passing blocking controls | Retained counterexamples, separate from the passing cohorts |

The ten-file passing roster is [controller-host-selection.json](controller-host-selection.json).
The serialized heavy follow-up roster is
[controller-followup-roster.json](controller-followup-roster.json), 26 proposed
cases covering actual source recognition, generic unroll/tiling and replay,
recursive configuration hoisting, bounded default memory, real known exclusions,
and admitted corrupt targets across tensor, Q/DQ, MACC and host fusion. It has
not been launched by this reviewer. Root owns its memory-limited execution.
Every launch must retain all pytest temporary evidence and use a fresh directory.

The root's subsequent fresh generic TILE wrong-inner-bound run is retained
under `newproofs-work/test_supported_generic_tile_wr0/worker/0001-wrong-actual-inner-upper/`.
It blocks the wrong output, but the worker verdict is **unavailable**, not
`not_established`. All imported proof modules compiled successfully and the
query process exited 1. The raw query reports an unsolved goal comparing
iteration up to 257 against iteration up to `min(base+32,257)`. No heartbeat or
resource exhaustion is reported. The checker labels this an unknown/mixed
diagnostic, so it must not be promoted to a proved-false closed obligation or
counted as a classified semantic refusal. The existing test allows either
verdict and consequently passes.

This adds a reporting issue to the reused configuration terminology. The generic
loop worker's reason starts "configuration query" because it shares the runner.
A separate reviewed reporting change could distinguish generated obligations
left unproved from actual resource failures and use a family-neutral label.
Changing that label must not silently turn an unsolved goal into a semantic
counterexample. No runtime fix is applied here. Missing tools, resource kills
and timeouts remain distinct from deliberately corrupted outputs being blocked.

Appropriate paper wording remains conditional on coverage. Merlin checks
admitted individual pass boundaries and reports original-source exclusions as
unverified. These controller tests do not prove all repository kernels, native
execution, numerical policy approval or upstream MLIR transformations.
