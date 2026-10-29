---
title: Gate dead-code pruning and cse on Merlin's Lean proofs
kind: guide
status: current
owner: verification
last_verified: 2026-10-29
related: [verification, compiler_verification, lean_compiler_stack]
code_refs:
  - src/merlin/verify/pass_validation.py
  - src/merlin/verify/gate_checks.py
  - src/merlin/verify/speculative_gate.py
  - src/merlin/verify/lean/check.py
  - src/merlin/verify/lean/receipt_session.py
  - src/merlin/verify/gate_worker.py
  - src/merlin/llvmlower/passes_xdsl.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/leading_cse.py
  - src/merlin/llvmlower/pipeline.py
  - src/merlin/compile/command.py
  - merlin/lean/MerlinVerify/Check/ExactCse.lean
  - merlin/lean/MerlinVerify/Check/AlphaFast.lean
  - merlin/lean/MerlinVerify/Rules/Dce.lean
  - merlin/lean/LEDGER.md
  - merlin/lean/lakefile.toml
  - build_tools/scripts/check_lean_guard.py
  - build_tools/scripts/verify_lean_pass_models.py
  - merlin/tests/ir/test_lean_pass_gate.py
  - merlin/tests/ir/lean_gate_chunks.py
  - merlin/tests/ir/test_lean_pass_gate_fuzz.py
  - merlin/tests/ir/test_lean_receipt_session.py
  - merlin/tests/ir/test_speculative_gate.py
  - merlin/tests/rvv/test_lean_pass_gate_spike.py
---

# Gate dead-code pruning and cse on Merlin's Lean proofs

For the integrated loop and private-buffer gates, their compiler locations and
examples, see [Lean verification through the compiler stack](lean_compiler_stack.md).
This guide details the two tensor checks.

Compiler admission now follows the [bounded profile catalogue](lean_bounded_pass_profiles.md).
A known exclusion of the original source runs ordinary lowering and is listed
unverified at invocation end, with zero proof credit. An admitted changing pair
still requires an accepted actual-boundary check; unsupported targets, proof
failures and unavailable tools block. The worker verdict table below describes
submitted jobs, not source exclusions filtered before submission. Final
`coverage.json` distinguishes these cases after worker join.

Merlin can check two of its compile steps against the `MerlinVerify` proofs while it compiles: the
dead-code pruning in preprocessing and the tensor-level `cse` every lowering pipeline starts with.
Each step's exact IR before and after goes to a worker process; compilation carries on as if the
check passes, and a check that does not accept stops the compile, prints the check, the reason and
the paths of the input and output, and renames any `.ll`, `.so` or `.o` already produced to
`*.rejected`.

| Validated pass | Where it runs | Hook |
| --- | --- | --- |
| `prune_dead_pure_tensor_ops` | `lower_model` preprocessing (xDSL route) | `passes_xdsl._preprocess_module` |
| `cse` | every tensor-level `cse` the build's runner runs (the pipeline's leading one), on every route and with any features | `leading_cse.bind_runner`, `leading_cse.submit` |

The `cse` check is recorded inside the build's own runner, after whatever the selected features and
source stages do before the pass manager, so it checks the `cse` the build actually ran and the
build's output is byte-identical to an ungated build. Quantization lowering (`lower_quant_ext`, the
textual whole-model preprocessing) is not gated here.

## What is proved

For every validated boundary the worker requires all of:

1. `MerlinVerify.Check.TypedModule.check_sound` on the boundary, inputs of the declared types, both
   buffers exported by MLIR's own parser and round-tripped identically through Lean;
2. `MerlinVerify.Check.supportCheck_sound` on every function of the input (it never gets stuck),
   which makes the refinement strict rather than conditional;
3. `compose_chain` validating and composing both, with `refinement_verdict: verified`.
   Only live results from the same job can avoid duplicate checker execution, as described below.

It attempts two strengthenings and reports them without requiring them: the universal boundary
conditional theorem (`moduleCheck_sound`, for all inputs) and source definedness on typed,
nonpoison inputs. The strict claim
retains the declared input types, where source support was established; a universal conditional
boundary alone does not widen that domain. The receipt's `claim` says exactly what was established.
Axioms: `propext`, `Classical.choice`, `Quot.sound`.

* Pruning is proved by the C1 DCE rule. Dead source ops are erased before alignment
  (`synthC1Live`, and `synthC1Unused` for what upstream `cse` erases), so a dead op is never matched
  with an identical live one.
* `cse` is proved by `Check/ExactCse.lean`, in one forward walk as upstream `cse` makes it: a later
  effect-free op with the same head as an earlier one, operands holding the same values and regions
  reading the same outside values computes the earlier op's value in the same run, so its uses are
  renamed to it. Facts come from the earlier op's actual execution plus an alias per merge, so
  merges cascade and regions may read outside constants; nothing is merged into an op nothing reads,
  matching `cse`'s erasure of unused ops. Both programs are normalized this way before comparison.

## Human review before relying on it

Lean checks every proof step; review establishes that what is proved is what is intended.

1. **The op semantics the claims are stated over** (`merlin/lean/LEDGER.md` A1–A8, T3): the
   `MerlinVerify` meaning of `arith`, `math`, `tensor` and `linalg` ops, which both checks rely on.
2. **The purity scope** (`Rules/Dce.lean` `pureOp`/`erasable`, ledger A9): which ops are treated as
   only computing their results, and so may be erased when dead and reused by `cse`. A9 extends this
   to the three modeled `quant_ext` ops, so a dead quantization op can be pruned.
3. **Source Q/DQ meanings where these operations remain** (`Spec/Dequant.lean`,
   `Spec/Quantize.lean`): a boundary whose input contains modeled `quant_ext` ops needs their
   meaning to establish source support, even when the transformation only prunes or reuses them.
   This does not check the separate quantization-lowering boundary.
4. **The trusted components** (ledger T1–T6): the Lean kernel and compiler, the `torch_mlir` exporter
   and the Python bridge, and everything below the tensor stage.
5. **The locked statements** (`merlin/lean/SPEC_LOCK`): the hashes pin the semantics files and the
   theorem statements the gate cites; review confirms they say what the claims need.

Not covered: `canonicalize`, the trailing LLVM-dialect `canonicalize,cse`, and lowering below the
tensor stage stay trusted. Only the 64-bit index width is admitted (LEDGER S6).

## Set it up once

```
uv venv --python 3.12 <venv>
grep '^torch-mlir' build_tools/toolchains/lean-mlir-requirements.txt > torch-mlir.txt
uv pip install --python <venv>/bin/python --no-deps --require-hashes -r torch-mlir.txt
uv pip install --python <venv>/bin/python --no-deps numpy==2.4.6   # numpy carries no hash in the file

cd merlin/lean
PATH=<lean-4.35.0>/bin:$PATH lake exe cache get        # prebuilt Mathlib, about 7 GB
PATH=<lean-4.35.0>/bin:$PATH taskset -c 0-7 nice -n 19 lake build MerlinVerify merlin-lean merlin-lean-audit
```

A checker that is out of date with `merlin/lean` (`lake build --no-build`) is refused.
The freshness check covers the proof library and both executables, including audit-only
modules that the native checker does not import.
`build_tools/scripts/check_lean_guard.py` audits every theorem's axioms and the locked files.
On Linux, workers also require util-linux `setpriv` for parent-death signals. Missing cleanup
tooling is `unavailable`, never an accepted check.

## Turn it on

```
merlin-compile ... --validate-passes all --lean-sdk-bin <lean-4.35.0>/bin
```

or set `MERLIN_VALIDATE_PASSES` (`all` or a comma list) and `MERLIN_LEAN_SDK_BIN`, with
`MERLIN_COMPILER_PYTHON=<venv>/bin/python`, or pass `pass_validation=PassValidation(gate, tools)` to
`lower_model`. It is off by default; a missing or stale tool stops the compile before it starts. A
lowering called inside an active validation joins it, and a different explicit validation is refused.

Checks run in owned worker process groups. A failed check rejects speculative compiler products;
closing the gate stops its groups. On Linux, a compile killed by TERM or KILL also stops its
workers and their checker children. A startup parent-identity check closes the interval before
the parent-death signal was armed. Worker crashes are followed by bounded group cleanup too.

## Cost and implemented speedups

Two optimizations preserve the existing proof obligations:

* `Check/AlphaFast.lean` replaces list-based correspondence lookup and image membership with
  persistent ordered maps/sets. Lean proves equality to `alphaOps` for every correspondence and
  pair of operation lists; `@[csimp]` selects that equal implementation in compiled callers.
  Frozen semantics, the original checker definition and its soundness statements stay unchanged.
* Within a gate job, `receipt_session` records results issued by actual successful checks.
  Composition validates their schema, provenance, axioms, exports and byte bindings and reuses
  the live result instead of executing the same checker twice. This authority exists only in
  memory in that process/job, and is discarded on exit. Saved or edited JSON cannot declare
  itself fresh. Ordinary external composition still reruns the checks. This is trusted host
  orchestration under ledger T2, not a new Lean axiom.

The original profiling of `SY_model_tiny_llama` (4,659 source operations) measured about 61 minutes
in the accepting normalization plus another replay of similar cost. The indexed implementation
reduced the measured accepting normalization to about 98 seconds; eliminating duplicate replay
removes the second execution. These are historical component measurements from the performance
worktree, not a bound on complete gate latency. Exact timings depend on the module and host load.

## Further optimizations deferred

The following ideas were identified during profiling and are deliberately not implemented here:

* Order cheaper, likely accepting `TypedModule.check` alternatives first, or index the rule search
  in `synthC1Live`. Historical failed alternatives cost about 73 seconds, including 66 seconds in
  `synthC1Live` alone.
* Batch `ExactCse` substitutions into one checked pass per block. It currently performs a
  continuation rename for each merge (about 1,800 merges in the profiled module).
* Index the scans in `FromElementsSelect` and `DenseExtract`, measured at roughly 18 and 2 seconds
  respectively in that same profile.

Each would need separate correctness and qualification work. Current rule ordering and these
normalizers are retained.

## Quantization handoff

The tensor checks gate pruning and tensor CSE only. Quantization proof progress, branches, evidence,
review obligations and integration boundaries are recorded in the separate
[quantization handoff](../../out/artifacts/handoff/quantization-lean-20261028/README.md).
Its numerical lowering and proof coverage are not extended by this deliverable.

## Outcomes

| Verdict | Meaning | Compile |
| --- | --- | --- |
| `accepted` | the three required checks passed | continues |
| `not_established` | the boundary or the composition did not verify | stops |
| `unsupported` | MLIR's parser refused the IR, or source support is not established | stops |
| `unavailable` | a tool could not run or the check ran out of time | stops |

A pass that changes nothing submits nothing.

## Tests

* `merlin/tests/ir/test_lean_pass_gate.py` — the gate's behaviour (injected prune and `cse` bugs stop
  the compile and leave no artifact, nesting, tool and freshness refusals, features around the
  runner's `cse`), small exact `cse` cases (cascades, captured constants, unused-op erasure, wrong
  merges), and every representative chunk of the retained capsules (`lean_gate_chunks.py`) through
  real lowering, live and with dead code, with byte-identical gated output.
* `merlin/tests/ir/test_lean_pass_gate_fuzz.py` — seeded random programs through the real passes;
  completeness against an independent reference and soundness against a numpy oracle.
* `merlin/tests/ir/test_lean_receipt_session.py` — actual proof invocation counts, external replay,
  forged verdicts and edited live receipts.
* `merlin/tests/ir/test_speculative_gate.py` — worker ownership, bounded cancellation, parent/worker
  death and cleanup-tool refusal.
* `merlin/tests/rvv/test_lean_pass_gate_spike.py` — whole-model builds on Spike: a gated build matches
  its golden output, and a wrong `cse` is stopped before any ELF exists.

Set `MERLIN_LEAN_REQUIRED=1` to turn a missing tool into a failure instead of a skip. Whole models
are qualified with `build_tools/scripts/verify_lean_pass_models.py --capture <mlir> ...`, which
compiles each with and without the gate and requires byte-identical LLVM output.
The [integration qualification](../../out/artifacts/handoff/cse-dce-integration-20261028/README.md)
records the qualified source points, executed checks, retained evidence and scope limits.

## Development integration and gate scope

`feat/pass-verif-integration` retains the original DCE/CSE integration and the
upstream pass-verification and independent-review histories. It registers ten
selectors. `--validate-passes all` selects the five defaults:
`prune_dead_pure_tensor_ops`, `cse`, `ordered_loop_unrolling`,
`ordered_loop_segmentation` and `buffer_reuse`, while transformations themselves
remain separately opt-in. Configuration-write elision, configuration hoisting
and host graph fusion require their explicit selectors; see the stack guide.
The explicit `lower_quant_ext` selector checks actual xDSL/textual Q/DQ
lowering under the proposed source semantics and stays outside `all`.
Optional original-source qualification is separate from specification approval;
see [the Q/DQ gate guide](lean_qdq_gate.md).
The separately registered `bf16_f32acc_draft_contract` check requires explicit
selection and remains outside `all`. `--macc-f32acc` or
`Options(macc_f32acc=True)` selects its actual production transformation;
the accepted claim is draft-reference conformance, with numerical approval
still abstracted. See the stack guide for its interface and limits.
The loop and memory checks have their own narrower domains and relations;
see [the compiler stack](lean_compiler_stack.md). The DCE/CSE historical
qualification retains its original source/branch names and scope.

Other quantization proofs and approved numerical-policy gates remain separate
review and handoff tooling. No numerical review admission is required to select
DCE/CSE validation. Existing numerical compiler lowerings are inherited from
development; their correctness is not claimed by the DCE/CSE checks.

The separate shared native-prefix verification API retains its existing explicit
opt-in scope; it is not selected by `--validate-passes all`. Its trusted native
suffix and unsupported transport combinations remain separate API constraints.
