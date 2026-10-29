# DCE/CSE integration qualification — 2026-10-28

Status: implementation complete and ready for integration review within the
scope below. Human semantic/trust approval has not been recorded.

## Integration source

Worktree: `/scratch/grammy/blast/merlin-lean-dce-cse`.
Review branch: `feat/pass-verification-dce-cse`, preserving the qualified history
from the former local name `integrate/lean-dce-cse-gate`.
Base: `origin/main` at `e45205bf9`; starting integration commit: `15de277ec`.
The proof and enforcement implementation qualified below is at `d73cbcfe8`; `4e59d5476`
adds documentation only. `6990bece6` fixes test environment isolation and makes
the absence of baseline gate instrumentation an assertion. `5c564cd11`
clarifies the optional definedness report and adds its exact-domain regression.
Proof/enforcement logic and the Lean binary remain unchanged since `d73cbcfe8`.
The feature branch is published for integration review; it is not merged into
`main` and no pull request is opened by this delivery.

The branch includes the Lean checker library plus two compiler gates:
`prune_dead_pure_tensor_ops` in xDSL preprocessing, and the pipeline's leading
tensor `cse` in the actual compiler runner on both routes. Enable them with
`merlin-compile ... --validate-passes all --lean-sdk-bin <sdk>/bin`, or the
environment/API described in the [gate guide](../../../../docs/guides/lean_pass_gate.md).
Validation is opt-in. Missing/stale tools and unsupported or rejected checks
stop the requested validated compilation; unchanged boundaries need no check.

## Changes completed

* `5c564cd11`: state optional source definedness on typed, nonpoison inputs.
  Four real-checker regressions pass; the strict refinement domain and actual
  verification decisions are unchanged.
* `8925f5b44`: close the worker startup parent-death race, require Linux
  cleanup tooling, and stop owned checker groups even when their worker crashes.
* `6affac56b`: complete the indexed alpha-comparison implementation. Lean proves
  `alphaOps_eq_fast` for every correspondence and pair of operation lists;
  compiled callers select it using `csimp`. Frozen semantics stay unchanged.
* `6990bece6`: restore the CLI test environment even when settings were initially
  absent; explicitly disable and assert the absence of validation on comparison
  baselines. The focused compiler-gate recheck retains its complete artifacts.
* `d73cbcfe8`: remove duplicate proof execution only for successful results
  actually issued in the same live process/job. Composition still validates
  provenance, exports, axioms and byte bindings. Saved, forged, edited and
  previous-job receipts cannot authorize reuse. Clarify that strict refinement
  retains the declared input types even if universal conditional checking passes.

## Verification and evidence

The local evidence root is
`out/build/cse-dce-production-20261028/`. Its `environment.sh` records exact
SDK/interpreter/tool paths; validation uses two workers per suite, Lean threads 2,
BLAS/OMP threads 1 and nice 19. The two suites overlapped with four pytest
workers, sharing the same total ceiling of four pinned CPUs on the shared VM.
See `validation-cpu-budget.json` for the recorded transition.

* Lean build: all 3,341 build jobs succeeded (`lean-build.log`).
* Full guard: 566 source modules covered, 15,300 shared-module theorems,
  zero axiom violations; frozen hashes and pinned statements pass
  (`full-guard.log`). Imported theorem counts overlap across audit groups.
* Main regression suite: **779 passed**, zero failures/skips, in 4,763.80 seconds
  (`full-tests.log`, `MERLIN_LEAN_REQUIRED=1`), including three Spike tests.
  It emitted six configuration warnings because pytest-timeout was not loaded;
  gate/checker subprocess deadlines remained active.
* Focused compiler-gate/fuzz recheck: **75 passed**, zero failures/skips, in
  2,154.99 seconds (`final-gate-tests.log`). This run loads pytest-timeout
  and retains its artifacts. It is the authority for the
  corrected baseline comparisons: the CLI flag test in the original run could
  leave validation enabled in one worker. The fix restores the environment and
  asserts that baseline validation artifacts/instrumentation are absent.
  All 40 representative cases used uninstrumented baselines. Of these, 39
  emitted byte-identical baseline/gated LLVM; one preserved a baseline lowering
  failure on both builds. Their 54 gate boundaries accepted (20 pruning, 34 CSE).
  Three seeded fuzz programs reported zero
  checker gaps, 10 distinguishable mutations rejected and one indistinguishable
  mutation counted separately.

* Final receipt regressions: **4 passed** in 84.07 seconds
  (`receipt-tests-final.log`, pytest-timeout loaded). They verify the final
  domain wording, live invocation counts, saved-receipt replay and tampering
  refusal. Only report wording/docstrings differ from the earlier production
  code; `scope-audit.json` records an AST comparison of the enforcement logic.
* Independent targeted runs: 30 worker-lifecycle tests, four initial real-checker
  receipt-session tests, and six guard-infrastructure tests passed.
  `worker-tests.log` preserves an initial race in the process-liveness test probe;
  the corrected passing run is `worker-tests-v2.log`.
* Source structure, documentation, artifact layout and changed-Python lint pass.
  The no-target-name scan reports existing baseline debt only.

The one chunk with baseline failure parity is retained at
`final-gate-tests/popen-gw0/test_representative_kernels_lo3/` under the evidence
root. An isolated replay of its saved ungated native lowering recipe also
returns SIGSEGV (-11), without LLVM output or a diagnostic. The replay command
and result are in `baseline-chunk-replay/result.json`. This is outside the
recorded DCE/CSE boundary proofs and is excluded from successful LLVM exports;
no numerical or native-lowering fix is introduced here.

The final `completion-qualification.json` audits actual receipt/export/LLVM,
checker, SPEC_LOCK and log hashes for the delivery. Whole-model qualification
is indexed, with byte hashes, in
`out/build/cse-dce-production-20261028/model-qualification.json`:

| Capture | Qualified routes | Outcome |
| --- | --- | --- |
| `SY_micro_model` | xDSL | CSE accepted; identical gated/ungated LLVM |
| `SY_model_resnet50` | xDSL | CSE accepted; identical gated/ungated LLVM |
| `SY_model_tiny_llama` | xDSL | CSE accepted; identical gated/ungated LLVM |
| `small_llama_int8_consistent` | xDSL and textual | DCE/CSE checks accepted where run; identical LLVM |
| `tiny_llama_fp32_consistent` | xDSL and textual | CSE accepted; identical LLVM |

Seven model/route exports passed, with eight accepted boundaries. Their original
receipts, source/export hashes, checker hash, SPEC_LOCK hash and composed strict
refinement verdicts were inspected. Products:

* `out/artifacts/verification/v1/verification_v1_20261028T223434Z_d73cbcf`
* `out/artifacts/verification/v1/verification_v1_20261028T224639Z_4e59d54`

Three direct raw-capsule textual attempts failed in the **ungated baseline**
at an unlowered `quant_ext` parser boundary. They are preserved in the first
product, excluded from successes and handed to the quantization agent.
Compatible captures above qualify the textual CSE route separately.

The large TinyLlama xDSL comparison took 451.583 seconds end to end under
concurrent two-CPU validation. Historical profiling on the performance branch
reduced accepting normalization from about 61 minutes to 98 seconds, then
identified duplicate replay. Those are different sources and component timings;
they do not establish an apples-to-apples whole-compile speedup ratio.

## Claim and human review

The required result is strict tensor-stage refinement on inputs of the declared
types, under `merlin/lean/LEDGER.md`. Review the modeled operation semantics,
DCE erasability/CSE purity (including ledger A9 for retained source Q/DQ ops),
SPEC_LOCK statements, exporter/host orchestration and ledger T1–T6 before
relying on the gate. Human review has not been recorded as approved.
In particular, the `linalg.memoized_indexing_maps` exemption needs confirmation
that cached maps agree with the actual operation: xDSL named-op accessors can
read this attribute. Changes to `llvm.emit_c_interface` are separately reported
as ABI changes trusted to downstream lowering, outside the tensor claim.
Optional source definedness concerns typed, nonpoison inputs; universal
conditional refinement alone does not establish strict refinement for all inputs.
Canonicalization, numerical preparation, bufferization, LLVM lowering and
hardware remain outside this proof. Only 64-bit index width is admitted.

## Deferred work and quantization handoff

The three further performance ideas are documented in the guide and deliberately
not implemented: rule-alternative ordering/indexed `synthC1Live`, batched
`ExactCse` continuation substitutions, and indexed `FromElementsSelect`/
`DenseExtract` scans. `scope-audit.json` records their unchanged source hashes.

[Quantization handoff](../quantization-lean-20261028/README.md) records its
published proof/lowering branch, implemented numerical scope, evidence and
remaining human review/integration work. This deliverable does not advance
quantization proofs or numerical compiler corrections. The existing source
semantics, specifications and SPEC_LOCK match the published reference; the
production quantization lowerings retain the `origin/main` behavior.

## Review branch

The completed DCE/CSE compiler-gate implementation is published at
[feat/pass-verification-dce-cse](https://github.com/ucb-bar/merlin/tree/feat/pass-verification-dce-cse).
Publication updates branch-location/review notes and removes one extra EOF
blank line from a standalone proof fixture. The proof/compiler implementation
and qualification source points above are unchanged. Other proof families in the
shared library retain their stated provisional scope.
