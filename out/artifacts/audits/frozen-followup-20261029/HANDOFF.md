# Frozen compiler gate audit handoff

Snapshot dated 2026-10-29 America/Los_Angeles. This is an **incomplete qualification**, with confirmed results and preserved failures. Production functionality, proofs, specifications, locks, the assumption ledger and approval flags were not changed. Proposed fixes require separate review.

Publication update follows the audit. The branch was fast-forwarded to Graham's `b7626bbf4`, which makes `all` select all ten registered gates. Results below still belong to `591687da8`, where `all` selected five. The new default has not been requalified by this audit. This published packet contains this handoff, the controller and proof-domain reviews, the source cut and [compact results](RESULTS.json). Other evidence links refer to local retained files and are not bundled in Git.

Publication checks ran on the report atop `b7626bbf4`. `git diff --cached --check` passed. `check_structure.py` failed with 16 existing problems in unchanged source files, consisting of core-to-optional dependency imports and module size limits. `check_artifact_layout.py` failed on existing stale generated-root literals and two tracked event logs outside the output root. AGENT coverage and documentation checks within the structure check passed. These repository checks remain failed; the report publication does not repair them.

## Source and scope

- Repository `/home/jeremymills/projects/merlin`
- Branch `feat/pass-verif-integration`
- Current tested commit `591687da8bf6a7f99807dbe0978ff297669867d3`
- Previous interrupted campaign `8f30c77acbd92e4c4637b9a75286a917b1a30a49`
- Exact source and executable bindings in [SOURCE-CUT.json](SOURCE-CUT.json)
- Current scope is compiler gate behavior, proof domains, real boundaries, malformed inputs, numerical references and evidence custody. Upstream MLIR correctness, W8A8 policy correctness and accelerator performance are separate claims.

Three parallel agents reviewed corpus tooling, controller failures and proof domains. Parallel reviews were stopped after the user raised token usage. Their completed evidence is retained below. This publication adds reports and an index link only. The untracked user PDF was not modified.

## Completed on the current commit

| Check | Result and limit |
| --- | --- |
| Lean library, checker and auditor build | Passed, 3349 Lake jobs. [Build log](build.log), [exit code](build.exit) |
| Specification lock file hashes | Passed. [Log](lock-only.log). This does **not** rerun the 235 elaborated theorem pins |
| Full Lean guard | Passed, exit 0, including the elaborated-statement lock comparison. Audit covered 571 source modules with zero violations; the shared group reported 15350 theorems. Imported theorem counts overlap between groups. [Log](lean-guard.log), [exit code](lean-guard.exit) |
| Existing policy, controller and new proof tests | **207 distinct cases passed, zero skips** after removing overlap among four cohorts. [Accounting](current-test-accounting.json) |
| New generic UNROLL, TILE and recursive configuration suites | 38 passed within the 207 above, including actual canonical workers and fresh replay. [Log](newproofs.log), [JUnit](newproofs.xml) |
| Audit runner bookkeeping | 26 pure tests passed, zero skips. These validate reporting, not compiler transformations. [Log](bookkeeping-tests.log) |
| Source admission fault injections | Six expected regression failures and five passing controls. Unexpected projector exceptions can be treated as nonblocking exclusions. [Review](controller-review.md) |
| Incomplete worker result injections | Nine expected regression failures. Three old handlers accept injected incomplete returns. [Results](incomplete-handler-expected-failures.json) |
| Additional actual UNROLL semantic negative | One positive passed, one strict negative test failed because the rejected bad target was classified `unavailable`. The incorrect output was blocked. [Evidence](generic-unroll-semantic-negative/README.md) |

The four overlapping existing-test cohorts contain 91 policy cases, 156 host controller cases, five worker-failure cases and 38 new-proof cases. Their raw sum of 290 must not be reported as 290 independent tests. Expected-failure probes and the additional UNROLL experiment are outside the 207 count. All retained new pytest runs request retention of successful and failed temporary evidence.

The separate full `check_lean_guard.py` run finished successfully before this handoff was finalized. This is an audit of the checked theorem declarations and allowed axioms, not evidence that every compiler boundary or native numerical case has been exercised.

## Findings requiring review

### Unexpected admission errors can let compilation continue

`compiler_loop_generic_profile.py` converts unexpected `ValueError`, `KeyError`, `TypeError` and `IndexError` from projection into an outside-profile decision. The scalar loop and memory source recognizers similarly swallow unexpected `ValueError`.

On an otherwise admitted generic source, fault injection caused the actual transform controller to run the ordinary producer, submit no proof job and record a completed compilation with one unverified boundary. The scalar and memory probes establish the classification defect. These are implementation-error injections, not counterexamples to a Lean theorem. Coverage correctly gives zero proof credit, but the intended blocking behavior is violated.

Proposed fix is a dedicated exception for known domain exclusions, with unexpected implementation errors remaining blocking. Preserve intentional scalar-to-generic fallback. Six failing regressions and five blocking controls are retained in [controller-review.md](controller-review.md), [generic probes](test_source_admission_faults.py) and [other probes](test_source_admission_other_faults.py). No fix was applied.

### Three older worker handlers lack complete result guards

`tensor_refinement`, `compiler_loop_refinement` and `compiler_buffer_refinement` accept mocked handler returns of `None`, an empty dictionary or a pending status. Nine current-commit protocol probes reproduce this. No ordinary canonical handler was observed returning these values, and this is not evidence that a false theorem passed Lean.

An older, unapplied [guard proposal](../proof-cutoff-20261029/proposals/README.md) includes a patch and regressions. It needs review and rebasing against this commit. Its 48 mock tests were run with the proposal installed before the frozen audit; they are not evidence that the frozen runtime contains the fix.

### Generic loop failures need more precise reporting

Valid wrong-target UNROLL and TILE examples reached Lean with successful imports and failed closed queries without a resource timeout. Both blocked delivery, but their verdict was `unavailable`. The existing TILE test permits this and therefore passes. The stricter new UNROLL test requires `not_established` and remains failed.

Preserve the raw queries and diagnostics. A proposed generic-loop diagnostic classification fix should distinguish a failed obligation from tooling/resource failures without treating every unsolved goal as a proof of inequivalence. See [UNROLL evidence](generic-unroll-semantic-negative/README.md) and `newproofs-work/test_supported_generic_tile_wr0/`.

### Corpus reporting had to account for source exclusions

The previous audit script treated successful LLVM with no proof jobs as unchanged. That is invalid under the new source-admission policy. A test-only replacement now binds `coverage.json` and separates verified, unverified, identity and selected-but-unobserved boundaries. Mixed verified and excluded output remains partially covered.

The replacement is [qualify_bounded_cutoff.py](qualify_bounded_cutoff.py), with [frozen hashes](audit-tool-freeze.json), [26 regressions](test_bounded_cutoff.py) and [usage notes](bounded-runner-notes.md). The older runner and its old evidence were preserved. No production reporting code was changed.

## Agustin kernel set and models

The requested `KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md` was found on `origin/comparisons/tvm-workload-tuning` at `ebf58f2c11816370ee27f99557a6bd0b5363643f`, under `examples/gemmini/comparisons/tvm/`. The handoff, canonical oracle and supporting source are archived in [agustin16/source](agustin16/source/), with [provenance](agustin16/source-provenance.json).

The set contains six integer GEMMs, three convolutions and seven transformer kernels. The complete set has 81 logical output tensors across 78 matrix products and three convolutions. T01 and T02 each require all 32 heads with shared K/V groups; T04 and T06 require every output. Convolutions use the supplied real input border, VALID cross-correlation and bias once. Scaled epilogues use the supplied f32 round-to-nearest-even contract. Do not replace these with single-head or output-prefix shortcuts. See the [independent contract review](agustin16-contract-review.md).

Fixture preparation is **unfinished**. The preparation agent reported an installed xDSL named-matmul parsing problem with mixed i8 inputs and i32 accumulation. The `agustin16/prepared-v1/` directory contains no completed fixture artifacts at this checkpoint, so this reported failure still needs a retained minimal reproduction. A proposed test-fixture workaround is an explicit signed-extension `linalg.generic` contraction. This reconstructs the handoff's mathematical definitions and must be labeled trusted fixture construction, not authenticated capture-producer evidence. No new 16-kernel compiler or native run has completed.

The intended current campaign is 16 complete kernels plus these six retained full models, each on xDSL and textual routes, giving **44 route cases**. The final external input manifest and independent fixture review are still required.

- `SY_micro_model`
- `M0_small_llama_gemmini`
- `M2_microvit_gemmini`
- `M4_smolvla_denoise_gemmini`
- `SY_model_resnet50`
- `SY_model_tiny_llama`

Model sources are `merlin/contract/capsules/model/<name>/capsule.interface.mlir`. Lowering these models does not establish numerical model accuracy or hardware execution.

## Proof and coverage boundaries

| Family | Scope and exclusions that remain |
| --- | --- |
| Tensor rewrites and Q/DQ | Typed refinement for supported operations under pinned operation models. Definedness, framework/IEEE fidelity and actual IR projection remain separate obligations. Q/DQ source qualification must bind the interpreter, producer, wheel and dependencies |
| Scalar and generic loops | Equality under the selected evaluator, including modeled error and world behavior. The new generic profile admits one top-level loop, no carries, one zero-result opaque event and a captured return. UNROLL is factor two with a nonempty bulk and one tail. TILE size is 2 through 65536. Generic UNROLL-to-TILE composition and arbitrary tensor loop bodies are excluded |
| Configuration elision and hoisting | Selected state/effect model and actual compiler consumer. Recursive hoisting still requires universal `Recursive.Models`, total exact setters and the A10 ISA-effect assumptions |
| Buffer and memory transforms | Only stated memory, aliasing and observation domains. Possible-success refinement must not be rewritten as unrestricted equality. Changed memory CSE currently receives unverified coverage |
| MACC | Draft f32 accumulation policy conformance, not equality with original half-precision accumulation. Numerical approval and unsupported initialization/maps remain separate |
| Host graph fusion | Conditional operation-model and extf assumptions. Hardware NaN behavior and target injectivity must not be silently assumed |
| All families | T7 trusts Python/xDSL projection. A10 covers selected ISA-effect/Models premises where applicable. Receipt custody and fresh replay do not prove encoder, LLVM or hardware fidelity |

[Proof-domain review](proof-domain-review.md) gives exact generic-loop restrictions, recursive configuration assumptions and source references. The main assumption ledger remains `merlin/lean/LEDGER.md` at the tested commit.

At this commit, `DEFAULT_CHECKS` and `all` still select only five checks. Q/DQ, MACC, host fusion and both configuration checks require explicit selection. The new campaign tool explicitly selects all ten registered checks. Selection does not activate optional transformations or guarantee that a boundary occurs. Known source exclusions intentionally continue ordinary compilation with an unverified record; accepted compiler output is therefore not synonymous with fully verified output.

## Earlier interrupted evidence

The previous campaign is retained under [proof-cutoff-20261029](../proof-cutoff-20261029/). Its results belong to `8f30c77ac`, not the current commit.

| Earlier campaign | Completed evidence and limits |
| --- | --- |
| 40 retained kernel windows on two routes | 16 of 80 route cases completed. Eleven checked successes, two identity cases, two baseline native crashes and one proof not established. Fourteen accepted changed boundaries. Not Agustin's complete 16-kernel set |
| Six retained full models on two routes | Nine of 12 route cases completed successfully, with 14 accepted jobs. Three cases unfinished. Some completed case records were not yet incorporated into the coordinator summary |
| Malformed inputs | Eight route cases blocked before final products, six parse errors and two verifier errors. Fresh current-commit rerun prepared but not executed |
| Memory observer mutation | One real positive and one changed-observer refusal retained |
| Pinned Q/DQ numerical cohort | Four native source cases completed before interruption. Not a complete current-cut qualification |
| Legacy differential prerequisite | Failed setup because `third_party/llvm-install/lib/libmlir_c_runner_utils.so` is missing. Not a pass or an optional skip |

The two native crashes and the refused DCE target remain preserved. Neither should be relabeled as successful baseline/gated parity. One older nine-test partial mutation cohort lost successful temporary receipts because its inherited retention policy kept only failures. It has stdout evidence, not retained proof receipts. The corrected campaigns retain all temporary evidence.

The earlier heavy workers were stopped during swap pressure. Interrupted or killed jobs have no semantic verdict. Subsequent heavy runs use one process cohort at a time, with a 6 GiB memory high watermark, 8 GiB maximum and no additional swap under a user systemd scope.

## Reproduction and next owner

Run from the repository root. Do not change source files or the ledger while generating receipts. Use fresh output directories. Exact completed cohort details are in [controller-test-results.json](controller-test-results.json), JUnit files and retained work trees.

```bash
source out/artifacts/audits/proof-cutoff-20261029/numerical-environment.sh
git rev-parse HEAD

# Full guard replay
systemd-run --user --scope --unit=merlin-frozen-audit-replay \
  -p MemoryHigh=6G -p MemoryMax=8G -p MemorySwapMax=0 \
  .venv/bin/python -u build_tools/scripts/check_lean_guard.py

# Representative actual-boundary suites, retaining all evidence
.venv/bin/python -m pytest -q -ra --tb=short \
  -o tmp_path_retention_policy=all \
  --basetemp=out/artifacts/audits/frozen-replay-newproofs \
  merlin/tests/ir/test_compiler_loop_generic.py \
  merlin/tests/ir/test_compiler_loop_generic_tile.py \
  merlin/tests/ir/test_compiler_configuration_hoist_recursive.py
```

The environment file selects Lean 4.35.0, the compiler interpreter and clang, plus the pinned Q/DQ CPython 3.12.3, Torch 2.10.0 CPU wheel, model2MLIR producer `905d2c918f839596c9d1198cbdeb42d4c12a325b` and capture dependency directory. These are local `out/build/lean/` resources, not portable dependencies bundled with this handoff.

For the pending corpus, finalize an external manifest with schema `merlin.pass-cutoff.inputs.v1` and `id`, `path`, `sha256` for every input. Then run the following serially under the same memory scope, substituting fresh paths.

```bash
AUDIT=out/artifacts/audits/frozen-followup-20261029
.venv/bin/python "$AUDIT/qualify_bounded_cutoff.py" inventory \
  --external-inputs /absolute/path/to/final-inputs.json \
  --output "$AUDIT/final-inventory" --routes xdsl textual
.venv/bin/python "$AUDIT/qualify_bounded_cutoff.py" run \
  --inventory "$AUDIT/final-inventory/inventory.json" \
  --output "$AUDIT/final-run" --timeout 300 --job-timeout 240
.venv/bin/python "$AUDIT/qualify_bounded_cutoff.py" audit \
  --output "$AUDIT/final-run"
```

The final command audits retained bytes and classifications. It sets `fresh_proof=false`; fresh replay requires executing the proofs again. Native reference execution is a separate step and is not performed by this corpus command.

Outstanding work in priority order

1. Retain the completed full guard result and verify the same source/dependency bindings before restarting qualification. Do not change locked specifications.
2. Finish and independently review all 16 reconstructed fixtures, preserve the failed preparation, and freeze the 22-input manifest.
3. Run all 44 compiler route cases, accounting for every exclusion, identity, selected-but-unobserved check and admitted failure.
4. Execute complete native kernel outputs against the supplied deterministic oracle, then run fresh malformed and meaningful wrong-rewrite controls. Prepared malformed scripts are [test_malformed_current.py](test_malformed_current.py) and [run_malformed_current.py](run_malformed_current.py).
5. Complete current-source numerical qualification, receipt byte audits and fresh replay. Keep unavailable tools, timeouts and unrun cases explicit.
6. Review the admission exception and worker result-guard fixes separately, with retained failing regressions. Requalify any changed source cut.

Most of the audit directory is gitignored; the published reports are explicitly allowlisted. Transfer the complete local directory with the earlier evidence directory and necessary local dependencies, or select and package the referenced evidence. The Git report preserves the summary but not its proof receipts or reproduction environment.

## Paper wording supported by this checkpoint

Merlin validates supported transformation instances against explicit semantic models using Lean checked proofs. Compiler gates bind admitted before and after boundaries to fresh checking results, while coverage records distinguish verified rewrites, unchanged boundaries and source exclusions. The guarantees remain conditional on the documented projection and operation-model assumptions.

This checkpoint does not support a claim that every kernel or model has passed, that every emitted transformation is verified, or that the whole compiler and hardware execution are proved correct.
