# Integration test inventory — 2026-10-29

This is a metadata inventory, not another test run. Counts below belong to separate, often overlapping source cuts. **Do not add them.** Host protocols and fake jobs grant no Lean proof credit; skipped workload tests were not executed. Historical receipts require a new live check before delivery.

## Main team integration cohorts

| Cohort | Result | Scope and limits | Exact evidence |
| --- | --- | --- | --- |
| All-ten selection host controls | 25 pass / 0 fail / 0 error / 0 skip; 12.34s | Host/controller tests; no simulated acceptance is proof credit. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-all-checks-20261029/host-v1/results.xml) |
| All-ten initial real compiler | 3 pass / 1 fail / 0 error / 0 skip; 440.33s | 3 pass/1 stale expected-verdict assertion; all seven real workers accepted. Corrected by compiler-v2; this cohort is not green. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-all-checks-20261029/compiler-v1/results.xml) |
| All-ten corrected environment/textual DQ | 2 pass / 0 fail / 0 error / 0 skip; 290.69s | 2 real compiler cases pass; 290.71s log. Proposed/draft/ISA ownership limits unchanged. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-all-checks-20261029/compiler-v2/results.xml) |
| Kernel portable host collection | 20 pass / 0 fail / 0 error / 16 skip; 0.40s | 20 small host cases; 16 opt-in full-size cases skipped explicitly, not claimed executed. | [HOST-PORTABLE-v1.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/kernel-benchmark-formal-gate-20261029/HOST-PORTABLE-v1.xml) |
| Kernel layout controls latest | 26 pass / 0 fail / 0 error / 16 skip; 0.57s | 26 small host/probe-argument/corruption controls; 16 heavy cases skipped. | [HOST-LAYOUT-v2.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/kernel-benchmark-formal-gate-20261029/HOST-LAYOUT-v2.xml) |
| Root bounded source/controller policy | 30 pass / 0 fail / 0 error / 0 skip; 15.09s | Host source-admission/coverage policy; no numerical or pass-boundary proof credit. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/root-policy-v1/results.xml) |
| Corrected focused host regression | 246 pass / 0 fail / 0 error / 0 skip; 139.40s | 246 pass;27 deliberately deselected according to retained run.log. Overlaps other host cohorts. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/host-final-v1/results.xml) |
| Registry and owner identity | 11 pass / 0 fail / 0 error / 0 skip; 1.67s | 11 small canonical registry/owner controls. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/registry-current-v1/results.xml) |
| Final source policy deltas | 32 pass / 0 fail / 0 error / 0 skip; 3.55s | 32 host cases = 23 intrinsic memory-width  + 3 prior-worker-failure identity  + 6 controller checks; overlaps component cohorts. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/policy-final-deltas-v1/results.xml) |
| Native-reader source/export custody | 37 pass / 0 fail / 0 error / 0 skip; 1.48s | 37 host controls; parser grammar exclusion, source/export hashes and original failure identity. No semantic worker acceptance implied. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/native-source-export-binding-v2/results.xml) |
| Current policy actual compiler paths | 4 pass / 0 fail / 0 error / 0 skip; 40.27s | 4 actual compiler cases: genuine admitted host mutation rejection/quarantine; source-excluded overflowing memory DCE and per-channel i64 DQ on both routes continue ordinary compilation with zero proof jobs/credit. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/final-qualification/policy-entry-v3/results.xml) |
| Positive source recognizers | 5 pass / 0 fail / 0 error / 0 skip; 27.14s | 5 real source-adapter tests, 10 source decisions; no boundary-proof or numerical approval credit. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/final-qualification/source-admission-positive-v4/results.xml) |
| Bounded final memory on old five-check bundle | 2 pass / 0 fail / 0 error / 0 skip; 1238.92s | 2 real cases / 4 accepted workers,24 native observations + 6 untouched-entry checks,81 hashes unchanged. Untiled3 and tiled32-tail257; predates all-ten default change. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/final-qualification/memory-final-v2/results.xml) |
| Generic UNROLL/TILE ABI compatibility | 54 pass / 0 fail / 0 error / 0 skip; 6.68s | 54 host ABI/profile controls; not 54 proofs. | [FINAL-HOST.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/buffer-rank-one-20261029/llm-context-tile-gate/FINAL-HOST.xml) |
| Generic UNROLL/TILE final public+replay | 2 pass / 0 fail / 0 error / 0 skip; 232.48s | 2 actual public tests/4 fresh proofs,worker+replay per kind; UNROLL41roots/940 declarations,TILE47/929,STD3. Actual native context occurrences were added probes; not optimization of original model loops. | [FINAL-SDK-v1.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/buffer-rank-one-20261029/llm-context-tile-gate/FINAL-SDK-v1.xml) |
| Configuration original source predicates | 30 pass / 0 fail / 0 error / 0 skip; 8.10s | 30 host source-only checks; optimizers/proposed targets not consulted. | [canonical-host30.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/buffer-configuration/canonical-host30.xml) |
| Configuration runtime/replay host protocol | 21 pass / 0 fail / 0 error / 0 skip; 25.90s | 21 host protocol cases,including marked private ledger fixtures; no fake completion/acceptance receives proof credit. | [canonical-runtime-host21.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/buffer-configuration/canonical-runtime-host21.xml) |
| Configuration retained final workers/replay | 3 pass / 0 fail / 0 error / 0 skip; 175.85s | 3 actual tests/4 fresh queries, 2 accepted workers + fresh original C64 replay + genuine operand NotEstablished, 27 tiny native observations;139 refs unchanged. | [public.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/buffer-configuration/installed-qualification-final-v2/public.xml) |
| Original scalar/dense NaN custody | 137 pass / 0 fail / 0 error / 0 skip; 20.69s | 137 host admissions/custody cases. Source+target/nested metadata/resource and exact dense-loop-byte controls; no SDK proof success claimed. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/default-all-memory-20261029/scalar-nan-refusal-v1/canonical-host-v1/results.xml) |
| NaN refusal classification | 26 pass / 0 fail / 0 error / 0 skip; 6.90s | 26 host classifications: known source exclusions Unsupported; resources/missing SDK remain Unavailable; known proof false Rejected. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/default-all-memory-20261029/nan-loop-classification-v1/canonical/results.xml) |
| Host fusion restored source/worker qualification | 40 pass / 0 fail / 0 error / 0 skip; 163.56s | 40 total cases = 20 actual SDK/native cases + 20 synthetic protocol cases. Those 20 synthetic cases are not proofs. | [results.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/remote-8f30c77-integration-20261029/host-fusion-qualified-v1/results.xml) |
| Rank-one uniform initializer installed qualification | 12 pass / 0 fail / 0 error / 0 skip; 6.74s | 12 host cases; companion sdk.xml has 21 SDK/native cases at this recorded cut. No sum with overlapping memory/cohort tests. | [host.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/buffer-rank-one-20261029/installed-qualification-v1/host.xml) |
| Rank-one uniform initializer actual compiler/native/replay | 21 pass / 0 fail / 0 error / 0 skip; 6092.44s | 21 SDK/native tests at recorded rank-one i32 fixture cut; extents 1/3/65536, both routes, fresh process replay, classified mutants,250 roots/2545 audited theorems per recorded gates. | [sdk.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/buffer-rank-one-20261029/installed-qualification-v1/sdk.xml) |
| Recursive configuration host | 73 pass / 0 fail / 0 error / 0 skip; 41.20s | 73 host cases at V6 cut; 5 legacy reconstructions are separate metadata controls. | [host.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/configuration-hoist-nested-20261029/installed-qualification-v1/host.xml) |
| Recursive configuration public native/replay/refusals | 5 pass / 0 fail / 0 error / 0 skip; 369.28s | 5 public pytest cases exercise 3 accepted original-owner workers, 1 replay, 2 genuine refusals, 27 finite native rows; not six independent pytest cases. | [public.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/configuration-hoist-nested-20261029/installed-qualification-v1/public.xml) |
| Default-all original memory integration | 2 pass / 0 fail / 0 error / 0 skip; 1207.78s | 2 actual compiler cases / 4 accepted workers at old five-check source cut. Separate from later final memory-v2 refresh. | [compiler.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/default-all-memory-20261029/installed-qualification-v3/compiler.xml) |
| Source guard intrinsic index64 correction | 23 pass / 0 fail / 0 error / 0 skip; 2.85s | 23 host source-only width checks; zero, MAX63 and a huge unused view positive; overflow constants/DIM results excluded;three actual DCE pairs but no SDK. | [host-v1.xml](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/memory-source-bounds-fix/host-v1.xml) |

The JSON companion records every selected test ID, current suite-file location, available exact command argv and source-cut records. JUnit suite elapsed time is shown above; wall/process summaries may differ slightly. For example, compiler-v2 reports 290.71s in its run log. A missing command record is left missing rather than reconstructed as an executed command. Nearby source-cut files are locators; this inventory does not independently prove their association to every historical XML.

## Main suite source locations

These are current canonical suite locations; their historical bytes remain bound only by each cohort's own receipts and source records.

| Test suite | Main inventory cohorts |
| --- | --- |
| [merlin/tests/ir/test_admission_existing_failure.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_admission_existing_failure.py) | Final source policy deltas; Native-reader source/export custody |
| [merlin/tests/ir/test_compiler_bounded_policy.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_bounded_policy.py) | Root bounded source/controller policy; Final source policy deltas; Current policy actual compiler paths; Positive source recognizers |
| [merlin/tests/ir/test_compiler_buffer_rank_one.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_buffer_rank_one.py) | Rank-one uniform initializer installed qualification; Rank-one uniform initializer actual compiler/native/replay |
| [merlin/tests/ir/test_compiler_configuration.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_configuration.py) | Configuration retained final workers/replay |
| [merlin/tests/ir/test_compiler_configuration_coverage.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_configuration_coverage.py) | Corrected focused host regression; Configuration runtime/replay host protocol |
| [merlin/tests/ir/test_compiler_configuration_hoist_c_consumer.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_configuration_hoist_c_consumer.py) | Recursive configuration host |
| [merlin/tests/ir/test_compiler_configuration_hoist_host.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_configuration_hoist_host.py) | Recursive configuration host |
| [merlin/tests/ir/test_compiler_configuration_hoist_multi.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_configuration_hoist_multi.py) | Recursive configuration host |
| [merlin/tests/ir/test_compiler_configuration_hoist_recursive.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_configuration_hoist_recursive.py) | Configuration retained final workers/replay; Recursive configuration host; Recursive configuration public native/replay/refusals |
| [merlin/tests/ir/test_compiler_coverage.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_coverage.py) | Root bounded source/controller policy; Corrected focused host regression |
| [merlin/tests/ir/test_compiler_default_memory.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_default_memory.py) | Bounded final memory on old five-check bundle; Default-all original memory integration |
| [merlin/tests/ir/test_compiler_family_source_profiles.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_family_source_profiles.py) | Corrected focused host regression; Configuration original source predicates |
| [merlin/tests/ir/test_compiler_host_graph_fusion.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_host_graph_fusion.py) | Current policy actual compiler paths; Host fusion restored source/worker qualification |
| [merlin/tests/ir/test_compiler_loop_generalization.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_loop_generalization.py) | Registry and owner identity |
| [merlin/tests/ir/test_compiler_loop_generic.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_loop_generic.py) | Generic UNROLL/TILE ABI compatibility; Generic UNROLL/TILE final public+replay |
| [merlin/tests/ir/test_compiler_loop_generic_tile.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_loop_generic_tile.py) | Generic UNROLL/TILE ABI compatibility; Generic UNROLL/TILE final public+replay |
| [merlin/tests/ir/test_compiler_loop_source_profile.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_loop_source_profile.py) | Generic UNROLL/TILE ABI compatibility |
| [merlin/tests/ir/test_compiler_macc_wiring.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_macc_wiring.py) | All-ten selection host controls; All-ten initial real compiler; Corrected focused host regression |
| [merlin/tests/ir/test_compiler_memory_origin.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_memory_origin.py) | Corrected focused host regression; Original scalar/dense NaN custody |
| [merlin/tests/ir/test_compiler_qdq.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_compiler_qdq.py) | All-ten selection host controls; All-ten initial real compiler; All-ten corrected environment/textual DQ; Corrected focused host regression |
| [merlin/tests/ir/test_host_fusion_input_binding.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_host_fusion_input_binding.py) | Host fusion restored source/worker qualification |
| [merlin/tests/ir/test_kernel_benchmark_formal_gates.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_kernel_benchmark_formal_gates.py) | Kernel portable host collection; Kernel layout controls latest |
| [merlin/tests/ir/test_lean_compiler_gate_scope.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_lean_compiler_gate_scope.py) | All-ten selection host controls; Corrected focused host regression; Registry and owner identity; Generic UNROLL/TILE ABI compatibility |
| [merlin/tests/ir/test_lean_pass_gate.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_lean_pass_gate.py) | All-ten initial real compiler; All-ten corrected environment/textual DQ |
| [merlin/tests/ir/test_loop_nan_admission.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_loop_nan_admission.py) | Corrected focused host regression; Original scalar/dense NaN custody; NaN refusal classification |
| [merlin/tests/ir/test_memory_nan_admission.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_memory_nan_admission.py) | Corrected focused host regression; Original scalar/dense NaN custody |
| [merlin/tests/ir/test_memory_source_exclusion_compiler.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_memory_source_exclusion_compiler.py) | All-ten initial real compiler; Current policy actual compiler paths |
| [merlin/tests/ir/test_memory_source_profile_bounds.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_memory_source_profile_bounds.py) | Final source policy deltas; Native-reader source/export custody |
| [merlin/tests/ir/test_source_admission_binding.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_source_admission_binding.py) | Corrected focused host regression; Native-reader source/export custody |
| [merlin/tests/ir/test_source_export_classification.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_source_export_classification.py) | Native-reader source/export custody |
| [merlin/tests/ir/test_transform_input_binding.py](/scratch/grammy/blast/merlin-lean-dce-cse/merlin/tests/ir/test_transform_input_binding.py) | Root bounded source/controller policy; Corrected focused host regression |
| [out/build/pass-bounded-models-20261029/memory-source-bounds-fix/candidates/merlin/tests/ir/test_memory_source_profile_bounds.py](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/memory-source-bounds-fix/candidates/merlin/tests/ir/test_memory_source_profile_bounds.py) | Source guard intrinsic index64 correction |

## Protocol, parser and setup controls

| Control | What actually ran | Evidence |
| --- | --- | --- |
| Parallel runner protocol and tiny guardian | 16 fake tasks, 8 exclusive CPU leases/non-FIFO completions,4 retained controlled failures. One real tiny detached-child timeout guardian cleanup. No compiler, SDK, full-size arrays or kernel code. | [RESULT.json](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/kernel-benchmark-formal-gate-20261029/parallel-host-v1/RESULT.json) |
| Real empty host ABI probe | Small headerless C pointer-width/LLVM layout probe only; no workload compilation, reference arrays, native kernel or Lean proof. | [HOST-INDEX-OBSERVATION-v2.json](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/kernel-benchmark-formal-gate-20261029/HOST-INDEX-OBSERVATION-v2.json) |
| Sixteen exact full-size source fixtures | 16 xDSL verifier checks+16 registered native parser checks+16 host contract, signature and actual-operand-binding checks. No arrays computed, LLVM/native kernel or SDK proof. | [FIXTURE_MANIFEST.json](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/kernel-benchmark-formal-gate-20261029/fixtures/FIXTURE_MANIFEST.json) |
| Buffer width/rank bridge host audit | 19 admitted real producer→prepare_chain cases (16 width/rank/routes + two mixed ranks + one reused sibling IV name), three source-excluded bridge refusals. No new Lean/native width evidence. | [AUDIT.json](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/pass-bounded-models-20261029/buffer-domain-audit/AUDIT.json) |
| Initial sixteen-kernel setup attempt | All 16 cases fail at the same missing selected host data-layout prerequisite before proof or native kernel work. Layout-repair controls and fresh next campaign are separate; no pass credit. | [CAMPAIGN-RESULT.json](/scratch/grammy/blast/merlin-lean-dce-cse/out/build/kernel-benchmark-formal-gate-20261029/campaign-v1/CAMPAIGN-RESULT.json) |

## Retained failures, interruptions and corrections

| Earlier attempt | Finding | Later correction or scope |
| --- | --- | --- |
| pass-all-checks/compiler-v1(3 pass / 1 fail) | stale expected-verdict list omitted newly selected accepted Q/DQ check | pass-all-checks/compiler-v2(2 pass);seven initial workers accepted |
| bounded final-qualification/compiler-v1(11 pass / 1 fail) | old host-mutation injection hook failed to mutate newer implementation | policy-entry-v3 actual host negative now genuinely rejected/quarantined |
| bounded host-regressions-v1(210 pass1fail) | same stale host-mutation test | host-final-v1(246 pass, 27 deselected) |
| configuration runtime-host-v1(8 pass / 2 fail) | host test assumed an 8-bit field,actual fixture was 32-bit; the diagnostic changed | runtime-host-v2(10 pass),v3(21 pass),canonical-runtime-host21(21 pass) |
| configuration installed-qualification-v1/public.xml(3 pass) | pytest deleted passed receipt trees under default retention policy | installed-qualification-final-v2/public.xml(3 pass),retention enabled and 139-reference independent audit |
| configuration-hoist-multi/installed-host-v1(77 pass / 2 fail) | single-loop V3 diagnostics were replaced by broad V4 fallback | actual multi-loop dispatchguard;installed-host-v2(79 pass)+canonical 11 pass |
| default-all-memory installed-qualification-v1(first compiler failure) | native property dictionary order differed from original parent generic print | property-order 46 host controls;later compiler-v3 2 pass |
| default-all-memory installed-qualification-v2(setup error) | new additive incoming proof modules lacked current built checker products | bounded rebuild and later compiler-v3 2 pass; not observed mathematical rejection |
| kernel campaign-v1(all 16 fail) | missing exact selected compiler data layout before gate/native work | HOST-LAYOUT-v2(26 pass / 16 skip)+real empty ABI probe;fresh campaign still separate |
| interrupted default-memory source-custody attempt | root intentionally stopped own run for NaN custody repair; not a passing or semantic-false case | 137 and 26 host hardening,final memory-v2 2 pass;see handoff for exact cut |

## Prior reported numerical evidence

The imported colleague reports of 143 MACC passes (two missing-artifact skips) and 71 Q/DQ passes are prior evidence. They are not presented as new reruns by this team. The real Q/DQ/MACC compiler cases above are the focused team reruns, with their own source cuts.


## Exhaustive retained cohorts

[ALL-TEST-COHORTS.csv](ALL-TEST-COHORTS.csv) lists all 248 identified retained JUnit files: the original 232 plus the 16 successful campaign-v2 records. It includes failures, superseded attempts and explicit skips. The original local JSON/Markdown inventory is preserved unchanged as a historical 232-file snapshot; this consolidated CSV supplies the count corrections below and the 16 later records. Individual test IDs, failure details and available exact command records remain in that local snapshot and the per-case campaign artifacts. No overlapping grand total is claimed.

## Current full-size campaign

Campaign v2 subsequently passed all 16 full-size cases with eight concurrent workers, 106 immutable input arrays, 81 exact output arrays, seven accepted typed rewrite checks and 41 unchanged observed boundaries. [RESULTS.csv](RESULTS.csv) lists each case. The initial campaign-v1 failures were test-harness setup failures and remain retained.

`elapsed_seconds` in [RESULTS.csv](RESULTS.csv) and the per-case summary is full
pytest-case walltime, including setup, compilation, gates, reference/native
comparisons and cleanup. It is not isolated kernel latency or kernel performance.
The CSV's JUnit `suite_seconds` is a separate suite-header measurement.

## Historical JUnit count anomalies

The consolidated CSV keeps declared header counts and independently parsed child
counts in separate columns. Its `passed`/`failed`/`errors`/`skipped` columns use
the header; `count_status` and `count_anomaly` identify the five exceptions below.
An inconsistent or partial row is not a confirmed complete passing run. Parsed
failure counts count failed test cases, rather than multiple `<failure>` elements
within one case. Nameless empty children earn no inferred pass credit.

| Retained XML (relative to `out/build/`) | Declared header | Parsed children | Accounting |
| --- | --- | --- | --- |
| `default-all-memory-20261029/post-format-qualification-v1/results.xml` | 0 tests | 1 nameless empty child | Incomplete/interrupted; **0 passes**, no credit |
| `pass-verif-integration-20261028/resume-loop-replay.xml` | 0 tests | 1 nameless empty child | Incomplete/interrupted; **0 passes**, no credit |
| `pass-proof-completion-20261029-v1/final.xml` | 51 tests, 0 failures | 52 children, 51 named | Header records 51 passes; the extra empty child does not establish a 52nd pass |
| `pass-verif-integration-20261028/dce-cse-regression.xml` | 664 tests, 3 failures | 665 children, 664 named, 3 failed cases | Header-derived 661 passes; not 662; failed historical cohort |
| `pass-verif-integration-20261028/resume-owned-host.xml` | 59 tests, 6 failures | 57 named children, 5 failed cases containing 6 failure elements | Inconsistent/partial; header-derived 53 passes conflicts with 52 parsed named successes; neither count establishes a complete green run |

## Newly imported colleague audit

The separate [frozen audit handoff](../frozen-followup-20261029/HANDOFF.md),
[compact results](../frozen-followup-20261029/RESULTS.json),
[controller review](../frozen-followup-20261029/controller-review.md) and
[proof-domain review](../frozen-followup-20261029/proof-domain-review.md) report
results at `591687da8bf6a7f99807dbe0978ff297669867d3`, before the all-ten default.
These are imported colleague evidence, **not reruns by this team**. Their raw
JUnit and proof logs are not bundled locally in this report; they do not add
rows or pass totals to the 248-file inventory.

| Imported finding | Scope |
| --- | --- |
| 207 distinct existing tests passed, zero skips | Four overlapping cohorts deduplicated by that audit; their raw sum of 290 is not 290 independent tests; also overlaps our earlier suites |
| Full Lean guard passed | Reported exit 0 at that frozen source cut, including lock comparison; not a universal compiler-boundary claim |
| Six unexpected admission-error regression failures, five passing controls | Unexpected projector errors can become nonblocking exclusions; controller gap remains open |
| Nine incomplete-handler-result protocol failures | Mocked `None`, empty or pending results accepted by three older handlers; no canonical false theorem acceptance was demonstrated |
| One extra UNROLL positive passed; strict negative assertion failed | Valid wrong-target output was blocked, but verdict was `unavailable` rather than `not_established` |

The successful 16-kernel campaign does not resolve these controller regressions.
Fixes and current-cut requalification remain separate work; the historical audit
is preserved unchanged.
