# Larger-model gate tests — 2026-10-29

This is a curated evidence index, not another test run. Counts belong to separate,
overlapping source cuts; **do not add them**. Host protocols, parser checks and
mocked proof engines grant no Lean transformation-proof credit. JUnit seconds
are suite walltime, not model or kernel performance.

The fetched source base is `fbe68e88846f109c792f7e2eb0aa43be99d858a1`, including
Jeremy's worker-result guards, typed model refusals and generic-query diagnostic
classification. The current worktree also has the separately reviewed resource
policy, test-owner isolation and demonstration helpers. Exact JUnit hashes,
component suite paths, overlaps and current source hashes are in the
[host test record](../../../build/larger-model-gate-demo-20261029/test-record-current-v3/HOST-TEST-RECORD.json).
Its current-file snapshot is not retroactive attribution of every earlier run.

## Current host and repository checks

| Cohort | Result | Scope and retained evidence |
| --- | --- | --- |
| Current canonical deadline and model harness | **58 pass**, zero errors/skips; 2.03s | 20 intake/audit protocols, 18 campaign protocols and 20 native guards. Whole shared-build timeout is 1,200 seconds; ordinary codegen retains 900 seconds per command; maximum case timeout is 5,400 seconds. [JUnit](../../../build/larger-model-gate-demo-20261029/CURRENT-DEMO-HOST-v4.xml) |
| Corrected combined source/controller policy | **217 pass**, zero errors/skips; 13.918s | 48 worker-result protocols, 17 typed source-admission fault controls, 30 query/cleanup controls, 69 memory-NaN, 26 loop-NaN, 26 resource-policy tests and one original configuration-selection test. [JUnit](../../../build/larger-model-gate-demo-20261029/CURRENT-POLICY-HOST-v2.xml) |
| Initial combined policy attempt | **215 pass / 1 fail**; 12.119s | Source-owner/test-order fixture failure in `test_complete_current_source_closure_includes_actual_local_imports`; retained as failed evidence, followed by the isolation correction and combined rerun. [JUnit](../../../build/larger-model-gate-demo-20261029/CURRENT-POLICY-HOST-v1.xml) |
| Isolated configuration diagnostic | **30 pass**; 3.755s | Query classification, source custody and cleanup controls; these same test IDs also occur in the corrected combined run. [JUnit](../../../build/larger-model-gate-demo-20261029/CONFIG-UNIT-ISOLATED-v1.xml) |
| Broader model-demo host controls | **53 pass**; 1.272s | 20 manifest/receipt-audit protocols, 15 campaign protocols and 18 native guards. This precedes the separately retained two-GiB refresh below. [JUnit](../../../build/larger-model-gate-demo-20261029/CURRENT-DEMO-HOST-v1.xml) |
| Latest native allocation-cap refresh | **18 pass**; 0.643s | Current **2 GiB logical total inputs / 64 MiB total outputs**, inclusive/+1 boundaries and tiny-cap sentinels before RNG, copies and MLP reference arrays. Huge source signatures remain metadata-only. [JUnit](../../../build/larger-model-gate-demo-20261029/CURRENT-NATIVE-CAP-HOST-v2.xml) |
| Earlier native allocation/campaign controls | **33 pass**; 1.036s | 18 native guards plus 15 campaign protocols at the previous **1 GiB** input-cap cut. Qualified predecessor helpers and exact hashes remain archived. [JUnit](../../../build/larger-model-gate-demo-20261029/native-array-budget-20261029/host-v1/results.xml), [source review](../../../build/larger-model-gate-demo-20261029/native-array-budget-20261029/SOURCE-REVIEW.json) |
| Draft tensor resource controls | **26 pass**; 0.563s | Candidate-overlay source-intake tests; the promoted tests run against the canonical policy within the 217-case cohort. This draft is not a separate semantic qualification. [JUnit](../../../build/larger-model-gate-demo-20261029/resource-policy-draft/HOST-v1.xml), [applied source pins](../../../build/larger-model-gate-demo-20261029/resource-policy-applied-v1/MANIFEST.json) |
| Layout/access regression | **111 pass / 2 skip**; 9.740s | Source layout and access-registry tests. The real bubblewrap check skipped because bubblewrap is absent; installed release-wheel coverage skipped because core/experiments wheels were not built. [JUnit](../../../build/larger-model-gate-demo-20261029/CURRENT-LAYOUT-ACCESS-v1.xml) |

All listed JUnit headers match their unique named child records. The 217-case
run contains the earlier 216 test IDs plus one configuration-selection case.
The isolated 30, latest native 18 and earlier combined 33 overlap the larger
cohorts. They are not additional independent passes to sum.

The associated documentation check passed. The structure check still reports
the same **16 inherited findings: two dependency and fourteen size findings**;
it is not a clean structure-gate pass. The skipped real bubblewrap and installed
wheel checks do not establish those release/security properties.

The scratch deadline candidate separately passes 38 overlapping host controls
([JUnit](../../../build/larger-model-gate-demo-20261029/native-build-budget-v4/HOST-v2.xml));
the promoted canonical tests are rerun in the 58-case cohort above. The earlier
20-case native-only draft is also retained. These simulated timeout/failure
controls execute no compiler, native library or Lean proof.

The tensor-family resource policy retains a 4 MiB raw-source ceiling, with
32,768 walked operations, 64 MiB aggregate decoded dense storage and 8,388,608
dense elements. Dense limits are charged before splat expansion. These are
finite host intake ceilings, not enlarged Lean theorems or numerical approvals.
The native byte cap bounds logical arrays, **not peak RSS**, compiler workspaces
or all temporary copies. Q/DQ remains proposed-model; MACC remains draft.

## Actual model qualification is separate

[README.md](README.md) and `summary.json` report the actual compiler workers,
complete native outputs, original source cuts and scope. The frozen **14-route campaign-v3 is terminal: twelve native passes and two
TinyLlama native-build timeouts** at the observer's 300-second outer deadline.
All fourteen compiler phases passed on that recorded v3 source cut. A finite-budget two-case retry is being
prepared; its outcome is not inferred from the successful formal checks. Host results above do not establish
that campaign's success.

The preceding ten-route campaign retained successful compile/receipt audits but
eight native cases failed because its first isolated linker lacked the normal
runtime's `rsqrtf` or `memrefCopy` provider. Those failures remain unchanged. The subsequent
native-only v2 qualification passed ten cases/twenty seeded scenarios using the
ordinary runtime builder, while re-auditing the **25 prior proof jobs**. It did
not execute fresh pass proofs or turn saved results into live delivery authority.
Those are earlier source-cut records, distinct from current campaign-v3.

Selecting all ten gates does not enable every transformation. The manifests
choose the actual transformations, and coverage records distinguish verified
changing boundaries, unchanged boundaries, exclusions and selectors that were
not observed. Neither the model runs nor the host tests prove LLVM, hardware,
pretrained-model accuracy or every possible transformation of the model.

## Suite locations

| Area | Current source locations |
| --- | --- |
| Worker result protocol | [test_gate_builtin_results.py](../../../../merlin/tests/ir/test_gate_builtin_results.py) |
| Typed admission failures and finite intake | [test_source_admission_faults.py](../../../../merlin/tests/ir/test_source_admission_faults.py), [test_tensor_source_resource_policy.py](../../../../merlin/tests/ir/test_tensor_source_resource_policy.py) |
| Query classification, cleanup and selected configuration owner | [test_configuration_elision_gate_unit.py](../../../../merlin/tests/ir/test_configuration_elision_gate_unit.py), [test_compiler_configuration.py](../../../../merlin/tests/ir/test_compiler_configuration.py) — the 217-case cohort selects only `test_current_candidate_binds_its_actual_runner_and_refuses_runner_mutation` from the latter |
| NaN/parser custody | [test_memory_nan_admission.py](../../../../merlin/tests/ir/test_memory_nan_admission.py), [test_loop_nan_admission.py](../../../../merlin/tests/ir/test_loop_nan_admission.py) |
| Demo intake and evidence audit | [test_larger_model_gate_demo.py](../../../../merlin/tests/ir/test_larger_model_gate_demo.py), [larger_model_gate_demo.py](../../../../merlin/tests/ir/larger_model_gate_demo.py) |
| CPU leases, cleanup, source cuts and model orchestration | [test_larger_model_campaign.py](../../../../merlin/tests/ir/test_larger_model_campaign.py), [larger_model_campaign.py](../../../../merlin/tests/ir/larger_model_campaign.py) |
| Native ABI, guarded arrays and allocation caps | [test_larger_model_native_demo.py](../../../../merlin/tests/ir/test_larger_model_native_demo.py), [larger_model_native_demo.py](../../../../merlin/tests/ir/larger_model_native_demo.py) |
| Portable original-source preparation and constructed-network definition | [larger_model_prepare.py](../../../../merlin/tests/ir/larger_model_prepare.py), [larger_model_mlp.py](../../../../merlin/tests/ir/larger_model_mlp.py) |
| Native-only prior-compile re-audit | [larger_model_native_replay.py](../../../../merlin/tests/ir/larger_model_native_replay.py); [scratch protocol controls](../../../build/larger-model-gate-demo-20261029/native-v2-draft/test_native_replay_protocol.py) |
| Layout/access | [test_source_layout_gates.py](../../../../merlin/tests/infra/test_source_layout_gates.py), [test_access_registry.py](../../../../merlin/tests/infra/test_access_registry.py) |

## Reproduction

Use the configured compiler/test Python, Lean SDK and selected Clang described
in the [compiler gate guide](../../../../docs/guides/lean_pass_gate.md). Run from
the repository root. The following are reproduction commands, not reconstructed
historical argv; no additional tests were executed while writing this file.

```sh
export PYTHONPATH="src:merlin/tests/ir${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q \
  merlin/tests/ir/test_gate_builtin_results.py \
  merlin/tests/ir/test_source_admission_faults.py \
  merlin/tests/ir/test_configuration_elision_gate_unit.py \
  merlin/tests/ir/test_memory_nan_admission.py \
  merlin/tests/ir/test_loop_nan_admission.py \
  merlin/tests/ir/test_tensor_source_resource_policy.py \
  merlin/tests/ir/test_compiler_configuration.py::test_current_candidate_binds_its_actual_runner_and_refuses_runner_mutation

python -m pytest -q \
  merlin/tests/ir/test_larger_model_gate_demo.py \
  merlin/tests/ir/test_larger_model_campaign.py \
  merlin/tests/ir/test_larger_model_native_demo.py

python -m pytest -q \
  merlin/tests/infra/test_source_layout_gates.py \
  merlin/tests/infra/test_access_registry.py
```

To prepare the complete model inputs without compiling or generating arrays,
choose fresh output directories:

```sh
python merlin/tests/ir/larger_model_prepare.py \
  --out out/build/larger-model-inputs-rerun \
  --include-large-captures \
  --recaptures /path/to/original/recaptures

nice -n 19 taskset -c 28-31 python merlin/tests/ir/larger_model_campaign.py \
  --manifest-list out/build/larger-model-inputs-rerun/CAMPAIGN-MANIFEST.json \
  --out out/build/larger-model-campaign-rerun \
  --jobs 4 --cpus 28..31 --case-timeout 5400
```

`--include-large-captures` retains the stored ResNet50 and TinyLlama graphs
without changing their shapes, types or operations. The original small-LLaMA
recapture is optional: omit `--recaptures` if unavailable, and unset
`MERLIN_RECAPTURES` to prevent implicit selection. Explicitly requested missing
recaptures fail. With the two large captures, preparation yields twelve routes
without the optional recapture or fourteen with it; every source uses xDSL and
textual routes. Existing capsule scale/provenance and the separately constructed
twelve-layer f16 network remain explicitly labeled; no pretrained/accuracy claim
is supplied by preparation.

The campaign leases at most four available single-CPU lanes, sets nice 19 and
one numerical/Lean thread per case, and uses seeds **20261029 and 20261030**.
It requests all ten gates and requires the manifest's actual changing/verified
selectors. Native comparisons begin only after the compile/receipt audit passes.
The constructed MLP additionally compares every full output against its
independent ordered draft-policy reference and retains every layer's hash and
batch diversity. Unknown cleanup quarantines a lane; source drift, timeouts and
incomplete evidence fail the case. Raw logs, failed cases and all temporary
proof/native products remain retained.

## Retained cohort-level JUnit files

These seventeen earlier files index historical iterations as well as current runs;
the three later deadline/canonical records are linked above.
Each row is a file/cohort, not a unique set of tests. Draft copies and successive
harness iterations are not additional independent test or proof credit. Temporary
fixture-input copies are not indexed as test cohorts.

| Relative to `out/build/larger-model-gate-demo-20261029/` | Pass / fail / error / skip | JUnit seconds | Attribution |
| --- | ---: | ---: | --- |
| [CONFIG-UNIT-ISOLATED-v1.xml](../../../build/larger-model-gate-demo-20261029/CONFIG-UNIT-ISOLATED-v1.xml) | 30 / 0 / 0 / 0 | 3.755 | Isolated diagnostic; overlaps corrected policy cohort |
| [CURRENT-DEMO-HOST-v1.xml](../../../build/larger-model-gate-demo-20261029/CURRENT-DEMO-HOST-v1.xml) | 53 / 0 / 0 / 0 | 1.272 | Combined demonstration host controls |
| [CURRENT-LAYOUT-ACCESS-v1.xml](../../../build/larger-model-gate-demo-20261029/CURRENT-LAYOUT-ACCESS-v1.xml) | 111 / 0 / 0 / 2 | 9.740 | Current infrastructure controls; two unavailable-tool/product skips |
| [CURRENT-NATIVE-CAP-HOST-v2.xml](../../../build/larger-model-gate-demo-20261029/CURRENT-NATIVE-CAP-HOST-v2.xml) | 18 / 0 / 0 / 0 | 0.643 | Latest 2 GiB input-cap refresh |
| [CURRENT-POLICY-HOST-v1.xml](../../../build/larger-model-gate-demo-20261029/CURRENT-POLICY-HOST-v1.xml) | 215 / 1 / 0 / 0 | 12.119 | Retained source-owner fixture failure |
| [CURRENT-POLICY-HOST-v2.xml](../../../build/larger-model-gate-demo-20261029/CURRENT-POLICY-HOST-v2.xml) | 217 / 0 / 0 / 0 | 13.918 | Corrected canonical policy run |
| [HOST-v1.xml](../../../build/larger-model-gate-demo-20261029/HOST-v1.xml) | 14 / 0 / 0 / 0 | 0.383 | Earlier demo-harness iteration |
| [HOST-v2.xml](../../../build/larger-model-gate-demo-20261029/HOST-v2.xml) | 19 / 0 / 0 / 0 | 0.379 | Earlier demo-harness iteration |
| [HOST-v3.xml](../../../build/larger-model-gate-demo-20261029/HOST-v3.xml) | 20 / 0 / 0 / 0 | 0.647 | Demo-harness iteration; overlaps current combined run |
| [NATIVE-HOST-v1.xml](../../../build/larger-model-gate-demo-20261029/NATIVE-HOST-v1.xml) | 9 / 0 / 0 / 0 | 2.324 | Original ABI/input guards, before allocation-budget controls |
| [NATIVE-HOST-v2.xml](../../../build/larger-model-gate-demo-20261029/NATIVE-HOST-v2.xml) | 9 / 0 / 0 / 0 | 0.707 | Original guards rerun; same test IDs |
| [campaign-runner-host-v1/results.xml](../../../build/larger-model-gate-demo-20261029/campaign-runner-host-v1/results.xml) | 14 / 0 / 0 / 0 | 0.420 | Earlier runner protocol iteration |
| [campaign-runner-host-v2/results.xml](../../../build/larger-model-gate-demo-20261029/campaign-runner-host-v2/results.xml) | 15 / 0 / 0 / 0 | 0.438 | Runner protocols; no model/compiler execution |
| [native-array-budget-20261029/host-v1/results.xml](../../../build/larger-model-gate-demo-20261029/native-array-budget-20261029/host-v1/results.xml) | 33 / 0 / 0 / 0 | 1.036 | Previous 1 GiB cap plus runner protocols |
| [native-v2-draft/host-v1/results.xml](../../../build/larger-model-gate-demo-20261029/native-v2-draft/host-v1/results.xml) | 11 / 0 / 0 / 0 | 0.089 | Scratch native-replay protocol draft |
| [native-v2-draft/host-v2/results.xml](../../../build/larger-model-gate-demo-20261029/native-v2-draft/host-v2/results.xml) | 11 / 0 / 0 / 0 | 0.103 | Final scratch protocol cut; overlaps draft v1 |
| [resource-policy-draft/HOST-v1.xml](../../../build/larger-model-gate-demo-20261029/resource-policy-draft/HOST-v1.xml) | 26 / 0 / 0 / 0 | 0.563 | Candidate-overlay resource tests, later rerun canonically |

Raw JUnit/process/proof/native products live locally under the indicated
`out/build/` roots; this curated document does not bundle them or restore live
proof authority. Earlier integration and sixteen-kernel evidence is separately
indexed in the [previous test handoff](../kernel-benchmark-formal-gate-20261029/TESTS.md)
and its [248-cohort inventory](../kernel-benchmark-formal-gate-20261029/ALL-TEST-COHORTS.csv).
Those overlapping historical source cuts are not relabeled as current results.
This work did not run the full repository suite.
