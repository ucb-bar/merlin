Final restored-tree run: **908 passed, 44 skipped, 22 failed, 2 errors** (976 cases, 375.26 seconds). All 24 failing/erroring shared cases also fail/error on C10. No new shared-case failures; 967 cases are shared; the two new LITE regressions and seven restored historical retained-stage cases pass. 103 C10 case identities are absent after restoring/deleting the integration tests with their interfaces.

C10 controls: restored areas **763 passed, 44 skipped, 17 failed, 2 errors** (826 cases); advisory composition/fusion files **245 passed, 5 failed** (250 cases). The two C10 batches overlap in six passing cases: their deduplicated union is 1,070 cases (1,002 passed, 44 skipped, 22 failed, 2 errors). This is a bounded comparison, not a claim of a clean repository suite. A separate run of the new no-record synthesis test against C10 fails at its required-record gate as expected.

The required interpreter was `/scratch/loren/core-aten-snap/venvs/merlin-wtC/bin/python -I -B`, run from each checkout under `nice -n 19`, with explicit checkout `src`/`packages/*/src` imports, one BLAS/OpenMP thread and a shared maximum six-CPU affinity. The venv's existing editable paths refer to another checkout, so the main test process explicitly prepends these source roots. A 60-second per-test ceiling was applied equally to both trees to honor the 40-minute limit. Logs and JUnit results are under `/scratch/loren/restore1/artifacts/` (`c10-tests`, `c10-advisory`, `lite-final`); `comparison.json` records exact common/new/retired case identities.

Baseline problems include missing Torch in this interpreter, absent `MERLIN_M2M_PYTHON` in the test environment, two child-interpreter failures, seven neutral-evidence fixture failures, five forkserver `PermissionError` failures, one bounded large-graph timeout, and existing quantization/policy/float-mock assertions. These were not repaired as part of gate restoration. The initial restoration exposed a zero-budget regression; the independent fix was reapplied and both affected cases pass in the final run. The numerical/golden, target freeze/tool selection, rank-zero, complete-output, issuer and recipe regression tests that can run in this environment remain in the selection.

C10 baseline failures/errors for the selected restored areas and retained advisory modules.

| Test | Baseline failure |
|---|---|
| `merlin.tests.targetgen.test_conformance_application_inventory::test_real_vision_and_vla_graphs_are_fully_accounted_but_not_claimed_covered` | Failed: Timeout (>60.0s) from pytest-timeout. |
| `merlin.tests.targetgen.test_quantization_spec::test_selected_recipe_and_observed_admission_do_not_license_host_or_framework` | AssertionError: assert ['ineligible', 'ineligible'] == ['candidate', 'ineligible'] |
| `merlin.tests.targetgen.test_quantization_spec::test_multi_format_candidates_never_borrow_the_selected_format_readout` | AssertionError: assert 'ineligible' == 'unknown' |
| `merlin.tests.targetgen.test_quantization_spec::test_float_semantics_and_unknowns_are_explicit_and_refusals_fail_closed` | AssertionError: assert 'ineligible' == 'unknown' |
| `merlin.tests.targetgen.test_recipe_quantizer_families::test_pt2e_helper_does_not_mutate_torch_globals` | KeyError: 'MERLIN_M2M_PYTHON' |
| `merlin.tests.targetgen.test_recipe_quantizer_stored_weight::test_named_owner_selection_shares_weight_identity_and_geometry[True]` | failed on setup with "ModuleNotFoundError: No module named 'torch'" |
| `merlin.tests.targetgen.test_recipe_quantizer_stored_weight::test_named_owner_selection_shares_weight_identity_and_geometry[False]` | failed on setup with "ModuleNotFoundError: No module named 'torch'" |
| `merlin.tests.targetgen.test_quant_recipe_policy::test_selected_scalar_native_and_single_dynamic_plane_preserve_other_arithmetic[tiny_llama]` | ValueError: tensor-scale derivation changed another arithmetic policy |
| `merlin.tests.targetgen.test_quant_recipe_policy::test_successor_reporting_config_preserves_geometry_and_exact_quantized_comparison` | AssertionError: assert ({'kind': 'causal', 'prefill_tokens': 128, 'decode_steps': 32}, (Program(name='prefill', stage='prefill', inputs=({'from': 'tokens', 'slice': 'prefill'},), steps=1), Program(name='decode', stage='decode', inputs=({'from': 'tokens', 'slice': 'next'}, {'from': 'output', 'program': 'prefill', 'output': 1, 'carry': 1}, {'from': 'output', 'program': 'prefill', 'output': 2, 'carry |
| `merlin.tests.targetgen.test_conformance_regime_dtype::test_regime_dtype_tie_is_stable_across_python_hash_seeds` | subprocess.CalledProcessError: Command '['/scratch/loren/core-aten-snap/venvs/merlin-wtC/bin/python', '-c', "\nimport json\nfrom merlin.targetgen.conformance import Cell, regime_dtype_selection\ncells = [Cell('contraction', 'bf16', None) for _ in range(6)] + [\n    Cell('contraction', 'fp8_e4m3', None) for _ in range(6)\n]\nprint(json.dumps(regime_dtype_selection(cells), sort_keys=True))\n"]' retu |
| `merlin.tests.targetgen.test_conformance_regime_dtype::test_host_only_dtype_map_is_stable_across_python_hash_seeds` | subprocess.CalledProcessError: Command '['/scratch/loren/core-aten-snap/venvs/merlin-wtC/bin/python', '-c', "\nimport json\nfrom merlin.targetgen import conformance as cf, model_coverage as mc\nclass Region:\n    in_dtype = 'f32'\n    def __init__(self, family):\n        self.family = family\n    def resolved_family(self):\n        return self.family\nmc.load_module = lambda path: None\nmc.regions |
| `packages.merlin-experiments.tests.test_phase0_evidence::test_neutral_evidence_never_reads_provider_hooks_headers_or_lowering[missing]` | Failed: neutral evidence touched provider/compiler evidence |
| `packages.merlin-experiments.tests.test_phase0_evidence::test_neutral_evidence_never_reads_provider_hooks_headers_or_lowering[rtl]` | Failed: neutral evidence touched provider/compiler evidence |
| `packages.merlin-experiments.tests.test_phase0_evidence::test_neutral_evidence_never_reads_provider_hooks_headers_or_lowering[host]` | Failed: neutral evidence touched provider/compiler evidence |
| `packages.merlin-experiments.tests.test_phase0_evidence::test_neutral_selected_readout_metadata_reaches_all_facet_inputs` | Failed: selected readout metadata discovered executable support |
| `packages.merlin-experiments.tests.test_phase0_evidence::test_neutral_selected_readout_metadata_refuses_malformed_data[malformed0]` | Failed: selected readout metadata discovered executable support |
| `packages.merlin-experiments.tests.test_phase0_evidence::test_neutral_selected_readout_metadata_refuses_malformed_data[malformed1]` | Failed: selected readout metadata discovered executable support |
| `packages.merlin-experiments.tests.test_phase0_evidence::test_neutral_selected_readout_metadata_refuses_malformed_data[malformed2]` | Failed: selected readout metadata discovered executable support |
| `packages.merlin-experiments.tests.test_phase0_software_spec::test_selected_float_arithmetic_is_forwarded_and_legacy_is_explicit` | TypeError: test_selected_float_arithmetic_is_forwarded_and_legacy_is_explicit.<locals>.<lambda>() got an unexpected keyword argument 'palette' |
| `packages.merlin-experiments.tests.test_fusion_validation::test_suite_refuses_resealed_forged_payloads[input_values-4]` | PermissionError: [Errno 1] Operation not permitted |
| `packages.merlin-experiments.tests.test_fusion_validation::test_suite_refuses_resealed_forged_payloads[references-4]` | PermissionError: [Errno 1] Operation not permitted |
| `packages.merlin-experiments.tests.test_fusion_validation::test_suite_refuses_resealed_forged_payloads[source-4]` | PermissionError: [Errno 1] Operation not permitted |
| `packages.merlin-experiments.tests.test_fusion_validation::test_suite_refuses_resealed_forged_payloads[nodes-4]` | PermissionError: [Errno 1] Operation not permitted |
| `packages.merlin-experiments.tests.test_fusion_validation::test_parallel_payload_verification_matches_serial` | PermissionError: [Errno 1] Operation not permitted |

Exact restored-tree file selection (C10 controls use the same preexisting paths, with its historical versions of restored test files; the new `test_phase0_lite.py` is tested separately against C10):

```text
merlin/tests/targetgen/test_application_inventory_rank_zero.py
merlin/tests/targetgen/test_conformance_derivation.py
merlin/tests/targetgen/test_conformance_application_inventory.py
merlin/tests/targetgen/test_capsule_source.py
merlin/tests/targetgen/test_quantization_spec.py
merlin/tests/targetgen/test_corpus_synth.py
merlin/tests/targetgen/test_quant_recipe.py
merlin/tests/targetgen/test_recipe_quantizer_families.py
merlin/tests/targetgen/test_recipe_quantizer_stored_weight.py
merlin/tests/targetgen/test_recipe_quantizer_semantic_mac.py
merlin/tests/targetgen/test_phase0_empty_values.py
merlin/tests/targetgen/test_derived_column_readouts.py
merlin/tests/targetgen/test_quant_recipe_policy.py
merlin/tests/targetgen/test_conformance_regime_dtype.py
merlin/tests/targetgen/test_frontend_trace_reader.py
merlin/tests/targetgen/test_functional_truth_compositions.py
packages/merlin-experiments/tests/test_phase0_evidence.py
packages/merlin-experiments/tests/test_phase0_freeze.py
packages/merlin-experiments/tests/test_phase0_frozen_target_data.py
packages/merlin-experiments/tests/test_phase0_execution_runtime.py
packages/merlin-experiments/tests/test_native_host_outputs.py
packages/merlin-experiments/tests/test_phase0_performance_scale.py
packages/merlin-experiments/tests/test_sealed_m2m_issuer_admission.py
packages/merlin-experiments/tests/test_phase0_core_aten_stage.py
packages/merlin-experiments/tests/test_component_generation.py
packages/merlin-experiments/tests/test_phase0_software_spec.py
packages/merlin-experiments/tests/test_software_derivation_diagnostics.py
packages/merlin-experiments/tests/test_phase0_spec_drift_identity.py
packages/merlin-experiments/tests/test_phase0_generation_scope.py
packages/merlin-experiments/tests/test_phase0_materialized_capsules.py
packages/merlin-experiments/tests/test_phase0_writer_attestations.py
packages/merlin-experiments/tests/test_operation_coverage_ledger.py
packages/merlin-experiments/tests/test_operation_coverage_composition_aliases.py
packages/merlin-experiments/tests/test_release_composition_constraints.py
packages/merlin-experiments/tests/test_validated_graph_runtime_golden.py
packages/merlin-experiments/tests/test_phase0_capture_selection.py
packages/merlin-experiments/tests/test_phase0_capture_worker_limit.py
packages/merlin-experiments/tests/test_phase0_inventory_blocker.py
packages/merlin-experiments/tests/test_phase0_frozen_relocations.py
packages/merlin-experiments/tests/test_completed_phase0_artifact.py
packages/merlin-experiments/tests/test_phase0_output_destination.py
packages/merlin-experiments/tests/test_phase0_explicit_plan.py
packages/merlin-experiments/tests/test_phase0_lite.py
merlin/tests/targetgen/test_composition_corpus.py
packages/merlin-experiments/tests/test_fusion_validation.py
packages/merlin-experiments/tests/test_release_composition_constraints.py
packages/merlin-experiments/tests/test_validated_graph_runtime_golden.py
```

Source-pinning runner used with the isolated interpreter, from `merlin/`:

```python
import sys, pathlib,os
root=pathlib.Path.cwd()
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:6]))
sys.path[:0]=[str(root),str(root/'src'),*[str(p) for p in sorted((root/'packages').glob('*/src'))]]
os.environ.setdefault('MERLIN_REPO_ROOT',str(root))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
```

Reproduce this workspace's final run:

```bash
mapfile -t lite_tests < ../artifacts/final-test-files.txt
nice -n 19 /scratch/loren/core-aten-snap/venvs/merlin-wtC/bin/python -I -B   ../artifacts/run-tests.py -vv --timeout=60 "${lite_tests[@]}"
```
