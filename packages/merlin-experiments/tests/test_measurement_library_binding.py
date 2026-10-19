"""Real paired storage/owner transport with synthetic numerical and cycle observations."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import measurement_evidence as ME
from merlin_experiments.phase2 import measurement_library_binding as MB
from merlin_experiments.phase2 import paired_measurement as PM
from test_ordinary_search_policy import ordinary_inputs  # noqa: F401
from test_paired_measurement import _plan
from test_prepared_compiler_route import prepared  # noqa: F401

from merlin.benchharness import hash_tree


@pytest.fixture(params=[False, True], ids=["library", "prepared"])
def observation(request, tmp_path):
    fixture = request.getfixturevalue("prepared")
    root = tmp_path / "paired-evidence"
    root.mkdir()
    old = _plan(root)
    use_preparation = request.param
    handoff = SimpleNamespace(
        compiler_library_binding=fixture.library_binding,
        compiler_library_selection=fixture.library_snapshot,
        compiler_library_host_selection=fixture.library,
        compiler_pass_preparation=fixture.record if use_preparation else None,
        compiler_pass_selection=fixture.passes if use_preparation else None,
        compiler_pass_preparation_selection=fixture.choice if use_preparation else None,
    )
    baseline = Path(fixture.record["baseline" if use_preparation else "original"]["path"])
    candidate = fixture.arguments["candidate"]
    inputs = PM.PairedInputs(
        SimpleNamespace(run_id="original", digest=fixture.record["original"]["sha256"]),
        handoff,
        SimpleNamespace(capsules=(old.schedule[0].member,)),
        old.phase,
        baseline,
        hash_tree(baseline)["sha256"],
        candidate,
        hash_tree(candidate)["sha256"],
        old.schedule[0].gsim_certificate,
    )
    plan = PM.build_measurement_plan(inputs)
    out = root / "observations"
    out.mkdir()

    def executor(spec, *args, **kwargs):
        return dict(
            execution=spec.as_dict(),
            measurement=dict(
                status="pass",
                numeric="pass",
                gsim_qualification=dict(admitted=True),
                per_sim=dict(
                    spike=dict(correct=True),
                    gsim=dict(
                        correct=True,
                        cycles=100,
                        provenance=dict(
                            tier="L3",
                            simulator="gsim",
                            oracle_kind="rtl_gsim",
                            derived_from_rtl=True,
                            cycle_accurate=True,
                            elf_sha256="f" * 64,
                            reused_measurement=False,
                        ),
                    ),
                ),
            ),
        )

    rows, _ = PM.execute_schedule(
        plan,
        out,
        contract_root=tmp_path,
        timeout=1,
        target_experiment=None,
        rtl_identity={},
        hardware_counters=False,
        executor=executor,
        progress=lambda value: None,
    )
    result_path = out / "paired-cells.json"
    PM.write_json(result_path, dict(schema="paired_arm4_result_cells_v2", cells=rows))
    identity = dict(
        baseline_sha256=inputs.baseline_sha256,
        candidate_sha256=inputs.candidate_sha256,
        compiler_library_binding_sha256=fixture.library_binding["sha256"],
    )
    manifest = dict(
        schema="paired_arm4_performance_campaign_v2",
        status="GO",
        phase=inputs.phase,
        functional_run_id=inputs.functional.run_id,
        functional_submission_sha256=inputs.functional.digest,
        candidate_record_sha256="b" * 64,
        candidate_sha256=inputs.candidate_sha256,
        gsim_certificate=dict(sha256="c" * 64),
        frozen_corpus=dict(manifest_sha256="d" * 64, capsules_sha256="e" * 64, visibility=inputs.phase),
        measurement_plan=plan.declaration,
        measurement_plan_sha256=plan.declaration_sha256,
        completion=ME.completion_report(rows, plan.expected),
        engine_policy=dict(
            rtl_execution_backends=["gsim"],
            timing_authority="gsim",
            verilator="prelaunch_certificate_qualification_only",
        ),
        identity_before=identity,
        identity_after=copy.deepcopy(identity),
        fork_before=dict(ok=True),
        fork_after=dict(ok=True),
        raw_results=dict(paired_cells=str(result_path), paired_cells_sha256=PM.sha256_file(result_path)),
    )
    if use_preparation:
        manifest.update(
            schema="paired_arm4_prepared_performance_campaign_v1",
            comparison_baseline_sha256=inputs.baseline_sha256,
            compiler_passes=fixture.passes.record(),
            compiler_preparation=fixture.choice.record(),
        )
        for key in ("identity_before", "identity_after"):
            manifest[key]["compiler_pass_preparation_sha256"] = fixture.record["sha256"]
    path = out / "campaign.json"
    PM.write_json(path, manifest)
    binding = ME.MeasurementBinding(
        inputs.phase,
        inputs.functional.run_id,
        inputs.functional.digest,
        "b" * 64,
        inputs.candidate_sha256,
        "d" * 64,
        "e" * 64,
        "c" * 64,
        **MB.binding_arguments(handoff),
    )
    return SimpleNamespace(path=path, binding=binding, manifest=manifest, rows=rows, prepared=use_preparation)


def test_both_selected_treatments_reopen_actual_paired_worker_products(observation):
    result = ME.verify_paired_measurement(observation.path, expected=observation.binding)
    assert len(result.rows) == 8
    if observation.prepared:
        assert result.manifest["functional_submission_sha256"] != result.manifest["comparison_baseline_sha256"]


@pytest.mark.parametrize(
    "damage",
    [
        "implicit",
        "plan_release",
        "arm",
        "identity",
        "baseline_parent",
        "policy",
        "raw_release",
        "raw_execution",
        "raw_preparation",
    ],
)
def test_self_consistent_saved_labels_cannot_change_selected_treatments(observation, damage):
    value = observation.manifest
    expected = observation.binding
    if damage == "implicit":
        expected = replace(expected, compiler_library_binding=None, compiler_pass_preparation=None)
    elif damage == "plan_release":
        value["measurement_plan"]["additional_compiler_library"]["sha256"] = "0" * 64
    elif damage == "arm":
        value["measurement_plan"]["schedule"][0]["package_sha256"] = "0" * 64
    elif damage == "identity":
        for key in ("identity_before", "identity_after"):
            value[key]["baseline_sha256"] = expected.candidate_sha256
    elif damage == "baseline_parent":
        value["comparison_baseline_sha256"] = expected.functional_submission_sha256
    elif damage == "policy":
        value["compiler_preparation"] = {"saved_success": True}
    else:
        result_path = Path(value["raw_results"]["paired_cells"])
        results = json.loads(result_path.read_bytes())
        raw_path = Path(results["cells"][0]["raw_result_path"])
        raw_path.chmod(0o644)
        raw = json.loads(raw_path.read_bytes())
        if damage == "raw_release":
            raw["compiler_library_binding_sha256"] = "0" * 64
        elif damage == "raw_preparation":
            raw["compiler_pass_preparation_sha256"] = "0" * 64
        else:
            raw["execution"]["package_sha256"] = "0" * 64
        PM.write_json(raw_path, raw)
        raw_path.chmod(0o444)
        for row in results["cells"]:
            if row["raw_result_path"] == str(raw_path):
                row["raw_result_sha256"] = PM.sha256_file(raw_path)
        PM.write_json(result_path, results)
        value["raw_results"]["paired_cells_sha256"] = PM.sha256_file(result_path)
    value["measurement_plan_sha256"] = PM.sha256_bytes(PM.canonical_bytes(value["measurement_plan"]))
    PM.write_json(observation.path, value)
    with pytest.raises(ME.MeasurementEvidenceError):
        ME.verify_paired_measurement(observation.path, expected=expected)
