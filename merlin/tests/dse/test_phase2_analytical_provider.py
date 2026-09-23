from __future__ import annotations

import hashlib
import json

import pytest

from merlin.perf.phase2_analytical_provider import build_fast_evaluator_installation
from merlin.perf.phase2_portfolio import evaluate_fast_portfolio


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


MODELS = tuple(_sha(f"member-{index}") for index in range(4))
CORPORA = {model: _sha(f"corpus-{index}") for index, model in enumerate(MODELS)}
RECEIPTS = tuple(_sha(f"receipt-{index}") for index in range(5))


def _calibration():
    return {
        "schema": "phase2_host_analytical_calibration_v1",
        "target_sha256": _sha("target"),
        "evidence_sha256s": list(RECEIPTS),
        "composition": {
            "operator": "sum", "eta": 0.0, "provenance_sha256": RECEIPTS[0],
        },
        "movement_balance": {
            "schema": "merlin_movement_balance_v1", "status": "derived",
            "peak_bytes_per_cycle": 4.0, "base_latency_cycles": 1.0,
            "domain_bytes": [8, 128], "n_distinct_sizes": 4,
            "residual_cycles": [0.0, 0.0, 0.0, 0.0],
            "provenance_sha256": RECEIPTS[1],
        },
        "accelerator_compute_roles": ["execute"],
        "risk_score": 0.1,
        "features": [
            {
                "id": "target-compute", "pointer": "/target_activity/issued/compute_instructions",
                "resource": "arithmetic", "kind": "compute",
                "cycles_per_unit": {"lo": 1.0, "hi": 1.0},
                "floor_cycles_per_unit": 0.5, "effects": ["tiling"],
                "provenance_sha256": RECEIPTS[2],
            },
            {
                "id": "target-movement", "pointer": "/target_activity/issued/movement_instructions",
                "resource": "transfer", "kind": "movement",
                "physical_bytes_per_unit": 8.0, "commands_per_unit": 1,
                "floor_cycles_per_unit": 1.0, "effects": ["movement"],
                "provenance_sha256": RECEIPTS[3],
            },
            {
                "id": "verified-encoding", "pointer": "/encoding_activity/count",
                # Conversion traffic uses the same serialized transfer resource in this calibration.
                # A separate resource would require evidence that partitions aggregate overlap.
                "resource": "transfer", "kind": "encoding",
                "cycles_per_unit": {"lo": 2.0, "hi": 2.0},
                "physical_bytes_per_unit": 16.0, "commands_per_unit": 1,
                "transitions_per_unit": 1, "floor_cycles_per_unit": 1.0,
                "effects": ["encoding"], "provenance_sha256": RECEIPTS[3],
            },
            {
                "id": "host-control", "pointer": "/host_activity/dynamic_operations/scalar",
                "resource": "control", "kind": "fixed",
                "cycles_per_unit": {"lo": 1.0, "hi": 1.0},
                "floor_cycles_per_unit": 0.5, "effects": ["lowering"],
                "provenance_sha256": RECEIPTS[4],
            },
        ],
    }


class _Sentinel:
    def __init__(self, member: str):
        self.capsule_sha256 = member


def _task(index, sources, *, accelerated, reads, writes, kind):
    return ({
        "task_index": index, "kind": kind, "source_op_indices": list(sources),
        "reads": list(reads), "writes": list(writes),
        "instruction_start": index, "instruction_end": index + 1,
    }, {
        "task_index": index, "source_op_indices": list(sources),
        "declared_task_kind": kind,
        "role_counts": {"execute": 1} if accelerated else {},
    })


def _arm(label: str, *, compute: int, movement: int, encodings: int,
         host_operations: int, fused: bool):
    if fused:
        pairs = [
            _task(0, range(3), accelerated=True, reads=("input",),
                  writes=("middle0",), kind="fused"),
            _task(1, range(3, 6), accelerated=True, reads=("middle0",),
                  writes=("output",), kind="fused"),
        ]
    else:
        pairs = [
            _task(0, (0, 1), accelerated=False, reads=("input",), writes=("middle0",),
                  kind="preprocess"),
            _task(1, (2, 3), accelerated=True, reads=("middle0",), writes=("middle1",),
                  kind="compute"),
            _task(2, (4, 5), accelerated=False, reads=("middle1",), writes=("output",),
                  kind="postprocess"),
        ]
    tasks, evidence_tasks = zip(*pairs, strict=True)
    global_plan = {
        "schema": "mixed_program_plan_v1", "source_op_count": 6,
        "tasks": list(tasks), "entry_bindings": ["input"],
        "output_bindings": ["output"],
    }
    command = {
        "commands": [],
        "tensors": {
            name: {"shape": [4], "dtype": "i8", "role": role}
            for name, role in (("input", "input"), ("middle0", "intermediate"),
                               ("middle1", "intermediate"), ("output", "output"))
        },
        "params": {"global_program_plan": global_plan},
    }
    command_text = json.dumps(command, sort_keys=True, separators=(",", ":"))
    command_sha, lowered_sha, plan_sha = (_sha(command_text), _sha(label), _sha("plan-" + label))
    transitions = [{
        "id": f"transition-{index}", "status": "verified",
        "load_payload_bytes": 8, "store_payload_bytes": 8,
    } for index in range(encodings)]
    plan = {
        "status": "verified", "source_operations": 6,
        "candidate_command_buffer_sha256": command_sha,
        "candidate_lowered_sha256": lowered_sha, "plan_digest": plan_sha,
        "host_activity": {"status": "derived",
                          "dynamic_operations": {"scalar": host_operations}},
        "physical_transition_evidence": {"status": "verified", "transitions": transitions},
    }
    task_evidence = {
        "status": "static_ownership_verified", "tasks": list(evidence_tasks),
        "binding": {"command_buffer_sha256": command_sha, "lowered_sha256": lowered_sha,
                    "plan_digest": plan_sha},
    }
    retained = {
        "command_buffer_text": command_text, "lowered_sha256": lowered_sha,
        "task_instruction_evidence": task_evidence,
    }
    activity = {
        "status": "decoded", "encoding_resolution": {"status": "complete"},
        "issued": {
            "compute_instructions": compute, "movement_instructions": movement,
        },
    }
    return retained, plan, activity


def _analysis_and_artifacts(member: str, *, only_coverage: bool = False):
    baseline = _arm("baseline-" + member, compute=100, movement=10, encodings=2,
                    host_operations=10, fused=False)
    candidate = _arm("candidate-" + member,
                     compute=100 if only_coverage else 70,
                     movement=10 if only_coverage else 8,
                     encodings=2 if only_coverage else 1,
                     host_operations=10 if only_coverage else 5,
                     fused=True)
    baseline_artifacts, baseline_plan, baseline_activity = baseline
    candidate_artifacts, candidate_plan, candidate_activity = candidate
    analysis = {
        "workload": {"capsule_sha256": member},
        "emission": {
            "baseline_command_buffer_sha256": baseline_plan["candidate_command_buffer_sha256"],
            "baseline_lowered_sha256": baseline_plan["candidate_lowered_sha256"],
            "candidate_command_buffer_sha256": candidate_plan["candidate_command_buffer_sha256"],
            "candidate_lowered_sha256": candidate_plan["candidate_lowered_sha256"],
        },
        "diagnostics": {
            "verified_baseline_global_plan_emission": baseline_plan,
            "verified_global_plan_emission": candidate_plan,
            "target_artifact_activity": {
                "baseline": baseline_activity, "candidate": candidate_activity,
            },
            "arms": {"baseline": {}, "candidate": {}},
        },
    }
    artifacts = {**candidate_artifacts, "baseline_artifacts": baseline_artifacts}
    return analysis, artifacts


def _quality_observer(**kwargs):
    model = kwargs["sentinel"].capsule_sha256
    analysis = kwargs["analysis"]
    plans = analysis["diagnostics"]
    classification = model == MODELS[0]
    baseline_values = ({"top1_degradation_percentage_points": 0.0} if classification else
                       {"cosine_similarity": 1.0,
                        "normalized_root_mean_square_error": 0.0})
    candidate_values = ({"top1_degradation_percentage_points": 0.4} if classification else
                        {"cosine_similarity": 0.995,
                         "normalized_root_mean_square_error": 0.01})
    return {
        "baseline": {
            "model_sha256": model, "corpus_sha256": kwargs["corpus_sha256"],
            "artifact_sha256": plans["verified_baseline_global_plan_emission"][
                "candidate_lowered_sha256"],
            "evidence_sha256": _sha("baseline-quality-" + model),
            "values": baseline_values, "complete": True,
        },
        "candidate": {
            "model_sha256": model, "corpus_sha256": kwargs["corpus_sha256"],
            "artifact_sha256": plans["verified_global_plan_emission"][
                "candidate_lowered_sha256"],
            "evidence_sha256": _sha("candidate-quality-" + model),
            "values": candidate_values, "complete": True,
        },
    }


def _installation(corpora=CORPORA):
    return build_fast_evaluator_installation(
        MODELS, classification_member_sha256=MODELS[0],
        corpus_sha256_by_member=corpora, calibration=_calibration(),
        quality_observer=_quality_observer, quality_observer_sha256=_sha("quality-observer"),
    )


def _row(installation, member, *, only_coverage=False):
    analysis, artifacts = _analysis_and_artifacts(member, only_coverage=only_coverage)
    return {"model_id": member, **installation.provider(
        analysis=analysis, artifacts=artifacts, sentinel=_Sentinel(member),
        target_descriptor=None, target_sha256=_sha("target"),
        portfolio_sha256=_sha("portfolio"),
        provider_binding=installation.provider_binding,
    )}


def test_factory_falls_back_to_exact_only_without_all_corpora():
    installation = build_fast_evaluator_installation(
        MODELS, classification_member_sha256=MODELS[0],
        corpus_sha256_by_member={MODELS[0]: CORPORA[MODELS[0]]},
        calibration=_calibration(), quality_observer=_quality_observer,
        quality_observer_sha256=_sha("quality-observer"),
    )

    assert installation.experiment_kwargs() == {}
    assert installation.quality_schema.mode == "exact_only"
    assert installation.fallback["status"] == "exact_only_fallback"
    assert installation.fallback["approximation_allowed"] is False


def test_provider_composes_bound_whole_program_metrics_and_is_directly_installable():
    installation = _installation()
    row = _row(installation, MODELS[0])
    candidate = row["candidate"].to_dict()

    assert set(installation.experiment_kwargs()) == {
        "fast_evaluation_provider", "fast_evaluation_policy", "quality_budgets",
        "fast_evaluation_provider_binding",
    }
    assert candidate["cycles"]["resolved"] is True
    assert candidate["movement_scope"] == "physical"
    assert candidate["movement_bytes"] == 80.0
    assert candidate["occupancy"]["compute_utilization"] is not None
    assert candidate["occupancy"]["latency_hiding_efficiency"] is not None
    assert candidate["encoding_conversions"]["count"] == 1
    assert candidate["coverage"]["supported_work_placed_fraction"] == 1.0
    assert candidate["coverage"]["largest_connected_region_fraction"] == 1.0
    assert candidate["coverage"]["connected_region_work"] == [6.0]
    assert candidate["roofline"]["lower_bound_cycles"] > 0
    required_hashes = (
        candidate["provenance"], candidate["cycles"]["provenance"],
        candidate["occupancy"]["provenance"], candidate["coverage"]["provenance"],
        candidate["roofline"]["provenance"],
        candidate["encoding_conversions"]["cycles"]["provenance"],
    )
    assert all(any("sha256:" in item for item in provenance) for provenance in required_hashes)
    assert row["provider_provenance"]["candidate_lowered_sha256"] in candidate["provenance"][0]


def test_provider_backed_gate_rejects_coverage_only_improvement():
    installation = _installation()
    rows = [_row(installation, model, only_coverage=True) for model in MODELS]

    report = evaluate_fast_portfolio(
        rows, quality_budgets=installation.quality_schema.budget_map,
        policy=installation.policy, expected_models=MODELS,
    )

    assert all("supported_work_placed_fraction" in row["robust_improvements"]
               for row in report["models"])
    assert report["status"] == "reject"
    assert "increased placement alone is insufficient" in report["failures"][0]


def test_missing_emitted_feature_remains_unknown_and_cannot_retain():
    installation = _installation()
    rows = []
    for model in MODELS:
        analysis, artifacts = _analysis_and_artifacts(model)
        del analysis["diagnostics"]["target_artifact_activity"]["candidate"]["issued"][
            "movement_instructions"]
        rows.append({"model_id": model, **installation.provider(
            analysis=analysis, artifacts=artifacts, sentinel=_Sentinel(model),
            target_descriptor=None, target_sha256=_sha("target"),
            portfolio_sha256=_sha("portfolio"),
            provider_binding=installation.provider_binding,
        )})

    report = evaluate_fast_portfolio(
        rows, quality_budgets=installation.quality_schema.budget_map,
        policy=installation.policy, expected_models=MODELS,
    )
    assert report["status"] == "needs_evidence"
    assert all(row["candidate"]["cycles"]["resolved"] is False for row in report["models"])
    assert all(row["candidate"]["movement_bytes"] is None for row in report["models"])


def test_provider_refuses_inputs_not_bound_to_verified_emission():
    installation = _installation()
    analysis, artifacts = _analysis_and_artifacts(MODELS[0])
    artifacts["command_buffer_text"] += " "

    with pytest.raises(ValueError, match="verified emitted artifacts"):
        installation.provider(
            analysis=analysis, artifacts=artifacts, sentinel=_Sentinel(MODELS[0]),
            target_descriptor=None, target_sha256=_sha("target"),
            portfolio_sha256=_sha("portfolio"),
            provider_binding=installation.provider_binding,
        )


def test_partial_target_decode_keeps_calibrated_metrics_unknown():
    installation = _installation()
    analysis, artifacts = _analysis_and_artifacts(MODELS[0])
    analysis["diagnostics"]["target_artifact_activity"]["candidate"][
        "encoding_resolution"] = {"status": "partial"}

    row = installation.provider(
        analysis=analysis, artifacts=artifacts, sentinel=_Sentinel(MODELS[0]),
        target_descriptor=None, target_sha256=_sha("target"),
        portfolio_sha256=_sha("portfolio"),
        provider_binding=installation.provider_binding,
    )

    assert row["candidate"].cycles.resolved is False
    assert row["candidate"].movement_bytes is None


def test_movement_fit_does_not_extrapolate_past_its_transfer_domain():
    calibration = _calibration()
    calibration["features"][1]["physical_bytes_per_unit"] = 256.0
    installation = build_fast_evaluator_installation(
        MODELS, classification_member_sha256=MODELS[0],
        corpus_sha256_by_member=CORPORA, calibration=calibration,
        quality_observer=_quality_observer,
        quality_observer_sha256=_sha("quality-observer"),
    )
    analysis, artifacts = _analysis_and_artifacts(MODELS[0])

    row = installation.provider(
        analysis=analysis, artifacts=artifacts, sentinel=_Sentinel(MODELS[0]),
        target_descriptor=None, target_sha256=_sha("target"),
        portfolio_sha256=_sha("portfolio"),
        provider_binding=installation.provider_binding,
    )

    assert row["baseline"].cycles.resolved is False
    assert row["candidate"].cycles.resolved is False
