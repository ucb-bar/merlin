"""Ordinary feedback uses original public replies, without inspecting private runs."""

import json

import pytest
from merlin_experiments.phase2 import contracts as K
from merlin_experiments.phase2 import corpus as C
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import paired_measurement as PME
from test_ordinary_search_policy import ordinary_inputs  # noqa: F401 -- shared harmless admission fixture

from merlin.perf.work_volume import command_buffer_evidence


def _corpus(tmp_path):
    root = tmp_path / "public"
    path = root / "capsules/F/M"
    path.mkdir(parents=True)
    descriptor = {
        "name": "M",
        "label": "dev",
        "performance": {"family": "F"},
        "inputs": [{"name": "a", "shape": [2, 3]}, {"name": "b", "shape": [3, 4]}],
        "operation": {"op": "matmul", "attributes": {"lhs": "a", "weight": "b"}},
    }
    (path / "capsule.yaml").write_text(json.dumps(descriptor))
    tree = K.exact_tree_record(path)
    member = C.PerformanceCapsule("F", "M", path, "F/M", descriptor, tree["sha256"], tree["n_files"], tree["n_bytes"])
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"capsules": [{"family": "F", "capsule": "M", **tree, "snapshot_sha256": tree["sha256"]}]})
    )
    return C.FrozenPerformanceCorpus(
        root,
        root / "capsules",
        manifest,
        K.sha256_file(manifest),
        K.exact_tree_record(root / "capsules")["sha256"],
        (member,),
    )


def _reply(spec, *, cycles=8):
    cb = {
        "tensors": {"a": {"shape": [2, 3], "dtype": "i8"}, "b": {"shape": [3, 4], "dtype": "i8"}},
        "commands": [{"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "b"}}],
    }
    work, artifact = command_buffer_evidence(cb, compiler_provenance="synthetic original process")
    tier = {
        "status": "pass",
        "cycles": cycles,
        "measured_now": True,
        "derived_from_rtl": True,
        "cycle_accurate": True,
        "timing": {"sim_active_s": 2.5},
        "concurrency": {"workers": 1},
    }
    return {
        "schema": "paired_arm4_raw_execution_v2",
        "execution": spec.as_dict(),
        "byte_guards": {
            "package_before": spec.package_sha256,
            "package_after": spec.package_sha256,
            "capsule_before": spec.member.source_sha256,
            "capsule_after": spec.member.source_sha256,
        },
        "measurement": {
            "status": "pass",
            "numeric": "pass",
            "work_volume": work,
            "command_buffer_artifact": artifact,
            "measurement_identity": {
                "program": {"sha256": artifact["artifact_sha256"]},
                "inputs": {"sha256": spec.member.source_sha256},
                "toolchain": {"frozen_submission_sha256": spec.package_sha256},
            },
            "execution_outcome": {"gsim": {"tier_outcome": tier}},
            "gsim_execution": {"gsim": {"cycles": cycles, "reused_measurement": False}},
            "gsim_qualification": {"admitted": True, "decision": spec.gsim_decision.to_dict()},
            "per_sim": {"spike": {"correct": True}, "gsim": {"correct": True, "cycles": cycles}},
        },
    }


@pytest.fixture
def feedback(request, tmp_path, monkeypatch):
    inputs = request.getfixturevalue("ordinary_inputs")
    arguments = inputs.arguments
    arguments["corpus"] = _corpus(tmp_path)
    # Matching public names and attractive private values would have affected
    # both recursive ceiling harvest and sibling-run scheduling history.
    for parent in (
        inputs.functional.run_dir / "grading_public",
        inputs.functional.run_dir / "grading_hidden",
        tmp_path / "other-run",
    ):
        parent.mkdir(parents=True, exist_ok=True)
        (parent / "capsule_result.json").write_text(
            json.dumps(
                {
                    "capsule": "M",
                    "status": "pass",
                    "tiers": {"L3": {"cycles": 1, "timing": {"sim_active_s": 0.001}, "concurrency": {"workers": 1}}},
                }
            )
        )
        generated = parent / "generated"
        generated.mkdir()
        (generated / "command_buffer.json").write_text(
            json.dumps(
                {
                    "tensors": {"a": {"shape": [1, 1000]}, "b": {"shape": [1000, 1]}},
                    "commands": [{"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "b"}}],
                }
            )
        )
    monkeypatch.setattr(OS.DF.GATE, "validate_execution", lambda *args: {"admitted": True})
    selected = OS.prepare_ordinary_search_policy(**arguments)
    evaluator = selected.feedback_evaluator
    assert evaluator.achievable_macs_per_cycle is None and evaluator.member_cost == {}
    evaluator.executed_profile_timeout_s = 0
    return evaluator


def test_normal_preparation_ignores_history_and_learns_only_current_public_baseline(feedback, tmp_path, monkeypatch):
    monkeypatch.setattr(PME, "run_execution", lambda spec, *args, **kwargs: _reply(spec))
    member = feedback.corpus.capsules[0]
    decision = feedback.decisions[(member.family, member.capsule)]
    feedback._execute(
        arm="baseline",
        package=feedback.baseline,
        package_sha256=feedback.baseline_sha256,
        member=member,
        decision=decision,
        workspace=tmp_path / "live",
        timeout_s=2,
    )
    point = feedback._baseline_points[("F", "M")]
    assert (point.macs, point.cycles, point.reduction_depths) == (24, 8, (3,))
    feedback._refresh_achievable()
    assert feedback.member_cost == {"M": 2.5}
    assert len(feedback._achievable_points) == 1
    assert feedback.achievable_macs_per_cycle is None  # one sample cannot qualify a ceiling


@pytest.mark.parametrize(
    "mutation",
    [
        "hidden",
        "final",
        "hidden_label",
        "validation_label",
        "unknown_label",
        "descriptor",
        "failed",
        "missing_status",
        "numeric",
        "missing_numeric",
        "spike_failed",
        "carried",
        "not_measured",
        "reused",
        "other_source",
        "other_package",
        "candidate",
        "other_engine",
        "other_certificate",
        "missing_runtime",
        "wrong_cycles",
        "work",
    ],
)
def test_ineligible_reply_never_changes_ordinary_rates_or_scheduling(feedback, tmp_path, monkeypatch, mutation):
    member = feedback.corpus.capsules[0]

    def execute(spec, *args, **kwargs):
        raw = _reply(spec)
        m = raw["measurement"]
        tier = m["execution_outcome"]["gsim"]["tier_outcome"]
        if mutation in {"hidden", "final"}:
            raw["execution"]["phase"] = mutation
        elif mutation in {"hidden_label", "validation_label", "unknown_label"}:
            member.descriptor["label"] = mutation.removesuffix("_label")
        elif mutation == "descriptor":
            member.descriptor["inputs"][0]["shape"] = [200, 3]
        elif mutation == "failed":
            tier["status"] = "fail"
        elif mutation == "missing_status":
            m.pop("status")
        elif mutation == "numeric":
            m["numeric"] = "fail"
        elif mutation == "missing_numeric":
            m.pop("numeric")
        elif mutation == "spike_failed":
            m["per_sim"]["spike"]["correct"] = False
        elif mutation == "carried":
            tier["carried"] = {"carried": True}
        elif mutation == "not_measured":
            tier.pop("measured_now")
        elif mutation == "reused":
            m["gsim_execution"]["gsim"]["reused_measurement"] = True
        elif mutation == "other_source":
            raw["byte_guards"]["capsule_after"] = "0" * 64
        elif mutation == "other_package":
            raw["execution"]["package_sha256"] = "0" * 64
        elif mutation == "candidate":
            raw["execution"]["arm"] = "candidate"
        elif mutation == "other_engine":
            raw["execution"]["gsim_decision"]["selected_engine"] = "spike"
        elif mutation == "other_certificate":
            raw["execution"]["gsim_decision"]["certificate_sha256"] = "0" * 64
        elif mutation == "missing_runtime":
            m.pop("gsim_execution")
        elif mutation == "wrong_cycles":
            m["gsim_execution"]["gsim"]["cycles"] = 1
        elif mutation == "work":
            m["work_volume"]["exact_macs"] = 1000
        return raw

    monkeypatch.setattr(PME, "run_execution", execute)
    feedback._execute(
        arm="baseline",
        package=feedback.baseline,
        package_sha256=feedback.baseline_sha256,
        member=member,
        decision=feedback.decisions[("F", "M")],
        workspace=tmp_path / "live",
        timeout_s=2,
    )
    feedback._refresh_achievable()
    assert feedback._baseline_points == {} and feedback.member_cost == {}
