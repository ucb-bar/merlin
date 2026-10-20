"""Normal AWS command and fixed evaluator routing using synthetic observations."""

import json
import os

import pytest
import test_ordinary_search_policy as NORMAL
from merlin_experiments.phase2 import ordinary_calibration as CAL
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import ordinary_timing_study as STUDY
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, sha256_file
from test_ordinary_fast_feedback import extend_corpus

revisions = NORMAL.revisions


@pytest.fixture
def ordinary_inputs(tmp_path, monkeypatch):
    return NORMAL.ordinary_inputs.__wrapped__(tmp_path, monkeypatch)


def request_for(fixture, tmp_path):
    extend_corpus(fixture, tmp_path)
    args = fixture.arguments
    edit = tmp_path / "edit.json"
    edit.write_text(json.dumps(args["edit_authority"].contract))

    def pin(path):
        return {"path": str(path), "sha256": sha256_file(path)}

    work = tmp_path / "allocated"
    work.mkdir()
    members = [[r.family, r.capsule] for r in args["corpus"].capsules]
    return {
        "schema": STUDY.SCHEMA,
        "normal_inputs": {
            "functional_runs_root": str(args["functional_runs_root"]),
            "functional_run_id": args["functional_run_id"],
            "functional_submission_sha256": args["functional_submission_sha256"],
            "corpus_root": str(args["corpus"].root),
            "corpus_manifest_sha256": args["corpus"].manifest_sha256,
            "corpus_sha256": args["corpus"].capsules_sha256,
            "baseline": str(args["baseline"]),
            "candidate": str(args["candidate"]),
            "target_descriptor": pin(args["target_experiment"].path),
            "contract_root": str(args["contract_root"]),
            "edit_contract": pin(edit),
            "certificate": pin(args["certificate_path"]),
            "rtl_facts": pin(args["rtl_facts_path"]),
        },
        "resources": {
            "provider": "aws",
            "instance_id": "i-synthetic",
            "cpu_ids": [0],
            "memory_bytes": 1 << 20,
            "disk_bytes": 1 << 20,
            "work_root": str(work),
        },
        "environment_sha256": document_sha256(dict(os.environ)),
        "replicates": 2,
        "max_stage_seconds": 10,
        "public_partition": {
            "schema": CAL.PARTITION_SCHEMA,
            "calibration": members[:1],
            "held_validation": members[1:],
        },
        "alternatives": [
            {"path": str(args["candidate"]), "sha256": "a" * 64},
            {"path": str(args["baseline"]), "sha256": "b" * 64},
        ],
    }


def test_default_command_plans_without_runtime_admission_or_aws(ordinary_inputs, tmp_path, monkeypatch):
    request = request_for(ordinary_inputs, tmp_path)
    path, output = tmp_path / "request.json", tmp_path / "plan.json"
    path.write_text(json.dumps(request))

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not call AWS or normal execution")

    monkeypatch.setattr(STUDY, "_load", forbidden)
    monkeypatch.setattr(STUDY.COST, "_allocated", forbidden)
    assert STUDY.main(["--request", str(path), "--output", str(output)]) == 0
    plan = json.loads(output.read_text())
    assert plan["execution_performed"] is False
    assert plan["screening_policy"]["minimum_decided"] == 100
    assert plan["cache_state"] == "UNKNOWN"


def test_request_export_reuses_actual_original_normal_selection(ordinary_inputs, tmp_path):
    request = request_for(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    exported = STUDY.request_from_policy(
        policy,
        resources=request["resources"],
        alternatives=(policy.candidate, policy.search_baseline),
        public_partition=request["public_partition"],
        edit_contract_path=request["normal_inputs"]["edit_contract"]["path"],
        max_stage_seconds=10,
    )
    assert exported["normal_inputs"] == request["normal_inputs"]
    assert exported["environment_sha256"] == request["environment_sha256"]
    assert STUDY.plan(exported)["execution_performed"] is False


def test_study_retains_complete_costs_and_results_when_calibration_is_unavailable(
    ordinary_inputs, tmp_path, monkeypatch
):
    request = request_for(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    adapter = STUDY.OrdinaryCorpusCpuStudyAdapter(policy)
    observed = []

    def profile(self, **kwargs):
        observed.append(kwargs["kind"])
        return {"kind": kwargs["kind"], "wall_seconds": 1.0, "result": {"synthetic": True}}

    def unavailable(*args, **kwargs):
        raise CAL.CalibrationUnavailable("original compiled join unavailable")

    monkeypatch.setattr(STUDY.COST, "_allocated", lambda request: None)
    monkeypatch.setattr(STUDY, "_load", lambda request: (adapter, (candidate, baseline), baseline))
    monkeypatch.setattr(STUDY.OrdinaryCorpusCpuStudyAdapter, "profile", profile)
    monkeypatch.setattr(CAL, "produce_calibration", unavailable)
    result = STUDY.execute(request)
    assert observed == [FAST.KIND, "corpus-rtl-feedback"] * 4
    assert len(result["profiles"]) == 8
    assert result["qualification"]["status"] == "UNKNOWN" and result["qualification"]["exposable"] is False
    assert result["calibration_sha256"] is None
    assert result["setup"]["wall_seconds"] >= 0
    assert result["calibration_and_held_validation"]["cpu_seconds"] >= 0
    assert len(list(tmp_path.joinpath("allocated").glob("profile_*.json"))) == 8
    assert result["promotion"] == "NONE"


def test_allocated_profile_selects_only_fixed_normal_route(ordinary_inputs, tmp_path, monkeypatch):
    request = request_for(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    calls, allocations = [], []

    def evaluate(self, *args, **kwargs):
        calls.append((args, kwargs))
        kwargs["work_root"].mkdir()
        return {"synthetic": True}

    monkeypatch.setattr(OS.OrdinarySearchPolicy, "evaluate_retained", evaluate)
    monkeypatch.setattr(STUDY.COST, "_allocated", lambda req: allocations.append(req))
    monkeypatch.setattr(STUDY.os, "sched_setaffinity", lambda *args: None)
    from merlin_experiments.phase2 import supervised_feedback

    monkeypatch.setattr(
        supervised_feedback, "bounded_feedback", lambda function, **kwargs: function(**kwargs["kwargs"])
    )
    adapter = STUDY.OrdinaryCorpusCpuStudyAdapter(policy)
    profile = adapter.profile(
        request=request,
        revision=candidate,
        baseline_revision=baseline,
        workspace=tmp_path / "allocated/profile",
        kind=FAST.KIND,
        call_index=0,
    )
    assert calls[0][0] == (candidate, baseline)
    assert calls[0][1]["kind"] == FAST.KIND
    assert allocations == [request, request]
    assert profile["wall_seconds"] >= 0
    assert profile["cache_state"] == "UNKNOWN"
    request["normal_inputs"]["certificate"]["sha256"] = "f" * 64
    with pytest.raises(StageGateError, match="changed"):
        adapter.verify(request)


def test_calibration_producer_requires_aws_and_multiple_public_alternatives(ordinary_inputs, tmp_path, monkeypatch):
    request = request_for(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(policy, tmp_path)

    def refuse(request):
        raise StageGateError("not on allocated AWS")

    monkeypatch.setattr(STUDY.COST, "_allocated", refuse)
    with pytest.raises(StageGateError, match="multiple distinct"):
        CAL.produce_calibration(
            policy,
            (candidate,),
            baseline,
            partition=request["public_partition"],
            work_root=tmp_path / "allocated/calibration",
            timeout_s=10,
            workers=1,
            allocation=request["resources"],
        )
    with pytest.raises(StageGateError, match="allocated AWS"):
        CAL.produce_calibration(
            policy,
            (candidate, baseline),
            baseline,
            partition=request["public_partition"],
            work_root=tmp_path / "allocated/calibration",
            timeout_s=10,
            workers=1,
            allocation=request["resources"],
        )
    assert not ordinary_inputs.executions


def test_live_calibration_producer_calls_both_original_routes_and_preserves_held_groups(
    ordinary_inputs, tmp_path, monkeypatch
):
    """Routing spies supply synthetic evidence; they establish no engine qualification."""
    request = request_for(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    calls, held = [], []

    def evaluate(self, revision, original, **kwargs):
        kind = kwargs.get("kind", "corpus-rtl-feedback")
        calls.append((kind, revision, original))
        root = kwargs["work_root"]
        root.mkdir()
        cells = []
        for index, member in enumerate(sorted(self.search_corpus.capsules, key=lambda row: (row.family, row.capsule))):
            row = {"family": member.family, "capsule": member.capsule, "source_sha256": member.source_sha256}
            if kind == FAST.KIND:
                for arm in ("baseline", "candidate"):
                    evidence = root / f"m{index:03d}_{arm}" / "ordinary_observation.json"
                    evidence.parent.mkdir()
                    row[arm] = {
                        "functional_status": "PASS",
                        "compiled": {
                            "status": "observed",
                            "features": {FAST.FEATURES[0]: float(index + 1), FAST.FEATURES[1]: float(index + 2)},
                        },
                    }
                    evidence.write_text(json.dumps({**row[arm], "original_grade": {"synthetic": True}}))
                    row[arm]["evidence_sha256"] = sha256_file(evidence)
            else:
                row.update(measured=True, comparable=True, baseline_gsim_cycles=10, candidate_gsim_cycles=9)
            cells.append(row)
        document = {"cells": cells}
        if kind == FAST.KIND:
            (root / "ordinary_fast_feedback.json").write_text(json.dumps(document))
        return document

    def join(root, cell, arm, compiled, package_sha):
        evidence = root / (cell["capsule"] + arm + ".json")
        evidence.write_text("synthetic joined process, not runtime authority")
        return document_sha256([package_sha, cell["source_sha256"], arm]), (str(evidence), sha256_file(evidence)), ()

    def qualify(rows, digest):
        held.extend(rows)
        return {"exposable": False, "reasons": ["synthetic sample count is insufficient"]}

    monkeypatch.setattr(STUDY.COST, "_allocated", lambda request: None)
    monkeypatch.setattr(OS.OrdinarySearchPolicy, "evaluate_retained", evaluate)
    monkeypatch.setattr(CAL, "_join", join)
    monkeypatch.setattr(CAL, "_qualify", qualify)
    monkeypatch.setattr(
        CAL,
        "_bounded_observe",
        lambda policy, revision, original, **kwargs: CAL._observe_fixed(
            policy=policy, revision=revision, baseline_revision=original, options=kwargs
        ),
    )
    owner = CAL.produce_calibration(
        policy,
        (candidate, baseline),
        baseline,
        partition=request["public_partition"],
        work_root=tmp_path / "allocated/calibration",
        timeout_s=10,
        workers=1,
        allocation=request["resources"],
    )
    assert [kind for kind, *_ in calls] == [FAST.KIND, "corpus-rtl-feedback"] * 2
    assert {row.group for row in held} != {"calibration"}
    assert any(row.group == "calibration" for row in held)
    assert all(row.workload in {m.source_sha256 for m in policy.search_corpus.capsules} for row in held)
    policy.install_screen(owner)
    assert owner.screen(policy, {"cells": []})["status"] == "UNKNOWN"
    artifact = owner.artifacts[0][0]
    from pathlib import Path

    Path(artifact).write_text("changed synthetic observation")
    with pytest.raises(StageGateError, match="evidence changed"):
        owner.verify(policy)
