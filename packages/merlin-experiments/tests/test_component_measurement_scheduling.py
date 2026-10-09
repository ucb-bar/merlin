"""Owned process seam diagnostics; no qualified production plan is fabricated.

Only the verifier method is replaced in the dispatch controls. The public
issuer remains unavailable and its private registry is never populated here.
Small process stdout is synthetic feedback, never physical cycle evidence.
"""

import json
import subprocess
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_measurement_scheduling as S
from merlin_experiments.phase2 import development_feedback as D
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, sha256_file

from merlin.benchharness import hash_tree
from merlin.common import invocation_record as I
from merlin.perf.component_measurement_plan import DevelopmentMeasurementDecision


def unissued(**updates):
    kwargs = {field.name: None for field in fields(S.DevelopmentMeasurementPlan)}
    kwargs.update(updates)
    return S.DevelopmentMeasurementPlan(**kwargs)


def test_saved_or_copied_plan_cannot_issue_development_deferral(tmp_path):
    evaluator = D.DevelopmentGsimFeedback(None, None, tmp_path, "a" * 64, None, {}, tmp_path / "work", {})
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler").write_text("owned")
    for plan in (unissued(), {"status": "qualified", "selected": []}):
        with pytest.raises(StageGateError, match="live qualified|live qualified producer"):
            evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=5, measurement_plan=plan)
    assert not evaluator.work_root.exists()


def test_missing_actual_runtime_refuses_before_analytical_provider_or_engine(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import component_analytical as A

    binding = A.ComponentAnalyticalBinding(**{field.name: None for field in fields(A.ComponentAnalyticalBinding)})
    evaluator = D.DevelopmentGsimFeedback(None, None, tmp_path, "a" * 64, None, {}, tmp_path / "work", {})
    monkeypatch.setattr(A, "_evaluate", lambda *a, **k: pytest.fail("unqualified feature dispatch"))
    with pytest.raises(StageGateError, match="independent runtime"):
        S.prepare_development_measurements(
            binding=binding,
            evaluator=evaluator,
            candidate=tmp_path,
            evidence_root=tmp_path / "plan",
            timeout_s=3,
            max_measurements=1,
            member_timeout_s=3,
        )
    assert not (tmp_path / "plan").exists()


def setup_dispatch(tmp_path, monkeypatch, *, selected=True, defect=None, workers=1, parse_defect=False, delay=0):
    candidate, baseline = tmp_path / "candidate", tmp_path / "baseline"
    for path in (candidate, baseline):
        path.mkdir()
        (path / "compiler").write_text("candidate" if path == candidate else "baseline")
    worker = tmp_path / "owned_worker.py"
    worker.write_text(
        f"import json,sys,time\nfrom pathlib import Path\ntime.sleep({delay!r})\n"
        "compiler,member=map(Path,sys.argv[1:3])\n"
        "print(json.dumps({'compiler':compiler.read_text(),'member':member.read_text()}))\n"
    )
    original = []
    for index in range(3):
        source = tmp_path / f"original_{index}.mlir"
        source.write_text(f"independent source {index}")
        original.append(SimpleNamespace(family="family", capsule=f"member{index}", descriptor={}, source=source))
    corpus = SimpleNamespace(capsules=tuple(original), capsules_sha256="c" * 64)
    ids = tuple(document_sha256([member.family, member.capsule]) for member in original)
    decisions = tuple(
        DevelopmentMeasurementDecision(identity, "MEASURE" if index == 1 else "DEFER", "diagnostic deferral")
        for index, identity in enumerate(ids)
    )
    plan = unissued(selected=(ids[1],) if selected else (), decisions=decisions, member_timeout_s=3)
    candidate_sha, baseline_sha = (hash_tree(path)["sha256"] for path in (candidate, baseline))
    pins = {path: sha256_file(path) for path in (worker, *(member.source for member in original))}
    verifies, events = [], []

    def diagnostic_verify(self, evaluator, *, candidate=None, package=None, arm=None, member=None):
        # Explicit diagnostic replacement; no live-owner or qualification claim.
        verifies.append((arm, member.capsule if member is not None else None))
        for path, digest in pins.items():
            if sha256_file(path) != digest:
                raise StageGateError("diagnostic source/tool changed")
        if candidate is not None and hash_tree(candidate)["sha256"] != candidate_sha:
            raise StageGateError("diagnostic candidate changed")
        if package is not None and hash_tree(package)["sha256"] != (
            baseline_sha if arm == "baseline" else candidate_sha
        ):
            raise StageGateError("diagnostic compiler changed")
        if member is not None and document_sha256([member.family, member.capsule]) not in self.selected:
            raise StageGateError("diagnostic membership changed")

    monkeypatch.setattr(S.DevelopmentMeasurementPlan, "verify", diagnostic_verify)
    monkeypatch.setattr(D, "sweep_workers", lambda: workers)

    def execute(**kwargs):
        member, package = kwargs["member"], kwargs["package"]
        workspace = kwargs["workspace"]
        workspace.mkdir(parents=True)
        result = I.run(
            (sys.executable, "-I", str(worker), str(package / "compiler"), str(member.source)),
            cwd=tmp_path,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            directory=workspace,
            inputs=(worker, package / "compiler", member.source),
            timeout=kwargs["timeout_s"],
            stage="owned_scheduling_control",
            capture_output=True,
        )
        (record_path,) = workspace.glob("invocations/*/invocation.json")
        I.verify(record_path)
        observed = json.loads(result.stdout)
        assert observed["compiler"] == ("baseline" if kwargs["arm"] == "baseline" else "candidate")
        assert observed["member"] == member.source.read_text()
        events.append((kwargs["arm"], member.capsule, kwargs["timeout_s"], str(record_path)))
        if defect == "tool":
            worker.write_text(worker.read_text() + "# changed\n")
        elif defect == "source":
            member.source.write_text("changed source")
        elif defect == "candidate":
            (package / "compiler").write_text("changed compiler")
        return {"synthetic": observed}

    evaluator = D.DevelopmentGsimFeedback(
        SimpleNamespace(sha256="e" * 64),
        corpus,
        baseline,
        baseline_sha,
        None,
        {},
        tmp_path / "work",
        {(member.family, member.capsule): SimpleNamespace(scope="diagnostic only") for member in original},
        executor=execute,
    )

    def parse(raw, decision, **kwargs):
        if parse_defect:
            baseline.joinpath("compiler").write_text("changed during parse")
        return {
            "correct": True,
            "gsim_cycles": 10 if kwargs["arm"] == "baseline" else 9,
        }

    monkeypatch.setattr(evaluator, "_redact_execution", parse)
    monkeypatch.setattr(evaluator, "_refresh_achievable", lambda: None)
    monkeypatch.setattr(evaluator, "_matched_achievable", lambda member: (None, "unqualified diagnostic", None))
    monkeypatch.setattr(evaluator, "_stopping", lambda *a, **k: {"status": "diagnostic legacy", "verdicts": []})
    return evaluator, candidate, plan, events, verifies


@pytest.mark.parametrize("workers", [1, 2])
def test_actual_process_subset_retains_every_deferred_row_and_no_convergence(tmp_path, monkeypatch, workers):
    evaluator, candidate, plan, events, verifies = setup_dispatch(tmp_path, monkeypatch, workers=workers)
    result = evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=15, measurement_plan=plan)
    assert sorted((arm, member) for arm, member, _, _ in events) == [("baseline", "member1"), ("candidate", "member1")]
    assert all(0 < seconds <= 3 for _, _, seconds, _ in events)
    assert len(result["cells"]) == 3 and result["summary"]["members"] == 3
    deferred = [row for row in result["cells"] if not row["measured"]]
    assert len(deferred) == 2
    assert all(row["candidate_gsim_cycles"] is None and row["baseline_correct"] is None for row in deferred)
    assert result["stopping"]["status"] == "undeterminable"
    assert result["summary"]["achievable_macs_per_cycle"] is None
    assert result["summary"]["recoverable"]["status"] == "unavailable"
    assert all(row["verdict"] == "undeterminable" for row in result["cells"] if row["measured"])
    assert evaluator._totals is None and evaluator._spend is None
    assert verifies.count(("baseline", "member1")) >= 2 and verifies.count(("candidate", "member1")) >= 2
    assert len(S._ISSUED) == 0


def test_default_ordinary_route_still_executes_complete_original_roster(tmp_path, monkeypatch):
    evaluator, candidate, _, events, _ = setup_dispatch(tmp_path, monkeypatch)
    result = evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=15)
    assert len(events) == 6 and len(result["cells"]) == 3
    assert all(row["measured"] for row in result["cells"])
    assert result["stopping"]["status"] == "diagnostic legacy"


def test_no_selected_measurements_cannot_become_measured_or_complete(tmp_path, monkeypatch):
    evaluator, candidate, plan, events, _ = setup_dispatch(tmp_path, monkeypatch, selected=False)
    result = evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=15, measurement_plan=plan)
    assert not events and len(result["cells"]) == 3
    assert result["summary"]["comparable"] == 0 and result["stopping"]["status"] == "undeterminable"
    assert all(not row["measured"] for row in result["cells"])


@pytest.mark.parametrize("defect", ["tool", "source", "candidate"])
def test_actual_changed_consumption_context_refuses_after_native_return(tmp_path, monkeypatch, defect):
    evaluator, candidate, plan, events, _ = setup_dispatch(tmp_path, monkeypatch, defect=defect)
    with pytest.raises(StageGateError, match="diagnostic .*changed"):
        evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=15, measurement_plan=plan)
    assert len(events) == 1


def test_legacy_redacted_baseline_cache_is_not_used_by_new_pinned_selection(tmp_path, monkeypatch):
    evaluator, candidate, plan, events, _ = setup_dispatch(tmp_path, monkeypatch)
    evaluator._baseline_cache = {("family", "member1"): {"correct": True, "gsim_cycles": 99999}}
    result = evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=15, measurement_plan=plan)
    assert len(events) == 2
    assert next(row for row in result["cells"] if row["measured"])["baseline_gsim_cycles"] == 10
    assert evaluator._baseline_cache["family", "member1"]["gsim_cycles"] == 99999


def test_changed_baseline_during_parser_refuses_before_candidate_dispatch(tmp_path, monkeypatch):
    evaluator, candidate, plan, events, _ = setup_dispatch(tmp_path, monkeypatch, parse_defect=True)
    with pytest.raises(StageGateError, match="diagnostic compiler changed"):
        evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=15, measurement_plan=plan)
    assert [(arm, member) for arm, member, _, _ in events] == [("baseline", "member1")]


def test_unselected_object_cannot_run_lower_native_boundary(tmp_path):
    evaluator = D.DevelopmentGsimFeedback(None, None, tmp_path, "a" * 64, None, {}, tmp_path / "work", {})
    counterfeit = SimpleNamespace(member_timeout_s=3, verify=lambda *a, **k: pytest.fail("counterfeit verifier ran"))
    kwargs = dict(
        arm="candidate",
        package=tmp_path,
        package_sha256="b" * 64,
        member=None,
        decision=None,
        workspace=tmp_path,
        timeout_s=3,
        measurement_plan=counterfeit,
    )
    for method in (evaluator._execute, evaluator._execute_once):
        with pytest.raises(StageGateError, match="live qualified plan"):
            method(**kwargs)


def test_changed_tool_selection_refuses_before_any_native_dispatch(tmp_path, monkeypatch):
    evaluator, candidate, plan, events, _ = setup_dispatch(tmp_path, monkeypatch)
    worker = tmp_path / "owned_worker.py"
    worker.write_text(worker.read_text() + "# changed before dispatch\n")
    with pytest.raises(StageGateError, match="diagnostic source/tool changed"):
        evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=15, measurement_plan=plan)
    assert not events and not evaluator.work_root.exists()


def test_owned_process_respects_selected_native_budget_and_retains_timeout(tmp_path, monkeypatch):
    evaluator, candidate, plan, events, _ = setup_dispatch(tmp_path, monkeypatch, delay=2)
    plan = replace(plan, member_timeout_s=1)
    with pytest.raises(subprocess.TimeoutExpired):
        evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=15, measurement_plan=plan)
    (path,) = evaluator.work_root.rglob("invocation.json")
    record = json.loads(path.read_text())
    assert record["status"] == "interrupted" and record["error"] == "TimeoutExpired"
    assert not events and len(S._ISSUED) == 0
