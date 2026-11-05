"""Performance feedback reaches every test case: per-case observation budgets and a headroom ranking."""

import json
import time

import pytest

from merlin.common import cpu_pool
from merlin_experiments.phase1.feedback import joint_performance as J
from merlin_experiments.phase1.feedback import joint_performance_batch as B
from merlin_experiments.phase1.feedback import joint_test_scope as T
from merlin_experiments.phase1.feedback import performance_headroom as H


def _select(tmp_path, monkeypatch, **fields):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(dict(schema=J.SCHEMA, **fields)))
    monkeypatch.setenv(J.ENV, json.dumps(dict(path=str(path), sha256=J.F.digest(path))))


def test_each_observation_gets_its_own_budget_within_the_request_deadline(tmp_path, monkeypatch):
    _select(tmp_path, monkeypatch, per_request_seconds=900, per_observation_seconds=120)
    assert 119 < J.observation_budget(time.monotonic() + 3600) <= 120
    assert 9 < J.observation_budget(time.monotonic() + 10) <= 10
    assert J.observation_budget(time.monotonic() - 5) == 0
    _select(tmp_path, monkeypatch, per_request_seconds=300)  # back-compat: per-request is the per-case default
    assert 299 < J.observation_budget(time.monotonic() + 3600) <= 300
    _select(tmp_path, monkeypatch, per_request_seconds=30, per_observation_seconds=901)
    with pytest.raises(ValueError):
        J.selected()


@pytest.mark.parametrize("width", [1, 4])
def test_a_slow_first_case_no_longer_starves_later_ones(tmp_path, monkeypatch, width):
    _select(tmp_path, monkeypatch, per_request_seconds=900, per_observation_seconds=60)
    budgets = {}

    def attach(row, root, *, result, context, compiler, capsule, remaining_s, **_kwargs):
        budgets[row["capsule"]] = remaining_s
        if row["capsule"] == "first":
            time.sleep(0.3)  # stands in for a long observation
        row["performance"] = {"status": "PROVISIONAL"}

    monkeypatch.setattr(J, "attach", attach)
    monkeypatch.setattr(cpu_pool, "compute_capacity", lambda *a, **k: width)
    rows = [{"capsule": name} for name in ("first", "second", "third")]
    jobs = [dict(row=row, root=tmp_path, result={}, compiler=tmp_path, capsule={}) for row in rows]
    B.attach_all(jobs, context=None, request_deadline=time.monotonic() + 3600)
    assert all(budget > 59 for budget in budgets.values())
    assert all("observation_seconds" in row["performance"] for row in rows)


def _verdict_row(name, l2="pass", performance=None):
    return {"capsule": name, "tiers": {"L2": l2}, **({"performance": performance} if performance else {})}


def _roofline(floor, useful=4096, peak=256, read=None):
    roofline = {"structural_compute_floor_cycles": floor, "useful_macs": useful, "structural_peak_macs_per_cycle": peak}
    if read is not None:
        roofline["logical_request_ideal_service"] = {"read": {"ideal_requested_service_cycles": read}}
    return roofline


def test_headroom_ranks_every_passing_case_with_measured_before_modeled(tmp_path):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    (ws / ".qa_channel").mkdir()
    (ws / "selfcheck_out" / "dense").mkdir(parents=True)
    (ws / "selfcheck_out" / "dense" / "command_buffer.json").write_text("{}")
    (ws / "qa" / "tier_state.json").write_text(json.dumps({
        "dense": {"L2": {"status": "pass", "digest": "a"}, "L3": {"status": "pass", "digest": "a",
                                                                   "execution_digest": "e1"}},
        "conv": {"L2": {"status": "pass", "digest": "new"}, "L3": {"status": "pass", "digest": "old",
                                                                    "execution_digest": "e2"}},
    }))
    (ws / ".qa_channel" / "simresp_1.json").write_text(json.dumps({"sim": "gsim", "per_capsule": [
        {"capsule": "dense", "execution_digest": "e1", "barrier_cycles": 1600, "pass": True,
         "performance": {"structural_roofline": _roofline(16)}},
        {"capsule": "conv", "execution_digest": "e2", "barrier_cycles": 500, "pass": True,
         "performance": {"structural_roofline": _roofline(10, read=100)}},
        {"capsule": "dense", "execution_digest": "unbound", "barrier_cycles": 1, "pass": True},
    ]}))
    modeled = {"status": "PROVISIONAL", "complete_event_accounting": True, "estimated_whole_soc_cycles": 900,
               "structural_roofline": _roofline(300),
               "scoped_resource_feedback": {"resources": [{"name": "cpu.issue", "capacity_normalized_activity": 0.9},
                                                          {"name": "accumulator", "capacity_normalized_activity": 0.2}]}}
    verdict = {"feedback_brief": {"performance_priority": {"correctness_first": []}}, "per_capsule": [
        _verdict_row("dense"), _verdict_row("conv"), _verdict_row("modeled_only", performance=modeled),
        _verdict_row("no_evidence"), _verdict_row("failing", l2="fail", performance=modeled)]}
    block = H.attach(ws, verdict)["feedback_brief"]["performance_priority"]["development_headroom"]
    ranked = block["largest_headroom"]
    assert [e["test_case"] for e in ranked] == ["dense", "modeled_only", "conv"]
    dense, modeled_only, conv = ranked
    assert dense["cycles"] == 1600 and dense["evidence"].startswith("measured")  # bound digest, not the stray row
    assert dense["array_utilization_pct"] == pytest.approx(100 * 4096 / (1600 * 256), abs=0.01)
    assert dense["inspect"].startswith("python isa_tools.py debug selfcheck_out/dense/")
    assert conv["bound_by"] == "read requests" and conv["measured_on_earlier_source"] is True
    assert modeled_only["dominant_modeled_resource"]["name"] == "cpu.issue"
    assert block["counts"] == {"passing_l2": 4, "ranked": 3, "measured": 2, "modeled": 1, "without_evidence": 1,
                               "performance_scope_excluded": 0}
    assert block["without_evidence"] == ["no_evidence"] and block["promotion_authority"] is False
    assert verdict["feedback_brief"]["performance_priority"]["correctness_first"] == []  # left intact


def test_performance_scope_excluded_cases_never_rank(tmp_path):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    modeled = {"status": "PROVISIONAL", "complete_event_accounting": True, "estimated_whole_soc_cycles": 900,
               "structural_roofline": _roofline(300)}
    verdict = {"feedback_brief": {"performance_priority": {}},
               "joint_test_selection": {"performance_scope": {"excluded": ["duplicate"]}},
               "per_capsule": [_verdict_row("scorer", performance=modeled),
                               _verdict_row("duplicate", performance=modeled)]}
    block = H.attach(ws, verdict)["feedback_brief"]["performance_priority"]["development_headroom"]
    assert [e["test_case"] for e in block["largest_headroom"]] == ["scorer"]
    assert block["counts"]["passing_l2"] == 2 and block["counts"]["performance_scope_excluded"] == 1


def test_a_witness_that_is_also_a_scalability_case_is_ranked_as_primary_only(tmp_path, monkeypatch):
    root, submission = tmp_path / "core", tmp_path / "compiler"
    submission.mkdir()
    paths = {}
    for name, parent in [("small", root), ("diagnostic", root), ("large", tmp_path / "scale")]:
        path = parent / name
        path.mkdir(parents=True)
        (path / "capsule.yaml").write_text(f"name: {name}\nlabel: public\nkind: operator\ninputs: []\n")
        paths[name] = path
    state = {"policy": {"correctness_curriculum": {"original_capsules": ["small"]},
                        "gsim": {"witness_capsules": ["diagnostic", "large"]}, "final_only_capsules": []}}
    monkeypatch.setattr(T.joint_control, "selected", lambda: state)
    monkeypatch.delenv("MERLIN_JOINT_GRADE_MILESTONE", raising=False)
    _, selection = T.select(root, submission=submission, additional_cases={"large": paths["large"]})
    cohorts = selection["performance_cohorts"]
    assert cohorts["primary"] == ["large"] and cohorts["calibration_diagnostics"] == ["diagnostic"]
