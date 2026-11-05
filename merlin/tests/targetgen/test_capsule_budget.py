"""A per-capsule budget below a capsule's measured cost is raised from that measurement and reported.

Measured on one run: a capsule whose gSIM run completed in a self-check (622 s adapter wall) was killed
after 287 s under a 300 s per-capsule budget in every later grade and the final qualification, and
stayed cert_not_measured with a reason that quoted only the clipped tail of the killed command.
"""

from __future__ import annotations

import json

import pytest

from merlin.targetgen import capsule_budget as CB
from merlin.targetgen import capsule_grade as CG
from merlin.targetgen import capsule_scheduling as CS

_STOPPED = {"category": "testbench_timeout", "plane": "elaborated_rtl", "tier": "L3", "detail": "timed out"}


def _result(root, name, *, wall, partial=False, measured=True, failure=None, status="pass"):
    path = root / "runs" / "suite" / name / "capsule_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    timing = {"adapter_wall_s": wall, **({"partial": True} if partial else {"build_s": 9.0, "sim_active_s": 559.0})}
    document = {
        "capsule": name,
        "status": status,
        "tiers": {
            "L2": {"status": "pass", "measured_now": measured, "timing": {"adapter_wall_s": 9.5}},
            "L3": {"status": "fail" if partial else "pass", "measured_now": measured, "timing": timing},
        },
        **({"failure": failure} if failure else {}),
    }
    path.write_text(json.dumps(document))
    return root


def test_a_completed_measurement_raises_the_budget_within_its_cap(tmp_path):
    history = _result(tmp_path / "selfcheck_a", "SLOW", wall=622.1)
    budget = CB.measured_capsule_budget([{"name": "SLOW"}, {"name": "FAST"}], [history], timeout=300)
    assert budget["requested_s"] == 300 and budget["raised"] and budget["capped"]
    assert budget["budget_s"] == 1200  # 2 x 622 s, capped at 4 x the requested 300 s
    assert budget["capsule"] == "SLOW" and budget["tier"] == "L3" and budget["measured"] == "completed"
    assert "SLOW L3 completed in 622 s in selfcheck_a" in budget["basis"]


def test_a_wall_clock_stop_escalates_instead_of_repeating(tmp_path):
    history = _result(tmp_path / "runs_00", "SLOW", wall=301.0, partial=True, failure=_STOPPED, status="x")
    budget = CB.measured_capsule_budget([{"name": "SLOW"}], [history], timeout=300)
    assert budget["budget_s"] == 602 and budget["measured"] == "stopped on its wall clock"


def test_unselected_carried_or_unrelated_history_never_raises_it(tmp_path):
    history = _result(tmp_path / "a", "OTHER", wall=900.0)
    _result(tmp_path / "a", "CARRIED", wall=900.0, measured=False)
    _result(tmp_path / "a", "CRASHED", wall=900.0, partial=True, failure={"category": "tool_crash", "tier": "L3"})
    for capsules in ([{"name": "SELECTED"}], [{"name": "CARRIED"}], [{"name": "CRASHED"}], []):
        budget = CB.measured_capsule_budget(capsules, [history, tmp_path / "missing"], timeout=300)
        assert budget == {"requested_s": 300, "budget_s": 300, "raised": False, "basis": None}
    assert CB.measured_capsule_budget([{"name": "OTHER"}], [history], timeout=3600)["raised"] is False


def _grade(monkeypatch, tmp_path, results, history):
    seen = {}

    def run_suite(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return results

    monkeypatch.setattr(CG, "load_package", lambda *a, **k: type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CG, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *a, **k: [{"name": r["capsule"]} for r in results])
    monkeypatch.setattr(CG.CR, "run_suite", run_suite)
    with CS.timing_history([history]):
        score = CG.grade(
            "pkg", capsules_root=["root"], runs_root=str(tmp_path / "now"), target="gemmini", max_workers=1, timeout=300
        )
    return score, seen["timeout"]


@pytest.mark.parametrize("measured", [True, False])
def test_the_grade_uses_and_reports_the_budget(monkeypatch, tmp_path, measured, capsys):
    history = _result(tmp_path / "selfcheck_a", "SLOW" if measured else "UNKNOWN", wall=622.1)
    stopped = {
        "capsule": "SLOW",
        "label": "public",
        "status": "cert_not_measured",
        "tiers": {},
        "failure": {**_STOPPED, "tier_reason": "elaborated_rtl exceeded its wall-clock budget"},
    }
    score, used = _grade(monkeypatch, tmp_path, [stopped], history)
    assert used == (1200 if measured else 300)
    assert score["per_capsule_budget"]["budget_s"] == used
    row = score["timed_out"]["SLOW"]
    assert row["tier"] == "L3" and row["budget_s"] == used and row["requested_s"] == 300
    assert "--qa-timeout" in row["remedy"]
    out = capsys.readouterr().out
    assert f"[timeout] SLOW L3: stopped by the {used} s per-capsule wall budget" in out
    assert ("per-capsule budget: SLOW L3 completed in 622 s" in out) is measured
