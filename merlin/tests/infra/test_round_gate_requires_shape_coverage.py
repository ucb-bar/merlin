"""Passing every public capsule must no longer be enough to call a round converged.

The public suite is a fixed set of shapes. A backend keyed on exactly those shapes passes it by
construction, so `all_pass` was reachable by a compiler that lowers nothing else -- which is the state
one submission actually froze in, at 14/26, with its self-check clean on everything it had.

These pin the gate and, just as importantly, the failure modes of the gate itself: a probe that could
not run must never read as clean, and a run with the baseline down must not be told which axes failed.
"""
from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))


@pytest.fixture()
def loop(monkeypatch):
    import run_baseline_qa_loop as L
    return L


def _run(loop, monkeypatch, tmp_path, verdict, cov=None, boom=None):
    from merlin.targetgen import lowering_coverage as LC
    if boom is not None:
        monkeypatch.setattr(LC, "sweep", lambda *a, **k: (_ for _ in ()).throw(boom))
    else:
        monkeypatch.setattr(LC, "sweep", lambda *a, **k: cov)
    loop._attach_shape_generalization(verdict, tmp_path / "cand", tmp_path, 0, timeout=60)
    return verdict


_COVERED = {"tile_edge": 16, "baseline_tile_lowered": True, "all_covered": True,
            "multi_tile_axes_uncovered": [], "emitted_work": {"tile": 29, "m_2tiles": 37},
            "corners": [{"corner": "tile", "outcome": "lowered"},
                        {"corner": "m_2tiles", "outcome": "lowered"}]}

_M_UNCOVERED = {"tile_edge": 32, "baseline_tile_lowered": True, "all_covered": False,
                "multi_tile_axes_uncovered": ["m"],
                "emitted_work": {"tile": 418, "m_2tiles": 5},
                "corners": [{"corner": "tile", "outcome": "lowered"},
                            {"corner": "m_2tiles", "outcome": "collapsed",
                             "detail": "cannot compute more by doing less"}]}


def test_all_public_passing_is_not_convergence_when_an_axis_is_uncovered(loop, monkeypatch, tmp_path):
    """The exact state that shipped: a clean public suite over a backend that lowers one tile."""
    v = _run(loop, monkeypatch, tmp_path, {"all_pass": True, "n_passed": 26, "n_capsules": 26},
             cov=_M_UNCOVERED)
    assert v["all_pass"] is False
    assert "axis/axes ['m']" in v["not_converged_reason"]
    assert v["shape_coverage"]["multi_tile_axes_uncovered"] == ["m"]


def test_a_covered_backend_still_converges(loop, monkeypatch, tmp_path):
    """The gate must not block a compiler that does generalize -- otherwise nothing can ever finish."""
    v = _run(loop, monkeypatch, tmp_path, {"all_pass": True, "n_passed": 25, "n_capsules": 25},
             cov=_COVERED)
    assert v["all_pass"] is True
    assert "not_converged_reason" not in v
    assert v["shape_coverage"]["all_covered"] is True


def test_the_gate_never_flips_a_failing_round_to_passing(loop, monkeypatch, tmp_path):
    v = _run(loop, monkeypatch, tmp_path, {"all_pass": False, "n_passed": 3, "n_capsules": 26},
             cov=_COVERED)
    assert v["all_pass"] is False


def test_a_probe_that_could_not_run_is_recorded_as_not_run(loop, monkeypatch, tmp_path):
    """An absent measurement reading as a pass is the same bug class as an unavailable oracle scoring
    as one. It must be visible, and it must not silently certify."""
    v = _run(loop, monkeypatch, tmp_path, {"all_pass": True}, boom=RuntimeError("no oracle venv"))
    sc = v["shape_coverage"]
    assert sc["ran"] is False
    assert "no oracle venv" in sc["error"]
    assert "NOT a pass" in sc["note"]


def test_a_down_baseline_is_reported_without_naming_axes(loop, monkeypatch, tmp_path):
    """With the one-tile baseline failing, nothing can be attributed to shape -- so nothing is."""
    cov = {"tile_edge": 32, "baseline_tile_lowered": False, "all_covered": False,
           "multi_tile_axes_uncovered": [], "corners": [{"corner": "tile", "outcome": "error"}],
           "unmeasured": "the single-tile baseline did not lower, so nothing here can be attributed"}
    v = _run(loop, monkeypatch, tmp_path, {"all_pass": True}, cov=cov)
    assert v["all_pass"] is False
    assert "the baseline tile itself" in v["not_converged_reason"]
    assert v["shape_coverage"]["multi_tile_axes_uncovered"] == []
    assert v["shape_coverage"]["unmeasured"]


def test_the_report_is_persisted_for_audit(loop, monkeypatch, tmp_path):
    _run(loop, monkeypatch, tmp_path, {"all_pass": True}, cov=_M_UNCOVERED)
    p = tmp_path / "qa_history" / "shape_coverage_round_00.json"
    assert p.is_file()
    assert json.loads(p.read_text())["multi_tile_axes_uncovered"] == ["m"]
