"""A phase-2 round must seal its BEST authored revision, not its last.

A round used to seal whatever the agent left in its workspace when the budget expired, and the next
round resumed from that seal -- so an improvement the agent found and then moved off was discarded.
Measured over two runs / five rounds: revisions that removed host allocations were never sealed and
four consecutive rounds sealed one neutral revision. These tests pin the selection.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "run_global_perf_experiment_under_test", _SCRIPTS / "run_global_perf_experiment.py")
assert _SPEC is not None and _SPEC.loader is not None
GPE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = GPE
_SPEC.loader.exec_module(GPE)

COST = GPE.GlobalPerfExperiment.authored_host_cost
BEST = GPE.GlobalPerfExperiment.best_authored_candidate


def _analysis(load: int | None, store: int, allocations: int) -> dict:
    host: dict = {"store_payload_bytes": store, "static_operations": {"allocation": allocations}}
    if load is not None:
        host["load_payload_bytes"] = load
    return {"diagnostics": {"verified_global_plan_emission": {"host_activity": host}}}


def _row(tmp_path: Path, iteration: int, *, load: int | None, store: int, allocations: int,
         ready: bool = True, drift: bool = False) -> dict:
    snap = tmp_path / f"submission_{iteration:04d}"
    snap.mkdir()
    (snap / "manifest.yaml").write_text(f"iteration: {iteration}\n", encoding="utf-8")
    from merlin.benchharness import hash_tree
    digest = hash_tree(snap)["sha256"]
    if drift:                        # the snapshot no longer holds the analyzed bytes
        (snap / "manifest.yaml").write_text("edited after analysis\n", encoding="utf-8")
    return {
        "iteration": iteration,
        "candidate_sha256": digest,
        "submitted_snapshot": str(snap),
        "readiness": {"status": "ready_for_probe_admission" if ready else "blocked"},
        "analysis": _analysis(load, store, allocations),
    }


class _Stub:
    """Only `iterations` and the cost helper are used by the selection."""
    # staticmethod(): `COST` is the plain function once read off the class, and assigning a bare
    # function as a class attribute would rebind it as an instance method.
    authored_host_cost = staticmethod(COST)

    def __init__(self, iterations):
        self.iterations = iterations


def test_cost_is_bytes_then_allocations() -> None:
    assert COST(_analysis(100, 50, 3)) == (150, 3)
    # an analysis that emitted no verified plan is INELIGIBLE, never implicitly best
    assert COST(None) is None
    assert COST({}) is None
    assert COST(_analysis(None, 50, 3)) is None
    assert COST({"diagnostics": {}}) is None


def test_seals_the_cheapest_revision_not_the_last(tmp_path) -> None:
    rows = [
        _row(tmp_path, 0, load=1000, store=100, allocations=10),   # seed
        _row(tmp_path, 1, load=900, store=90, allocations=9),      # the win
        _row(tmp_path, 2, load=1000, store=100, allocations=10),   # reverted
        _row(tmp_path, 3, load=1000, store=100, allocations=10),   # neutral, and LAST
    ]
    best = BEST(_Stub(rows))

    assert best is not None
    assert best["iteration"] == 1, "the cheapest revision must win, not the final one"
    assert best["host_payload_bytes"] == 990
    assert best["candidate_sha256"] == rows[1]["candidate_sha256"]
    assert best["considered"] == 4


def test_ties_keep_the_latest_revision(tmp_path) -> None:
    rows = [_row(tmp_path, 0, load=500, store=50, allocations=5),
            _row(tmp_path, 1, load=500, store=50, allocations=5)]

    assert BEST(_Stub(rows))["iteration"] == 1


def test_blocked_and_unanalyzed_revisions_are_ineligible(tmp_path) -> None:
    rows = [_row(tmp_path, 0, load=1000, store=100, allocations=10),
            _row(tmp_path, 1, load=1, store=1, allocations=1, ready=False)]   # cheapest but blocked

    best = BEST(_Stub(rows))

    assert best["iteration"] == 0, "a blocked revision must never be sealed however cheap"
    assert best["considered"] == 1


def test_a_snapshot_that_drifted_from_its_analysis_is_skipped(tmp_path) -> None:
    rows = [_row(tmp_path, 0, load=1000, store=100, allocations=10),
            _row(tmp_path, 1, load=1, store=1, allocations=1, drift=True)]  # cheapest but not the
                                                                            # bytes it was graded on
    best = BEST(_Stub(rows))

    assert best["iteration"] == 0
    assert best["considered"] == 1


def test_no_eligible_revision_returns_none_so_the_caller_keeps_its_own(tmp_path) -> None:
    rows = [_row(tmp_path, 0, load=1000, store=100, allocations=10, ready=False)]

    assert BEST(_Stub(rows)) is None
