"""Command console progress: background grades in flight, the published copy, certificates, the series."""

import json
from datetime import UTC, datetime
from pathlib import Path

from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import Context
from merlin_experiments.tracking.console.sources import progress

T0 = 1792810000.0


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat()


def _json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _rows(passed: int, failed: int, errors: int = 0) -> list[dict]:
    statuses = ["pass"] * passed + ["fail"] * failed + ["error"] * errors
    return [{"capsule": f"c{i}", "status": s} for i, s in enumerate(statuses)]


def _grade(run: Path, token: int, owner_state: str, reason: str, doc: dict | None, verdict: dict | None = None):
    work = run / "_qa_work"
    _json(
        work / "background_snapshot_owners" / f"{token}.json",
        {
            "schema": "merlin.background_snapshot_owner.v1",
            "token": str(token),
            "source": "a" * 64,
            "started_at_epoch": token / 1e9,
            "state": owner_state,
            "admission_reason": reason,
        },
    )
    if doc is not None:
        _json(work / f"fruns_{token}" / "snapshot_progress.json", {"schema": "merlin.phase1.grade_progress.v1", **doc})
    if verdict is not None:
        _json(run / "qa_history" / f"verdict_fast_{token}.json", verdict)


def _run(tmp_path: Path) -> tuple[Path, Path]:
    run, workspace = tmp_path / "run", tmp_path / "workspace"
    stopped, superseded, grading = (int((T0 + k * 600) * 1e9) for k in (0, 1, 2))
    _grade(
        run,
        stopped,
        "complete",
        "snapshot owner stopped new admission",
        {"state": "complete", "n_selected": 252, "n_completed": 0, "n_passed": 0, "updated_at": _iso(T0 + 60)},
        {"graded_at": _iso(T0 + 60), "n_passed": 0, "n_capsules": 252, "per_capsule": []},
    )
    _grade(
        run,
        superseded,
        "complete",
        "snapshot admission deadline reached",
        {
            "state": "superseded_snapshot",
            "n_selected": 252,
            "n_completed": 24,
            "n_passed": 4,
            "updated_at": _iso(T0 + 1100),
            "per_capsule": _rows(4, 19, 1),
        },
        {"graded_at": _iso(T0 + 1100), "n_passed": 4, "n_capsules": 252},
    )
    _grade(
        run,
        grading,
        "draining",
        "snapshot admission deadline reached",
        {
            "state": "running",
            "n_selected": 232,
            "n_completed": 10,
            "n_passed": 6,
            "updated_at": _iso(T0 + 1700),
            "per_capsule": _rows(6, 4),
        },
    )
    # A round-end grade of a different capsule subset: not a point of the background series.
    _json(
        run / "qa_history" / "verdict_round_00.json",
        {"graded_at": _iso(T0 + 900), "n_passed": 0, "n_capsules": 13, "per_capsule": _rows(0, 13)},
    )
    _json(
        workspace / "qa" / "progress.json",
        {
            "schema": "merlin.phase1.grade_progress.v1",
            "state": "running",
            "n_selected": 232,
            "n_completed": 6,
            "n_passed": 4,
            "updated_at": _iso(T0 + 200),
        },
    )
    _json(
        workspace / "qa" / "tier_state.json",
        {
            "c0": {"<certs>": {"L2": {"x": {"status": "pass"}}, "L3": {"y": {"status": "pending"}}}},
            "c1": {"L2": {"status": "pass"}, "L3": {"status": "pass"}},
        },
    )
    _json(
        workspace / "qa" / "joint_physical_correctness.json",
        {"schema": "merlin.joint_physical_correctness.v1", "required": 3, "passed": 1, "failed": 0, "pending": 2},
    )
    return run, workspace


def _ctx(run: Path, workspace: Path, *, live: bool) -> Context:
    experiment = {"id": str(run), "run_id": run.name, "run_dir": str(run), "workspace": str(workspace)}
    return Context(
        experiment=experiment, reader=Reader(), state={}, operator_private=True, now=T0 + 1800, shared={}, live=live
    )


def test_grades_in_flight_come_from_the_background_grader(tmp_path):
    section = progress.read(_ctx(*_run(tmp_path), live=True))
    grading, superseded, stopped = section["grades"]
    assert (grading["state"], grading["n_completed"], grading["n_passed"], grading["n_failed"]) == ("grading", 10, 6, 4)
    assert abs(grading["updated_at"] - (T0 + 1700)) < 1e-3
    assert (superseded["state"], superseded["n_failed"], superseded["n_errors"]) == ("superseded", 19, 1)
    assert (stopped["state"], stopped["reason"]) == ("stopped", "snapshot owner stopped new admission")
    # The agent's published copy is kept apart: it trails the grades.
    assert section["published"]["n_completed"] == 6 and abs(section["published"]["updated_at"] - (T0 + 200)) < 1e-3


def test_a_stopped_run_leaves_its_grades_unfinished(tmp_path):
    section = progress.read(_ctx(*_run(tmp_path), live=False))
    assert section["grades"][0]["state"] == "unfinished"


def test_series_keeps_one_scope_and_skips_empty_grades(tmp_path):
    section = progress.read(_ctx(*_run(tmp_path), live=True))
    assert section["series_scope"] == "background grades"
    assert section["series"]["n_passed"] == [4, 6] and section["series"]["n_completed"] == [24, 10]


def test_certificates_count_ledger_and_physical_obligation(tmp_path):
    certificates = progress.read(_ctx(*_run(tmp_path), live=True))["certificates"]
    assert certificates["L2"] == {"pass": 2, "pending": 0, "fail": 0}
    assert certificates["L3"] == {"pass": 1, "pending": 1, "fail": 0}
    assert certificates["physical"] == {"required": 3, "passed": 1, "failed": 0, "pending": 2}


def test_round_grades_make_the_series_without_background_grades(tmp_path):
    run, workspace = tmp_path / "run", tmp_path / "workspace"
    _json(
        run / "qa_history" / "verdict_round_00.json",
        {"graded_at": _iso(T0), "n_passed": 2, "n_capsules": 13, "per_capsule": _rows(2, 11)},
    )
    section = progress.read(_ctx(run, workspace, live=True))
    assert section["grades"] == [] and section["series_scope"] == "round grades"
    assert section["series"]["n_passed"] == [2]


def test_results_of_an_older_compiler_are_stale(tmp_path):
    from merlin_experiments.tracking.console.sources import capsules

    run = tmp_path / "run"
    old, new = int(T0 * 1e9), int((T0 + 600) * 1e9)
    for token, source, state, status, at in (
        (old, "a" * 64, "running", "fail", T0 + 900),
        (new, "b" * 64, "running", "pass", T0 + 700),
    ):
        _json(
            run / "_qa_work" / f"fruns_{token}" / "snapshot_progress.json",
            {
                "schema": "merlin.phase1.grade_progress.v1",
                "state": state,
                "graded_source_sha256": source,
                "updated_at": _iso(at),
                "per_capsule": [
                    {"capsule": "shared", "status": status},
                    {"capsule": f"only_{status}", "status": status},
                ],
            },
        )
    ctx = _ctx(run, tmp_path / "workspace", live=True)
    # The old compiler's grade wrote last, but the newer compiler's result wins.
    ranked = max((r for r in capsules._results(ctx, run, None) if r[2]["capsule"] == "shared"), key=lambda r: r[:2])
    assert ranked[2]["status"] == "pass"
    assert capsules._current_sources(ctx, run) == {"a" * 64, "b" * 64}
    # Once the old grade is superseded, only the newer compiler is current.
    doc = json.loads((run / "_qa_work" / f"fruns_{old}" / "snapshot_progress.json").read_text())
    doc["state"] = "superseded_snapshot"
    _json(run / "_qa_work" / f"fruns_{old}" / "snapshot_progress.json", doc)
    assert capsules._current_sources(_ctx(run, tmp_path / "workspace", live=True), run) == {"b" * 64}


def test_lock_inventory_streams_a_large_proc_locks(tmp_path):
    import os

    from merlin_experiments.tracking import terminal_probe

    proc = tmp_path / "proc"
    proc.mkdir()
    filler = "".join(f"{i}: POSIX  ADVISORY  WRITE 1 00:00:{i} 0 EOF\n" for i in range(1, 120000))
    (proc / "locks").write_text(filler + "999: FLOCK  ADVISORY  WRITE 4242 08:01:77 0 EOF\n")
    assert (proc / "locks").stat().st_size > 4 << 20
    key = (os.makedev(8, 1), 77)
    assert terminal_probe.locks(proc)[key] == [{"pid": 4242, "lock_kind": "FLOCK", "mode": "WRITE"}]
    assert list(terminal_probe.locks(proc, wanted={key})) == [key]


def test_a_loop_pass_waiting_for_its_certificate_is_counted_apart():
    waiting = {
        "capsule": "a",
        "status": "incomplete",
        "tiers": {"L2": "pass", "L3": "unavailable"},
        "failure_tier": "L3",
        "failure_plane": "oracle_unavailable",
    }
    other = {"capsule": "b", "status": "incomplete", "tiers": {"L2": "fail"}, "failure_tier": "L2"}
    assert progress._tally({"per_capsule": [waiting, other, {"capsule": "c", "status": "pass"}]}) == {
        "pass": 1,
        "fail": 0,
        "error": 0,
        "l2": 1,
    }
