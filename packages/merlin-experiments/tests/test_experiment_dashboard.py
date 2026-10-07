"""The experiment dashboard and ``watch`` read only existing records and say what is not recorded.

Fixture runs are tiny synthetic records in the owners' own schemas: a phase-1 run's graded history, a
phase-2 measured run with its store (one job per failure class), and runs missing their records.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml
from merlin_experiments.cli import main
from merlin_experiments.phase2.whole_model_measured import batch as B
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import ledger as L
from merlin_experiments.phase2.whole_model_measured import runs as RUNS
from merlin_experiments.phase2.whole_model_measured import sessions as S
from merlin_experiments.tracking import html, records, text, write_dashboard

from merlin.perf import whole_model_verdict as V

TARGET = "toy"
HOUR = 3600.0


def _stamp(epoch: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(epoch))


def _write(path: Path, document) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return path


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "out"
    for sub in ("runs", "artifacts", "build"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    return root


# --------------------------------------------------------------------------- phase-2 fixture
def _job(store: Path, key: str, *, requested: float, state: str = J.DONE, **extra) -> Path:
    job = {
        "schema": J.JOB_SCHEMA,
        "package_sha256": key.partition(".")[0],
        "job_key": key,
        "label": extra.pop("label", f"candidate {key[:4]}"),
        "role": extra.pop("role", J.ROLE_CANDIDATE),
        "state": state,
        "requested_at": _stamp(requested),
        "requested_epoch": requested,
        "replicate": extra.pop("replicate", 0),
        "target": TARGET,
        **extra,
    }
    return _write(store / key / "job.json", job)


def _result(store: Path, key: str, *, finished: float, status: str, cycles: int | None = None, **extra) -> None:
    verdict = extra.pop("verdict", None)
    if verdict is None and cycles is not None:
        verdict = {"timing_status": status, "whole_window_cycles": cycles, "groups": []}
    _write(
        store / key / "result.json",
        {
            "schema": J.RESULT_SCHEMA,
            "package_sha256": key.partition(".")[0],
            "finished_at": _stamp(finished),
            "timing_status": status,
            "objective_cycles": cycles if status == V.TIMING_MEASURED else None,
            **({"verdict": verdict} if verdict is not None else {}),
            **extra,
        },
    )


def phase2_run(root: Path, t0: float, *, store_recorded: bool = True) -> tuple[Path, Path]:
    """A measured run started at ``t0`` whose store holds one job per outcome class."""
    run_dir = root / "runs" / TARGET / "phase2" / f"{_stamp(t0)}_whole_model_measured_abc1234"
    store = root / "artifacts" / "perf-studies" / "whole-model" / TARGET / "store0"
    reference = _write(
        root / "artifacts" / "perf-studies" / "whole-model" / TARGET / "ref" / "result.json",
        {
            "schema": J.RESULT_SCHEMA,
            "label": "vendor reference",
            "timing_status": V.TIMING_MEASURED,
            "verdict": {
                "whole_window_cycles": 700_000,
                "groups": [
                    {"group": "1", "kind": "conv", "cycles": 300_000},
                    {"group": "2", "kind": "conv", "cycles": 200_000},
                    {"group": "3", "kind": "add", "cycles": 200_000},
                ],
            },
        },
    )
    _write(
        run_dir / "run.json",
        {
            "schema": RUNS.RUN_SCHEMA,
            "mode": "whole_model_measured",
            "target": TARGET,
            "method": "whole_model_measured",
            "prohibited_instruction_roles": ["loop_role"],
        },
    )
    _write(
        run_dir / "resumed_seed.json",
        {
            "schema": RUNS.RESUMED_SEED_SCHEMA,
            "seed_package_sha256": "5eed" * 16,
            "store_roots": {"screen": str(store)} if store_recorded else {},
            "lineage_kind": "phase1_freeze",
            "origin_kind": "phase1_freeze",
            "frozen": True,
            "why": "fixture run",
        },
    )
    _write(
        run_dir / RUNS.CONFIG_NAME,
        {
            "plateau_hours": 6,
            "plateau_min_sessions": 2,
            "screen": {"reference": str(reference)},
            "orientation": [{"label": "analytical floor", "whole_model_cycles": 400_000, "note": "not a measurement"}],
        },
    )
    rows = [
        {"kind": "candidate", "n": 1, "package_sha256": "a" * 64, "label": "seed", "commit": {"commit": "c1"}},
        {
            "kind": "measured",
            "n": 1,
            "tag": "measured/1",
            "timing_status": "MEASURED",
            "objective_cycles": 1_000_000,
            "at": _stamp(t0 + 1 * HOUR),
        },
        {"kind": "best", "n": 1, "package_sha256": "a" * 64, "at": _stamp(t0 + 1 * HOUR)},
    ]
    (run_dir / L.ITERATIONS).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    routes = {
        "groups": [{"group": "1", "on": "package"}, {"group": "2", "on": "package"}, {"group": "3", "on": "library"}]
    }
    reference_pin = {"reference": {"path": str(reference), "sha256": "r" * 64}}
    _job(store, "a" * 64, requested=t0 + 0.1 * HOUR, label="seed", **reference_pin)
    _result(store, "a" * 64, finished=t0 + 1 * HOUR, status=V.TIMING_MEASURED, cycles=1_000_000, build=routes)
    _job(store, "b" * 64, requested=t0 + 1.5 * HOUR, label="faster", **reference_pin)
    _result(
        store,
        "b" * 64,
        finished=t0 + 2 * HOUR,
        status=V.TIMING_MEASURED,
        cycles=900_000,
        build=routes,
        verdict={"whole_window_cycles": 900_000, "groups": [{"group": "1", "kind": "conv", "cycles": 400_000}]},
        diagnostics={
            "per_group": {"1": {"roofline": {"status": "derived", "roofline_cycles": 250_000, "limiter": "compute"}}}
        },
    )
    _job(store, "c" * 64, requested=t0 + 2.5 * HOUR)
    _result(
        store,
        "c" * 64,
        finished=t0 + 3 * HOUR,
        status=V.TIMING_MEASURED_INVALID,
        cycles=800_000,
        verdict={"whole_window_cycles": 800_000, "invalid_reason": "g2 output differs", "groups": []},
    )
    _job(store, "d" * 64, requested=t0 + 3.1 * HOUR)
    _result(
        store,
        "d" * 64,
        finished=t0 + 3.2 * HOUR,
        status=V.TIMING_REFUSED,
        refusal="isa_prohibited: loop_role; no board time",
        isa_prohibited={"clean": False},
    )
    _job(store, "e" * 64, requested=t0 + 3.3 * HOUR)
    _result(
        store,
        "e" * 64,
        finished=t0 + 3.4 * HOUR,
        status=V.TIMING_REFUSED,
        refusal="coverage_regression: 1 group(s) declined to the library",
        coverage_regression={"declined_groups": ["2"]},
    )
    _job(store, "f" * 64, requested=t0 + 3.5 * HOUR)
    _result(
        store,
        "f" * 64,
        finished=t0 + 3.6 * HOUR,
        status=V.TIMING_REFUSED,
        refusal=f"{J.INFRA_WORKER_LOST}: measurement lost",
        infra_worker_lost=True,
    )
    _job(store, "1" * 64, requested=t0 + 3.7 * HOUR, state=J.SCREEN_FAILED)
    _result(
        store,
        "1" * 64,
        finished=t0 + 3.8 * HOUR,
        status=V.TIMING_REFUSED,
        screen_failed=True,
        refusal="screen_failed: the required capsule screen failed",
        pre_measure_check={"passed": False, "output_tail": "2 capsules mismatched", "summary": {"error": None}},
    )
    _job(store, "2" * 64, requested=t0 + 3.9 * HOUR)
    _result(store, "2" * 64, finished=t0 + 4.0 * HOUR, status=V.TIMING_REFUSED, refusal="no admissible reading")
    _job(store, "3" * 64, requested=t0 + 4.1 * HOUR, state=J.BOARD, board_ready_epoch=t0 + 4.2 * HOUR)

    _write(
        store / S.PLATEAU_FILE,
        {
            "schema": S.PLATEAU_SCHEMA,
            "rule": {"hours": 6.0, "min_sessions": 2},
            "last_improvement": {
                "at": _stamp(t0 + 2 * HOUR),
                "epoch": t0 + 2 * HOUR,
                "cycles": 900_000,
                "package_sha256": "b" * 64,
                "hours_since": 1.5,
                "sessions_since": 1,
            },
            "trace": [
                {
                    "at": _stamp(t0 + 2 * HOUR),
                    "epoch": t0 + 2 * HOUR,
                    "run": str(run_dir / "stage"),
                    "session": 1,
                    "cycles": 900_000,
                    "package_sha256": "b" * 64,
                }
            ],
        },
    )
    (store / B.BOARD_OUTAGES).write_text(
        json.dumps(
            {
                "opened_at": _stamp(t0),
                "opened_epoch": t0,
                "closed_at": _stamp(t0 + HOUR),
                "failures": [{"reason": "infra_board_unavailable: flash failed"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write(store / B.SOLO_STREAK_FILE, {"count": 2})
    return run_dir, store


# --------------------------------------------------------------------------- phase-1 fixture
def _capsule(name: str, status: str, tiers: dict, **extra) -> dict:
    return {"capsule": name, "label": "public", "status": status, "tiers": tiers, **extra}


def phase1_run(root: Path, t0: float) -> Path:
    run_dir = root / "runs" / TARGET / "phase1" / "fixture-arm-run"
    passing = {"L0": "pass", "L1": "pass", "L2": "pass", "L3": "pass"}
    grades = [
        (
            t0,
            [
                _capsule("cap_add", "pass", {"L0": "pass", "L1": "pass"}),
                _capsule(
                    "cap_mm",
                    "fail",
                    {"L0": "pass", "L1": "fail"},
                    failure_plane="numeric",
                    failure_detail="4 mismatches",
                ),
                _capsule(
                    "cap_model", "error", {}, failure_plane="runner_internal", failure_detail="RUNNER_CRASH: no facts"
                ),
            ],
        ),
        (
            t0 + HOUR,
            [
                _capsule("cap_add", "pass", passing),
                _capsule("cap_mm", "pass", {"L0": "pass", "L1": "pass", "L2": "pass"}),
                _capsule(
                    "cap_model", "error", {}, failure_plane="runner_internal", failure_detail="RUNNER_CRASH: no facts"
                ),
            ],
        ),
    ]
    for index, (at, capsules) in enumerate(grades):
        _write(
            run_dir / "qa_history" / f"verdict_round_{index:02d}.json",
            {
                "graded_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(at)),
                "n_passed": sum(c["status"] == "pass" for c in capsules),
                "n_capsules": len(capsules),
                "all_pass": False,
                "highest_tier": "L3" if index else "L1",
                "first_failure_planes": {"runner_internal": 1} | ({} if index else {"numeric": 1}),
                "per_capsule": capsules,
            },
        )
    _write(
        run_dir / "plateau.json",
        {
            "stuck": False,
            "sentence": "2/3 passing after 2 grade(s)",
            "n_grades": 2,
            "best_passed": 2,
            "latest_passed": 2,
            "never_passed": ["cap_model"],
            "regressed": [],
        },
    )
    (run_dir / "oot_commits.jsonl").write_text(
        json.dumps(
            {
                "label": "round",
                "key": "01",
                "n_passed": 2,
                "n_capsules": 3,
                "oot": {"commit": "f" * 40, "committed_at": _stamp(t0 + HOUR), "package_digest": "9" * 64},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "environment.yaml").write_text(yaml.safe_dump({"task_scope": {"target": TARGET}}), encoding="utf-8")
    return run_dir


def _snapshot(*roots: Path) -> dict:
    return {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for root in roots
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------- phase 2
def test_failure_classes_follow_the_owners_record_fields(out_root):
    t0 = 1_790_000_000.0
    run_dir, store = phase2_run(out_root, t0)
    summary = records.run_summary(run_dir, now=t0 + 5 * HOUR)
    p2 = summary["phase2"]
    by_key = {c["key"][:1]: c["class"] for c in p2["candidates"]}
    assert by_key == {
        "a": "measured",
        "b": "measured",
        "c": "correctness",
        "d": "prohibited_instruction",
        "e": "declined",
        "f": "infra",
        "1": "correctness",
        "2": "refused",
        "3": "pending",
    }
    assert p2["best"]["cycles"] == 900_000 and p2["bar"]["cycles"] == 700_000
    assert [row["cycles"] for row in p2["best_line"]] == [1_000_000, 900_000]
    assert p2["board"]["queue"] == 1 and p2["board"]["solo_streak"] == 2
    assert len(p2["board"]["closed_outages"]) == 1
    seed = next(c for c in p2["candidates"] if c["key"].startswith("a"))
    assert seed["coverage"] == {"groups_answered": 2, "groups_total": 3, "priced_share": 0.7143}
    assert p2["roofline"]["by_kind"] == [{"kind": "conv", "groups": 1, "ours": 400_000, "roofline": 250_000}]
    assert [e["n"] for e in p2["ledger"]["best"]] == [1]


def test_an_earlier_attempts_verdict_stands_over_a_host_lost_retry(out_root):
    """A re-queued job whose retry was lost to the host still shows the reading it already had."""
    t0 = 1_790_000_000.0
    run_dir, store = phase2_run(out_root, t0)
    key = "9" * 64
    _job(store, key, requested=t0 + 2.5 * HOUR, label="re-queued")
    archived = store / key / J.ATTEMPTS_DIR / "1"
    _write(
        archived / J.RESULT_FILE,
        {
            "schema": J.RESULT_SCHEMA,
            "package_sha256": key,
            "finished_at": _stamp(t0 + 3 * HOUR),
            "timing_status": V.TIMING_MEASURED,
            "objective_cycles": 850_000,
            "verdict": {"timing_status": V.TIMING_MEASURED, "whole_window_cycles": 850_000, "groups": []},
        },
    )
    _write(archived / J.ATTEMPT_RECORD, {"kind": "requeue", "why": "worker lost"})
    _result(
        store,
        key,
        finished=t0 + 4 * HOUR,
        status=V.TIMING_REFUSED,
        refusal="worker lost to host pressure",
        infra_worker_lost=True,
    )
    p2 = records.run_summary(run_dir, now=t0 + 5 * HOUR)["phase2"]
    row = next(c for c in p2["candidates"] if c["key"] == key)
    assert row["class"] == "measured" and row["cycles"] == 850_000 and row["from_attempt"] == 1
    assert p2["best"]["cycles"] == 850_000


def test_html_carries_the_recorded_numbers_and_is_self_contained(out_root, tmp_path):
    t0 = 1_790_000_000.0
    run_dir, store = phase2_run(out_root, t0)
    before = _snapshot(run_dir, store)
    result = write_dashboard(run_dir=run_dir, out=tmp_path / "page.html", now=t0 + 5 * HOUR)
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert _snapshot(run_dir, store) == before  # read-only: nothing re-measured, nothing rewritten
    assert result["state"] == records.LIVE
    for expected in (
        "900,000",
        "700,000",
        "analytical floor",
        "71.4%",
        "1.60x",
        "prohibited_instruction",
        "declined",
        "isa_prohibited: loop_role",
        "Board queue",
        "<svg",
        "this run",
    ):
        assert expected in page, expected
    assert "http://" not in page and "https://" not in page and "<link" not in page  # nothing fetched
    assert page.count("<script>") == 1


def test_stalled_when_no_candidate_was_measured_past_the_threshold(out_root):
    t0 = 1_790_000_000.0
    run_dir, _ = phase2_run(out_root, t0)
    fresh = records.run_summary(run_dir, now=t0 + 5 * HOUR)["liveness"]
    assert fresh["state"] == records.LIVE and fresh["last_measured"] == t0 + 3 * HOUR  # the wrong-output reading
    stale = records.run_summary(run_dir, now=t0 + 3 * HOUR + 7 * HOUR)["liveness"]
    assert stale["state"] == records.STALLED and stale["hours"] == 7.0
    assert records.run_summary(run_dir, now=t0 + 10 * HOUR, stall_hours=12)["liveness"]["state"] == records.LIVE
    _write(
        run_dir / "stage" / "sessions.json",
        {
            "schema": S.SEQUENCE_SCHEMA,
            "sessions": [],
            "stopped": {"kind": "plateau", "reason": "no improvement for 6 h"},
        },
    )
    stopped = records.run_summary(run_dir, now=t0 + 30 * HOUR)["liveness"]
    assert stopped["state"] == records.STOPPED and "plateau" in stopped["detail"]


def test_a_relaunched_run_is_not_called_stalled(out_root):
    t0 = 1_790_000_000.0
    run_dir, _ = phase2_run(out_root, t0)
    later = run_dir.parent / f"{_stamp(t0 + 20 * HOUR)}_whole_model_measured_abc1234"
    _write(later / "run.json", {"schema": RUNS.RUN_SCHEMA, "target": TARGET, "resumed_from_run": str(run_dir)})
    liveness = records.run_summary(run_dir, now=t0 + 30 * HOUR)["liveness"]
    assert liveness["state"] == records.RELAUNCHED and later.name in liveness["detail"]


# --------------------------------------------------------------------------- phase 1
def test_phase1_view_shows_tiers_over_grades_and_failing_reasons(out_root, tmp_path):
    t0 = 1_790_000_000.0
    run_dir = phase1_run(out_root, t0)
    summary = records.run_summary(run_dir, now=t0 + 2 * HOUR)
    p1 = summary["phase1"]
    assert summary["target"] == TARGET and summary["phases"] == ["1"]
    assert p1["tiers"] == ["L0", "L1", "L2", "L3"]
    assert [g["n_passed"] for g in p1["grades"]] == [1, 2]
    assert [c["capsule"] for c in p1["failing"]] == ["cap_model"]
    assert {c["capsule"]: c["highest_pass"] for c in p1["latest"]["capsules"]} == {
        "cap_add": "L3",
        "cap_mm": "L2",
        "cap_model": None,
    }
    assert p1["liveness"]["state"] == records.LIVE
    assert records.run_summary(run_dir, now=t0 + 9 * HOUR)["phase1"]["liveness"]["state"] == records.STALLED
    page = html.render(summary)
    for expected in ("cap_mm", "RUNNER_CRASH: no facts", "pass, highest tier L3", "2 / 3", "runner_internal"):
        assert expected in page, expected
    assert 'Freeze (<span class="mono">freeze.json</span>): <span class="nr">not recorded' in page
    _write(
        run_dir / "freeze.json",
        {"frozen_at": "2026-09-21T13:00:00+00:00", "submission_sha256": "9" * 64, "oot": {"frozen_commit": "f" * 40}},
    )
    assert records.run_summary(run_dir, now=t0 + 90 * HOUR)["liveness"]["state"] == records.FINISHED


# --------------------------------------------------------------------------- missing records
def test_missing_records_render_as_not_recorded_never_a_crash(out_root, tmp_path):
    bare = out_root / "runs" / TARGET / "phase2" / "20260920T000000Z_whole_model_measured_abc1234"
    _write(bare / "run.json", {"schema": RUNS.RUN_SCHEMA, "target": TARGET, "method": "whole_model_measured"})
    summary = records.run_summary(bare, now=1_790_000_000.0)
    p2 = summary["phase2"]
    assert p2["store"]["root"] is None and p2["bar"] is None and p2["plateau"] is None and p2["ledger"] is None
    assert p2["liveness"]["state"] == records.STALLED and "no measurement store" in p2["liveness"]["detail"]
    page = html.render(summary)
    assert "not recorded" in page and "Measurement store" in page
    states = {row["record"]: row["state"] for row in summary["inventory"]}
    assert states["resumed_seed.json"] == "absent" and states["run.json"] == "read"

    empty = tmp_path / "nothing_here"
    empty.mkdir()
    nothing = records.run_summary(empty, now=1_790_000_000.0)
    assert nothing["phases"] == [] and nothing["liveness"]["state"] == records.UNKNOWN
    assert "No phase records were found" in html.render(nothing)


def test_unreadable_and_foreign_schema_records_are_named_in_the_inventory(out_root):
    t0 = 1_790_000_000.0
    run_dir, store = phase2_run(out_root, t0)
    (store / S.PLATEAU_FILE).write_text("{not json", encoding="utf-8")
    seed = json.loads((run_dir / "resumed_seed.json").read_text())
    _write(run_dir / "resumed_seed.json", {**seed, "schema": "some_older_seed_v1"})
    summary = records.run_summary(run_dir, now=t0 + 5 * HOUR)
    states = {row["record"]: row for row in summary["inventory"]}
    assert states["store plateau.json"]["state"] == "unreadable"
    assert states["resumed_seed.json"]["state"] == "unexpected schema"
    assert summary["phase2"]["plateau"] is None


# --------------------------------------------------------------------------- CLI
def test_dashboard_cli_writes_under_the_declared_home(out_root, capsys):
    from merlin.common.storage_cli import declared_concerns

    run_dir, _ = phase2_run(out_root, time.time() - 20 * HOUR)
    assert main(["dashboard", str(run_dir)]) == 0
    result = json.loads(capsys.readouterr().out)
    expected = out_root / "artifacts" / "experiments" / TARGET / "dashboard" / f"{run_dir.name}.html"
    assert Path(result["dashboard"]) == expected and expected.is_file()
    assert result["state"] == records.STALLED  # the fixture's last reading is 17 h old
    assert "experiments" in declared_concerns()


def test_dashboard_cli_refuses_ambiguous_or_untargeted_requests(out_root, tmp_path, capsys):
    run_dir, _ = phase2_run(out_root, time.time())
    assert main(["dashboard", str(run_dir), "--target", TARGET]) == 2
    assert main(["dashboard"]) == 2
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    assert main(["dashboard", str(orphan)]) == 2
    assert "records no target" in capsys.readouterr().err
    assert main(["dashboard", str(tmp_path / "missing"), "--out", str(tmp_path / "x.html")]) == 2


def test_target_view_lists_runs_and_the_recorded_champion_lineage(out_root, capsys):
    from merlin.targetgen import target_index

    now = time.time()
    phase2_run(out_root, now - 20 * HOUR)
    phase1_run(out_root, now - 2 * HOUR)
    index = target_index.index_path(TARGET)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        yaml.safe_dump(
            {
                "schema": target_index.SCHEMA,
                "target": TARGET,
                "champions": [
                    {
                        "package_id": "toy_champion_v1",
                        "firesim": {"cycles": 900_000, "control_in_batch": True},
                        "lineage": {
                            "phase1_run": f"{TARGET}/phase1/fixture-arm-run",
                            "frozen_commit": "f" * 40,
                            "phase2_run": f"{TARGET}/phase2/x",
                            "best_commit": "b" * 40,
                            "corpus_seal_digest": "a" * 64,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["dashboard", "--target", TARGET]) == 0
    written = Path(json.loads(capsys.readouterr().out)["dashboard"])
    page = written.read_text(encoding="utf-8")
    assert written == out_root / "artifacts" / "experiments" / TARGET / "dashboard" / "target.html"
    for expected in (
        "toy_champion_v1",
        "900,000",
        "fixture-arm-run",
        "whole_model_measured_abc1234",
        "STALLED",
        "LIVE",
    ):
        assert expected in page, expected


def test_watch_once_prints_the_same_summary(out_root, capsys):
    run_dir, store = phase2_run(out_root, time.time() - 20 * HOUR)
    assert main(["watch", str(run_dir), "--once"]) == 0
    out = capsys.readouterr().out
    assert "\x1b[" not in out  # not a terminal: plain text
    for expected in (
        "STATE  STALLED",
        "best 900,000",
        "bar 700,000",
        "prohibited_instruction 1",
        "declined 1",
        "board  queue 1",
        "plateau  rule 6.0 h / 2 sessions",
    ):
        assert expected in out, expected


def test_watch_refreshes_until_interrupted(out_root):
    import io

    run_dir = phase1_run(out_root, time.time())
    stream, ticks = io.StringIO(), []

    def sleep(seconds):
        ticks.append(seconds)
        if len(ticks) == 2:
            raise KeyboardInterrupt

    from merlin_experiments.tracking import watch

    assert watch(run_dir, interval=5, stream=stream, sleep=sleep, colour=True) == 0
    output = stream.getvalue()
    assert output.count(text.CLEAR) == 2 and ticks == [5, 5]
    assert "\x1b[32mLIVE\x1b[0m" in output
