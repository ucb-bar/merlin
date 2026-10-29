"""A refresh's byte budget never starves the panels that judge by the small authoritative records.

Fixtures keep a long live run's proportions at a quarter of its size against a quarter of the
interactive budget: a 1.9 MB tier ledger, a 1.3 MB latest completed grade and 65 fast verdicts of
0.7 MB each (45 MB), 40 self-check responses of 0.9 MB read early by the event feed (36 MB, more
than the whole budget) and one response over the per-file bound.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from merlin_experiments.tracking.console import model
from merlin_experiments.tracking.console.io import Reader

BUDGET = model.REFRESH_BUDGET // 4
CASES = 150


def _write(path: Path, document) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def _padded_rows(count: int, size: int) -> list[dict]:
    pad = "x" * max(0, size // count - 80)
    return [{"capsule": f"case{i:03d}", "status": "pass", "failure_detail": pad} for i in range(count)]


def _live_run(tmp_path: Path) -> dict:
    run = tmp_path / "out" / "runs" / "tgt" / "phase1" / "long-run"
    ws = tmp_path / "out" / "build" / "ws" / "long-run" / "workspace"
    pad = "y" * (1_900_000 // CASES)
    ledger = {f"case{i:03d}": {"L2": {"status": "pass", "pad": pad}, "L3": {"status": "pass"}} for i in range(CASES)}
    _write(ws / "qa" / "tier_state.json", ledger)
    _write(
        ws / "qa" / "joint_physical_correctness.json",
        {
            "schema": "merlin.joint_physical_correctness.v1",
            "obligations": {f"case{i:03d}": {"status": "pass"} for i in range(CASES)},
        },
    )
    _write(
        ws / "qa" / "latest_completed_grade.json",
        {"per_capsule": _padded_rows(CASES, 1_300_000), "graded_source_sha256": "f" * 64},
    )
    _write(ws / "qa" / "verdict.json", {"per_capsule": _padded_rows(CASES, 200_000)})
    for i in range(65):
        _write(
            run / "qa_history" / f"verdict_fast_{1792950000000000000 + i * 10**9}.json",
            {"per_capsule": _padded_rows(CASES, 700_000), "n_capsules": CASES, "n_passed": CASES},
        )
    channel = ws / ".qa_channel"
    for i in range(40):  # newest self-check responses, read early by the fast event feed: 36 MB
        _write(
            channel / f"resp_4_{1792960000000000000 + i * 10**9}_{i:032x}.json",
            {"all_pass": False, "per_capsule": _padded_rows(CASES, 900_000)},
        )
    with open(channel / "resp_3_1792963767835670698_oversized.json", "wb") as stream:
        stream.truncate(9 << 20)  # sparse: only its size is ever looked at
    (run / "rounds").mkdir(parents=True)
    return {
        "id": str(run),
        "run_id": "long-run",
        "run_dir": str(run),
        "workspace": str(ws),
        "state": "LIVE",
        "task": str(tmp_path),
    }


def _summary(snapshot: dict, panel: str, label: str):
    return next((s["value"] for s in snapshot["sections"][panel].get("summary") or [] if s["label"] == label), None)


def test_bulk_scans_no_longer_starve_the_certification_and_grader_panels(tmp_path, monkeypatch):
    experiment = _live_run(tmp_path)
    # The previous behaviour: no prefetch and no reserve. The verdict scan spends the whole budget first.
    starved = model.Collector(experiment, budget=BUDGET)
    starved.reader.reserve = 0
    monkeypatch.setattr(starved, "prefetch", lambda: None)
    before = starved.refresh("all")
    assert before["sections"]["certification"]["totals"]["L2"] == 0  # the ledger was never read
    assert _summary(before, "grader", "verdicts done / delivered") == "no verdict yet"

    collector = model.Collector(experiment, budget=BUDGET)
    snapshot = collector.refresh("all")
    assert f"L3 {CASES}/{CASES} (100%)" in _summary(snapshot, "certification", "functional tier certification")
    # The headline repeats the CASES panel's own count from the same refresh.
    cases = _summary(snapshot, "cases", "cases passing")
    assert cases and _summary(snapshot, "certification", "cases passing numerics") == cases
    verdicts = _summary(snapshot, "grader", "verdicts done / delivered")
    assert verdicts and verdicts != "no verdict yet"
    assert snapshot["sections"]["agent_view"]["latest_completed"]["rows"] == CASES
    # Bounded: one interactive refresh never reads more than its budget.
    assert 0 <= collector.reader.remaining and collector.reader.budget == BUDGET
    assert any("byte budget" in n for n in snapshot["notices"])


def test_a_one_shot_dump_reads_every_record(tmp_path):
    experiment = _live_run(tmp_path)
    snapshot = model.Collector(experiment, budget=None).refresh("all")
    assert _summary(snapshot, "grader", "verdicts done / delivered").startswith("65 /")
    assert not any("byte budget" in n for n in snapshot["notices"])
    assert any("exceeds the 8 MiB per-file read bound" in n for n in snapshot["notices"])


def test_serve_dump_is_not_budget_limited(tmp_path, capsys):
    from merlin_experiments.tracking.console import serve

    experiment = _live_run(tmp_path)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    argv = ["--tasks-root", str(tasks), "--run-dir", experiment["run_dir"], "--dump", "--no-placement"]
    assert serve.main(argv) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert not any("byte budget" in n for n in snapshot["notices"])
    assert _summary(snapshot, "grader", "verdicts done / delivered").startswith("65 /")


def test_insight_reserve_holds_back_bytes_from_ordinary_reads(tmp_path):
    path = _write(tmp_path / "record.json", {"pad": "z" * 900})
    reader = Reader(budget=2000, reserve=1500)
    assert reader.document(path) is None  # 1 KB does not fit in the 500 bytes ordinary reads may use
    reader.priority = True
    assert reader.document(path) == {"pad": "z" * 900}
    spent = reader.budget - reader.remaining
    reader.priority = False
    reader.begin()
    assert reader.document(path) is not None and reader.remaining == reader.budget  # unchanged: a cache hit
    assert spent > 900


def test_oversized_records_are_skipped_unread_and_rewrites_show_the_last_read(tmp_path):
    big = tmp_path / "resp_big.json"
    with open(big, "wb") as stream:
        stream.truncate(9 << 20)
    reader = Reader(budget=64 << 20)
    assert reader.document(big) is None and reader.remaining == reader.budget
    assert any("per-file read bound" in n for n in reader.notices)

    path = _write(tmp_path / "tier_state.json", {"version": 1})
    reader = Reader(budget=4096)
    assert reader.document(path, stale_ok=True) == {"version": 1}
    _write(path, {"version": 2, "pad": "p" * 8000})  # rewritten larger than this refresh can read
    os.utime(path, ns=(1, 1))
    reader.begin()
    assert reader.document(path) is None  # an identity-keyed caller is never handed an old parse
    assert reader.document(path, stale_ok=True) == {"version": 1}
    assert any("as last read" in n for n in reader.notices)
