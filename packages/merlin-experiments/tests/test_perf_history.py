"""The per-test-case performance ledger a completed grade appends to, and the console reading it."""

from __future__ import annotations

import json
import time
from pathlib import Path

from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import perf_history
from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import Context
from merlin_experiments.tracking.console.sources import case_history, cases


def _roofline(floor, useful=4096, peak=256):
    return {"structural_compute_floor_cycles": floor, "useful_macs": useful, "structural_peak_macs_per_cycle": peak}


MODELED = {
    "status": "PROVISIONAL",
    "complete_event_accounting": True,
    "estimated_whole_soc_cycles": 900,
    "structural_roofline": _roofline(300),
    "scoped_resource_feedback": {"resources": [{"name": "cpu.issue", "capacity_normalized_activity": 0.9}]},
}


def _ws(tmp_path: Path, cert: str = "e1", cycles: int = 1600) -> Path:
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True, exist_ok=True)
    (ws / ".qa_channel").mkdir(exist_ok=True)
    _certify(ws, cert, cycles)
    # The author's Spike self-check of the certified executable: a same-executable pair.
    (ws / ".qa_channel" / "resp_1_a.json").write_text(
        json.dumps(
            {
                "sim": "spike",
                "per_capsule": [
                    {
                        "capsule": "dense",
                        "execution_digest": "e1",
                        "pass": True,
                        "performance": {
                            **MODELED,
                            "estimated_whole_soc_cycles": 1500,
                            "structural_roofline": _roofline(16),
                        },
                    }
                ],
            }
        )
    )
    return ws


def _certify(ws: Path, digest: str, cycles: int) -> None:
    (ws / "qa" / "tier_state.json").write_text(
        json.dumps(
            {
                "dense": {
                    "L2": {"status": "pass", "execution_digest": digest},
                    "L3": {"status": "pass", "execution_digest": digest},
                },
                "modeled_only": {"L2": {"status": "pass", "execution_digest": "m1"}},
            }
        )
    )
    (ws / ".qa_channel" / f"simresp_promo0_{digest}.json").write_text(
        json.dumps(
            {
                "sim": "gsim",
                "per_capsule": [
                    {
                        "capsule": "dense",
                        "execution_digest": digest,
                        "barrier_cycles": cycles,
                        "barrier_tier": "L3",
                        "pass": True,
                        "performance": {"structural_roofline": _roofline(16)},
                    }
                ],
            }
        )
    )


def _grade(source: str, modeled: dict | None = MODELED) -> dict:
    rows = [
        {"capsule": "dense", "status": "pass", "tiers": {"L2": "pass"}},
        {
            "capsule": "modeled_only",
            "status": "pass",
            "tiers": {"L2": "pass"},
            **({"performance": modeled} if modeled else {}),
        },
        {"capsule": "failing", "status": "fail", "tiers": {"L2": "fail"}, "performance": MODELED},
    ]
    return {
        "per_capsule": rows,
        "graded_source_sha256": source,
        "graded_at": f"at-{source}",
        "snapshot_current": False,
        "feedback_brief": {"performance_priority": {"advisory_only": True}},
    }


def _ledger(ws: Path) -> list[dict]:
    path = ws / "qa" / perf_history.FILENAME
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_a_completed_grade_records_each_passing_case_once_then_only_what_changed(tmp_path):
    ws = _ws(tmp_path)
    assert CG.record(ws, _grade("s1"), 100)
    rows = _ledger(ws)
    kinds = sorted((r["test_case"], r["kind"]) for r in rows)
    assert kinds == [
        ("dense", "measured"),
        ("dense", "pair"),
        ("dense", "status"),
        ("modeled_only", "modeled"),
        ("modeled_only", "status"),
    ]  # the failing case is not a passing case
    measured = next(r for r in rows if r["kind"] == "measured")
    assert measured["schema"] == perf_history.SCHEMA and measured["cycles"] == 1600
    assert measured["execution_digest"] == "e1" and measured["roofline_bound_cycles"] == 16
    assert measured["array_utilization_pct"] == round(100 * 4096 / (1600 * 256), 2)
    assert measured["grade_token"] == "100" and measured["graded_source_sha256"] == "s1"
    modeled = next(r for r in rows if r["kind"] == "modeled")
    assert modeled["cycles"] == 900 and modeled["evidence"] == "modeled: Spike timing model (provisional)"
    assert modeled["dominant_modeled_resource"]["name"] == "cpu.issue"
    pair = next(r for r in rows if r["kind"] == "pair")
    assert (pair["execution_digest"], pair["modeled"], pair["measured"]) == ("e1", 1500, 1600)
    # A newer grade seeing the same numbers adds nothing.
    assert CG.record(ws, _grade("s2"), 200)
    assert len(_ledger(ws)) == len(rows)
    # A faster certificate and a new estimate: exactly those two rows.
    _certify(ws, "e2", 1200)
    assert CG.record(ws, _grade("s3", modeled={**MODELED, "estimated_whole_soc_cycles": 700}), 300)
    new = _ledger(ws)[len(rows) :]
    assert sorted((r["test_case"], r["kind"], r["cycles"]) for r in new) == [
        ("dense", "measured", 1200),
        ("modeled_only", "modeled", 700),
    ]


def test_the_ledger_never_affects_the_grade(tmp_path, monkeypatch):
    ws = _ws(tmp_path)

    def broken(self, *args):
        raise RuntimeError("boom")

    monkeypatch.setattr(perf_history.Recorder, "_observe", broken)
    assert CG.record(ws, _grade("s1"), 100)
    latest = json.loads((ws / "qa" / CG.FILENAME).read_text())
    assert latest["feedback_brief"]["performance_priority"]["development_headroom"]["counts"]["ranked"] == 2
    assert _ledger(ws) == []


def test_the_ledger_stops_growing_past_its_bound(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    monkeypatch.setattr(perf_history, "MAX_BYTES", 10)
    assert CG.record(ws, _grade("s1"), 100)
    before = len(_ledger(ws))
    _certify(ws, "e2", 1200)
    assert CG.record(ws, _grade("s2"), 200)
    assert len(_ledger(ws)) == before


def test_the_console_prefers_the_ledger_and_tails_it(tmp_path):
    ws = _ws(tmp_path)
    run = tmp_path / "out" / "runs" / "tgt" / "phase1" / "r"
    (run / "qa_history").mkdir(parents=True)
    CG.record(ws, _grade("s1"), 100)
    experiment = {
        "id": str(run),
        "run_id": "r",
        "run_dir": str(run),
        "workspace": str(ws),
        "state": "LIVE",
        "task": None,
        "runner_pid": None,
    }
    state, reader = {}, Reader(budget=None)

    def ctx():
        return Context(
            experiment=experiment,
            reader=reader,
            state=state,
            operator_private=True,
            now=time.time(),
            shared={},
            live=True,
        )

    section = cases.read(ctx())
    assert section["history"]["basis"] == "ledger" and "qa/perf_history.jsonl" in section["summary"][-1]["value"]
    columns = section["table"]["columns"]
    rows = {r["key"]: dict(zip(columns, r["cells"], strict=True)) for r in section["table"]["rows"]}
    assert rows["dense"]["gSIM cyc"] == "1,600" and rows["dense"]["mod÷meas"] == "×0.94"
    assert rows["modeled_only"]["model cyc"] == "900" and rows["modeled_only"]["busiest"] == "cpu.issue"
    history = state["history"]
    assert isinstance(history, case_history.History) and history.digests == {}  # nothing derived
    _certify(ws, "e2", 800)
    CG.record(ws, _grade("s2"), 200)
    rows = {r["key"]: dict(zip(columns, r["cells"], strict=True)) for r in cases.read(ctx())["table"]["rows"]}
    assert rows["dense"]["gSIM cyc"] == "800" and rows["dense"]["gSIM best/now"] == "×2.00/×2.00"
    drawer = cases.inspect(ctx(), "dense")
    grades = next(s["lines"] for s in drawer["sections"] if s["title"].startswith("Grades"))
    assert any("L2 pass" in line for line in grades)
