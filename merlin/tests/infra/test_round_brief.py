"""The cross-round memory brief: it must summarize the redacted per-round verdicts (progress log), echo
the agent's own notes, flag a regression, and warn when the agent stops journaling — all without ever
reading a golden. Hermetic: synthetic verdicts + notes on tmp dirs, no oracle, no model venv.
"""
from __future__ import annotations

import importlib.util
import json
import sys

from merlin.common.paths import merlin_dir, repo_root


def _load():
    p = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "round_brief.py"
    spec = importlib.util.spec_from_file_location("round_brief", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verdict(n_passed, planes, mism):
    return {"n_passed": n_passed, "n_capsules": 11, "first_failure_planes": planes,
            "per_capsule": [{"capsule": f"C{i}", "mismatch_count": m} for i, m in enumerate(mism)]}


def _write_round(run_dir, rnd, verdict):
    d = run_dir / "qa_history"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"verdict_round_{rnd:02d}.json").write_text(json.dumps(verdict))
    (run_dir / "rounds").mkdir(parents=True, exist_ok=True)
    (run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl").write_text("{}\n")


def test_progress_log_and_regression_note(tmp_path):
    RB = _load()
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    _write_round(run_dir, 0, _verdict(0, {"atlas-functional": 9}, [1024, 256]))
    _write_round(run_dir, 1, _verdict(0, {"L2": 10}, []))          # regressed: halting -> not halting
    brief = RB.build(run_dir, ws, 1)
    assert "Progress log" in brief
    assert "| 0 |" in brief and "| 1 |" in brief                  # both rounds tabulated
    assert "atlas-functional:9" in brief and "L2:10" in brief     # planes rendered
    assert "REGRESSED" in brief or "moved" in brief               # transition flagged


def test_echoes_notes_and_flags_stale(tmp_path):
    RB = _load()
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    notes = ws / "submission" / "docs" / "iteration_notes.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("## Iter 1\n- kernel produces all zeros; need DMA load/store\n")
    _write_round(run_dir, 0, _verdict(0, {"atlas-functional": 10}, [1024]))
    # first write stamps the notes hash (not stale yet)
    RB.write(run_dir, ws, 0)
    b1 = (ws / "qa" / "round_brief.md").read_text()
    assert "need DMA load/store" in b1                            # the agent's own notes are echoed back
    assert "did NOT update" not in b1                             # first sighting is not 'stale'
    # next round graded, notes UNCHANGED -> must be flagged stale
    _write_round(run_dir, 1, _verdict(0, {"atlas-functional": 10}, [1024]))
    RB.write(run_dir, ws, 1)
    b2 = (ws / "qa" / "round_brief.md").read_text()
    assert "did NOT update" in b2                                 # journaling-stopped nudge fires


def test_empty_notes_prompts_creation(tmp_path):
    RB = _load()
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    _write_round(run_dir, 0, _verdict(0, {"schema": 1}, []))
    brief = RB.build(run_dir, ws, 0)
    assert "iteration_notes.md is EMPTY" in brief

    # never leaks a golden value: the brief only ever contains counts/planes, never expected outputs
    assert "golden" not in brief.lower() or "no goldens" in brief.lower()


def test_prelaunch_refresh_delivers_late_erratum_and_preserves_resume_note(tmp_path):
    """An operator erratum written after the last grade must still reach the resumed agent turn."""
    RB = _load()
    harness = repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"
    sys.path.insert(0, str(harness))
    try:
        import resume_on_quota as ROQ
    finally:
        sys.path.remove(str(harness))

    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    notes = ws / "submission" / "docs" / "iteration_notes.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("old advice: use logical packed stride\n", encoding="utf-8")
    _write_round(run_dir, 0, _verdict(1, {"spike": 1}, [0]))

    # This is the measured ordering: a grade wrote the brief, a cut-short turn added RESUME, and the
    # operator authored ERRATA.md only afterwards.  Before the fix, the next launch reads a stale brief.
    RB.write(run_dir, ws, 0)
    ROQ.prepend_resume_note(ws, ROQ.REASON_TIMEOUT)
    erratum = "## RETRACTION — derive physical stride again"
    (run_dir / "ERRATA.md").write_text(erratum + "\n", encoding="utf-8")

    RB.refresh_before_launch(run_dir, ws, 0)
    first = (ws / "qa" / "round_brief.md").read_text(encoding="utf-8")
    assert first.startswith("> ## RESUME")
    assert first.count("> ## RESUME") == 1
    assert first.count(erratum) == 1
    assert first.index(erratum) < first.index("Your iteration_notes.md")
    assert "old advice: use logical packed stride" in first

    # Relaunch retries can call the refresh repeatedly without accumulating either injected section.
    RB.refresh_before_launch(run_dir, ws, 0)
    assert (ws / "qa" / "round_brief.md").read_text(encoding="utf-8") == first


def test_fresh_prelaunch_refresh_builds_brief_with_erratum_before_notes(tmp_path):
    RB = _load()
    run_dir, ws = tmp_path / "run", tmp_path / "ws"
    run_dir.mkdir(parents=True)
    (run_dir / "ERRATA.md").write_text("## RETRACTION — re-derive\n", encoding="utf-8")
    notes = ws / "submission" / "docs" / "iteration_notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("prior finding\n", encoding="utf-8")

    RB.refresh_before_launch(run_dir, ws, 0)
    brief = (ws / "qa" / "round_brief.md").read_text(encoding="utf-8")
    assert brief.index("## RETRACTION") < brief.index("Your iteration_notes.md")


def test_every_qa_loop_agent_launch_refreshes_the_brief_first():
    src = (repo_root()
           / "merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py").read_text(
               encoding="utf-8")
    launches = []
    start = 0
    while True:
        pos = src.find("launch_agent(", start)
        if pos < 0:
            break
        if not src[max(0, pos - 5):pos].endswith("def "):
            launches.append(pos)
        start = pos + 1
    assert len(launches) == 3, "new launch paths must join the pre-launch brief refresh contract"
    for pos in launches:
        assert "refresh_before_launch(run_dir, ws," in src[max(0, pos - 250):pos]
