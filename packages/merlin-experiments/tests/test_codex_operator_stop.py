"""Local synthetic author processes: bounded host stop without touching the evaluator."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from merlin_experiments.phase1.providers import codex_agent as driver


def test_operator_stop_interrupts_author_group_and_retains_partial_work(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("offline fixture")
    fake = tmp_path / "offline-author"
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import json,os,pathlib,subprocess,sys,time\n"
        "sys.stdin.read()\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        "pathlib.Path('child.pid').write_text(str(p.pid))\n"
        "pathlib.Path('partial.py').write_text('unfinished = True\\n')\n"
        "print('retained stderr',file=sys.stderr,flush=True)\n"
        "for e in [{'type':'thread.started','thread_id':'offline'},"
        "{'type':'turn.started'},{'type':'item.started','item':"
        "{'id':'tool','type':'command_execution','command':'offline long command'}}]:\n"
        " print(json.dumps(e),flush=True)\n"
        "print('partial raw bytes',end='',flush=True)\n"
        "time.sleep(60)\n"
    )
    fake.chmod(0o700)
    monkeypatch.setattr(driver, "cli_version", lambda *a: "offline")
    monkeypatch.setattr(driver, "_CONTROL_POLL_S", 0.05)
    real_popen = subprocess.Popen

    def launch(argv, **kwargs):
        if "--pid" in argv:
            raise OSError("fixture sampler disabled")
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    unrelated = real_popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        if (workspace / "child.pid").exists() and calls >= 3:
            return "reviewed evaluator upgrade"
        return None

    start = time.monotonic()
    try:
        rc, transcript = driver.run_round(
            workspace, tmp_path / "run", "offline", {}, None, "none", 0, 30,
            effective_model="offline", continue_session=True, continuation_stop=stop,
            codex_binary=fake,
        )
        assert time.monotonic() - start < 10
        assert rc == driver.OPERATOR_INTERRUPTED_EXIT_CODE
        assert unrelated.poll() is None
        assert (workspace / "partial.py").read_text() == "unfinished = True\n"
        rows = [json.loads(line) for line in transcript.read_text().splitlines()]
        interrupt = next(row for row in rows if row.get("type") == "codex_operator_interrupt")
        assert driver._process_group_quiescent(interrupt["author_process_group"])
        assert interrupt["turn_incomplete"] is True
        assert rows[-1]["subtype"] == "interrupted" and rows[-1]["is_error"] is True
        assert not any(row.get("type") == "result" and row.get("subtype") == "success" for row in rows)
        turns = [row for row in rows if row.get("type") == "codex_session_turn"]
        assert len(turns) == 1 and turns[0]["continuing"] is False
        summary = json.loads((transcript.parent / "round_00.codex_summary.json").read_text())
        assert summary["operator_interrupted"] and summary["turn_incomplete"]
        assert not summary["timed_out"] and not summary["usage_complete"]
        assert summary["process_exit_code"] != 0
        assert b"partial raw bytes" in (transcript.parent / "round_00.codex_events.raw.jsonl").read_bytes()
        assert "retained stderr" in (transcript.parent / "round_00.codex_stderr.log").read_text()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_never_signal_evaluator_group():
    import pytest
    with pytest.raises(ValueError, match="unsafe process-group"):
        driver._terminate_process_group(os.getpgrp())
