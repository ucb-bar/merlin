"""Transient model-capacity refusals resume the same continuous session."""
import json
import subprocess
import sys

from merlin_experiments.phase1.providers import codex_agent as driver
from merlin_experiments.phase1.providers import codex_capacity as capacity

REFUSAL = "Selected model is at capacity. Please try a different model."


def test_refusal_matches_wrapped_compaction_error_only_when_every_error_is_capacity():
    assert capacity.refusal([REFUSAL])
    assert capacity.refusal(["Error running remote compact task: " + REFUSAL])
    assert not capacity.refusal([])
    assert not capacity.refusal([REFUSAL, "sandbox denied"])


def test_backoff_grows_and_is_bounded():
    waits = [capacity.backoff_s(n) for n in range(1, 10)]
    assert waits == sorted(waits) and waits[0] == capacity._CAPACITY_BACKOFF_S
    assert max(waits) == capacity._CAPACITY_BACKOFF_MAX_S


def test_compaction_refusal_resumes_session_and_reports_stopped_helpers(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("offline fixture")
    fake = tmp_path / "offline-author"
    # Invocation 1 and 2 are refused (the second wrapped by the compaction task),
    # invocation 3 completes. More refusals than one session-wide count allowed.
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import json,pathlib,sys\n"
        "sys.stdin.read()\n"
        "c=pathlib.Path('calls'); n=int(c.read_text()) if c.exists() else 0; c.write_text(str(n+1))\n"
        f"refusal={REFUSAL!r}\n"
        "msg=[refusal,'Error running remote compact task: '+refusal,None][min(n,2)]\n"
        "for e in [{'type':'thread.started','thread_id':'offline'},{'type':'turn.started'}]:\n"
        " print(json.dumps(e),flush=True)\n"
        "if msg:\n"
        " print(json.dumps({'type':'error','message':msg}),flush=True)\n"
        " print(json.dumps({'type':'turn.failed','error':{'message':msg}}),flush=True)\n"
        " sys.exit(1)\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}),flush=True)\n"
    )
    fake.chmod(0o700)
    monkeypatch.setattr(driver, "cli_version", lambda *a: "offline")
    monkeypatch.setattr(driver, "_CONTINUE_MIN_S", 0)
    monkeypatch.setattr(driver, "_CONTINUE_MAX_TURNS", 3)
    waits = []
    monkeypatch.setattr(capacity, "backoff_s", lambda attempt: waits.append(attempt) or 0)
    real_popen = subprocess.Popen

    def launch(argv, **kwargs):
        if "--pid" in argv:
            raise OSError("fixture sampler disabled")
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    rc, transcript = driver.run_round(
        workspace, tmp_path / "run", "offline", {}, None, "none", 0, 600,
        effective_model="offline", continue_session=True, codex_binary=fake,
    )
    assert (workspace / "calls").read_text() == "3"
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    retries = [row for row in rows if row.get("type") == "codex_capacity_retry"]
    assert [row["consecutive"] for row in retries] == [1, 2]
    assert waits[:2] == [1, 2]
    summary = json.loads((transcript.parent / "round_00.codex_summary.json").read_text())
    assert summary["capacity_retries"] == 2 and summary["unrecovered_errors"] == []
    assert rc == 0
    resumed = transcript.parent / "round_00.prompt.cont01.txt"
    assert capacity.RESUME_NOTE.strip() in resumed.read_text()


def _scripted_author(tmp_path, plan):
    """A fake CLI: invocation n emits the completed items ``plan[n][0]`` then is refused unless ``plan[n][1]``."""
    fake = tmp_path / "scripted-author"
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import json,pathlib,sys\n"
        "sys.stdin.read()\n"
        "c=pathlib.Path('calls'); n=int(c.read_text()) if c.exists() else 0; c.write_text(str(n+1))\n"
        f"plan={plan!r}\n"
        "items,completes=plan[min(n,len(plan)-1)]\n"
        "for e in [{'type':'thread.started','thread_id':'offline'},{'type':'turn.started'}]:\n"
        " print(json.dumps(e),flush=True)\n"
        "for i,kind in enumerate(items):\n"
        " item={'id':f'{n}_{i}','type':kind,'text':'x','command':'true','status':'completed'}\n"
        " print(json.dumps({'type':'item.completed','item':item}),flush=True)\n"
        "if not completes:\n"
        f" print(json.dumps({{'type':'error','message':{REFUSAL!r}}}),flush=True)\n"
        f" print(json.dumps({{'type':'turn.failed','error':{{'message':{REFUSAL!r}}}}}),flush=True)\n"
        " sys.exit(1)\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}),flush=True)\n"
    )
    fake.chmod(0o700)
    return fake


def _run_scripted(tmp_path, monkeypatch, plan):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("offline fixture")
    fake = _scripted_author(tmp_path, plan)
    monkeypatch.setattr(driver, "cli_version", lambda *a: "offline")
    monkeypatch.setattr(driver, "_CONTINUE_MIN_S", 0)
    monkeypatch.setattr(driver, "_CONTINUE_MAX_TURNS", len(plan))
    waits = []
    monkeypatch.setattr(capacity, "backoff_s", lambda attempt: waits.append(attempt) or 0)
    real_popen = subprocess.Popen

    def launch(argv, **kwargs):
        if "--pid" in argv:
            raise OSError("fixture sampler disabled")
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    rc, transcript = driver.run_round(
        workspace, tmp_path / "run", "offline", {}, None, "none", 0, 600,
        effective_model="offline", continue_session=True, codex_binary=fake,
    )
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    retries = [row["consecutive"] for row in rows if row.get("type") == "codex_capacity_retry"]
    return rc, retries, waits, int((workspace / "calls").read_text())


def test_progress_resets_the_refusal_count_and_backoff(tmp_path, monkeypatch):
    # More refusals than the consecutive bound, each after a turn that did real work: one long session
    # with refusals spread over it is not an outage, so none of them may accumulate or end the round.
    worked = [(["command_execution", "file_change"], False), (["agent_message"], False), (["mcp_tool_call"], False)]
    plan = [*worked * 3, ([], True)]
    assert len(plan) - 1 > capacity.MAX_CONSECUTIVE
    rc, retries, waits, calls = _run_scripted(tmp_path, monkeypatch, plan)
    assert retries == [1] * (len(plan) - 1)
    assert waits[: len(retries)] == [1] * len(retries)  # the backoff restarts from its first step
    assert calls == len(plan) and rc == 0


def test_back_to_back_refusals_still_accumulate_and_reasoning_is_not_progress(tmp_path, monkeypatch):
    plan = [([], False), (["reasoning"], False), (["command_execution"], False), ([], False), ([], True)]
    rc, retries, waits, calls = _run_scripted(tmp_path, monkeypatch, plan)
    assert retries == [1, 2, 1, 2]  # the third turn did work before its refusal: the count restarts there
    assert waits[: len(retries)] == [1, 2, 1, 2]
    assert calls == len(plan) and rc == 0


def test_consecutive_refusals_without_progress_still_end_the_round(tmp_path, monkeypatch):
    plan = [([], False)] * (capacity.MAX_CONSECUTIVE + 2)
    rc, retries, _, calls = _run_scripted(tmp_path, monkeypatch, plan)
    assert retries == list(range(1, capacity.MAX_CONSECUTIVE + 1))
    assert calls == capacity.MAX_CONSECUTIVE + 1 and rc != 0
