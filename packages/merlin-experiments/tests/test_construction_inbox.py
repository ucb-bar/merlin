"""A warm author client admits new work without restarting or blocking workers."""

import json
import sys
import threading
import time

from merlin_experiments.phase1.construction_actions import SCHEMA, enqueue_actions, execute


def action(name, code, **kwargs):
    return dict(id=name, kind="command", boundary="integration", capability="ordinary-test-work",
                argv=[sys.executable, "-c", code], **kwargs)


def test_live_intake_overlaps_existing_work_and_preserves_cross_plan_dependencies(tmp_path):
    inbox = tmp_path / "inbox"
    started, finished, observed, dependent = [tmp_path / n for n in ("start", "finish", "overlap", "dependent")]
    initial = action("initial", "from pathlib import Path; import time; "
        f"Path({str(started)!r}).touch(); time.sleep(0.4); Path({str(finished)!r}).touch()")

    def publish():
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        enqueue_actions(inbox, dict(schema=SCHEMA, actions=[
            action("parallel", f"from pathlib import Path; Path({str(observed)!r}).write_text(str(Path({str(finished)!r}).exists()))"),
            action("dependent", f"from pathlib import Path; Path({str(dependent)!r}).write_text(str(Path({str(finished)!r}).exists()))",
                   after=["initial"]),
        ]))
        (inbox / "STOP").touch()

    thread = threading.Thread(target=publish)
    thread.start()
    result = execute(dict(schema=SCHEMA, actions=[initial]), output=tmp_path / "results",
                     timeout_s=10, plan_sha256="a" * 64, command_workers=2, inbox=inbox)
    thread.join()
    assert observed.read_text() == "False"
    assert dependent.read_text() == "True"
    assert set(result["actions"].values()) == {"completed"}
    assert len(result["inbox"]["admitted_plans"]) == 1


def test_invalid_late_plan_is_reported_without_replacing_existing_work(tmp_path):
    inbox = tmp_path / "inbox"
    enqueue_actions(inbox, dict(schema=SCHEMA, actions=[action("initial", "raise RuntimeError('must not replace')")]))
    (inbox / "STOP").touch()
    result = execute(dict(schema=SCHEMA, actions=[action("initial", "pass")]),
                     output=tmp_path / "results", timeout_s=5, plan_sha256="b" * 64, inbox=inbox)
    assert result["actions"] == {"initial": "completed"}
    assert not result["inbox"]["admitted_plans"]
    event = json.loads((tmp_path / "results/inbox-events.jsonl").read_text())
    assert event["status"] == "refused"
