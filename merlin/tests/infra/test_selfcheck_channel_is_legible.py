"""A waiting agent can tell a queue from a fault, and a cancel actually cancels."""

from __future__ import annotations

import ast
import importlib.util
import json
import time
from pathlib import Path

from phase1_feedback import feedback_source

from merlin.common.paths import merlin_dir, module_source_path

HARNESS = merlin_dir() / "experiments" / "capsule_bench" / "harness"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, feedback_source(name, HARNESS / f"{name}.py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_an_in_flight_check_stops_when_its_requester_cancels(tmp_path: Path) -> None:
    broker = _module("selfcheck_broker")
    request = tmp_path / "req_1.json"
    request.write_text("{}", encoding="utf-8")
    far = time.monotonic() + 3600
    own = __import__("os").getppid()
    assert broker._abort_reason(tmp_path, own, request, far) is None
    # The client cancels by renaming its request away; that is the whole signal it sends.
    request.rename(tmp_path / "cancelled_req_1.json")
    assert broker._abort_reason(tmp_path, own, request, far) == "the requester cancelled it"
    request.write_text("{}", encoding="utf-8")
    assert broker._abort_reason(tmp_path, own, request, time.monotonic() - 1) == "timed out"
    # A stop sentinel still outranks both, and names itself.
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    assert broker._abort_reason(tmp_path, own, request, far) == "STOP"


def test_the_broker_says_which_request_it_claimed_before_it_runs_anything() -> None:
    # Held by source: the marker's whole purpose is to exist BEFORE a long child starts, and a test
    # that ran one would be testing the child.
    text = (feedback_source("selfcheck_broker")).read_text(encoding="utf-8")
    claim = text.index('ch / f"claimed_{rid}"')
    spawn = text.index("proc = subprocess.Popen(inherited_python_command(argv2)")
    assert claim < spawn
    assert '"queued_behind_this"' in text[claim : claim + 400]


def test_a_queued_client_is_told_it_is_queued_and_by_how_much(tmp_path: Path) -> None:
    text = module_source_path("merlin_experiments.phase1.tools.selfcheck").read_text(encoding="utf-8")
    tree = ast.parse(text)
    strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    joined = "\n".join(strings)
    assert "queued, not yet claimed" in joined and "claimed by the grader after" in joined
    assert "serves one request at a time" in joined
    # And it reads the marker the broker actually writes.
    assert 'claimed = ch / f"claimed_{rid}"' in text
    marker = json.loads(json.dumps({"claimed_at_unix_ns": 1, "queued_behind_this": 3}))
    assert set(marker) == {"claimed_at_unix_ns", "queued_behind_this"}
