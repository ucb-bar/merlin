"""The self-check wait must be survivable by a caller whose shell kills commands at 600s.

MEASURED, and this is why the file exists. The agent's shell tool aborts any command at 600s. A
full-suite self-check runs 2.4-14.7 min, so the shim's blocking wait was routinely killed mid-flight:
the request stayed alive broker-side, the agent got back only "moved to the background", and with no
way to re-attach it hand-wrote ``until ... sleep`` poll loops -- which the same ceiling then truncated
too. One run spent 59.4 min across six such calls that returned no information whatsoever, against a
task brief that says "WAIT, DO NOT POLL" three times.

Prose lost to a mechanism, so the fix is a mechanism and these are its tests: the wait RETURNS inside
the ceiling carrying everything needed to resume, and resuming does not re-submit the work.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _module(name: str, at: Path | None = None):
    """Load the shim, optionally from a staged copy.

    The shim resolves its workspace as ``Path(__file__).parent`` because that is exactly how it is
    deployed -- staged into the sandbox as ``<ws>/agent_selfcheck.py``. So a test that wants a
    throwaway workspace has to stage it the same way; importing it in place would make the repo's own
    harness directory the workspace and write channel files into the source tree.
    """
    src = HARNESS / f"{name}.py"
    path = src if at is None else (at / "agent_selfcheck.py")
    if at is not None:
        path.write_text(src.read_text())
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _workspace(tmp_path: Path):
    """A staged workspace: the shim, a submission, and the channel dir it will use."""
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "manifest.yaml").write_text("name: t\n")
    shim = _module("selfcheck_shim", at=tmp_path)
    ch = tmp_path / ".qa_channel"
    ch.mkdir(exist_ok=True)
    return shim, ch


def test_wait_budget_default_is_under_the_callers_kill_ceiling():
    """At or above 600s the process is killed instead of returning, and we are back to square one."""
    shim = _module("selfcheck_shim")
    budget = shim._DEFAULT_WAIT_BUDGET_S
    assert 0 < budget < 600, (
        f"default wait budget is {budget}s; the calling agent's shell kills at 600s, so a default at or "
        "above it can never be observed returning"
    )


def test_in_progress_return_carries_what_resuming_needs(tmp_path):
    """A budget-expired wait returns rc=3 and names the request id and the resume command.

    rc=3 is deliberately neither 0 (a pass) nor 1 (a failing verdict) nor 2 (an error): an unfinished
    check is none of those, and collapsing it into any of them is how "still running" gets read as a
    result.
    """
    shim, ch = _workspace(tmp_path)
    rid = "rid1"
    (ch / f"progress_{rid}.json").write_text(
        json.dumps({"n_finished": 37, "n_expected": 104, "n_remaining": 67, "counts": {"pass": 30}})
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = shim.main(["--attach", rid, "--wait-budget", "1", "--capsules", "all"])

    assert rc == 3, f"expected rc=3 for an unfinished check, got {rc}"
    out = json.loads(buf.getvalue())
    assert out["status"] == "in_progress"
    assert out["request_id"] == rid
    assert (out["n_finished"], out["n_expected"]) == (37, 104)
    assert rid in out["resume_with"], "the resume command must name the request id"
    assert "poll" in out["note"].lower(), "the note must say not to poll; that is the whole point"


def test_attach_does_not_submit_a_second_request(tmp_path):
    """Resuming must not queue duplicate work behind the request it is waiting on."""
    shim, ch = _workspace(tmp_path)

    with contextlib.redirect_stdout(io.StringIO()):
        shim.main(["--attach", "rid2", "--wait-budget", "1", "--capsules", "all"])

    assert not list(ch.glob("req_*.json")), (
        "--attach wrote a request: a resumed wait must not re-submit, or every resume doubles the work "
        "queued behind the check it is waiting for"
    )


def test_a_fresh_request_is_still_submitted(tmp_path):
    """The bound must not have cost us the ordinary path: without --attach, a request IS written."""
    shim, ch = _workspace(tmp_path)

    with contextlib.redirect_stdout(io.StringIO()):
        shim.main(["--wait-budget", "1", "--capsules", "all"])

    assert list(ch.glob("req_*.json")), "a non-attached invocation must still submit its request"
