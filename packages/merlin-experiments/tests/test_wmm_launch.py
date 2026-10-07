"""A measured run is launched DETACHED -- its own session, stdin from /dev/null, output appended to a
file in the run directory, never a pipe (a loop once died when the pipe its launcher wrote to closed) --
and found again by its own launch record, never by a process-name match."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2.whole_model_measured import cli as MCLI
from merlin_experiments.phase2.whole_model_measured import launch as LAUNCH
from merlin_experiments.phase2.whole_model_measured import runs as RUNS


def _exec_done(child) -> None:
    """Wait until the child has exec'd: until then /proc shows an empty command line."""
    for _ in range(500):
        if Path(f"/proc/{child.pid}/cmdline").read_bytes():
            return
        time.sleep(0.01)
    raise AssertionError("the child never exec'd")


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"schema": RUNS.RUN_SCHEMA, "target": "toy", "method": "m"}))
    return run_dir


def test_a_launch_is_its_own_session_writing_to_a_file_in_the_run(tmp_path):
    run_dir = _run_dir(tmp_path)
    seen = {}

    def spawn(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return SimpleNamespace(pid=4242)

    document = LAUNCH.launch(run_dir, profile="p", round_driver="m:f", price_table=tmp_path / "prices", spawn=spawn)
    assert seen["start_new_session"] is True and seen["stdin"] == subprocess.DEVNULL
    assert seen["stderr"] == subprocess.STDOUT and Path(seen["stdout"].name) == run_dir / LAUNCH.LAUNCH_LOG
    assert seen["argv"][1:5] == ["-m", LAUNCH.MODULE, "start", str(run_dir.resolve())]
    assert "--price-table" in seen["argv"]
    assert json.loads((run_dir / LAUNCH.LAUNCH_RECORD).read_text())["pid"] == document["pid"] == 4242


def test_a_live_launcher_is_found_by_its_record_and_refuses_a_second_launch(tmp_path):
    run_dir = _run_dir(tmp_path)
    assert LAUNCH.launcher_alive(run_dir) is None
    # A live process whose command line names the run: a real child, so /proc says what it runs.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", str(run_dir.resolve())])
    _exec_done(child)
    try:
        LAUNCH.launch(run_dir, profile="p", round_driver="m:f", spawn=lambda argv, **kw: SimpleNamespace(pid=child.pid))
        assert LAUNCH.launcher_alive(run_dir) is True
        with pytest.raises(ValueError, match="live launcher"):
            LAUNCH.launch(run_dir, profile="p", round_driver="m:f", spawn=lambda argv, **kw: pytest.fail("spawned"))
    finally:
        child.kill()
        child.wait()
    assert LAUNCH.launcher_alive(run_dir) is False
    # A reused pid that names another run is not this run's launcher.
    (run_dir / LAUNCH.LAUNCH_RECORD).write_text(json.dumps({"pid": os.getpid()}))
    assert LAUNCH.launcher_alive(run_dir) is False


def test_the_command_line_launches_a_prepared_run_and_refuses_anything_else(tmp_path, monkeypatch, capsys):
    run_dir = _run_dir(tmp_path)
    monkeypatch.setattr(LAUNCH, "spawn_process", lambda argv, **kw: SimpleNamespace(pid=77))
    assert MCLI.main(["launch", str(run_dir), "--profile", "p"]) == 0
    assert json.loads(capsys.readouterr().out)["pid"] == 77
    with pytest.raises(SystemExit, match="neither a prepared"):
        MCLI.main(["launch", str(tmp_path), "--profile", "p"])
