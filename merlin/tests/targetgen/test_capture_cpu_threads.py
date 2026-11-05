"""The capture worker runs at its selected CPU thread count and records it.

Reference bytes depend on the intra-op thread count. The worker sets the selected count (one unless
the plan selects another), refuses to run at any other, and records the count it ran at in the
capture's determinism policy. It needs the capture interpreter's torch.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from merlin.common.paths import python_import_roots
from merlin.targetgen import _m2m_capture_worker as worker

SCRIPT = r"""
import importlib.util, os, sys, torch
spec = importlib.util.spec_from_file_location("capture_worker", sys.argv[1])
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
torch.set_num_threads(3)
policy = W._seed_capture(0, torch)
assert policy["cpu_threads"] == torch.get_num_threads() == 1
assert policy["cpu_dispatch"]["capability"] == torch.backends.cpu.get_cpu_capability()
assert policy["cpu_dispatch"]["environment_override"] == os.environ.get("ATEN_CPU_CAPABILITY")
if os.environ.get("ATEN_CPU_CAPABILITY") == "default":
    assert policy["cpu_dispatch"]["capability"] == "DEFAULT"
assert W._seed_capture(0, torch, 4)["cpu_threads"] == torch.get_num_threads() == 4
real = torch.set_num_threads
torch.set_num_threads = lambda _n: real(2)
try:
    W._seed_capture(0, torch, 4)
except RuntimeError as exc:
    assert "selected 4 CPU threads but runs 2" in str(exc)
else:
    raise AssertionError("a thread count other than the selected one must fail the capture")
print("OK")
"""


@pytest.mark.parametrize("cpu_override", [None, "default"])
def test_the_worker_runs_and_records_its_selected_thread_count(tmp_path, cpu_override):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("an explicit capture interpreter with torch is required")
    script = tmp_path / "check.py"
    script.write_text(SCRIPT)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in python_import_roots()))
    env.pop("ATEN_CPU_CAPABILITY", None)
    if cpu_override is not None:
        env["ATEN_CPU_CAPABILITY"] = cpu_override
    result = subprocess.run(
        [python, str(script), worker.__file__],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]
    assert result.stdout.strip().endswith("OK")


@pytest.mark.parametrize("bad", ["0", "257"])
def test_an_out_of_range_thread_count_is_refused_before_capture(tmp_path, bad, capsys):
    with pytest.raises(SystemExit):
        worker.main(
            [
                "--loader",
                str(tmp_path / "loader.py"),
                "--dtype",
                "fp32",
                "--out",
                str(tmp_path / "out"),
                "--cpu-threads",
                bad,
            ]
        )
    assert "--cpu-threads must be between 1 and 256" in capsys.readouterr().err


def test_missing_cpu_dispatch_api_is_unknown_not_a_fabricated_default(monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen._m2m_capture_artifacts import _framework_cpu_dispatch

    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "default")
    report = _framework_cpu_dispatch(SimpleNamespace())
    assert report["status"] == "unavailable" and report["capability"] is None
    assert report["environment_override"] == "default"
