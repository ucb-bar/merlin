"""Real native children prove model checks use and release the selected compute CPU."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from merlin.common.cpu_pool import shared_compute_pool
from merlin.model_check import full_build as FB
from merlin.model_check import replay as RP
from merlin.model_check.compute_execution import compute_step


@pytest.mark.parametrize("endpoint", ["full_tool", "replay_tool", "full_model", "host_twin", "kernel_replay"])
@pytest.mark.parametrize("selected", [True, False])
def test_native_step_uses_one_leased_cpu_and_restores_affinity(tmp_path, monkeypatch, endpoint, selected):
    original = os.sched_getaffinity(0)
    cpu = max(original)
    root = tmp_path / "slots"
    if selected:
        monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
        monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(root))
    else:
        monkeypatch.delenv("MERLIN_COMPUTE_CPUS", raising=False)
        monkeypatch.delenv("MERLIN_COMPUTE_CPU_SLOTS", raising=False)
    out = tmp_path / "observed.json"
    script = (
        "import fcntl,json,os; from pathlib import Path; "
        f"p=Path({str(root / ('cpu-' + str(cpu) + '.lock'))!r}); "
        "held=False\n"
        "if p.exists():\n"
        " with p.open('r+') as probe:\n"
        "  try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "  except BlockingIOError: held=True\n"
        f"Path({str(out)!r}).write_text(json.dumps([sorted(os.sched_getaffinity(0)),held])); "
        "print('MCDONE')"
    )
    argv = [sys.executable, "-c", script]
    if endpoint == "full_tool":
        FB._run(argv, str(tmp_path))
    elif endpoint == "replay_tool":
        RP._run_tool(argv, cwd=tmp_path, timeout=5)
    elif endpoint == "full_model":
        from merlin.runtime.backends import base

        def run(_elf, *, timeout, capture_bytes):
            assert 0 < timeout <= 5 and capture_bytes
            return subprocess.check_output(argv, timeout=timeout)

        monkeypatch.setattr(base, "get_backend", lambda _: SimpleNamespace(run_declared_model_elf=run))
        assert FB.run("fixture", tmp_path / "image.elf", timeout_s=5)[0] == b"MCDONE\n"
    elif endpoint == "host_twin":
        from merlin.runtime.backends import spike_model

        monkeypatch.setattr(spike_model, "declared_isa", lambda _: "rv64gc")
        monkeypatch.setattr(
            spike_model, "run_raw", lambda *a, **kw: subprocess.check_output(argv, timeout=kw["timeout"]),
        )
        assert FB.run_host(tmp_path / "image.elf", timeout_s=5)[0] == b"MCDONE\n"
    else:
        monkeypatch.setattr(
            RP, "spike_argv", lambda *a, **kw: (argv, {"cwd": str(tmp_path), "environment": dict(os.environ)}),
        )
        assert RP.run_image("fixture", tmp_path / "image.elf", dram_base=0, dram_bytes=4096, timeout=5)[0] == "MCDONE\n"
    observed, held = json.loads(out.read_text())
    assert observed == ([cpu] if selected else sorted(original))
    assert held is selected
    assert os.sched_getaffinity(0) == original
    if selected:
        with shared_compute_pool(timeout_s=0):
            pass  # The completed native step left no slot held.


def test_waiting_model_step_times_out_without_starting_work(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(min(os.sched_getaffinity(0))))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    started = []

    def waiting():
        with compute_step(0.05):
            started.append(True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with shared_compute_pool():
            future = pool.submit(waiting)
            with pytest.raises(TimeoutError):
                future.result(timeout=2)
    assert not started


def test_step_timeout_is_reduced_by_slot_wait(tmp_path, monkeypatch):
    from merlin.model_check import compute_execution as C

    monkeypatch.delenv("MERLIN_COMPUTE_CPUS", raising=False)
    monkeypatch.delenv("MERLIN_COMPUTE_CPU_SLOTS", raising=False)
    times = iter([10.0, 12.5])
    monkeypatch.setattr(C.time, "monotonic", lambda: next(times))
    with compute_step(5) as remaining:
        assert remaining == 2.5
