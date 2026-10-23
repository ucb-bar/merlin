"""Actual wrapper parsing retains selected environment after its restricted hop."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reserved_command_parses_and_restores_selected_runtime(tmp_path, monkeypatch):
    builder, wrapper = load("prepare_author_launch"), load("resource_reservation")
    selected = {
        "MERLIN_COMPUTE_CPUS": "9,10",
        "MERLIN_SPIKE": "/selected/spike",
        "LD_LIBRARY_PATH": "/selected/lib",
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
    }
    # The sole process this test executes prints environment data, never a campaign.
    probe = [
        sys.executable,
        "-c",
        "import os,json;print(json.dumps({k:os.environ[k] for k in " + repr(list(selected)) + "}))",
    ]
    reserved = builder.reservation_argv(
        probe,
        selected,
        task=tmp_path,
        output=tmp_path,
        run_id="run-with-hyphens-20261023-v2",
        cpus="9,10",
        duration_s=60,
    )
    parsed = []
    monkeypatch.setattr(wrapper, "launch", lambda args: parsed.append(args) or 0)
    monkeypatch.setattr(sys, "argv", reserved[1:])
    assert wrapper.main() == 0
    args = parsed[0]
    assert re.fullmatch(r"merlinphase1[a-z0-9]+", args.unit)
    assert args.command[:2] == ["/usr/bin/env", "-i"]
    assert args.command[-len(probe) :] == probe
    # All selected values survive even when absent from the wrapper's own
    # restricted environment. Actual env(1), not a mirrored Python projection.
    result = subprocess.run(args.command, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == selected
    assert not args.state.exists()  # no resource reservation was performed
