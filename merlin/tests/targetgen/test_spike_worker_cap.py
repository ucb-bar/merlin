"""Fresh worker interpreters honor explicit Spike fan-out without exceeding compute capacity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = """
import importlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace
root = Path(sys.argv[1])
sys.path[:0] = [str(root / 'src'), *map(str, sorted((root / 'packages').glob('*/src')))]
os.cpu_count = lambda: 32
from merlin.targetgen import rtl_engine_policy as policy
from merlin.targetgen import capsule_grade as grade
adapter = SimpleNamespace(_merlin_simulator_engine='spike')
initial = policy.capsule_worker_cap('spike')
bounded = grade._bounded_grade_workers(15, {'L2': adapter})
os.environ.pop('MERLIN_SPIKE_MAX_SLOTS', None)
importlib.reload(policy)
print(json.dumps({'initial': initial, 'bounded': bounded, 'unset_after_reload': policy.capsule_worker_cap('spike')}))
"""


def probe(tmp_path, value=None, compute_count=None):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    for key in ("MERLIN_SPIKE_MAX_SLOTS", "MERLIN_GSIM_MAX_SLOTS", "MERLIN_COMPUTE_CPUS", "MERLIN_COMPUTE_CPU_SLOTS"):
        env.pop(key, None)
    if value is not None:
        env["MERLIN_SPIKE_MAX_SLOTS"] = value
    if compute_count is not None:
        env["MERLIN_COMPUTE_CPUS"] = ",".join(map(str, range(16, 16 + compute_count)))
        env["MERLIN_COMPUTE_CPU_SLOTS"] = str(tmp_path / "compute-slots")
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, str(root)], env=env, capture_output=True, text=True, timeout=15,
    )


@pytest.mark.parametrize(
    "value,compute_count,expected",
    [(None, None, 8), ("", 15, 8), ("15", 15, 15), ("32", None, 32), ("32", 15, 15), ("15", 3, 3), ("1", 15, 1)],
)
def test_fresh_worker_cap_and_actual_grading_bound(tmp_path, value, compute_count, expected):
    result = probe(tmp_path, value, compute_count)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"initial": expected, "bounded": min(15, expected), "unset_after_reload": 8}


@pytest.mark.parametrize("value", ["0", "-1", "33", "1.5", "false"])
def test_invalid_operator_cap_refuses_in_fresh_worker(tmp_path, value):
    result = probe(tmp_path, value, 15)
    assert result.returncode != 0 and "MERLIN_SPIKE_MAX_SLOTS" in result.stderr
