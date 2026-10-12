import json
import os
import subprocess
import sys

import pytest

from merlin.common.cpu_pool import worker_pool_command


def test_affinity_applies_before_worker_python_starts():
    cpu = sorted(os.sched_getaffinity(0))[-1]
    command = [sys.executable, '-c', 'import json,os;print(json.dumps(sorted(os.sched_getaffinity(0))))']
    actual = subprocess.check_output(worker_pool_command(command, {'MERLIN_GRADE_CPUS': str(cpu)}), text=True)
    assert json.loads(actual) == [cpu]


def test_unselected_legacy_launch_and_invalid_roster():
    command = ['python', '-m', 'worker']
    assert worker_pool_command(command, {}) is command
    with pytest.raises(ValueError):
        worker_pool_command(command, {'MERLIN_GRADE_CPUS': '9-11'})
