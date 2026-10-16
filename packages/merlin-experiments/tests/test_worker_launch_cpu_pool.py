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


def test_preparation_affinity_and_compute_lease_are_disjoint(tmp_path):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 2:
        pytest.skip("two allowed CPUs are needed to test distinct preparation and compute pools")
    prep, compute = available[:2]
    environment = dict(os.environ)
    environment.update({
        'MERLIN_PREPARATION_CPUS': str(prep),
        'MERLIN_COMPUTE_CPUS': str(compute),
        'MERLIN_COMPUTE_CPU_SLOTS': str(tmp_path / 'slots'),
    })
    script = (
        'import json,os; from merlin.common.cpu_pool import shared_compute_pool; '
        'before=sorted(os.sched_getaffinity(0)); '
        'scope=shared_compute_pool(timeout_s=1); scope.__enter__(); '
        'inside=sorted(os.sched_getaffinity(0)); scope.__exit__(None,None,None); '
        'print(json.dumps([before,inside,sorted(os.sched_getaffinity(0))]))'
    )
    command = worker_pool_command([sys.executable, '-c', script], environment)
    actual = json.loads(subprocess.check_output(command, env=environment, text=True))
    assert actual == [[prep], [compute], [prep]]
    environment['MERLIN_PREPARATION_CPUS'] = f'{prep},{compute}'
    with pytest.raises(ValueError, match='overlap'):
        worker_pool_command(command, environment)
