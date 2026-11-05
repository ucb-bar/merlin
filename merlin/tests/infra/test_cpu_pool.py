"""Actual subprocess affinity and exclusive CPU slots across launch threads."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.common.cpu_pool import parse_roster, selected_pool


def test_children_use_exclusive_cpus_and_launch_threads_restore_affinity(tmp_path):
    original = os.sched_getaffinity(0)
    cpus = tuple(sorted(original)[:2])
    if len(cpus) < 2:
        pytest.skip("requires two available CPUs")
    script = (
        "import json,os,time; start=time.monotonic(); "
        "cpus=sorted(os.sched_getaffinity(0)); time.sleep(.15); "
        "print(json.dumps([cpus,start,time.monotonic()]))"
    )

    def launch(_):
        before = os.sched_getaffinity(0)
        with selected_pool(cpus, slot_root=tmp_path / "slots", timeout_s=5):
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        assert os.sched_getaffinity(0) == before
        return json.loads(result.stdout)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(launch, range(4)))
    assert {row[0][0] for row in results} == set(cpus)
    assert all(len(row[0]) == 1 for row in results)
    for cpu in cpus:
        intervals = sorted((row[1], row[2]) for row in results if row[0] == [cpu])
        assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert os.sched_getaffinity(0) == original


def test_roster_refuses_duplicate_and_non_numeric_cpus():
    assert parse_roster("") == ()
    assert parse_roster("1,3") == (1, 3)
    for raw in ("1,1", "1-3", "1,", "-1"):
        with pytest.raises(ValueError):
            parse_roster(raw)
