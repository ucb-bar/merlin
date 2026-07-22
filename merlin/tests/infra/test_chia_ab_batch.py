"""C1: the multi-arm A/B/C Chia fan-out plans one task per (arm × repeat) — the cross-arm matrix on the
Chia spine (verilator-slot gated) replacing the plain subprocess backgrounding. The plan is verified via
the driver's --dry-run (board-free); the chia execution itself needs the chia venv + sim."""
from __future__ import annotations

import subprocess
import sys

from merlin.common.paths import repo_root


def test_fanout_plan_is_arm_times_repeat():
    scripts = repo_root() / "merlin/experiments/gemmini_capsule_bench_v0/scripts"
    p = subprocess.run(
        [sys.executable, "chia_ab_batch.py", "--tag", "t", "--arms",
         "baseline,merlin,merlin_rtlchecks", "--repeats", "2", "--verilator-slots", "2", "--dry-run"],
        cwd=str(scripts), capture_output=True, text=True, timeout=90)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "6 tasks (3 arms x 2 repeats)" in out and "verilator_slots=2" in out
    # each arm appears with both repeats (distinct run-ids)
    for token in ("rb_t_r0", "rb_t_r1", "merlin_t_r0", "merlincirct_t_r1"):
        assert token in out, f"missing {token} in plan:\n{out}"
