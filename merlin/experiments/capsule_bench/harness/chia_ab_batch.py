"""Multi-arm A/B/C fan-out as a CHIA workflow — one ``@ChiaFunction`` per (arm × repeat), gated on the
logical ``verilator`` resource, replacing the plain ``subprocess`` backgrounding in ``launch_ab_batch``.

Same arms + per-arm commands as ``launch_ab_batch`` (reuses ``ARMS`` + ``_arm_cmd``), same verilator-slot
gating as ``chia_repeatability`` (``--verilator-slots``; 1 == sequential). So the whole cross-arm matrix
runs on the one Chia/Ray spine (unified profiler + resource gating) instead of N un-gated subprocesses.

Run under the isolated chia venv::

  build/chia-venv/bin/python chia_ab_batch.py --tag abc --arms baseline,merlin,merlin_rtlchecks \\
      --repeats 3 --verilator-slots 2

The task PLAN (which commands fan out) is a pure function (:func:`plan_tasks`), unit-testable without
chia; only the fan-out itself needs the chia venv + the sim.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import launch_ab_batch as LB  # reuse ARMS / _arm_cmd / _run_id / _run_preflight / C (don't reimplement)

try:  # the decorator needs chia (chia venv); a no-op shim lets the module import for planning/tests
    from chia.base.ChiaFunction import ChiaFunction
    _HAVE_CHIA = True
except Exception:  # noqa: BLE001
    _HAVE_CHIA = False

    def ChiaFunction(**_kw):  # type: ignore[no-redef]
        def deco(fn):
            return fn
        return deco


def plan_tasks(arms: list[str], repeats: int, tag: str, a, cond: str = "kernels") -> list[dict]:
    """The (arm × repeat) fan-out plan: one task per repeat of each arm, each with the arm's driver
    command from ``launch_ab_batch._arm_cmd``. Pure — no chia, no side effects."""
    tasks: list[dict] = []
    for arm in arms:
        if arm not in LB.ARMS:
            raise KeyError(f"unknown arm {arm!r} (have {sorted(LB.ARMS)})")
        for r in range(repeats):
            rid = LB._run_id(arm, f"{tag}_r{r}" if repeats > 1 else tag)
            tasks.append({"arm": arm, "repeat": r, "run_id": rid,
                          "cmd": LB._arm_cmd(arm, rid, a, cond)})
    return tasks


@ChiaFunction(resources={"verilator": 1}, num_cpus=1, max_retries=0)
def run_arm(cmd: list[str], cwd: str, run_id: str) -> dict:
    """Run one arm-repeat's driver command; holds one logical ``verilator`` unit for the duration."""
    proc = subprocess.run(cmd, cwd=cwd)
    return {"run_id": run_id, "returncode": proc.returncode}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", default="baseline,merlin,merlin_rtlchecks,cpp_merlininfra")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--verilator-slots", type=int, default=1,
                    help="how many arm-repeats may hold the logical 'verilator' resource at once (1 == sequential)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--max-rate-limit-waits", type=int, default=8)
    ap.add_argument("--round-timeout", type=int, default=14400)
    ap.add_argument("--experiment", default="realistic")
    ap.add_argument("--sandbox", default="bwrap")
    ap.add_argument("--skip-hidden", action="store_true")
    ap.add_argument("--cond", default="kernels")
    ap.add_argument("--dry-run", action="store_true", help="print the fan-out plan and exit (no chia)")
    a = ap.parse_args(argv)

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    tasks = plan_tasks(arms, a.repeats, a.tag, a, a.cond)
    print(f"=== chia_ab_batch: {len(tasks)} tasks ({len(arms)} arms x {a.repeats} repeats), "
          f"verilator_slots={a.verilator_slots} ===")
    for t in tasks:
        print(f"  {t['arm']:16s} r{t['repeat']} {t['run_id']}")
    if a.dry_run:
        return 0

    from merlin.benchharness.chia_bridge import chia_get, chia_run, require_chia
    require_chia()
    if a.verilator_slots < 1:
        print("--verilator-slots must be >= 1", file=sys.stderr)
        return 2
    LB._run_preflight()   # lock answer surfaces + verify_no_cheat before any spend
    with chia_run(suite="capsule-bench", method="chia_ab_batch", target="gemmini",
                  params={"arms": arms, "repeats": a.repeats, "verilator_slots": a.verilator_slots},
                  ray_resources={"verilator": a.verilator_slots}) as run:
        refs = [run_arm.chia_remote(t["cmd"], str(LB.C.REPO), t["run_id"]) for t in tasks]
        results = chia_get(refs)
    fails = [r for r in results if r.get("returncode")]
    print(f"=== done: {len(results) - len(fails)}/{len(results)} ok ===")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
