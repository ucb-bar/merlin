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
    """Run one arm-repeat's driver command; holds one logical ``verilator`` unit for the duration.

    The command is built by ``LB._arm_cmd`` with ``sys.executable`` — but on a Ray worker that is the
    CHIA venv (ray/pydantic), which lacks xdsl + the merlin framework the drivers need. Re-point the
    interpreter at the MAIN ``.venv`` (``driver_python()``, the same choice ``chia_repeatability`` makes)
    so the driver + its agent subprocess never see chia's env. Keeps Claude Code + xDSL out of chia's tree."""
    from merlin.benchharness.chia_bridge import driver_python
    if cmd and cmd[0] != driver_python():
        cmd = [driver_python()] + cmd[1:]
    proc = subprocess.run(cmd, cwd=cwd)
    return {"run_id": run_id, "returncode": proc.returncode}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", default="baseline,merlin,merlin_rtlchecks,cpp_merlininfra")
    ap.add_argument("--repeats", type=int, default=1)
    # AGENT DRIVER. LB._arm_cmd already forwards --driver to each arm script when it
    # is not "auto", so the fan-out only has to expose it — and gate the provider's
    # concurrency, which is a different scarce resource from the simulator's.
    ap.add_argument("--driver", choices=["auto", "converse", "claudecode", "opencode", "codex"],
                    default="auto", help="agent driver for every arm (codex = Codex CLI)")
    ap.add_argument("--codex-slots", type=int, default=1,
                    help="how many arm-repeats may hold the logical 'codex_slots' resource at once. A "
                         "provider quota is not a simulator slot: two arms can share a Verilator host and "
                         "still contend on one account, so it is gated separately (1 == one Codex call "
                         "in flight)")
    ap.add_argument("--verilator-slots", type=int, default=1,
                    help="how many arm-repeats may hold the logical 'verilator' resource at once (1 == sequential)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    # provider toggle (experiments-only) — threaded verbatim into each arm's driver cmd by LB._arm_cmd
    ap.add_argument("--provider", choices=["subscription", "bedrock"], default="subscription")
    ap.add_argument("--aws-region", default="us-east-1")
    ap.add_argument("--aws-profile", default="")
    # Schedule passthrough — _arm_cmd (reused from launch_ab_batch) forwards these to each arm's driver.
    ap.add_argument("--schedule", choices=("rounds", "continuous"), default="rounds")
    ap.add_argument("--plateau-rounds", type=int, default=None,
                    help="continuous only: forwarded to each arm's loop — stop when the best "
                         "score has not improved across this many rounds (0 disables). "
                         "Unset leaves the loop default, so a batch that omits it is unchanged.")
    ap.add_argument("--max-wall-s", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--max-rate-limit-waits", type=int, default=8)
    ap.add_argument("--round-timeout", type=int, default=14400)
    # Threaded into each arm's driver by LB._arm_cmd, like --max-wall-s and --plateau-rounds. Unset
    # leaves the loop's own defaults, so a batch that omits them is unchanged.
    ap.add_argument("--qa-timeout", type=int, default=None,
                    help="per-capsule oracle wall inside the round grade (s)")
    ap.add_argument("--qa-workers", type=int, default=None,
                    help="parallel capsules in the round grade")
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
    # Honour the verdict. This return code was DISCARDED, so a preflight that printed
    # "VERIFY_NO_CHEAT: FAIL -- DO NOT launch" went on to launch anyway, and the gate that exists
    # to stop a compromised run from spending was decorative. Observed on a live launch.
    _pf = LB._run_preflight()   # lock answer surfaces + verify_no_cheat before any spend
    if _pf:
        print("preflight FAILED — refusing to launch (nothing has been spent)", file=sys.stderr)
        return _pf
    # target is DERIVED from the active descriptor (MERLIN_TARGET_EXPERIMENT via _common), never hardcoded
    # — so the chia run dir + telemetry land under the right target (atlas/gemmini/…), no per-target branch.
    # The cluster must SUPPLY every resource a task requests, or Ray never schedules that task at all
    # (a Codex arm asks for a `codex_slots` unit below, so the pool has to declare one).
    cluster_resources = {"verilator": a.verilator_slots}
    if a.driver == "codex":
        if a.codex_slots < 1:
            print("--codex-slots must be >= 1", file=sys.stderr)
            return 2
        cluster_resources["codex_slots"] = a.codex_slots
    with chia_run(suite="capsule-bench", method="chia_ab_batch", target=LB.C.TARGET,
                  extra={"arms": arms, "repeats": a.repeats, "verilator_slots": a.verilator_slots,
                         "codex_slots": a.codex_slots if a.driver == "codex" else None,
                         "driver": a.driver, "model": a.model, "provider": a.provider},
                  ray_resources=cluster_resources) as run:
        # A Codex arm consumes provider quota, so it holds a `codex_slots` unit for
        # its duration in addition to the verilator unit. Requested at call time
        # because the resource set depends on the chosen driver; if the installed
        # chia cannot re-option a ChiaFunction, say so rather than silently running
        # an ungated fan-out against one account.
        launcher = run_arm
        if a.driver == "codex":
            opts = getattr(run_arm, "options", None)
            if callable(opts):
                launcher = opts(resources={"verilator": 1, "codex_slots": 1})
            else:
                print("WARNING: this chia build cannot re-option resources; the Codex fan-out is NOT "
                      "gated on codex_slots — run with --verilator-slots to bound concurrency instead.")
        refs = [launcher.chia_remote(t["cmd"], str(LB.C.REPO), t["run_id"]) for t in tasks]
        results = chia_get(refs)
    fails = [r for r in results if r.get("returncode")]
    print(f"=== done: {len(results) - len(fails)}/{len(results)} ok ===")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
