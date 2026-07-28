#!/usr/bin/env python3
"""Run the FULL 20+5 capsule suite through the SHARED QA loop — same loop, same grader as the pilot, so
both arms stay apples-to-apples.

Why a wrapper instead of editing run_baseline_qa_loop.py: that driver is shared and concurrently edited
(the raw-baseline operator session is actively running + modifying it). To avoid clobbering live work,
this wrapper reuses `run_baseline_qa_loop.main()` verbatim and only OVERRIDES, at the module level:
  - the capsule set  -> `merlin/contract/capsules` (all 20 public/dev + 5 hidden), via L.PILOT_SUBSET
  - the served task  -> `task/TASK_full.md` (+ merlin addendum), via L._build_task
  - per-round timing -> wraps L.launch_agent / L.qa_grade to record ACTIVE (agent) vs SIM-WAIT
    (oracle = spike+verilator) wall, then appends the split to qa_loop_summary.yaml.
Name resolution: L.main() looks these up as module globals at call time, so overriding L.* before
calling L.main() takes effect with no edit to the shared file.

The agent still iterates until ALL public/dev capsules pass at L3 (real RTL); hidden H0-H4 graded
post-freeze. Usage mirrors the loop driver:
  run_fullsuite.py --arm merlin_assisted --run-id merlin_full_01 --model claude-opus-4-8 --effort high \
      --max-rounds 14 --round-timeout 2700 --qa-timeout 1800 --sandbox none
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

import _common as C
import run_agent_experiment as RX
import run_baseline_qa_loop as L

FULL_CAPSULES = C.REPO / "merlin/contract" / "capsules"
TASK_FULL = C.EXP / "task" / "TASK_full.md"

_walls: dict[str, list[float]] = {"agent": [], "qa": []}


def _full_build(arm: str, ws: Path, run_dir: Path) -> None:
    """Stage the workspace TASK.md from TASK_full.md (both arms identical contract); merlin appends its
    addendum + stages its docs. Mirrors L._build_task but with the full-suite task."""
    base = TASK_FULL.read_text()
    ws_task = ws / "TASK.md"
    if arm == "merlin_assisted":
        bdir = C.BUNDLES / RX.ARM_BUNDLE[arm]
        add = (bdir / "TASK_ADDENDUM.md").read_text() if (bdir / "TASK_ADDENDUM.md").exists() else ""
        ws_task.write_text(base + "\n\n---\n\n" + add)
        for doc in L.MERLIN_WS_DOCS:
            s = bdir / doc
            if s.exists():
                shutil.copy(s, ws / doc)
    else:
        ws_task.write_text(base)
    shutil.copy(ws_task, run_dir / "TASK.md")


def _timed(orig, key: str):
    def w(*a, **k):
        t = time.time()
        try:
            return orig(*a, **k)
        finally:
            _walls[key].append(round(time.time() - t, 3))
    return w


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # locate the run dir for post-run timing append (does not consume args from L.main)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm", default="raw_baseline")
    ap.add_argument("--run-id", required=True)
    known, _ = ap.parse_known_args(args)
    run_dir = C.RUNS / known.arm / known.run_id

    # --- override the shared loop's capsule set + task + timing, WITHOUT editing it ---
    L.PILOT_SUBSET = FULL_CAPSULES          # qa_grade() + the final grade_agent_run --capsules use this
    L.CAPSULES_ROOT = FULL_CAPSULES         # harmless if the shared file later adds this global
    L._build_task = _full_build             # launch_agent calls _build_task as a module global
    L.launch_agent = _timed(L.launch_agent, "agent")
    L.qa_grade = _timed(L.qa_grade, "qa")

    if argv is not None:
        sys.argv = [sys.argv[0]] + args
    rc = L.main()  # parses sys.argv[1:]; uses the overridden globals/functions

    # --- agent-active vs sim-wait split (COMPLEMENTS the driver's active-vs-quota timing) ---
    # The shared driver records active_wall_s (agent+oracle combined) vs rate_limit_wait_s (quota
    # sleeps) in qa_loop_summary['timing'] — authoritative + cumulative across resumes. Here we add
    # the finer agent-vs-sim split the user asked for ("time waiting on the sim vs doing things"),
    # which the driver does not separate. RESUME-SAFE: accumulate into a sidecar (do NOT touch the
    # driver's qa_loop_summary), summing this invocation's _walls onto any prior invocations' totals.
    side = run_dir / "fullsuite_agent_sim_timing.yaml"
    prev = yaml.safe_load(side.read_text()) if side.exists() else {}
    prev = prev or {}
    side.write_text(yaml.safe_dump({
        "agent_active_s": round(prev.get("agent_active_s", 0.0) + sum(_walls["agent"]), 3),
        "sim_wait_s": round(prev.get("sim_wait_s", 0.0) + sum(_walls["qa"]), 3),
        "invocations": int(prev.get("invocations", 0)) + 1,
        "note": ("agent_active_s = agent subprocess wall (summed rounds, cumulative across resumes); "
                 "sim_wait_s = oracle grading wall (spike+verilator). These split the driver's "
                 "active_wall_s; the driver's rate_limit_wait_s (quota sleeps) is separate."),
    }, sort_keys=False))
    try:
        pub = sorted(c.get("capsule") or c.get("name") or Path(c.get("dir", "")).name
                     for c in L.CR.discover_capsules(FULL_CAPSULES, labels={"public", "dev"}))
    except Exception:
        pub = []
    ef = run_dir / "environment.yaml"
    if ef.exists():
        e = yaml.safe_load(ef.read_text()) or {}
        e["suite"] = "full"
        e["capsules_root"] = str(FULL_CAPSULES)
        e["task_file"] = str(TASK_FULL)
        if pub:
            e["public_dev_capsules"] = pub  # corrects the inherited pilot label
        ef.write_text(yaml.safe_dump(e, sort_keys=False))
    print(f"[fullsuite] {known.arm}/{known.run_id} rc={rc} "
          f"this-invocation agent={round(sum(_walls['agent']),1)}s sim={round(sum(_walls['qa']),1)}s "
          f"(cumulative in {side.name})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
