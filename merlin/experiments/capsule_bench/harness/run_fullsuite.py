#!/usr/bin/env python3
"""Run the FULL 20+5 capsule suite through the SHARED QA loop — same loop, same grader as the pilot, so
both arms stay apples-to-apples.

The shared controller receives an explicit, source-attributed Treatment: the legacy public corpus
root, the TASK_full.md stager (+ merlin addendum), and an invocation-local timing observer.
No controller function or process-global module entry is replaced. Agent-active versus QA wall time
accumulates in the existing fullsuite timing sidecar; hidden selection remains descriptor-derived.

The agent still iterates until ALL public/dev capsules pass at L3 (real RTL); hidden H0-H4 graded
post-freeze. Usage mirrors the loop driver:
  run_fullsuite.py --arm merlin_assisted --run-id merlin_full_01 --model claude-opus-4-8 --effort high \
      --max-rounds 14 --round-timeout 2700 --qa-timeout 1800 --sandbox none
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C
import run_baseline_qa_loop as L
import yaml
from merlin_experiments.phase1 import run_inputs as RI
from merlin_experiments.phase1.treatments import Treatment

from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

FULL_CAPSULES = C.REPO / "merlin/contract" / "capsules"
TASK_FULL = load_target_experiment(C.DESCRIPTOR).resource_path("task/TASK_full.md")


def _full_build(arm: str, ws: Path, run_dir: Path, *, bundle_dir: Path, sandbox: str = "bwrap") -> None:
    """Stage the workspace TASK.md from TASK_full.md (both arms identical contract); merlin appends its
    addendum + stages its docs. The sandbox keyword matches the native staging contract;
    this legacy task's bytes do not vary with the selected sandbox."""
    base = TASK_FULL.read_text()
    ws_task = ws / "TASK.md"
    if arm == "merlin_assisted":
        bdir = bundle_dir
        add = (bdir / "TASK_ADDENDUM.md").read_text() if (bdir / "TASK_ADDENDUM.md").exists() else ""
        ws_task.write_text(base + "\n\n---\n\n" + add)
        for doc in RI.MERLIN_WS_DOCS:
            s = bdir / doc
            if s.exists():
                shutil.copy(s, ws / doc)
    else:
        ws_task.write_text(base)
    shutil.copy(ws_task, run_dir / "TASK.md")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # locate the run dir for post-run timing append (does not consume args from L.main)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm", default="raw_baseline")
    ap.add_argument("--run-id", required=True)
    known, _ = ap.parse_known_args(args)
    run_dir = C.RUNS / known.arm / known.run_id

    _walls: dict[str, list[float]] = {"agent": [], "qa": []}

    def observe(kind: str, elapsed: float) -> None:
        _walls[kind].append(elapsed)

    treatment = Treatment(name="fullsuite", capsules_root=FULL_CAPSULES, stage_task=_full_build, on_duration=observe)
    rc = L.main(args, treatment=treatment)

    # --- agent-active vs sim-wait split (COMPLEMENTS the driver's active-vs-quota timing) ---
    # The shared driver records active_wall_s (agent+oracle combined) vs rate_limit_wait_s (quota
    # sleeps) in qa_loop_summary['timing'] — authoritative + cumulative across resumes. Here we add
    # the finer agent-vs-sim split the user asked for ("time waiting on the sim vs doing things"),
    # which the driver does not separate. RESUME-SAFE: accumulate into a sidecar (do NOT touch the
    # driver's qa_loop_summary), summing this invocation's _walls onto any prior invocations' totals.
    side = run_dir / "fullsuite_agent_sim_timing.yaml"
    prev = yaml.safe_load(side.read_text()) if side.exists() else {}
    prev = prev or {}
    side.write_text(
        yaml.safe_dump(
            {
                "agent_active_s": round(prev.get("agent_active_s", 0.0) + sum(_walls["agent"]), 3),
                "sim_wait_s": round(prev.get("sim_wait_s", 0.0) + sum(_walls["qa"]), 3),
                "invocations": int(prev.get("invocations", 0)) + 1,
                "note": (
                    "agent_active_s = agent subprocess wall (summed rounds, cumulative across resumes); "
                    "sim_wait_s = oracle grading wall (spike+verilator). These split the driver's "
                    "active_wall_s; the driver's rate_limit_wait_s (quota sleeps) is separate."
                ),
            },
            sort_keys=False,
        )
    )
    ef = run_dir / "environment.yaml"
    e = (yaml.safe_load(ef.read_text()) or {}) if ef.exists() else {}
    selected_root = FULL_CAPSULES
    selected_contract = None
    if e.get("public_corpus_input"):
        from merlin_experiments.phase1.corpus_inputs import resolve

        effective = yaml.safe_load((run_dir / "input_bundle_manifest.yaml").read_text())
        view = resolve(Path(e["workspace_path"]), effective, e["public_corpus_input"], repo=C.REPO)
        selected_root, selected_contract = view.public, view.contract
    try:
        pub = sorted(
            c.get("capsule") or c.get("name") or Path(c.get("dir", "")).name
            for c in L.CR.discover_capsules(selected_root, labels={"public", "dev"}, contract=selected_contract)
        )
    except Exception:
        pub = []
    if ef.exists():
        e["suite"] = "full"
        e["capsules_root"] = str(selected_root)
        e["task_file"] = str(TASK_FULL)
        if pub:
            e["public_dev_capsules"] = pub  # corrects the inherited pilot label
        ef.write_text(yaml.safe_dump(e, sort_keys=False))
    print(
        f"[fullsuite] {known.arm}/{known.run_id} rc={rc} "
        f"this-invocation agent={round(sum(_walls['agent']), 1)}s sim={round(sum(_walls['qa']), 1)}s "
        f"(cumulative in {side.name})"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
