#!/usr/bin/env python3
"""Launch the 3-approach agentic experiment (baseline · merlin · merlin+CIRCT) as fresh experiments.

The A/B/C contrast — same task, model, flags, grading; ONLY the authoring aids differ:
  • baseline        : run_baseline_qa_loop.py --arm raw_baseline      (no Merlin tools)
  • merlin          : run_baseline_qa_loop.py --arm merlin_assisted   (xDSL + all Merlin authoring aids,
                                                                       NO CIRCT checks)
  • merlin_rtlchecks: run_rtlchecks_qa_loop.py                        (same aids + CIRCT-compiled-from-RTL
                                                                       checks as advisory per-round feedback)
baseline lands in runs/raw_baseline/<id>; both merlin arms land in runs/merlin_assisted/<id> (the rtlchecks
driver writes a TRACK_RTLCHECKS marker so they're distinguishable downstream). Each arm = ONE fresh run.

--mode parallel  : background all selected arms at once (use when the 5h bucket has headroom).
--mode sequential: background a single chain that runs them one after another (tight-budget fallback).

This wrapper backgrounds the sessions; monitor via each run's qa_loop_state.yaml (NOT the .log — block-
buffered). --dry-run prints the exact commands + preflight and launches nothing.

Usage:
  launch_ab_batch.py --tag abc1 --dry-run
  launch_ab_batch.py --tag abc1 --mode parallel
  launch_ab_batch.py --tag abc1 --arms baseline,merlin,merlin_rtlchecks --mode parallel
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import _common as C

SCRIPTS = C.EXP / "scripts"
# arm -> (driver, extra args, run-id prefix, run-dir subdir)
# arm -> (driver, extra args, run-id prefix, run-dir subdir, realistic bundle id)
# realistic (abc3+) = HW-bringup info set: RTL + ISA + README + ONE example kernel (not the full suite).
ARMS = {
    "baseline":         (SCRIPTS / "run_baseline_qa_loop.py", ["--arm", "raw_baseline"], "rb", "raw_baseline", "raw_baseline_hwbringup_v0"),
    "merlin":           (SCRIPTS / "run_baseline_qa_loop.py", ["--arm", "merlin_assisted"], "merlin", "merlin_assisted", "merlin_assisted_hwbringup_v0"),
    "merlin_rtlchecks": (SCRIPTS / "run_rtlchecks_qa_loop.py", [], "merlincirct", "merlin_assisted", "merlin_assisted_rtlchecks_hwbringup_v0"),
}


def _run_id(arm: str, tag: str) -> str:
    return f"{ARMS[arm][2]}_{tag}"


def _run_dir(arm: str, run_id: str) -> Path:
    return C.RUNS / ARMS[arm][3] / run_id


def _bundle_for(arm: str, cond: str) -> str:
    b = ARMS[arm][4]                                   # *_hwbringup_v0 (kernels condition)
    if cond == "no-kernels":
        b = b.replace("_hwbringup_v0", "_hwbringup_nokernel_v0")
    return b


def _arm_cmd(arm: str, run_id: str, a, cond: str = "kernels") -> list[str]:
    driver, extra, _, _, _ = ARMS[arm]
    cmd = [sys.executable, str(driver), "--run-id", run_id,
           "--model", a.model, "--effort", a.effort,
           "--max-rounds", str(a.max_rounds), "--max-rate-limit-waits", str(a.max_rate_limit_waits),
           "--round-timeout", str(a.round_timeout)]
    cmd += extra
    if a.experiment == "realistic":
        cmd += ["--experiment", "realistic", "--bundle", _bundle_for(arm, cond)]
    if a.skip_hidden:
        cmd += ["--skip-hidden"]
    cmd += ["--sandbox", a.sandbox]
    return cmd


# answer surfaces locked (chmod 000) right before any spend — prior backends, hidden capsules, goldens,
# results, the experimenter's agent memory. Defence-in-depth on top of the per-bundle workspace assembly.
ANSWER_SURFACES = [
    "generated_targets/gemmini",
    "bench_contract/capsules/hidden",
    "results/gemmini",
]


def _run_preflight() -> int:
    """Lock answer surfaces + run verify_no_cheat.py (the authoritative gate). Returns 0 iff safe."""
    locked = []
    for rel in ANSWER_SURFACES:
        p = C.REPO / rel
        if p.exists():
            subprocess.run(["chmod", "-R", "000", str(p)], capture_output=True)
            locked.append(rel)
    print(f"  locked answer surfaces (chmod 000): {locked or '(none present)'}")
    vnc = SCRIPTS / "verify_no_cheat.py"
    if not vnc.is_file():
        print(f"  ⚠ verify_no_cheat.py not found at {vnc} — build it before launch (#167).")
        return 1
    print(f"  running {vnc.name} ...")
    r = subprocess.run([sys.executable, str(vnc)], cwd=str(C.REPO))
    return r.returncode


def _arm_env(arm: str, a, base_env: dict) -> dict:
    """Per-arm env. Realistic baseline = status-quo C++ OOT MLIR (PILOT_LANG=cpp); merlin arms = xDSL."""
    env = dict(base_env)
    if a.experiment == "realistic" and arm == "baseline":
        env["PILOT_LANG"] = "cpp"
    return env


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="batch label, e.g. abc1 -> rb_abc1 / merlin_abc1 / merlincirct_abc1")
    ap.add_argument("--arms", default="baseline,merlin,merlin_rtlchecks")
    ap.add_argument("--mode", choices=["parallel", "sequential"], default="parallel")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--max-rate-limit-waits", type=int, default=8)
    ap.add_argument("--round-timeout", type=int, default=14400, help="per-round agent wall cap (s); large = effectively no timeout")
    ap.add_argument("--skip-hidden", action="store_true")
    ap.add_argument("--experiment", choices=["full", "realistic"], default="full",
                    help="'realistic' (abc2): whole-repo + self-check tool + verilator barrier; baseline=C++")
    ap.add_argument("--account-config-dir", default="", help="CLAUDE_CONFIG_DIR for all arms (optional)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="N independent repeats per arm×condition (fresh tagged run-ids; N>1 -> _r{n} suffix). "
                         "The N>1 that the magnitude claims need (error bars).")
    ap.add_argument("--condition", choices=["kernels", "no-kernels", "both"], default="kernels",
                    help="info-set axis: 'kernels' (hwbringup + example kernels) vs 'no-kernels' "
                         "(RTL+ISA+README only — tests the CIRCT-shines-without-kernels hypothesis). "
                         "'both' expands the cross-product (run-id gets _nk for the no-kernels cell).")
    ap.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap",
                    help="bwrap (default, now that claude 2.1.185 runs under it): true filesystem allow-list "
                         "— only granted bundle files + the legit toolchain visible, all answers masked "
                         "(proven by test_sandbox.py 18/18 per arm). 'none' = legacy reachable-fs + audit.")
    ap.add_argument("--preflight", action="store_true",
                    help="lock all answer surfaces (chmod 000) + dry-run + assert cheat-clean BEFORE any "
                         "spend; launches NOTHING (use this immediately before a real launch).")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        print(f"unknown arms: {bad} (valid: {list(ARMS)})", file=sys.stderr); return 2
    if a.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr); return 2

    conditions = ["kernels", "no-kernels"] if a.condition == "both" else [a.condition]
    # each (arm, condition, repeat) -> a fresh tagged run. tag encodes the cell so dirs never collide.
    cells = []  # (arm, condition, tag)
    for cond in conditions:
        csfx = "_nk" if cond == "no-kernels" else ""
        for rep in range(1, a.repeats + 1):
            rsfx = f"_r{rep}" if a.repeats > 1 else ""
            for arm in arms:
                cells.append((arm, cond, f"{a.tag}{csfx}{rsfx}"))

    planned, problems = [], []
    for arm, cond, tag in cells:
        rid = _run_id(arm, tag); rd = _run_dir(arm, rid)
        if rd.exists():
            problems.append(f"run dir already exists: {rd} (fresh experiment requires a new --tag)")
        planned.append((arm, rid, rd, cond))
    acct = os.path.expanduser(a.account_config_dir) if a.account_config_dir else ""
    if acct and not Path(acct).exists():
        problems.append(f"account-config-dir not found: {acct}")

    print(f"=== A/B/C batch '{a.tag}' — arms={arms} mode={a.mode} model={a.model} effort={a.effort} "
          f"repeats={a.repeats} condition={a.condition} ({len(planned)} runs) ===")
    for arm, rid, rd, cond in planned:
        print(f"  {arm:16s} [{cond:10s}] run-id={rid:22s} -> {rd}")
        print(f"     $ {' '.join(_arm_cmd(arm, rid, a, cond))}")
    if problems:
        print("\nPREFLIGHT PROBLEMS (nothing launched):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2

    if a.preflight:
        print("\n=== --preflight: lock answer surfaces + assert cheat-clean (launches NOTHING) ===")
        rc = _run_preflight()
        print(f"\n[preflight] {'PASS — safe to launch (drop --preflight, keep flags)' if rc == 0 else 'FAILED — DO NOT launch'}")
        return rc

    if a.dry_run:
        print(f"\n[dry-run] preflight OK ({a.mode}); nothing launched. Drop --dry-run to launch.")
        return 0

    # HARD pre-flight: verilator timing must be FRESH (readiness_check measured it on a known-good backend
    # AFTER the current sim binary was built). This is the abc7 safeguard — never launch a run whose
    # verilator timeout/availability was never actually verified. (Skip for --sandbox none legacy.)
    if a.sandbox == "bwrap":
        timing = SCRIPTS / ".oracle_timing.json"
        sim = Path("/scratch2/agustin/chipyard/sims/verilator/simulator-chipyard.harness-GemminiAndOPUShuttleConfig")
        if not timing.is_file():
            print("REFUSING TO LAUNCH: scripts/.oracle_timing.json missing — run readiness_check.py "
                  "(it RUNS spike+verilator on the reference backend) first.", file=sys.stderr)
            return 3
        if sim.is_file() and timing.stat().st_mtime < sim.stat().st_mtime:
            print("REFUSING TO LAUNCH: .oracle_timing.json is STALE (older than the verilator binary) — "
                  "re-run readiness_check.py to re-measure.", file=sys.stderr)
            return 3
        print(f"[pre-flight] oracle timing fresh: {timing.read_text().strip()[:80]}")

    env = dict(os.environ)
    if acct:
        env["CLAUDE_CONFIG_DIR"] = acct
    manifest = {"tag": a.tag, "mode": a.mode, "launched_at": datetime.now(timezone.utc).isoformat(),
                "model": a.model, "effort": a.effort, "account_config_dir": acct or None, "runs": []}

    if a.mode == "parallel":
        for arm, rid, rd, cond in planned:
            rd.parent.mkdir(parents=True, exist_ok=True)
            log = rd.parent / f"{rid}.launch.log"
            with open(log, "w") as lf:
                p = subprocess.Popen(_arm_cmd(arm, rid, a, cond), cwd=str(C.REPO), stdout=lf,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=_arm_env(arm, a, env))
            manifest["runs"].append({"arm": arm, "run_id": rid, "condition": cond, "pid": p.pid, "log": str(log)})
            print(f"launched {arm}/{rid} [{cond}]  pid={p.pid}  log={log}")
    else:  # sequential: one backgrounded bash chain (cmd1 ; cmd2 ; cmd3)
        parts = []
        for arm, rid, rd, cond in planned:
            rd.parent.mkdir(parents=True, exist_ok=True)
            log = rd.parent / f"{rid}.launch.log"
            prefix = "PILOT_LANG=cpp " if (a.experiment == "realistic" and arm == "baseline") else ""
            parts.append(f"{prefix}{' '.join(_arm_cmd(arm, rid, a, cond))} > {log} 2>&1")
            manifest["runs"].append({"arm": arm, "run_id": rid, "condition": cond, "pid": None, "log": str(log)})
        chain = " ; ".join(parts)
        clog = C.RUNS / f"ab_batch_{a.tag}.chain.log"
        with open(clog, "w") as lf:
            p = subprocess.Popen(["bash", "-c", chain], cwd=str(C.REPO), stdout=lf,
                                 stderr=subprocess.STDOUT, start_new_session=True, env=env)
        manifest["chain_pid"] = p.pid
        print(f"launched sequential chain pid={p.pid}  chain-log={clog}\n  order: {[r['run_id'] for r in manifest['runs']]}")

    mpath = C.RUNS / f"ab_batch_{a.tag}.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nbatch manifest: {mpath}")
    print("monitor via each run's qa_loop_state.yaml (NOT .log). After convergence: full_suite_audit.py + "
          "agg_agentic_results.py + the agentic plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
