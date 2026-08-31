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

from merlin.common.paths import ext_path  # noqa: E402
import _common as C

SCRIPTS = C.EXP / "scripts"
# arm -> (driver, extra args, run-id prefix, run-dir subdir)
# arm -> (driver, extra args, run-id prefix, run-dir subdir, realistic bundle id)
# realistic (abc3+) = HW-bringup info set: RTL + ISA + README + ONE example kernel (not the full suite).
ARMS = {
    "baseline":         (SCRIPTS / "run_baseline_qa_loop.py", ["--arm", "raw_baseline"], "rb", "raw_baseline", "raw_baseline_hwbringup_v0"),
    "merlin":           (SCRIPTS / "run_baseline_qa_loop.py", ["--arm", "merlin_assisted"], "merlin", "merlin_assisted", "merlin_assisted_hwbringup_v0"),
    "merlin_rtlchecks": (SCRIPTS / "run_rtlchecks_qa_loop.py", [], "merlincirct", "merlin_assisted", "merlin_assisted_rtlchecks_hwbringup_v0"),
    "cpp_merlininfra":  (SCRIPTS / "run_baseline_qa_loop.py", ["--arm", "cpp_merlininfra"], "rbinfra", "cpp_merlininfra", "cpp_merlininfra_hwbringup_v0"),
    "merlin_eqsat":     (SCRIPTS / "run_eqsat_qa_loop.py", [], "merlineqsat", "merlin_assisted", "merlin_assisted_eqsat_hwbringup_v0"),
}


def _sim_via() -> str:
    """The target's declared bespoke sim (toolchain.sim_via) from the active descriptor — "" (arc-only)
    for atlas/npu_model/radiance/saturn, "chipyard" for gemmini. Drives the verilator-timing preflight."""
    import yaml
    desc = C.EXP / "target_experiment.yaml"       # C.EXP honors MERLIN_TARGET_EXPERIMENT
    if desc.is_file():
        d = yaml.safe_load(desc.read_text()) or {}
        return ((d.get("toolchain") or {}).get("sim_via") or "").strip()
    return ""


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
    if getattr(a, "min_rounds", 0):
        cmd += ["--min-rounds", str(a.min_rounds)]
    # Schedule passthrough. Forwarded ONLY when non-default so an unpatched/older driver in the batch is
    # invoked exactly as before — a batch runs several arms and a flag one of them does not understand
    # kills that cell, which reads as an arm-specific failure rather than a launcher mismatch.
    if getattr(a, "schedule", "rounds") != "rounds":
        cmd += ["--schedule", a.schedule]
    if getattr(a, "max_wall_s", 0):
        cmd += ["--max-wall-s", str(a.max_wall_s)]
    # Threaded like --max-wall-s: a per-arm loop terminator the batch must pass through, or every arm
    # silently falls back to the loop's default and the batch's own setting is a lie. Only forwarded when
    # the caller set it explicitly (None), so a batch that does not mention it stays byte-identical.
    if getattr(a, "plateau_rounds", None) is not None:
        cmd += ["--plateau-rounds", str(a.plateau_rounds)]
    # The whole-model capsule's wall-clock ceiling, forwarded like the two above (only when set, so an
    # older driver in the batch is invoked exactly as before). Without a ceiling a capstone that clears
    # its op-pass gate runs a cycle-accurate simulation of the entire model inside the round grade --
    # measured at 5h30m past a round's own 4h timeout, with the round never grading.
    if getattr(a, "model_budget_s", None) is not None:
        cmd += ["--model-budget-s", str(a.model_budget_s)]
    cmd += extra
    # Agent driver + optional tier-within-agent models (default "" -> the per-driver default tier).
    if getattr(a, "driver", "auto") != "auto":
        cmd += ["--driver", a.driver]
    if getattr(a, "subagent_model", ""):
        cmd += ["--subagent-model", a.subagent_model]
    if getattr(a, "background_model", ""):
        cmd += ["--background-model", a.background_model]
    if a.experiment == "realistic":
        cmd += ["--experiment", "realistic", "--bundle", _bundle_for(arm, cond)]
    if a.skip_hidden:
        cmd += ["--skip-hidden"]
    cmd += ["--sandbox", a.sandbox]
    # Provider (experiments-only): default subscription is a no-op flag; bedrock threads AWS region/profile.
    if getattr(a, "provider", "subscription") != "subscription":
        cmd += ["--provider", a.provider, "--aws-region", a.aws_region]
        if a.aws_profile:
            cmd += ["--aws-profile", a.aws_profile]
    # ABLATION passthrough: the same cell applies to EVERY arm in the batch, which is what makes an
    # add-one/subtract-one sweep a single command per cell instead of one per arm.
    for t in getattr(a, "with_tool", []) or []:
        cmd += ["--with-tool", t]
    for t in getattr(a, "without_tool", []) or []:
        cmd += ["--without-tool", t]
    return cmd


# Host-side answer surfaces are owner-only but remain readable to the host grader. The agent boundary is
# the deny-by-default bwrap mount table, not a same-UID chmod trick that also blinds grading and readiness.
# DERIVED from the descriptor so a new target protects ITS OWN surfaces: the per-target hidden holdout
# dir (the answers) + stale prior results. NOTE
# the target's contract dir (out/artifacts/targets/<target>) is deliberately NOT locked — it holds the
# capability contract, not an answer, and locking it would break the run's own contract resolution.
def _host_answer_surfaces(te) -> list[Path]:
    surfaces: list[Path] = []
    if te is not None:
        h = te.hidden_corpus()                                # per-target hidden set (capsules/<t>/hidden)
        if h:
            surfaces.append(C.REPO / h.rstrip("/"))
    surfaces.append(C.REPO / f"out/artifacts/capsule-bench/{C.TARGET}")  # stale prior results
    return surfaces


def _make_host_owner_only(root: Path) -> int:
    """Keep a surface host-readable while denying other host users; return entries protected.

    Symlinks are refused instead of followed: recursively changing a linked tree would expand the
    preflight's authority beyond the descriptor-derived surface.
    """
    if root.is_symlink():
        raise RuntimeError(f"answer surface must not be a symlink: {root}")
    entries = [root] + sorted(root.rglob("*"))
    for entry in entries:
        if entry.is_symlink():
            raise RuntimeError(f"answer surface contains a symlink: {entry}")
        entry.chmod(0o700 if entry.is_dir() else 0o600)
    return len(entries)


def _run_preflight() -> int:
    """Prepare host-only surfaces + run verify_no_cheat.py. Returns 0 iff safe."""
    C.require_scaffolding()   # fail loudly if MERLIN_TARGET_EXPERIMENT points at a scaffolding-less dir
    from merlin.targetgen.target_experiment import load_target_experiment, bundles_match_descriptor
    desc = C.EXP / "target_experiment.yaml"          # C.EXP honors MERLIN_TARGET_EXPERIMENT
    te = load_target_experiment(desc) if desc.is_file() else None
    # Recover from a prior launch that left mode-000 trees behind, then keep them owner-only. This must
    # precede verification because the host-side anti-cheat and hidden grader need to enumerate the set.
    protected = []
    try:
        for path in _host_answer_surfaces(te):
            if path.exists():
                protected.append(f"{path} ({_make_host_owner_only(path)} entries)")
    except (OSError, RuntimeError) as exc:
        print(f"  ⚠ could not prepare host-only answer surfaces: {exc}")
        return 1
    print(f"  host-only answer surfaces (dirs 0700, files 0600): {protected or '(none present)'}")
    vnc = SCRIPTS / "verify_no_cheat.py"
    if not vnc.is_file():
        print(f"  ⚠ verify_no_cheat.py not found at {vnc} — build it before launch (#167).")
        return 1
    print(f"  running {vnc.name} ...")
    r = subprocess.run([sys.executable, str(vnc)], cwd=str(C.REPO))
    if r.returncode:
        return r.returncode
    # Descriptor governs the shared hardware spec: refuse if any active arm bundle drifted from it (so
    # every arm gets exactly the ISA/RTL the target_experiment.yaml declares — a fair, honest run).
    try:
        if te is not None:
            bundles = C.BUNDLES
            manifests = [bundles / ARMS[arm][4] / "input_bundle_manifest.yaml" for arm in ARMS
                         if (bundles / ARMS[arm][4] / "input_bundle_manifest.yaml").is_file()]
            drift = bundles_match_descriptor(te, manifests)
            if drift:
                print(f"  ⚠ bundles drifted from {desc.name}'s shared spec: {drift}")
                return 1
            print(f"  descriptor consistency: OK ({len(manifests)} active arm bundles vs {desc.name})")
    except Exception as e:  # noqa: BLE001 — a missing/invalid descriptor must not silently pass a run
        print(f"  ⚠ descriptor consistency check errored: {e}")
        return 1
    return 0


def _arm_env(arm: str, a, base_env: dict) -> dict:
    """Per-arm env. Realistic baseline = status-quo C++ OOT MLIR (PILOT_LANG=cpp); merlin arms = xDSL."""
    env = dict(base_env)
    if a.experiment == "realistic" and arm in ("baseline", "cpp_merlininfra"):
        env["PILOT_LANG"] = "cpp"   # both C++ arms author a C++ OOT package (status-quo vs infra-assisted)
    return env


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="batch label, e.g. abc1 -> rb_abc1 / merlin_abc1 / merlincirct_abc1")
    ap.add_argument("--arms", default="baseline,merlin,merlin_rtlchecks")
    ap.add_argument("--mode", choices=["parallel", "sequential"], default="parallel")
    # THE AGENT IS DECLARED, NOT INFERRED. These experiments run on the codex subscription seat, and
    # both defaults used to point elsewhere: `--model claude-opus-4-8` with `--driver auto` launched
    # Claude Code, and `auto` can NEVER resolve to codex -- it routes a non-Anthropic id to the Bedrock
    # Converse loop (metered, real money against the campaign ceiling) and everything else to the
    # `claude` CLI. So a batch launched without both flags silently measured a different agent, on a
    # different account, than every stored result it would be compared against.
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="high")
    # Agent driver + tier-within-agent models (Claude-Code-like). auto preserves route-by-model-id behavior.
    ap.add_argument("--driver", choices=["auto", "converse", "claudecode", "opencode", "codex"],
                    default="codex",
                    help="agent driver for every arm. Default codex (the subscription seat these "
                         "experiments run on). NOTE that `auto` cannot select codex: it routes by "
                         "model id to the Bedrock Converse loop or the claude CLI, so it is a way to "
                         "run a different agent than intended, not a way to pick this one.")
    ap.add_argument("--subagent-model", default="", help="delegate/subagent model (alias or Bedrock id)")
    ap.add_argument("--background-model", default="", help="background/mechanical model (alias or Bedrock id)")
    # Provider for the agent CLI (experiments-only; interactive Claude Code keeps the subscription).
    # For --provider bedrock, --model must be a Bedrock inference-profile id.
    ap.add_argument("--provider", choices=["subscription", "bedrock"], default="subscription")
    ap.add_argument("--aws-region", default="us-east-1", help="AWS region for --provider bedrock")
    ap.add_argument("--aws-profile", default="", help="AWS profile (~/.aws) for --provider bedrock")
    ap.add_argument("--schedule", choices=("rounds", "continuous"), default="rounds",
                    help="forwarded to every arm's driver. rounds (default) is unchanged; continuous "
                         "stops using the round COUNT as a terminator (see the driver's --schedule).")
    ap.add_argument("--plateau-rounds", type=int, default=None,
                    help="continuous only: forwarded to each arm's loop — stop when the best "
                         "score has not improved across this many rounds (0 disables). "
                         "Unset leaves the loop default, so a batch that omits it is unchanged.")
    ap.add_argument("--max-wall-s", type=int, default=0,
                    help="forwarded with --schedule continuous: per-arm ACTIVE wall budget (0 = none).")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--min-rounds", type=int, default=0,
                    help="Decline a READY_FOR_BARRIER self-declaration before round N while the run is "
                         "still failing (0 = disabled). Passed through to the arm driver.")
    ap.add_argument("--max-rate-limit-waits", type=int, default=8)
    ap.add_argument("--round-timeout", type=int, default=14400, help="per-round agent wall cap (s); large = effectively no timeout")
    ap.add_argument("--model-budget-s", type=int, default=None,
                    help="wall-clock ceiling for ONE whole-model capsule inside a round grade (s). "
                         "Unset = the driver's default; 0 = no ceiling.")
    ap.add_argument("--max-spend-usd", type=float, default=0.0,
                    help="batch DOLLAR ceiling across ALL arms (0=off). Each arm appends its per-round cost "
                         "to a shared ledger and stops before its next round once the total crosses this "
                         "(enforces the org spend ceiling in code, e.g. --max-spend-usd 300).")
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
                    help="make host answer surfaces owner-only + dry-run + assert cheat-clean BEFORE "
                         "any spend; bwrap masks them from the agent; launches NOTHING.")
    ap.add_argument("--with-tool", action="append", default=[], metavar="NAME",
                    help="ABLATION: grant this arm-gated tool on top of every launched arm's rung "
                         "(repeatable). Generate the matching bundles first with "
                         "`python -m merlin.targetgen.generate_bundles --with-tool NAME`.")
    ap.add_argument("--without-tool", action="append", default=[], metavar="NAME",
                    help="ABLATION: withhold this arm-gated tool from every launched arm (repeatable).")
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

    # SAY WHICH AGENT AND WHOSE ACCOUNT, BEFORE ANY SPEND. A batch is only comparable with stored
    # results if it ran the same agent on the same account, and both are decided by flags whose effect
    # is invisible in the run-id. Resolve them through the loop's own routing (never re-implemented
    # here) and print the answer, so a batch that would silently bill Bedrock or drive the claude CLI
    # is caught while reading the banner rather than afterwards in the artifacts.
    _agent = f"driver={a.driver} model={a.model}"
    try:
        sys.path.insert(0, str(C.EXP / "harness"))
        import run_baseline_qa_loop as _L
        _L._DRIVER = a.driver
        _agent = (f"driver={a.driver} -> resolved={_L._driver_for(a.model)} model={a.model} "
                  f"billing={_L._billing_mode(a.model)}")
    except Exception as _e:  # noqa: BLE001 — a banner must never block a launch
        _agent += f" (resolution unavailable: {type(_e).__name__})"
    print(f"=== A/B/C batch '{a.tag}' — arms={arms} mode={a.mode} {_agent} effort={a.effort} "
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
        print("\n=== --preflight: protect host surfaces + assert cheat-clean (launches NOTHING) ===")
        rc = _run_preflight()
        print(f"\n[preflight] {'PASS — safe to launch (drop --preflight, keep flags)' if rc == 0 else 'FAILED — DO NOT launch'}")
        return rc

    if a.dry_run:
        print(f"\n[dry-run] preflight OK ({a.mode}); nothing launched. Drop --dry-run to launch.")
        return 0

    # HARD pre-flight: verilator timing must be FRESH (readiness_check measured it on a known-good backend
    # AFTER the current sim binary was built). This is the abc7 safeguard — never launch a run whose
    # verilator timeout/availability was never actually verified. (Skip for --sandbox none legacy.)
    # It is target-conditional: the freshness gate guards a chipyard verilator sim, so a target whose RTL
    # tier is the mlc arc model (sim_via != "chipyard": atlas/npu_model, radiance, saturn …) has no
    # verilator binary to time against — the gate is N/A and would spuriously refuse an otherwise-ready run.
    if a.sandbox == "bwrap" and _sim_via() != "chipyard":
        # Non-chipyard RTL tiers split two ways: a pure mlc-arc target (atlas/npu/saturn) has NO verilator
        # binary to time — the gate is genuinely N/A — but a bespoke SIMT sim (cyclotron/radiance) DOES run
        # a verilator RTL cert at L3 and needs a TARGET-SCOPED T_obs (it must not inherit gemmini's, which
        # would floor to 900s and mass-timeout L3). Surface which case this is so an unverified L3 timing is
        # never silent (the driver's _verilator_per_capsule_timeout falls back to a conservative bound).
        _tsc = SCRIPTS / f".oracle_timing.{C.TARGET}.json"
        if _tsc.is_file():
            print(f"[pre-flight] L3 timing target-scoped & present: {_tsc.name} = {_tsc.read_text().strip()[:80]}")
        else:
            print(f"[pre-flight] verilator-timing gate: no target-scoped {_tsc.name} — if this target runs a "
                  f"verilator L3 cert, the driver will use a CONSERVATIVE timeout (run the L3 measurement / "
                  f"readiness to record it). N/A for pure mlc-arc targets.")
    elif a.sandbox == "bwrap":
        timing = SCRIPTS / ".oracle_timing.json"
        if not timing.is_file():
            print("REFUSING TO LAUNCH: scripts/.oracle_timing.json missing — run readiness_check.py "
                  "(it RUNS spike+verilator on the reference backend) first.", file=sys.stderr)
            return 3
        # Resolve the verilator binary the SAME way the sandbox and readiness_check do (.env chipyard),
        # and take the design name from the timing record readiness_check wrote — a literal path here
        # silently never matched, so the staleness half of this gate could not fire at all.
        _cfg = (json.loads(timing.read_text()) or {}).get("config")
        sim = (ext_path("chipyard") / "sims" / "verilator"
               / f"simulator-chipyard.harness-{_cfg}") if _cfg else None
        if sim is None or not sim.is_file():
            print(f"[pre-flight] verilator staleness check SKIPPED (no binary at {sim}) — the timing "
                  f"record exists but cannot be compared against a build.")
        elif timing.stat().st_mtime < sim.stat().st_mtime:
            print("REFUSING TO LAUNCH: .oracle_timing.json is STALE (older than the verilator binary) — "
                  "re-run readiness_check.py to re-measure.", file=sys.stderr)
            return 3
        print(f"[pre-flight] oracle timing fresh: {timing.read_text().strip()[:80]}")

    env = dict(os.environ)
    if acct:
        env["CLAUDE_CONFIG_DIR"] = acct
    if a.max_spend_usd and a.max_spend_usd > 0:
        # Enforce the batch dollar ceiling in code: all arms share one spend ledger (keyed by tag) and stop
        # before their next round once the running total crosses the cap. Bounded overshoot = one in-flight
        # round per arm. Consumed by run_baseline_qa_loop._spend_over_cap.
        env["MERLIN_MAX_SPEND_USD"] = str(a.max_spend_usd)
        env["MERLIN_SPEND_LEDGER"] = str(C.RUNS / f"ab_batch_{a.tag}.spend_ledger.jsonl")
        print(f"[cost-cap] batch ceiling ${a.max_spend_usd:.2f} across all arms; shared ledger "
              f"{env['MERLIN_SPEND_LEDGER']} (each arm stops before its next round once crossed).")
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
            prefix = "PILOT_LANG=cpp " if (a.experiment == "realistic" and arm in ("baseline", "cpp_merlininfra")) else ""
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
