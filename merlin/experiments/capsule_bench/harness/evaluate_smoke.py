"""Evaluate a completed capsule-bench run dir → a per-(target, arm) GO / ISSUES verdict.

This is the "evaluate" half of the cheap-first smoke→evaluate→fix loop: point it at one or more run dirs
and it reports, for each, whether the round was BOUNDED (no hang), GRADED (the oracle actually ran),
CLEAN (no answer access), and CONFORMANT for the arm (used the right contracts/info/tools + developed in
xDSL with no regex — the per-arm expectations that follow the experiment ladder). It reads only run
artifacts (qa_loop_summary.yaml, cost_time_toolcalls.yaml, the round transcript, the submission, the
grade), so it never spends and never needs the oracle.

Conformance is a FLAG, not a hard gate: a non-conformant run is reported loudly (and should be treated as
NOT a valid arm-N demonstration), but this tool does not alter the run's oracle grade. Per-arm expectations
are derived from the arm (from run_manifest.yaml), so arm-1/2 are held only to the build+self-check floor
while arm-3 adds xDSL/no-regex/isa-tools and arm-4 adds the RTL-facts derivation.

Usage:
    evaluate_smoke.py <run_dir> [<run_dir> ...]
    evaluate_smoke.py --roots out/runs/radiance out/runs/atlas   # newest run under each target's arms
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _load_yaml(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:  # noqa: BLE001 — a missing/garbled artifact is reported as absent, never a crash
        return {}


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _arm_of(run_dir: Path) -> str:
    """The arm this run exercised. The RTL-checks track (arm-4) reuses the merlin_assisted arm string but
    marks its runs with TRACK_RTLCHECKS + a ``merlincirct`` run-id stem, so detect it FIRST (else it would
    read back as arm-3 and skip the RTL-derivation conformance check)."""
    if (run_dir / "TRACK_RTLCHECKS").exists() or run_dir.name.startswith("merlincirct"):
        return "merlin_rtlchecks"
    m = _load_yaml(run_dir / "run_manifest.yaml")
    arm = m.get("arm") or m.get("bundle_arm") or ""
    if arm:
        return arm
    name = run_dir.name
    if name.startswith("merlin"):
        return "merlin_assisted"
    if name.startswith("rb") or "raw" in name:
        return "raw_baseline"
    return "unknown"


def _transcript_commands(run_dir: Path) -> list[str]:
    """Every Bash/tool command string in the round transcripts (best-effort; the driver merges the agent's
    session into transcript.jsonl at round end)."""
    cmds: list[str] = []
    files = [run_dir / "transcript.jsonl"] + sorted((run_dir / "rounds").glob("round_*.transcript.jsonl"))
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:  # noqa: BLE001
                cmds.append(line)                      # keep the raw line as a scannable fallback
                continue
            cmds.append(json.dumps(ev))                # scan the whole event blob for tool/command mentions
    return cmds


def _submission_sources(run_dir: Path) -> list[Path]:
    sub = run_dir / "submission"
    return [p for p in sub.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".md", ".yaml", ".mlir")]


def _conformance(run_dir: Path, arm: str) -> dict:
    """Per-arm dev-conformance: did the agent use the right contracts/info/tools + develop in xDSL w/o
    regex, as the arm's ladder rung requires? Best-effort from the transcript + submission source."""
    blob = "\n".join(_transcript_commands(run_dir))
    assisted = ("merlin_assisted" in arm) or ("rtlchecks" in arm)
    is_arm4 = "rtlchecks" in arm
    checks: dict = {}
    viol: list[str] = []
    # floor (all arms): consulted the contract + ran the self-check oracle loop
    checks["contracts_read"] = ("merlin/contract" in blob)
    checks["self_check_ran"] = ("agent_selfcheck.py" in blob) or ("agent_selfcheck" in blob)
    if not checks["self_check_ran"]:
        viol.append("never ran agent_selfcheck (the required oracle iteration loop)")
    if assisted:                                       # arm-3 / arm-4
        srcs = _submission_sources(run_dir)
        src_text = "\n".join(p.read_text(errors="ignore") for p in srcs if p.suffix == ".py")
        checks["xdsl_authored"] = ("xdsl" in src_text.lower()) or ("xdsl" in blob.lower())
        # NO regex in the agent's development (the cardinal dev rule) — a submission `import re` / re.compile
        checks["no_regex"] = not (("import re" in src_text) or ("re.compile" in src_text)
                                  or ("re.match" in src_text) or ("re.search" in src_text))
        checks["isa_tools_ran"] = ("isa_tools.py" in blob)
        if not checks["xdsl_authored"]:
            viol.append("no evidence the backend was authored in xDSL")
        if not checks["no_regex"]:
            viol.append("submission uses regex (import re) — prohibited for development")
        if not checks["isa_tools_ran"]:
            viol.append("never ran isa_tools disasm/lint on its own emitted artifact")
    if is_arm4:                                        # arm-4 only: genuine RTL derivation
        checks["rtl_derived"] = ("targetgen/rtl" in blob) or ("rtl_facts" in blob) or ("circt" in blob.lower())
        if not checks["rtl_derived"]:
            viol.append("arm-4 but no evidence it derived from the RTL facts / ran the CIRCT checks")
    return {"checks": checks, "violations": viol, "conformant": not viol}


def evaluate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    summ = _load_yaml(run_dir / "qa_loop_summary.yaml")
    rounds = summ.get("rounds") or [{}]
    r0 = rounds[0] if rounds else {}
    arm = _arm_of(run_dir)
    pub = _load_json(run_dir / "grading_public" / "score_capsule.json")
    rc = r0.get("agent_rc")
    wall = summ.get("wall_seconds") or (summ.get("timing") or {}).get("wall_seconds")
    n_caps = r0.get("n_capsules") or 0
    n_pass = r0.get("n_passed") or 0
    tool_calls = r0.get("tool_calls")
    bounded = (rc != 124)                              # 124 = round-timeout hit = the agent stalled / hung
    graded = bool(n_caps) and bool(pub)
    submission = (run_dir / "submission").is_dir() and any((run_dir / "submission").iterdir())
    clean = bool(r0.get("answer_access_clean", True)) and not (r0.get("audit_hits") or [])
    conf = _conformance(run_dir, arm)
    issues: list[str] = []
    if not bounded:
        issues.append(f"NOT BOUNDED — round timed out (rc=124, wall={wall}s); the agent stalled (hang class)")
    if not submission:
        issues.append("no submission produced")
    if not graded:
        issues.append(f"NOT GRADED — n_capsules={n_caps} (oracle did not grade a submission)")
    if not clean:
        issues.append("ANSWER ACCESS / audit hit — possible cheat, investigate")
    if not conf["conformant"]:
        issues += [f"non-conformant: {v}" for v in conf["violations"]]
    return {"run": str(run_dir), "arm": arm, "bounded": bounded, "submission": submission,
            "graded": graded, "n_capsules": n_caps, "n_passed": n_pass, "tool_calls": tool_calls,
            "clean": clean, "conformance": conf, "wall_seconds": wall,
            "verdict": "GO" if not issues else "ISSUES", "issues": issues}


def _latest_runs(root: Path) -> list[Path]:
    """Newest run dir under each arm subtree of a target root (out/runs/<target>/capsule-bench/<arm>/<id>)."""
    out: list[Path] = []
    base = root / "capsule-bench"
    if not base.is_dir():
        return out
    for arm_dir in base.iterdir():
        if not arm_dir.is_dir():
            continue
        runs = [d for d in arm_dir.iterdir() if d.is_dir() and (d / "qa_loop_summary.yaml").is_file()]
        if runs:
            out.append(max(runs, key=lambda d: (d / "qa_loop_summary.yaml").stat().st_mtime))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate capsule-bench run dirs -> GO/ISSUES per (target, arm).")
    ap.add_argument("run_dirs", nargs="*", help="explicit run dirs to evaluate")
    ap.add_argument("--roots", nargs="*", default=[], help="target roots; evaluate the newest run per arm")
    ap.add_argument("--json", action="store_true", help="emit the full verdict JSON")
    a = ap.parse_args(argv)
    dirs = [Path(d) for d in a.run_dirs]
    for r in a.roots:
        dirs += _latest_runs(Path(r))
    if not dirs:
        print("no run dirs given (pass run dirs or --roots)", file=sys.stderr)
        return 2
    results = [evaluate(d) for d in dirs]
    if a.json:
        print(json.dumps(results, indent=2))
        return 0
    print(f"{'VERDICT':8} {'ARM':22} {'bnd':4} {'grd':4} {'cnf':4} {'pass/caps':10} run")
    for v in results:
        c = v["conformance"]["conformant"]
        print(f"{v['verdict']:8} {v['arm']:22} {str(v['bounded'])[:1]:4} {str(v['graded'])[:1]:4} "
              f"{str(c)[:1]:4} {str(v['n_passed'])+'/'+str(v['n_capsules']):10} {v['run']}")
        for iss in v["issues"]:
            print(f"    - {iss}")
    n_go = sum(1 for v in results if v["verdict"] == "GO")
    print(f"\n{n_go}/{len(results)} GO")
    return 0 if n_go == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
