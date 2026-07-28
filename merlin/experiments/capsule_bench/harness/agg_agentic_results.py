"""Aggregate the 3-approach agentic A/B/C (baseline · merlin · merlin+CIRCT) into a tidy
agentic_results.json for plotting.

Arms are keyed by the run's `environment.yaml::bundle_id` (NOT the run dir — both merlin arms live under
runs/merlin_assisted/):
  raw_baseline_public_v0            -> baseline
  merlin_assisted_public_v0         -> merlin
  merlin_assisted_rtlchecks_public_v0 -> merlin_rtlchecks   (the CIRCT-checks arm)

Per run it surfaces BOTH dimensions:
  • authoring effort  — cost/tokens/tool-calls/wall + per-round n_passed (from cost_time_toolcalls.yaml +
                        qa_loop_summary.yaml)
  • dialect completeness — passed X/25, public/hidden split, first-failure planes (from the full-suite
                        audit reports/full_suite_audit.json::backends[run_id], if it has been run)
Everything is read from artifacts already on disk (no agent re-run). Honest about scale: each run is marked
valid/invalid and N is recorded. -> reports/agentic_results.json
"""
from __future__ import annotations
import json
from pathlib import Path
import yaml

def _repo_root():
    from pathlib import Path as _P
    p = _P(__file__).resolve()
    while p != p.parent and not (p / "merlin" / "python").is_dir():
        p = p.parent
    return p
_ROOT = _repo_root()

EXP = Path(f"{_ROOT}/merlin/experiments/capsule_bench/targets/gemmini")
REPORTS = EXP.parents[2] / "artifacts" / "capsule-bench" / "gemmini"
RUN_DIRS = ["raw_baseline", "merlin_assisted"]   # both scanned; arm decided by bundle_id
BUNDLE_ARM = {
    "raw_baseline_public_v0": "baseline",
    "merlin_assisted_public_v0": "merlin",
    "merlin_assisted_rtlchecks_public_v0": "merlin_rtlchecks",
}
ARM_ORDER = ["baseline", "merlin", "merlin_rtlchecks"]


def _arm_of(d: Path) -> str | None:
    env = d / "environment.yaml"
    if env.is_file():
        bid = (yaml.safe_load(env.read_text()) or {}).get("bundle_id")
        if bid in BUNDLE_ARM:
            return BUNDLE_ARM[bid]
    # fallback for older runs without environment.yaml bundle_id
    bid = None
    man = d / "input_bundle_manifest.yaml"
    if man.is_file():
        bid = (yaml.safe_load(man.read_text()) or {}).get("bundle_id")
    if bid in BUNDLE_ARM:
        return BUNDLE_ARM[bid]
    if (d / "TRACK_RTLCHECKS").exists():
        return "merlin_rtlchecks"
    if d.parent.name == "raw_baseline":
        return "baseline"
    if d.parent.name == "merlin_assisted":
        return "merlin"
    return None


def _completeness(run_id: str, audit: dict) -> dict | None:
    bk = (audit.get("backends") or {}).get(run_id)
    if not bk:
        return None
    def _frac(s):
        try:
            n, t = str(s).split("/"); return {"passed": int(n), "total": int(t)}
        except Exception:
            return {"passed": None, "total": None}
    return {"all": _frac(bk.get("passed")), "public": _frac(bk.get("public_passed")),
            "hidden": _frac(bk.get("hidden_passed")),
            "first_failure_planes": bk.get("first_failure_planes"), "language": bk.get("language")}


def load_run(d: Path, audit: dict) -> dict | None:
    ct = d / "cost_time_toolcalls.yaml"
    if not ct.is_file():
        return None
    c = yaml.safe_load(ct.read_text()) or {}
    qa = d / "qa_loop_summary.yaml"
    q = yaml.safe_load(qa.read_text()) if qa.is_file() else {}
    rounds = [{"round": r.get("round"), "n_passed": r.get("n_passed"), "tool_calls": r.get("tool_calls"),
               "all_pass": r.get("all_pass")} for r in (q.get("rounds") or [])]
    man = {}
    mp = d / "run_manifest.yaml"
    if mp.is_file():
        man = yaml.safe_load(mp.read_text()) or {}
    converged = bool(q.get("converged"))
    valid = converged and c.get("available", True) and (c.get("wall_time_seconds", 0) > 60)
    return {
        "run_id": d.name, "valid": valid, "converged": converged,
        "wall_s": c.get("wall_time_seconds"), "cost_usd": c.get("estimated_cost_usd"),
        "tokens_total": c.get("tokens_total"), "tokens_input": c.get("tokens_input"),
        "tokens_cached": c.get("tokens_cached"), "tokens_output": c.get("tokens_output"),
        "tool_calls": c.get("tool_calls"), "thinking_blocks": c.get("thinking_blocks"),
        "n_rounds": q.get("n_rounds", len(rounds)), "rounds": rounds,
        "public_pass": man.get("public_dev_pass") or man.get("pass_public"),
        "hidden_pass": man.get("hidden_pass") or man.get("pass_hidden"),
        "fullsuite": _completeness(d.name, audit),
    }


def main():
    fa = REPORTS / "full_suite_audit.json"
    audit = json.loads(fa.read_text()) if fa.is_file() else {}
    out = {"arms": {a: [] for a in ARM_ORDER}, "n_valid": {}, "arm_order": ARM_ORDER}
    for sub in RUN_DIRS:
        base = EXP / "runs" / sub
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            arm = _arm_of(d)
            if arm is None:
                continue
            r = load_run(d, audit)
            if r:
                out["arms"][arm].append(r)
    for a in ARM_ORDER:
        out["n_valid"][a] = sum(1 for r in out["arms"][a] if r["valid"])
    if audit:
        out["coverage"] = {"class_coverage": audit.get("class_coverage"),
                           "n_capsules": audit.get("n_capsules"), "backends": list((audit.get("backends") or {}))}
    out["caveat"] = ("3-arm A/B/C. valid converged runs: " +
                     ", ".join(f"{a}={out['n_valid'][a]}" for a in ARM_ORDER) +
                     ". full-suite completeness present where full_suite_audit has been run.")
    p = REPORTS / "agentic_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    for a in ARM_ORDER:
        vs = [r for r in out["arms"][a] if r["valid"]]
        line = ", ".join(
            f"{r['run_id']}(${(r['cost_usd'] or 0):.0f},{r['n_rounds']}rd,{r['tool_calls']}tc" +
            (f",{r['fullsuite']['all']['passed']}/{r['fullsuite']['all']['total']}" if r.get('fullsuite') else "") + ")"
            for r in vs)
        print(f"  {a}: {len(vs)} valid -> {line or '(none yet)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
