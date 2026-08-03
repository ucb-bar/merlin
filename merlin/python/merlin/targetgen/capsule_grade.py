"""Grade a submitted mlir_oot_target_backend package through the capsule_bench_v0 fidelity suite.

This turns capsule_bench from a self-test into a reusable GRADER: given any contract-satisfying
package directory, run the public/dev capsule suite (and, post-freeze, the hidden suite) through the
shared command-buffer/reference/oracle ladder, run the integrity scan, decode + check the RoCC trace,
and emit a single machine-readable ``score_capsule.json``.

Run this OUTSIDE the agent sandbox (it needs spike/verilator + the hidden capsules). The package is
still only invoked via its 4 CLI entrypoints (subprocess) — never imported.

Usage:
    python -m merlin.targetgen.capsule_grade --package <pkg_dir> \
        --capsules merlin/contract/capsules --runs-root <out> [--hidden] [--labels public,dev] \
        [--score <out.json>] [--no-oracle]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import capsule_runner as CR
from . import coverage_report as CV
from .oot_runner import CertFailure, build_package, integrity_scan, load_package


def grade(package_dir: str | Path, *, capsules_root: str | Path, runs_root: str | Path,
          labels: set[str] | None = None, contract: str | Path | None = None,
          oracle_adapters: dict | None = None, timeout: int = 900,
          max_workers: int = 1, target: str, no_oracle: bool = False) -> dict:
    """Run the capsule suite over a submitted package; return a score dict (also schema-checkable).

    ``max_workers > 1`` fans the per-capsule oracle runs out in parallel (verilator/VCS instances).

    ``no_oracle`` marks an EXPLICIT structure-only smoke (typically paired with ``oracle_adapters={}``):
    a capsule that clears the structural tiers but whose mandatory numeric tier was deliberately not run
    is recorded ``not_gradeable_no_oracle`` (a withheld numeric verdict), NOT the fixable
    ``oracle_unavailable`` plane — so the report is honest and never claims a numeric pass. Graded runs
    (``no_oracle=False``) keep the ``not_run_is_not_pass`` behavior byte-for-byte."""
    labels = labels or {"public", "dev"}
    pkg_dir = Path(package_dir)

    score: dict = {
        "task": f"{target}-mlir-oot-capsule", "package": str(pkg_dir),
        "integrity_exempt": None, "integrity_status": None,
        "labels_graded": sorted(labels),
        "functional_pass": 0, "n_capsules": 0, "n_passed": 0,
        "public_passed": None, "hidden_passed": None,
        "per_capsule": [], "tier_reached": {}, "first_failure_planes": {},
        "numeric_all_exact": None, "trace_all_pass": None,
        "cycles_diagnostic": {}, "highest_tier": None,
        "timing_diagnostic": {}, "timing_rollup": {},
    }

    # K0/K1: load + integrity scan + build (fail-closed, recorded honestly)
    try:
        pkg = load_package(pkg_dir, contract=contract)
        score["integrity_exempt"] = pkg.integrity_exempt
        integrity_scan(pkg)
        score["integrity_status"] = "clean" if not pkg.integrity_exempt else "exempt"
        build_package(pkg)
    except CertFailure as cf:
        score["integrity_status"] = f"FAIL[{cf.plane}]: {cf.detail[:200]}"
        score["functional_pass"] = 0
        score["failure"] = {"plane": cf.plane, "category": str(cf.category), "detail": cf.detail}
        return score

    caps = CR.discover_capsules(capsules_root, labels=labels, contract=contract)
    import time as _time
    _suite_t0 = _time.perf_counter()
    results = CR.run_suite(caps, pkg_dir, runs_root=runs_root, contract=contract,
                           oracle_adapters=oracle_adapters, timeout=timeout,
                           max_workers=max_workers, target=target, no_oracle=no_oracle)
    _suite_wall = _time.perf_counter() - _suite_t0

    # collect decoded traces for coverage — read from the TARGET's own suite dir (run_capsule writes
    # under cfg.suite, e.g. atlas-capsule-bench), not the gemmini SUITE literal (which left the atlas
    # coverage dict silently empty; same root cause as the self-check n_capsules:0 blind loop).
    traces: dict[str, dict] = {}
    rr = Path(runs_root) / "runs" / CR.suite_for(target)
    for cap in caps:
        tp = rr / cap["name"] / "generated" / "instruction_trace.json"
        if tp.exists():
            try:
                traces[cap["name"]] = json.loads(tp.read_text())
            except Exception:
                pass

    n_pass = sum(1 for r in results if r["status"] == "pass")
    score["n_capsules"] = len(results)
    score["n_passed"] = n_pass
    score["functional_pass"] = int(n_pass == len(results) and len(results) > 0)
    # Structure-only smoke bookkeeping (honest, never a numeric pass): a capsule is structurally clean
    # when it did not FAIL a structural tier — status `pass` OR `not_gradeable_no_oracle` (numeric verdict
    # withheld under --no-oracle). `gradeable` says whether this run had a numeric oracle at all.
    n_not_gradeable = sum(1 for r in results if r["status"] == "not_gradeable_no_oracle")
    # Fail-closed on an empty suite: if NO capsule matched the requested labels at this root, nothing was
    # graded. `all([])` is vacuously True, so numeric_all_exact / trace_all_pass would read as a phantom
    # pass and `gradeable` as True — the exact vacuous-pass trap that made a mis-rooted hidden phase
    # (n_capsules:0) look green. Report the boolean flags as null and gradeable False, never a pass.
    _empty = len(results) == 0
    score["gradeable"] = (not no_oracle) and not _empty
    score["n_not_gradeable_no_oracle"] = n_not_gradeable
    score["n_structural_pass"] = n_pass + n_not_gradeable
    score["structural_pass"] = bool(not _empty and (n_pass + n_not_gradeable) == len(results))
    score["numeric_all_exact"] = None if _empty else all(
        r.get("numeric", {}).get("status") == "pass" for r in results)
    score["trace_all_pass"] = None if _empty else all(
        r.get("trace_check", {}).get("status") == "pass" for r in results)
    if _empty:
        score["note"] = ("no capsules matched the requested labels at this root — nothing graded; "
                         "flags are null (not a pass). Check the capsules root / labels.")

    pub = [r for r in results if r.get("label") in ("public", "dev")]
    hid = [r for r in results if r.get("label") == "hidden"]
    if pub:
        score["public_passed"] = f"{sum(1 for r in pub if r['status']=='pass')}/{len(pub)}"
    if hid:
        score["hidden_passed"] = f"{sum(1 for r in hid if r['status']=='pass')}/{len(hid)}"

    tiers = ["L0", "L1", "L2", "L3", "L4", "L5"]
    for t in tiers:
        score["tier_reached"][t] = sum(
            1 for r in results if r.get("tiers", {}).get(t, {}).get("status") == "pass")
    for t in reversed(tiers):
        if score["tier_reached"][t] == len(results) and len(results) > 0:
            score["highest_tier"] = t
            break

    _agg = {"build_s": 0.0, "sim_active_s": 0.0, "oracle_wait_s": 0.0}
    for r in results:
        l3 = r.get("tiers", {}).get("L3", {})
        if l3.get("cycles") is not None:
            score["cycles_diagnostic"][r["capsule"]] = l3["cycles"]
        # active-vs-waiting timing: sum across every tier that actually ran an oracle for this capsule
        cap_tm = {"build_s": 0.0, "sim_active_s": 0.0, "oracle_wait_s": 0.0, "by_tier": {}}
        for t in ("L2", "L3", "L4", "L5"):
            tm = (r.get("tiers", {}).get(t, {}) or {}).get("timing")
            if tm:
                cap_tm["by_tier"][t] = tm
                for k in ("build_s", "sim_active_s", "oracle_wait_s"):
                    v = tm.get(k) or 0.0
                    cap_tm[k] += v
                    _agg[k] += v
        if cap_tm["by_tier"]:
            score["timing_diagnostic"][r["capsule"]] = {k: round(cap_tm[k], 3)
                                                         for k in ("build_s", "sim_active_s",
                                                                   "oracle_wait_s")}
        if r.get("failure"):
            p = r["failure"]["plane"]
            score["first_failure_planes"][p] = score["first_failure_planes"].get(p, 0) + 1
        score["per_capsule"].append({
            "capsule": r["capsule"], "label": r.get("label"), "status": r["status"],
            "numeric": r.get("numeric", {}).get("status"),
            "trace": r.get("trace_check", {}).get("status"),
            "tiers": {t: r.get("tiers", {}).get(t, {}).get("status") for t in tiers
                      if t in r.get("tiers", {})},
        })

    # active-vs-waiting rollup: wall is the suite wall-clock (overlapped under parallelism); the sum of
    # active_sim across capsules can exceed wall (that ratio IS the parallel speedup). oracle_wait_s is
    # time blocked on a queue/FPGA slot (≈0 for local spike/verilator).
    _active = _agg["build_s"] + _agg["sim_active_s"]
    score["timing_rollup"] = {
        "suite_wall_s": round(_suite_wall, 3),
        "build_s": round(_agg["build_s"], 3),
        "sim_active_s": round(_agg["sim_active_s"], 3),
        "active_total_s": round(_active, 3),
        "oracle_wait_s": round(_agg["oracle_wait_s"], 3),
        "max_workers": max_workers,
        "parallel_speedup": round(_active / _suite_wall, 2) if _suite_wall > 0 else None,
    }

    # coverage aggregate (written alongside)
    cov = CV.aggregate(results, capsules=caps, traces=traces)
    score["coverage"] = {"by_tier_reached": cov["by_tier_reached"],
                         "instruction_class_coverage": cov["instruction_class_coverage"],
                         "mode_coverage": cov["mode_coverage"], "unavailable": cov["unavailable"]}
    return score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade a backend package through capsule_bench_v0")
    ap.add_argument("--package", required=True)
    ap.add_argument("--capsules", default="merlin/contract/capsules")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--target", required=True, help="target being graded (its config/oracle are derived)")
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--hidden", action="store_true", help="grade ONLY hidden capsules (post-freeze)")
    ap.add_argument("--no-oracle", action="store_true", help="L0/L1/trace only (skip spike/verilator)")
    ap.add_argument("--score", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=1, help="parallel oracle instances (verilator/VCS)")
    a = ap.parse_args(argv)

    labels = {"hidden"} if a.hidden else set(a.labels.split(","))
    adapters = {} if a.no_oracle else None
    score = grade(a.package, capsules_root=a.capsules, runs_root=a.runs_root, labels=labels,
                  contract=a.contract, oracle_adapters=adapters, timeout=a.timeout,
                  max_workers=a.workers, target=a.target, no_oracle=a.no_oracle)
    out = Path(a.score) if a.score else Path(a.runs_root) / "score_capsule.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score, indent=2), encoding="utf-8")
    print(f"wrote {out}: functional_pass={score['functional_pass']} "
          f"passed={score['n_passed']}/{score['n_capsules']} "
          f"public={score.get('public_passed')} hidden={score.get('hidden_passed')} "
          f"highest_tier={score.get('highest_tier')} integrity={score.get('integrity_status')}")
    return 0 if score["functional_pass"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
