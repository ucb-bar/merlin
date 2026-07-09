"""Target-parametric REDACTED self-check — the agent-facing grade both benches cloned.

Runs a submission package through the target's capsule runner over the public corpus and returns a
verdict that says *whether* each capsule passed (+ failure plane/category, a mismatch COUNT, cycles,
and the target's perf headline) but NEVER expected/golden values. This is the single source for
`agent_selfcheck.grade()`; per-target scripts wrap it with a ``BenchTargetSpec``.
"""
from __future__ import annotations

from pathlib import Path

from .spec import BenchTargetSpec


def redacted_grade(spec: BenchTargetSpec, submission: str, runs_root: str, timeout: int,
                   *, only: str | None = None, corpus_root: str | None = None) -> dict:
    """Grade ``submission`` over ``spec``'s public corpus; return a redacted verdict dict.

    ``corpus_root`` overrides the spec's corpus (e.g. capsules staged into an agent workspace)."""
    runner = spec.runner
    root = corpus_root if corpus_root is not None else str(spec.corpus_root)
    caps = runner.discover_capsules(root, labels=spec.labels, contract=spec.contract)
    if only:
        caps = [c for c in caps if c["name"] == only]
    per: list[dict] = []
    npass = 0
    pkg_fail = None
    for cap in caps:
        try:
            res = runner.run_capsule(cap, submission, runs_root=runs_root, run_id=cap["name"],
                                     contract=spec.contract, timeout=timeout)
        except Exception as e:                       # package didn't even load/build
            pkg_fail = {"plane": "package", "detail": str(e)[:300]}
            per.append({"capsule": cap["name"], "status": "error", "fail_plane": "package"})
            continue
        st = res["status"]
        npass += int(st == "pass")
        tier = res.get("tiers", {}).get(spec.perf_tier, {}) or {}
        fail = res.get("failure") or {}
        row = {
            "capsule": cap["name"], "status": st,
            "fail_plane": fail.get("plane"), "fail_category": fail.get("category"),
            # mismatch_count is a COUNT (safe); expected/got values are never surfaced
            "mismatch_count": (res.get("numeric") or {}).get("mismatch_count"),
            "cycles": tier.get("cycles"),
        }
        row.update(spec.perf_fields(tier))
        per.append(row)
    return {"all_pass": npass == len(caps) and len(caps) > 0,
            "n_passed": npass, "n_capsules": len(caps),
            "package_failure": pkg_fail, "per_capsule": per}


def print_verdict(spec: BenchTargetSpec, v: dict, *, perf_key: str | None = None,
                  perf_suffix: str = "") -> None:
    """Human-readable redacted print of a :func:`redacted_grade` verdict."""
    print(f"\n{spec.name} self-check (redacted) — {v['n_passed']}/{v['n_capsules']} pass")
    if v.get("package_failure"):
        print(f"  PACKAGE ERROR: {v['package_failure']['detail']}")
    for r in v["per_capsule"]:
        extra = ""
        if r.get("cycles"):
            perf = f"  {r.get(perf_key)}{perf_suffix}" if perf_key and r.get(perf_key) is not None else ""
            extra = f"  {r['cycles']} cyc{perf}"
        plane = f"  [{r['fail_plane']}/{r.get('fail_category')}]" if r["status"] != "pass" else ""
        mc = f"  mismatch={r['mismatch_count']}" if r.get("mismatch_count") else ""
        print(f"  [{r['status']:10s}] {r['capsule']}{extra}{plane}{mc}")
    print("\nALL PASS." if v["all_pass"] else "\nNot all pass yet — fix the failing planes above.")
