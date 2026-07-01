#!/usr/bin/env python
"""Agent-facing self-check for the Muon backend — REDACTED grade on cyclotron.

Runs the agent's submission package through the parallel Muon capsule runner over the public corpus
on **cyclotron** (the fast iterate+cert oracle, ~1 s/kernel), and prints pass/fail + failure plane +
mismatch_count + cycles + %FP-peak. It NEVER prints expected/golden values — only whether you matched
and how fast. Iterate on this until every capsule passes, then make your %FP-peak as high as you can.

  python agent_selfcheck.py                      # grade submission/ over all public capsules
  python agent_selfcheck.py --capsule MG00_gemm_16x16x16
  python agent_selfcheck.py --submission submission --json   # machine-readable

This is a harness tool (not part of your graded package) — it may import Merlin; YOUR package may not.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# locate the in-repo merlin/python (harness side)
_HERE = Path(__file__).resolve().parent
for _c in (_HERE, *_HERE.parents):
    if (_c / "merlin" / "python").is_dir():
        sys.path.insert(0, str(_c / "merlin" / "python"))
        _REPO = _c
        break


def grade(submission: str, capsules_root: str, runs_root: str, timeout: int,
          only: str | None = None) -> dict:
    """Return a REDACTED verdict: per-capsule pass/fail + plane + mismatch_count + cycles + %peak."""
    from merlin.targetgen import muon_capsule_runner as MR
    caps = MR.discover_capsules(capsules_root, labels={"public", "dev"})
    if only:
        caps = [c for c in caps if c["name"] == only]
    per: list[dict] = []
    npass = 0
    pkg_fail = None
    for cap in caps:
        try:
            res = MR.run_capsule(cap, submission, runs_root=runs_root, run_id=cap["name"],
                                 timeout=timeout)
        except Exception as e:                       # package didn't even load/build
            pkg_fail = {"plane": "package", "detail": str(e)[:300]}
            per.append({"capsule": cap["name"], "status": "error", "fail_plane": "package"})
            continue
        st = res["status"]
        npass += int(st == "pass")
        l2 = res["tiers"].get("L2", {})
        fail = res.get("failure") or {}
        per.append({
            "capsule": cap["name"], "status": st,
            "fail_plane": fail.get("plane"), "fail_category": fail.get("category"),
            # mismatch_count is a COUNT (safe); expected/got values are never surfaced
            "mismatch_count": res.get("numeric", {}).get("mismatch_count"),
            "cycles": l2.get("cycles"), "pct_fp_peak": l2.get("pct_fp_peak"),
        })
    return {"all_pass": npass == len(caps) and len(caps) > 0,
            "n_passed": npass, "n_capsules": len(caps),
            "package_failure": pkg_fail, "per_capsule": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Muon backend self-check (redacted, cyclotron)")
    ap.add_argument("--submission", default="submission")
    ap.add_argument("--capsule", default=None, help="grade only this capsule")
    # prefer the capsules staged into the agent's workspace (./capsules); else the repo corpus
    _default_caps = "capsules" if Path("capsules").is_dir() else str(
        _REPO / "experiments/muon_perf_bench_v0/kernels")
    ap.add_argument("--capsules-root", default=_default_caps)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    runs = Path(tempfile.mkdtemp(prefix="muon_selfcheck_"))
    v = grade(a.submission, a.capsules_root, str(runs), a.timeout, only=a.capsule)
    if a.json:
        print(json.dumps(v, indent=2))
        return 0 if v["all_pass"] else 1
    print(f"\nMuon self-check (cyclotron, redacted) — {v['n_passed']}/{v['n_capsules']} pass")
    if v["package_failure"]:
        print(f"  PACKAGE ERROR: {v['package_failure']['detail']}")
    for r in v["per_capsule"]:
        extra = ""
        if r.get("cycles"):
            extra = f"  {r['cycles']} cyc  {r['pct_fp_peak']}% peak"
        plane = f"  [{r['fail_plane']}/{r.get('fail_category')}]" if r["status"] != "pass" else ""
        mc = f"  mismatch={r['mismatch_count']}" if r.get("mismatch_count") else ""
        print(f"  [{r['status']:10s}] {r['capsule']}{extra}{plane}{mc}")
    print("\nALL PASS — now push %FP-peak up." if v["all_pass"]
          else "\nNot all pass yet — fix the failing planes above.")
    return 0 if v["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
