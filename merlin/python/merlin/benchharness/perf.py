"""Target-parametric single-backend perf bench — the loop+report both benches cloned.

Runs each capsule in a target's corpus through ONE backend package and reports the target's perf
headline (from ``BenchTargetSpec.perf_fields``) per kernel + a markdown table. This is the shared core
for the simple single-backend case (muon; a single gemmini approach). Gemmini's bespoke 8-approach
cross-backend matrix + bare-metal golden-C arm are NOT modeled here (that would push target
conditionals into the shared driver) — that script keeps its matrix and can call this per-kernel.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .spec import BenchTargetSpec


def run_perf(spec: BenchTargetSpec, *, package: str, run_id: str, out_dir: Path, timeout: int,
             flops_fn: Callable[[dict], int | None] | None = None,
             extra_tier: str | None = None) -> dict:
    """Run every capsule through ``package`` and write ``perf_results.json`` + ``perf_table.md`` under
    ``out_dir``. Returns the summary dict. ``flops_fn`` optionally adds a per-kernel flop count;
    ``extra_tier`` optionally records a second tier's pass/fail (e.g. an L3 cert)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = spec.discover()
    if not caps:
        raise FileNotFoundError(f"no capsules under {spec.corpus_root}")
    runs_root = out_dir / "_capsule_runs"
    rows = []
    for cap in sorted(caps, key=lambda c: c["name"]):
        res = spec.runner.run_capsule(cap, package, runs_root=str(runs_root), run_id=cap["name"],
                                      contract=spec.contract, timeout=timeout)
        tier = res.get("tiers", {}).get(spec.perf_tier, {}) or {}
        row = {"kernel": cap["name"], "status": res.get("status"), "cycles": tier.get("cycles")}
        if flops_fn is not None:
            row["flops"] = flops_fn(cap)
        row.update(spec.perf_fields(tier))
        if extra_tier:
            row[f"{extra_tier.lower()}_cert"] = res.get("tiers", {}).get(extra_tier, {}).get("status")
        rows.append(row)

    summary = {"target": spec.name, "package": package, "run_id": run_id, "peak_note": spec.peak_note,
               "kernels": rows, "passed": sum(1 for r in rows if r["status"] == "pass"),
               "total": len(rows)}
    (out_dir / "perf_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "perf_table.md").write_text(perf_table(summary), encoding="utf-8")
    return summary


def perf_table(summary: dict) -> str:
    """Render a summary dict (from :func:`run_perf`) as a markdown table over its row keys."""
    rows = summary["kernels"]
    cols = list(dict.fromkeys(k for r in rows for k in r))  # stable union of keys, first-seen order
    lines = [f"# {summary['target']} perf bench — {summary['package']} ({summary['run_id']})", ""]
    if summary.get("peak_note"):
        lines += [f"Reported against {summary['peak_note']}.", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "-") if r.get(c) is not None else "-")
                                       for c in cols) + " |")
    lines += ["", f"**{summary['passed']}/{summary['total']} pass.**"]
    return "\n".join(lines)
