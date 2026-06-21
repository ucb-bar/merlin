"""Aggregate per-capsule results + decoded traces into a coverage report.

Produces a ``coverage.json`` (validated against ``coverage.schema.json``) and a Markdown
``isa_coverage_report.md`` with explicit "not covered" rows -- nothing is implied covered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# every Gemmini instruction class the bench knows about (so 0-counts are explicit "not covered")
ALL_CLASSES = ["CONFIG_EX", "CONFIG_LD", "CONFIG_ST", "MVIN", "MVOUT", "PRELOAD",
               "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE", "FLUSH", "FENCE", "LOOP_WS", "LOOP_CONV"]
ALL_MODES = ["i8", "relu", "acc_scale", "k_accumulate", "resident_reuse",
             "conv2d", "movement", "padded_edge"]
TIERS = ["L0", "L1", "L2", "L3", "L4", "L5"]


def aggregate(results: list[dict], capsules: list[dict] | None = None,
              traces: dict[str, dict] | None = None) -> dict:
    """Aggregate capsule_result dicts (+ optional capsules/traces) into a coverage dict."""
    capsules = capsules or []
    cap_by_name = {c["name"]: c for c in capsules}
    traces = traces or {}

    by_kind: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_tier_reached = {t: 0 for t in TIERS}
    mode_cov = {m: 0 for m in ALL_MODES}
    class_cov = {c: 0 for c in ALL_CLASSES}
    unavail = {"vcs": 0, "firesim": 0}

    for r in results:
        by_kind[r.get("kind", "unknown")] = by_kind.get(r.get("kind", "unknown"), 0) + 1
        by_label[r.get("label", "unknown")] = by_label.get(r.get("label", "unknown"), 0) + 1
        for t in TIERS:
            tr = r.get("tiers", {}).get(t)
            if tr and tr.get("status") == "pass":
                by_tier_reached[t] += 1
            if t == "L4" and tr and tr.get("status") == "unavailable":
                unavail["vcs"] += 1
            if t == "L5" and tr and tr.get("status") == "unavailable":
                unavail["firesim"] += 1
        # modes from the capsule's declared expected.modes (only count when the capsule passed)
        cap = cap_by_name.get(r["capsule"])
        if cap and r.get("status") == "pass":
            for m, on in (cap.get("expected", {}).get("modes", {}) or {}).items():
                if on and m in mode_cov:
                    mode_cov[m] += 1
        # instruction classes from the decoded trace (what the backend actually emitted)
        tr = traces.get(r["capsule"])
        if tr:
            for c in {i["class"] for i in tr.get("instructions", [])}:
                if c in class_cov:
                    class_cov[c] += 1

    return {
        "total": len(results),
        "by_kind": by_kind,
        "by_label": by_label,
        "by_tier_reached": by_tier_reached,
        "mode_coverage": mode_cov,
        "instruction_class_coverage": class_cov,
        "unavailable": unavail,
    }


def render_markdown(cov: dict, results: list[dict]) -> str:
    L = ["# ISA / capsule coverage report (capsule_bench_v0)", "",
         f"Total capsules: **{cov['total']}**  ·  by kind: {cov['by_kind']}  ·  "
         f"by label: {cov['by_label']}", "",
         "## Oracle tiers reached (passing)", "",
         "| tier | capsules passing |", "|---|---|"]
    for t in TIERS:
        L.append(f"| {t} | {cov['by_tier_reached'].get(t, 0)} |")
    L += ["", "## Instruction-class coverage (explicit not-covered rows)", "",
          "| class | capsules exercising |", "|---|---|"]
    for c in ALL_CLASSES:
        n = cov["instruction_class_coverage"].get(c, 0)
        mark = "" if n else "  _(not covered)_"
        L.append(f"| {c} | {n}{mark} |")
    L += ["", "## Mode coverage", "", "| mode | passing capsules |", "|---|---|"]
    for m in ALL_MODES:
        n = cov["mode_coverage"].get(m, 0)
        mark = "" if n else "  _(not covered)_"
        L.append(f"| {m} | {n}{mark} |")
    L += ["", "## Heavy-oracle availability (honest)", "",
          f"- VCS (L4) recorded unavailable on **{cov['unavailable']['vcs']}** capsules",
          f"- FireSim (L5) recorded unavailable on **{cov['unavailable']['firesim']}** capsules",
          "", "_Not-run is not pass: a mandatory tier recorded unavailable yields capsule "
          "status=incomplete, never pass._"]
    return "\n".join(L) + "\n"


def write(cov: dict, out_json: str | Path, out_md: str | Path | None = None,
          results: list[dict] | None = None, *, contract: str | Path | None = None) -> None:
    from .contract import schemas
    schemas.validate(cov, "coverage", contract=contract)
    Path(out_json).write_text(json.dumps(cov, indent=2), encoding="utf-8")
    if out_md:
        Path(out_md).write_text(render_markdown(cov, results or []), encoding="utf-8")
