"""Aggregate per-capsule results + decoded traces into a coverage report.

Produces a ``coverage.json`` (validated against ``coverage.schema.json``) and a Markdown
``isa_coverage_report.md`` with explicit "not covered" rows -- nothing is implied covered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# BASELINE axes: always reported, so a 0 is an explicit "not covered" row rather than an absent one.
# These are gemmini's (the ISA the bench was written against) and they are NOT the vocabulary — a
# self-hosted-ISA or command-buffer target names its classes and modes differently, and a corpus may
# declare a mode no gemmini capsule has (radiance's `rmsnorm`). The axes actually reported are these
# UNIONED with what the capsules declare and the traces contain; see `_axes`. Counting only the baseline
# is how a mode could be declared, graded, and silently absent from its own coverage report.
BASELINE_CLASSES = ["CONFIG_EX", "CONFIG_LD", "CONFIG_ST", "MVIN", "MVOUT", "PRELOAD",
                    "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE", "FLUSH", "FENCE", "LOOP_WS", "LOOP_CONV"]
BASELINE_MODES = ["i8", "relu", "acc_scale", "k_accumulate", "resident_reuse",
                  "conv2d", "movement", "padded_edge"]
#: Back-compat aliases for the baseline sets (their former names).
ALL_CLASSES = BASELINE_CLASSES
ALL_MODES = BASELINE_MODES
TIERS = ["L0", "L1", "L2", "L3", "L4", "L5"]


def _axes(baseline: list[str], observed) -> list[str]:
    """Baseline axes first (stable report order), then anything else observed, sorted."""
    extra = sorted(set(observed) - set(baseline))
    return [*baseline, *extra]


def aggregate(results: list[dict], capsules: list[dict] | None = None,
              traces: dict[str, dict] | None = None) -> dict:
    """Aggregate capsule_result dicts (+ optional capsules/traces) into a coverage dict."""
    capsules = capsules or []
    cap_by_name = {c["name"]: c for c in capsules}
    traces = traces or {}

    by_kind: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_tier_reached = {t: 0 for t in TIERS}
    declared_modes = {m for c in capsules
                      for m in ((c.get("expected") or {}).get("modes") or {})}
    traced_classes = {i.get("class") for tr in traces.values()
                      for i in (tr.get("instructions") or []) if i.get("class")}
    mode_cov = {m: 0 for m in _axes(BASELINE_MODES, declared_modes)}
    class_cov = {c: 0 for c in _axes(BASELINE_CLASSES, traced_classes)}
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
    # Iterate the AGGREGATE's own axes, not the baseline list: a class or mode this corpus contributed
    # is in the counts, and rendering only the baseline would drop it from the report it belongs to.
    for c in _axes(BASELINE_CLASSES, cov["instruction_class_coverage"]):
        n = cov["instruction_class_coverage"].get(c, 0)
        mark = "" if n else "  _(not covered)_"
        L.append(f"| {c} | {n}{mark} |")
    L += ["", "## Mode coverage", "", "| mode | passing capsules |", "|---|---|"]
    for m in _axes(BASELINE_MODES, cov.get("mode_coverage") or {}):
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
