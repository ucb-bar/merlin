"""Aggregate per-capsule results + decoded traces into a coverage report.

Produces a ``coverage.json`` (validated against ``coverage.schema.json``) and a Markdown
``isa_coverage_report.md`` with explicit "not covered" rows -- nothing is implied covered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# LAST-RESORT reference vocabulary (the RoCC/systolic instruction classes + gemmini corpus mode tags),
# used ONLY when a target's ISA vocabulary cannot be derived AND no trace was observed — so a bare
# no-target aggregate call is still meaningful. A real graded run derives the class universe from the
# TARGET's own ISA + the decoded traces (see _isa_class_vocabulary / aggregate), so atlas MXU classes and
# radiance SIMT classes are counted, never silently dropped against a gemmini-shaped list.
_FALLBACK_CLASSES = ["CONFIG_EX", "CONFIG_LD", "CONFIG_ST", "MVIN", "MVOUT", "PRELOAD",
                     "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE", "FLUSH", "FENCE", "LOOP_WS", "LOOP_CONV"]
_FALLBACK_MODES = ["i8", "relu", "acc_scale", "k_accumulate", "resident_reuse",
                   "conv2d", "movement", "padded_edge"]
TIERS = ["L0", "L1", "L2", "L3", "L4", "L5"]


def _isa_class_vocabulary(target: str | None) -> list[str]:
    """The target's OWN ISA class names, DERIVED from its capability manifest — a self-hosted-ISA/SIMT
    target's decoded-instruction classes (contract ``interfaces[].instruction_classes``), or a
    RoCC/systolic target's ``encoding.semantic_class`` + ``config_subtype`` names (the config op is
    refined into its subtypes in the decoded trace). Returns ``[]`` when no manifest resolves — the
    caller then relies on the observed-trace union + the reference fallback. Never a target-name branch."""
    if not target:
        return []
    try:
        from .target_experiment import load_capability_manifest
        m = load_capability_manifest(target)
    except Exception:  # noqa: BLE001 — no resolvable manifest -> rely on observed traces
        return []
    out: list[str] = []
    for itf in (m.contract.get("interfaces") or []):     # self-hosted ISA / SIMT decoded classes
        out += list(itf.get("instruction_classes") or [])
    enc = m.encoding or {}
    sc = enc.get("semantic_class") or {}
    cst = enc.get("config_subtype") or {}
    for name in sc.values():                              # RoCC/systolic semantic classes
        if cst and str(name).upper() == "CONFIG":        # replaced by its subtypes in the trace
            continue
        out.append(name)
    out += list(cst.values())
    return [c for c in dict.fromkeys(out) if c]


def aggregate(results: list[dict], capsules: list[dict] | None = None,
              traces: dict[str, dict] | None = None, *, target: str | None = None) -> dict:
    """Aggregate capsule_result dicts (+ optional capsules/traces) into a coverage dict.

    The instruction-class universe is DERIVED from ``target``'s own ISA unioned with the classes the
    decoded traces actually exercised; the mode universe is the union of the modes the graded capsules
    declare. So the not-covered rows reflect THIS target's vocabulary, not a hardcoded gemmini list."""
    capsules = capsules or []
    cap_by_name = {c["name"]: c for c in capsules}
    traces = traces or {}

    universe_classes = set(_isa_class_vocabulary(target))
    for _tr in traces.values():
        universe_classes |= {i.get("class") for i in (_tr.get("instructions") or []) if i.get("class")}
    universe_modes: set[str] = set()
    for _c in capsules:
        universe_modes |= {m for m, on in ((_c.get("expected") or {}).get("modes") or {}).items() if on}
    classes = sorted(universe_classes) or list(_FALLBACK_CLASSES)
    modes = sorted(universe_modes) or list(_FALLBACK_MODES)

    by_kind: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_tier_reached = {t: 0 for t in TIERS}
    mode_cov = {m: 0 for m in modes}
    class_cov = {c: 0 for c in classes}
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
    for c, n in cov["instruction_class_coverage"].items():
        mark = "" if n else "  _(not covered)_"
        L.append(f"| {c} | {n}{mark} |")
    L += ["", "## Mode coverage", "", "| mode | passing capsules |", "|---|---|"]
    for m, n in cov["mode_coverage"].items():
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
