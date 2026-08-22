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
# UNIONED with the target's own ISA (see `_isa_class_vocabulary`), what the capsules declare, and what
# the traces contain; see `_axes` and `aggregate`. Counting only the baseline is how a mode could be
# declared, graded, and silently absent from its own coverage report.
BASELINE_CLASSES = ["CONFIG_EX", "CONFIG_LD", "CONFIG_ST", "MVIN", "MVOUT", "PRELOAD",
                    "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE", "FLUSH", "FENCE", "LOOP_WS", "LOOP_CONV"]
BASELINE_MODES = ["i8", "relu", "acc_scale", "k_accumulate", "resident_reuse",
                  "conv2d", "movement", "padded_edge"]
#: Back-compat aliases for the baseline sets (their former names).
ALL_CLASSES = BASELINE_CLASSES
ALL_MODES = BASELINE_MODES
TIERS = ["L0", "L1", "L2", "L3", "L4", "L5"]


def _ratio(num: int, den: int):
    """num/den, or ``None`` when there is no eligible work to recall."""
    return (num / den) if den else None


def _capsule_region(cap: dict):
    """A :class:`merlin.targetgen.eligibility.RegionDescriptor` for a capsule spec — its declared
    ``semantic.semantic_family`` (or, when omitted, the family derived from ``operation.op``) at the
    dtypes, rank and layout of its ACTUAL OPERANDS.

    ⚠️ The operand dtypes come from ``inputs[]`` selected by ``role``, NOT from ``numeric_policy.dtype``.
    That field is the ACCUMULATOR/compare dtype, and reading it as the operand dtype silently disabled
    this whole metric: a gemmini matmul presented ``i32`` against a contract declaring ``int8``, so EVERY
    gemmini capsule was judged ineligible, ``n_eligible`` was 0 and ARR reported ``None`` — the target's
    core capability excluded from its own denominator while an incidental ``movement`` capsule counted.
    Radiance only escaped by coincidence (it accumulates in f32 and also declares fp32 as an operand
    format), so its fp16 and MX capsules were being judged as f32/bf16 and the dtype axis was unenforced
    on every target. ``inputs[]`` is also the source
    :func:`merlin.targetgen.capsule_golden._recompute_golden` picks operands from, so the eligibility
    oracle and the golden engine now agree on what the operands are.

    ``rank``/``batch``/``layout`` are populated for the same reason: :class:`SemanticCapability` declares
    ``ranks``, ``batch`` and ``layouts``, and a descriptor that leaves them unset can never exercise them
    — gemmini's ``ranks: [2]`` was dead on this path.
    """
    from . import eligibility as _el
    from . import semantic_families as _sf

    sem = cap.get("semantic") or {}
    op = (cap.get("operation") or {}).get("op")
    fam = sem.get("semantic_family") or _sf.from_op(op)

    def _by_role(role: str) -> dict:
        for spec in cap.get("inputs") or []:
            if spec.get("role") == role:
                return spec
        return {}

    act, wgt = _by_role("input"), _by_role("weight")
    shape = act.get("shape") or []
    rank = len(shape) or None
    # A leading extent on a rank>=3 operand is the batch. Rank 1/2 operands are unbatched by definition,
    # and reporting batch=1 there is a fact, not a default.
    batch = int(shape[0]) if rank and rank >= 3 else 1
    return _el.RegionDescriptor(source=cap.get("name", ""), op=op, family=fam,
                                in_dtype=act.get("dtype"), weight_dtype=wgt.get("dtype"),
                                rank=rank, batch=batch, layout=act.get("layout"))


def _acceleratable_coverage(results: list[dict], cap_by_name: dict, target: str | None) -> dict:
    """Suite-level Acceleratable Region Recall: treat each capsule as one region, score the compiler's
    outcome (accelerated == passed on a real emitted-artifact / simulator tier) against the INDEPENDENT
    eligibility oracle (the hardware's declared semantic capability). An eligible capsule that did not
    accelerate is a ``false_fallback`` — the emit-layer gap ARR exists to surface. An empty capability
    map (target declares no ``semantic_capabilities`` yet) yields an honest all-ineligible, recall=None
    result rather than a fake 1.0."""
    from . import eligibility as _el
    from .capsule_runner import _TIER_SIM

    sim_tiers = tuple(_TIER_SIM)                      # tiers that run the emitted artifact on a simulator
    cap_map: dict = {}
    undetermined: frozenset = frozenset()
    if target:
        try:
            cap_map = _el.capability_map_for_target(target)
        except Exception:  # noqa: BLE001 — no resolvable contract -> honest empty denominator
            cap_map = {}
        undetermined = _el.undetermined_families_for_target(target)

    # Families this target can run ONLY fused behind another (a mesh whose only elementwise hardware is
    # the accumulator readout epilogue). Such a family is real hardware, but a STANDALONE region of it is
    # correctly ineligible -- so it silently never enters the denominator, and a reader who sees a high
    # recall assumes a coverage the number does not claim. Derived from the capability map, never named.
    fused_only = sorted(f for f, c in cap_map.items() if getattr(c, "composed_with", ()))
    n_fused_only_ineligible = 0
    by_family: dict[str, dict[str, int]] = {}

    per_capsule: list[dict] = []
    n_eligible = n_eligible_accelerated = n_accelerated = n_accel_eligible = 0
    n_undetermined = 0
    n_unclassified = 0
    unclassified_capsules: list[str] = []
    undetermined_capsules: list[str] = []
    false_fallback: list[str] = []
    must_accelerate_violations: list[str] = []
    # G0-G5 breakdown: recall per generalization axis (the axis each capsule PROBES)
    by_axis: dict[str, dict[str, int]] = {}
    for r in results:
        cap = cap_by_name.get(r["capsule"])
        if not cap:
            continue
        desc = _capsule_region(cap)
        sem = cap.get("semantic") or {}
        override = sem.get("eligible", "auto")
        if isinstance(override, bool):
            eligible, family, reason = override, desc.resolved_family(), "author override"
            is_undetermined = False
        else:
            v = _el.is_eligible(desc, cap_map, undetermined=undetermined)
            eligible, family, reason = v.eligible, v.family, v.reason
            is_undetermined = v.undetermined
        accelerated = any(r.get("tiers", {}).get(t, {}).get("status") == "pass" for t in sim_tiers)
        must = bool(sem.get("must_accelerate"))
        axis = sem.get("generalization_axis", "unspecified")
        if eligible:
            b = by_axis.setdefault(axis, {"n_eligible": 0, "n_eligible_accelerated": 0})
            b["n_eligible"] += 1
            if accelerated:
                b["n_eligible_accelerated"] += 1
        # must_accelerate contract: an ELIGIBLE region declared must_accelerate that did NOT reach the
        # accelerator is a violation — the fallback escape hatch cannot hide an emit-layer gap. An
        # ineligible region (or one with must_accelerate unset / fallback_allowed) legitimately falls back.
        violated = must and eligible and not accelerated
        per_capsule.append({"capsule": r["capsule"], "semantic_family": family, "eligible": eligible,
                            "accelerated": accelerated, "must_accelerate": must,
                            "must_accelerate_violated": violated, "reason": reason,
                            "undetermined": is_undetermined})
        if is_undetermined:
            n_undetermined += 1
            undetermined_capsules.append(r["capsule"])
        # UNCLASSIFIED is a THIRD state, distinct from both. `undetermined` means the evidence could not
        # decide whether this target supports the family; `unclassified` means OUR taxonomy could not name
        # the family at all, so eligibility failed closed on an op it has no entry for. Folding the latter
        # into the ineligible bucket makes "we do not have a word for this" read as "the hardware cannot
        # do it" -- the same defect in the denominator that the undetermined bucket already exists to
        # prevent. Reported separately; it is a gap in the vocabulary, not a fact about the device.
        if family is None:
            n_unclassified += 1
            unclassified_capsules.append(r["capsule"])
        # Per-family denominator, so "ARR = 1.000" can be read as the coverage it actually asserts
        # rather than as coverage of everything the target can do.
        fb = by_family.setdefault(family or "unclassified",
                                  {"n_regions": 0, "n_eligible": 0, "n_eligible_accelerated": 0})
        fb["n_regions"] += 1
        if eligible:
            fb["n_eligible"] += 1
            if accelerated:
                fb["n_eligible_accelerated"] += 1
        # Ineligible SPECIFICALLY because the family is fused-only -- decided structurally from the
        # capability, never by matching the verdict's prose.
        if not eligible and family in cap_map and getattr(cap_map[family], "composed_with", ()):
            n_fused_only_ineligible += 1
        if violated:
            must_accelerate_violations.append(r["capsule"])
        if accelerated:
            n_accelerated += 1
            if eligible:
                n_accel_eligible += 1
        if eligible:
            n_eligible += 1
            if accelerated:
                n_eligible_accelerated += 1
            else:
                false_fallback.append(r["capsule"])

    by_generalization_axis = {
        axis: {**b, "recall": _ratio(b["n_eligible_accelerated"], b["n_eligible"])}
        for axis, b in sorted(by_axis.items())
    }

    return {
        "denominator_source": "semantic_capabilities (independent eligibility oracle)",
        "n_eligible": n_eligible,
        # Regions whose family no evidence source could decide. In NEITHER the numerator nor the
        # denominator: scoring them either way would move ARR for a reason that is about our evidence,
        # not about the compiler. A large count means the recall below is computed over a small slice
        # of the corpus and should not be quoted alone -- the gate bounds it.
        "n_undetermined": n_undetermined,
        "undetermined_capsules": undetermined_capsules,
        # Regions our own taxonomy could not NAME (eligibility failed closed on an op with no family
        # entry). Reported apart from `undetermined` because the two have opposite owners: an
        # undetermined family is a gap in the target's evidence, an unclassified one is a gap in our
        # vocabulary. Neither is a statement that the hardware cannot do the work, and folding either
        # into the ineligible count would make it read as exactly that.
        "n_unclassified": n_unclassified,
        "unclassified_capsules": unclassified_capsules,
        "n_accelerated": n_accelerated,
        "n_eligible_accelerated": n_eligible_accelerated,
        # Which families the recall was actually computed over. A target may declare three families and
        # have only two of them reachable standalone, in which case the headline ratio is a statement
        # about those two and nothing else.
        "by_family": {f: {**b, "recall": _ratio(b["n_eligible_accelerated"], b["n_eligible"])}
                      for f, b in sorted(by_family.items())},
        "fused_only_families": fused_only,
        "n_fused_only_ineligible": n_fused_only_ineligible,
        # Families the target DECLARES and no capsule exercises. Distinct from fused_only, where the
        # exclusion is a hardware fact: here the hardware claim stands and the CORPUS is silent, so the
        # recall says nothing about them either way. Unnamed, a reader reasonably assumes a declared
        # family was measured -- which is how "7 of 8 families covered" and "the compiler handles 8
        # families" become the same sentence to someone reading quickly.
        "declared_unexercised_families": sorted(set(cap_map) - set(by_family)),
        "false_fallback": false_fallback,
        "must_accelerate_violations": must_accelerate_violations,
        "must_accelerate_pass": not must_accelerate_violations,
        "acceleratable_region_recall": _ratio(n_eligible_accelerated, n_eligible),
        "acceleration_precision": _ratio(n_accel_eligible, n_accelerated),
        "by_generalization_axis": by_generalization_axis,
        "per_capsule": per_capsule,
    }


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
def _axes(baseline: list[str], observed) -> list[str]:
    """Baseline axes first (stable report order), then anything else observed, sorted."""
    extra = sorted(set(observed) - set(baseline))
    return [*baseline, *extra]


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

    by_kind: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_tier_reached = {t: 0 for t in TIERS}
    # BASELINE axes unioned with what this target's ISA/traces/capsules exercised, so a corpus-declared
    # mode or a self-hosted-ISA class is reported (not silently filtered on the gemmini baseline) while a
    # baseline axis still shows an explicit not-covered 0.
    mode_cov = {m: 0 for m in _axes(BASELINE_MODES, universe_modes)}
    class_cov = {c: 0 for c in _axes(BASELINE_CLASSES, universe_classes)}
    # Heavy-oracle availability is tracked per heavy oracle tier; the substrate NAME for each tier is
    # DERIVED from the canonical tier->simulator map (single source of truth in capsule_runner), never
    # hardcoded as vcs/firesim here — so a target whose ladder names its heavy oracles differently is
    # counted under its own substrate labels.
    from .capsule_runner import _TIER_SIM
    heavy_tiers = tuple(t for t in ("L4", "L5") if t in _TIER_SIM)
    unavail = {_TIER_SIM[t]: 0 for t in heavy_tiers}

    for r in results:
        by_kind[r.get("kind", "unknown")] = by_kind.get(r.get("kind", "unknown"), 0) + 1
        by_label[r.get("label", "unknown")] = by_label.get(r.get("label", "unknown"), 0) + 1
        for t in TIERS:
            tr = r.get("tiers", {}).get(t)
            if tr and tr.get("status") == "pass":
                by_tier_reached[t] += 1
            if t in heavy_tiers and tr and tr.get("status") == "unavailable":
                unavail[_TIER_SIM[t]] += 1
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
        "acceleratable_coverage": _acceleratable_coverage(results, cap_by_name, target),
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
    # Acceleratable Region Recall. It was computed into coverage.json and rendered NOWHERE, so the
    # report a human actually reads never mentioned the headline generalization metric -- and never
    # showed the two buckets that say how much of the corpus the ratio was computed over.
    arr = cov.get("acceleratable_coverage") or {}
    if arr:
        _r = arr.get("acceleratable_region_recall")
        _p = arr.get("acceleration_precision")
        _n_e, _n_u, _n_c = arr.get("n_eligible", 0), arr.get("n_undetermined", 0), arr.get("n_unclassified", 0)
        L += ["", "## Acceleratable Region Recall", "",
              f"- eligible regions (the denominator): **{_n_e}**",
              f"- of those, accelerated: **{arr.get('n_eligible_accelerated', 0)}**",
              f"- **ARR = {'n/a' if _r is None else f'{_r:.3f}'}**  ·  "
              f"precision = {'n/a' if _p is None else f'{_p:.3f}'}",
              "",
              f"- undetermined (evidence could not decide the family): **{_n_u}**",
              f"- unclassified (this taxonomy has no name for the op): **{_n_c}**", ""]
        _bf = arr.get("by_family") or {}
        if _bf:
            L += ["| semantic family | regions | eligible | accelerated | recall |",
                  "|---|---|---|---|---|"]
            for _f, _b in _bf.items():
                _fr = _b.get("recall")
                L.append(f"| {_f} | {_b['n_regions']} | {_b['n_eligible']} | "
                         f"{_b['n_eligible_accelerated']} | "
                         f"{'n/a' if _fr is None else f'{_fr:.3f}'} |")
            L.append("")
        _fo = arr.get("fused_only_families") or []
        if _fo:
            L += [f"> This target runs {', '.join(_fo)} ONLY fused behind another family (the "
                  f"accumulator-readout epilogue, not a standalone engine), so "
                  f"{arr.get('n_fused_only_ineligible', 0)} standalone region(s) of it are ineligible "
                  f"BY HARDWARE and never enter the denominator. The recall above is a claim about the "
                  f"families in the table, not about everything the device can compute.", ""]
        if _n_u or _n_c:
            L += [f"> {_n_u + _n_c} region(s) are in NEITHER the numerator nor the denominator. "
                  f"Undetermined is a gap in the target's evidence; unclassified is a gap in our "
                  f"vocabulary. Neither says the hardware cannot do the work, and the recall above is "
                  f"computed over the remainder — do not quote it alone.", ""]
        if arr.get("must_accelerate_violations"):
            L += [f"> **must_accelerate violated** on {len(arr['must_accelerate_violations'])} capsule(s): "
                  f"{', '.join(arr['must_accelerate_violations'][:8])} — an ELIGIBLE region that fell back.",
                  ""]

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
