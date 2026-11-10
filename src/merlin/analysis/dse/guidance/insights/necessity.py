"""Classify abstraction necessity and expose its original decision predicates.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .workloads import _signal_by_workload

# one-sentence DSE-search-space implication shown beneath each presentation plot (structural only).
# =========================================================================== P16 decision-frontier
# Sharpen results into decision-discriminating form: a metric is useful only if it changes a DSE
# decision. Strict necessity (not permissive support), primitive-set frontier, operator Pareto,
# macro/micro + leave-one-workload-out robustness, and capture-fidelity as a first-class result.

_NEC_CLASSES = ("necessary", "useful", "possible", "blocked", "not_applicable")


def _classify_abstraction(spec: dict, w: str, sig: dict) -> tuple:
    """Strict per-(abstraction, workload) class + the reason (the predicate that fired)."""
    s = sig[w]
    name, sup = spec["abstraction"], spec["support"]
    is_ar = s["family"] in ("autoregressive_vla", "llm")
    has_control = s["family"] in ("flow_matching", "diffusion", "autoregressive_vla") and bool(s["control_rate"])
    if spec["kv"] and not is_ar:
        return "not_applicable", "KV abstraction; workload is not autoregressive"
    if sup == "decode" and not is_ar:
        return "not_applicable", "decode abstraction; workload is not autoregressive"
    if sup == "control_loop" and not has_control:
        return "not_applicable", "no control-rate loop in this workload"
    if spec["erased"]:
        return "blocked", "low-bit/packed structure dequantized in the capture"
    if spec["kv"]:
        return "blocked", "attention/KV lowered; structure not recovered"
    # Reasons below are class-invariant RULES (no per-workload scalar): the per-workload K /
    # weight bytes / MAC-fraction inputs live in predicate_audit_table.csv, so the necessity
    # rollup never presents one workload's K (e.g. "K=7") as a corpus constant.
    if name == "resident_weight_object":
        if s["invocations"] > 1 and s["weight_bytes"] > 1_000_000 and s["avoidable_reload"] > s["weight_bytes"]:
            return "necessary", (
                "repeated weight reuse (K>1), resident weights >1MB, "
                "avoidable_reload>weight_bytes (per-workload K/MB in predicate_audit)"
            )
        if s["invocations"] > 1:
            return "useful", "repeated weight reuse (K>1) but resident weights <=1MB"
        return "possible", "no repeated weight reuse recovered"
    if sup == "dense":
        d = s["dense_mac"]
        return (
            ("necessary", "dense (squareish_gemm) MAC fraction >0.5")
            if d > 0.5
            else ("useful", "dense MAC fraction in (0.1, 0.5]")
            if d > 0.1
            else ("possible", "dense MAC fraction <=0.1")
        )
    if sup == "gemv":
        g = s["gemv_mac"]  # true_gemv + skinny (engine serves both)
        return (
            ("necessary", "gemv/skinny MAC fraction >0.5 (true_gemv+skinny; split in predicate_audit)")
            if g > 0.5
            else ("useful", "gemv/skinny MAC fraction in (0.1, 0.5]")
            if g > 0.1
            else ("possible", "gemv/skinny MAC fraction <=0.1")
        )
    if name in ("partial_sum_object", "accumulator_merge"):
        if s["k_shard"] and not s["mn_shard"]:
            return "necessary", "K-shard reduction needed; reduction-free M/N insufficient"
        if s["k_shard"]:
            return "useful", "K-shard available but reduction-free M/N also possible"
        return "possible", "no attractive K-shard"
    if sup == "epilogue":
        if s["epi_scale_act"]:
            return "necessary", "epilogue has scale+activation"
        if s["epi_bias"]:
            return "useful", "bias-only epilogue (requant slot present)"
        return "possible", "no fused epilogue detected"
    if sup == "k_loop":  # bounded_loop_command / loop_carried_state_handle
        if s["K"] > 1:
            return "useful", (
                "K>1 cadence (configured/reference, not IR-recovered; needs a "
                "loop-preserving capture; per-workload K in predicate_audit)"
            )
        return "possible", "no loop"
    return "possible", "available; not gated by a discriminating signal"


def _capture_gate(spec: dict) -> str:
    if spec["erased"]:
        return "low-bit recapture (packed weights + scales + per-format accuracy)"
    if spec["kv"]:
        return "loop-preserving, attention-not-lowered capture"
    if spec["support"] in ("k_loop", "decode", "control_loop"):
        return "loop-preserving capture (K/cadence are configured/reference)"
    return ""


def abstraction_necessity(cs_dir: Path) -> dict:
    """The #1 fix: replace permissive 'supported by 4/4' with a strict necessity classification
    (necessary/useful/possible/blocked/not_applicable) per abstraction × workload, using the
    threshold signals already in the committed artifacts. Returns rows + a corpus rollup."""
    from merlin.analysis.dse.guidance.boundary_placement import catalog_rows

    sig, wls = _signal_by_workload(cs_dir)
    if not wls:  # partial mode: no per-workload signal artifacts present
        return {"workloads": [], "rows": [], "rollup": {k: 0 for k in _NEC_CLASSES}}
    rows = []
    for spec in catalog_rows():
        per = {w: _classify_abstraction(spec, w, sig) for w in wls}
        classes = {w: c for w, (c, _) in per.items()}
        counts = {k: sum(1 for c in classes.values() if c == k) for k in _NEC_CLASSES}
        applicable = sum(counts[k] for k in ("necessary", "useful", "possible"))
        if counts["blocked"] and applicable == 0:
            macro = "blocked"
        elif counts["not_applicable"] == len(wls):
            macro = "not_applicable"
        elif counts["necessary"] >= max(1, applicable // 2) and applicable:
            macro = "necessary"
        elif counts["necessary"] + counts["useful"] >= 1:
            macro = "useful"
        else:
            macro = "possible"
        # representative predicate: the highest-severity class-invariant RULE present (necessary ->
        # useful -> possible -> any). Reasons are now K-free rules, so this is honest — the rollup
        # describes the binding constraint and never presents one workload's K as a corpus constant
        # (the per-workload K / bytes / fractions live in predicate_audit_table.csv).
        reason = next(
            (r for w, (c, r) in per.items() if c == "necessary"),
            next(
                (r for w, (c, r) in per.items() if c == "useful"),
                next((r for w, (c, r) in per.items() if c == "possible"), next(iter(per.values()))[1]),
            ),
        )
        row = {
            "abstraction": spec["abstraction"],
            "support_tag": spec["support"],
            "macro_class": macro,
            "predicate": reason,
            "capture_gate": _capture_gate(spec),
        }
        for w in wls:
            row[w] = classes[w]
        for k in _NEC_CLASSES:
            row[f"n_{k}"] = counts[k]
        rows.append(row)
    order = {"necessary": 0, "useful": 1, "possible": 2, "blocked": 3, "not_applicable": 4}
    rows.sort(key=lambda r: (order[r["macro_class"]], -r["n_necessary"], r["abstraction"]))
    rollup = {k: sum(1 for r in rows if r["macro_class"] == k) for k in _NEC_CLASSES}
    return {"workloads": wls, "rows": rows, "rollup": rollup}


# =========================================================================== P17 adversarial audit
# Stress-test the P16 conclusions: do they survive leave-one-out, macro-vs-micro, and pad-waste
# threshold perturbation? Audit every necessity predicate (its inputs, thresholds, and whether it
# rests on configured/reference K or on capture-erased structure). Sweep the primitive-set frontier
# over thresholds + set sizes + extra candidate tiles. Structural only; every number is recomputed
# from the committed artifacts — no measurement, no performance claim.

_PRED_COLS = [
    "abstraction",
    "workload",
    "classification",
    "predicate",
    "predicate_inputs",
    "thresholds",
    "uses_configured_K",
    "uses_erased_capture",
    "has_negative_control",
    "is_discriminating",
    "suspicious",
]


def _predicate_io(spec: dict, w: str, sig: dict) -> tuple:
    """(predicate_inputs, thresholds, uses_configured_K, uses_erased_capture) for one cell. The
    per-workload scalars (K, weight bytes, MAC fractions) live HERE, not in the necessity rollup."""
    s = sig[w]
    name, sup = spec["abstraction"], spec["support"]
    if spec["erased"]:
        return "low-bit/packed weights+scales (dequantized in the f32 capture)", "n/a", "no", "yes"
    if spec["kv"]:
        return "attention/KV state (lowered in the flat capture)", "n/a", "no", "yes"
    if name == "resident_weight_object":
        return (
            f"K={s['invocations']}, weight_bytes={s['weight_bytes']}, avoidable_reload={s['avoidable_reload']}",
            "K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes",
            "yes",
            "yes",
        )
    if sup == "dense":
        return f"dense_mac={s['dense_mac']}", ">0.5 necessary; >0.1 useful", "no", "no"
    if sup == "gemv":
        return (
            f"gemv_mac={s['gemv_mac']} (true_gemv={s['true_gemv_mac']}, skinny_gemm={s['skinny_gemm_mac']})",
            ">0.5 necessary; >0.1 useful",
            "no",
            "no",
        )
    if name in ("partial_sum_object", "accumulator_merge"):
        return (f"k_shard={s['k_shard']}, mn_shard={s['mn_shard']}", "k_shard & not mn_shard -> necessary", "no", "no")
    if sup == "epilogue":
        return (
            f"epi_scale_act={s['epi_scale_act']}, epi_bias={s['epi_bias']}",
            "scale+act -> necessary; bias -> useful",
            "no",
            "no",
        )
    if sup == "k_loop":
        return f"K={s['K']}", "K>1 -> useful", "yes", "yes"
    return "not gated by a discriminating signal", "n/a", "no", "no"


def predicate_audit(cs_dir: Path) -> dict:
    """Part B: for every (abstraction, workload) emit the exact predicate, its numeric inputs, the
    thresholds, whether it depends on configured K or capture-erased structure, whether the
    abstraction has a negative control (some workload where it is NOT necessary/useful), whether it
    is discriminating (the class varies across the corpus), and a suspicious flag."""
    from merlin.analysis.dse.guidance.boundary_placement import catalog_rows

    sig, wls = _signal_by_workload(cs_dir)
    if not wls:
        return {"workloads": [], "rows": []}
    rows = []
    for spec in catalog_rows():
        per = {w: _classify_abstraction(spec, w, sig) for w in wls}
        classes = [c for c, _ in per.values()]
        is_disc = len({*classes}) > 1
        has_neg = any(c in ("possible", "blocked", "not_applicable") for c in classes) and any(
            c in ("necessary", "useful") for c in classes
        )
        for w in wls:
            cls, reason = per[w]
            inputs, thr, usesK, erased = _predicate_io(spec, w, sig)
            susp = []
            if cls == "necessary" and usesK == "yes":
                susp.append("necessity rests on configured/reference K (not IR-recovered)")
            if cls in ("necessary", "useful") and not is_disc:
                susp.append("class never varies across corpus (not discriminating)")
            rows.append(
                {
                    "abstraction": spec["abstraction"],
                    "workload": w,
                    "classification": cls,
                    "predicate": reason,
                    "predicate_inputs": inputs,
                    "thresholds": thr,
                    "uses_configured_K": usesK,
                    "uses_erased_capture": erased,
                    "has_negative_control": "yes" if has_neg else "no",
                    "is_discriminating": "yes" if is_disc else "no",
                    "suspicious": "; ".join(susp),
                }
            )
    return {"workloads": wls, "rows": rows}


# --------------------------------------------------------------------------- P16 emitters


def _necessity_cols(wls):
    return (
        ["abstraction", "support_tag", "macro_class"]
        + list(wls)
        + [f"n_{k}" for k in _NEC_CLASSES]
        + ["predicate", "capture_gate"]
    )


def _necessity_md(nec, scope) -> str:
    wls = nec["workloads"]
    r = nec["rollup"]
    L = [
        f"# Abstraction necessity ({scope})\n",
        "> Strict replacement for the permissive support table: each abstraction is classified per "
        "workload as **necessary / useful / possible / blocked / not_applicable** by a threshold "
        "predicate over the recovered signals (not `any-X` presence). 'possible' = available but not "
        "gated by a discriminating signal; 'blocked' = the capture erased the needed structure.\n",
        f"**Corpus rollup:** {r['necessary']} necessary · {r['useful']} useful · {r['possible']} "
        f"possible · {r['blocked']} blocked · {r['not_applicable']} not-applicable "
        f"(of {len(nec['rows'])} abstractions).\n",
        "| abstraction | macro | " + " | ".join(wls) + " | predicate | needs |",
        "|---|---|" + "|".join(["---"] * len(wls)) + "|---|---|",
    ]
    for row in nec["rows"]:
        L.append(
            f"| {row['abstraction']} | **{row['macro_class']}** | "
            + " | ".join(row[w] for w in wls)
            + f" | {row['predicate']} | {row['capture_gate'] or '—'} |"
        )
    return "\n".join(L) + "\n"


def _predicate_audit_md(pa, scope) -> str:
    rows = pa["rows"]
    susp = [r for r in rows if r["suspicious"]]
    L = [
        f"# Predicate audit ({scope})\n",
        "> Every necessity predicate, its numeric inputs, thresholds, and whether it rests on "
        "configured/reference K or on capture-erased structure. The per-workload K / weight-bytes / "
        "MAC-fraction scalars live HERE — the necessity rollup carries only the class-invariant "
        "rule, so it never presents one workload's K as a corpus constant.\n",
        f"**{len(susp)} suspicious cells** (necessity resting on configured K, or a non-discriminating predicate).\n",
        "| abstraction | workload | class | inputs | thresholds | uses_K | erased | "
        "neg_control | discriminating | suspicious |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['abstraction']} | {r['workload']} | {r['classification']} | "
            f"{r['predicate_inputs']} | {r['thresholds']} | {r['uses_configured_K']} | "
            f"{r['uses_erased_capture']} | {r['has_negative_control']} | "
            f"{r['is_discriminating']} | {r['suspicious'] or '—'} |"
        )
    return "\n".join(L) + "\n"
