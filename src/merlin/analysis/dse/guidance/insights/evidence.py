"""Classify source-traced facts and summarize their evidence strength.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

# ---- evidence labels -> tier inputs ----
# Tier A is reserved for IR/prov.fqn-recovered or measured facts (independently verifiable). A
# published config constant (recovered_from_model_config, e.g. K) is a *reference value*, not an
# IR-recovered or measured fact, so it tiers as C alongside assumed_reference — NOT A.
_TIER_A_LABELS = {"measured", "proxy_measured", "recovered_from_ir", "recovered_from_prov_fqn"}


_CONFIG = {"recovered_from_model_config"}


_DERIVED = "derived_requirement"


_ASSUMED = {"assumed_reference", "design_assumption"}


_ABSENT = {"unavailable", "unknown", ""}


EVIDENCE_LABELS = _TIER_A_LABELS | {_DERIVED} | _ASSUMED | {"unavailable"}


FORBIDDEN = (
    "speedup",
    "faster",
    "optimal",
    "best design",
    "predicted cycles",
    "cycle count",
    "performance improvement",
    "gap_closure",
    "energy",
    "area estimate",
)


FACT_COLUMNS = [
    "fact_id",
    "workload",
    "region",
    "phase",
    "op_id",
    "abstraction",
    "boundary_level",
    "metric_name",
    "metric_value",
    "metric_unit",
    "source_artifact",
    "source_phase",
    "evidence_type",
    "derivation_type",
    "evidence_tier",
    "metric_class",
    "dse_question",
    "verification_status",
    "verifying_check",
    "corroborated_by",
    "caveat",
    "dse_implication",
    "presentation_candidate",
]


# SIGNAL metrics = decision-relevant quantities a DSE engine acts on. Only these can become
# presentation candidates / findings. Everything else (row counts, provenance, redundant
# corroboration values) is CONTEXT — kept for traceability/coverage, never promoted to a finding.
# This is the de-noising gate: it keeps real single-source recovered signal (dense/skinny split,
# proof distribution, dominant shape) AND excludes "n_X = rows in artifact X" padding.
SIGNAL_METRICS = {
    "head_weight_bytes",
    "total_macs",
    "n_matmuls",
    "avoidable_weight_reload",
    "resident_int8_B",
    "available_parallelism",
    "serialization",
    "boundary_pressure_score",
    "coverage_under_10pct",
    "max_regret",
    "accuracy_int8_w8a8",
    "measured_dispatch_ratio",
    "matmul_bias_epilogues",
    "dominant_shape_class",
    "mac_fraction_dense_gemm",
    "mac_fraction_skinny_gemm_or_gemv",
    "compiler_proofs_proven_for_workload",
    "compiler_proofs_assumed",
    "compiler_proofs_unknown",
    "clean_8way_mn_shards",
    "head_cadence",
    "accumulator_dtype",
    "compute_dtype",
    "lowbit_storage_dequantized_finding",
    "overlap_candidates_yes",
    # NOTE: unit_multiplicity_implication is deliberately NOT signal — "heterogeneous" is an
    # interpretation, not a measured fact; it is kept as context and the report derives the
    # heterogeneity conclusion from the component metrics (dense/skinny split, parallelism, sharding).
}


def metric_class(metric: str, derivation: str) -> str:
    if derivation == "measured":
        return "measured"
    if metric in SIGNAL_METRICS:
        return "signal"
    return "context"  # row counts / provenance / redundant corroboration values (traceability only)


# every SIGNAL metric must answer ONE DSE-search-space question (the organizing rule). Context
# metrics get no question (they are traceability, never presented).
Q_PRIMITIVES = "Q_primitives: what compute primitives should DSE include?"


Q_HETERO = "Q_heterogeneity: should DSE explore heterogeneous / replicated units?"


Q_RESIDENCY = "Q_residency: should DSE explore weight residency / packed stores?"


Q_COMMAND = "Q_command: should DSE explore command/loop/dispatch abstractions?"


Q_LOWBIT = "Q_lowbit: should DSE explore low-bit formats / numerical placement?"


Q_BOUNDARY = "Q_boundary: where should the HW/SW boundary sit?"


Q_READINESS = "Q_readiness: what blocks quantitative ranking?"


DSE_QUESTIONS = [Q_PRIMITIVES, Q_HETERO, Q_RESIDENCY, Q_COMMAND, Q_LOWBIT, Q_BOUNDARY, Q_READINESS]


_METRIC_QUESTION = {
    "total_macs": Q_PRIMITIVES,
    "n_matmuls": Q_PRIMITIVES,
    "coverage_under_10pct": Q_PRIMITIVES,
    "max_regret": Q_PRIMITIVES,
    "dominant_shape_class": Q_PRIMITIVES,
    "available_parallelism": Q_HETERO,
    "serialization": Q_HETERO,
    "mac_fraction_dense_gemm": Q_HETERO,
    "mac_fraction_skinny_gemm_or_gemv": Q_HETERO,
    "clean_8way_mn_shards": Q_HETERO,
    "unit_multiplicity_implication": Q_HETERO,
    "head_weight_bytes": Q_RESIDENCY,
    "avoidable_weight_reload": Q_RESIDENCY,
    "resident_int8_B": Q_RESIDENCY,
    "measured_dispatch_ratio": Q_COMMAND,
    "head_cadence": Q_COMMAND,
    "overlap_candidates_yes": Q_COMMAND,
    "accuracy_int8_w8a8": Q_LOWBIT,
    "accuracy_gate_report_present": Q_LOWBIT,
    "matmul_bias_epilogues": Q_LOWBIT,
    "accumulator_dtype": Q_LOWBIT,
    "compute_dtype": Q_LOWBIT,
    "lowbit_storage_dequantized_finding": Q_LOWBIT,
    "boundary_pressure_score": Q_BOUNDARY,
    "compiler_proofs_proven_for_workload": Q_BOUNDARY,
    "compiler_proofs_assumed": Q_BOUNDARY,
    "compiler_proofs_unknown": Q_BOUNDARY,
}


# metric_name -> the SPECIFIC verify_implementation.py check that independently re-derives it (the
# per-metric verification basis, replacing the coarse artifact-membership proxy). A metric here is
# "verified"; one absent is verified only when corroborated by >=2 independent artifacts.
_METRIC_CHECK = {
    "head_weight_bytes": "B: head facts vs raw IR",
    "weight_bytes": "P9-A: region weight bytes == IR recompute",
    "total_macs": "P7-A: DAG total work == operator MAC sum",
    "n_matmuls": "A: matmul count raw==primitive==attributed",
    "macs": "P5-A: macs == M*N*K",
    "avoidable_weight_reload": "P9-B: avoidable == weight*(K-1)",
    "resident_int8_B": "P9-C: resident == dtype scaling (int8==WB/4)",
    "available_parallelism": "P7-C: available_parallelism == total/critical",
    "boundary_pressure_score": "P12: boundary matrix re-derivation",
    "matmul_bias_epilogues": "P10-B: every addmm flagged has_bias",
    "accuracy_int8_w8a8": "I: accuracy gating (measured)",
    "measured_dispatch_ratio": "measured dispatch coupling (committed measured data)",
    "coverage_under_10pct": "P5-C: primitive coverage aggregates recompute",
    "max_regret": "primitive_regret: max_regret == best - worst",
}


# --------------------------------------------------------------------------- tiers


def _derivation(evidence: str) -> str:
    if evidence in ("measured", "proxy_measured"):
        return "measured"
    if evidence in ("recovered_from_ir", "recovered_from_prov_fqn"):
        return "recovered"
    if evidence in _CONFIG:
        return "config_reference"
    if evidence == _DERIVED:
        return "derived"
    if evidence in _ASSUMED:
        return "assumed"
    return "unavailable"


def _fact_verified(metric: str, corroborated_by: int) -> bool:
    """A fact is independently verified iff a SPECIFIC harness check re-derives its metric, or it is
    corroborated by >=2 independent artifacts (not just artifact-set membership)."""
    return (metric in _METRIC_CHECK) or (corroborated_by >= 2)


def evidence_tier(evidence: str, metric: str = "", corroborated_by: int = 1) -> str:
    """A: IR-recovered/measured AND verified (per-metric check or >=2-artifact corroboration).
    B: recovered-but-unverified or derived+verified. C: assumed/config (labeled) or derived-unverified.
    D: unavailable/unknown."""
    verified = _fact_verified(metric, corroborated_by)
    if evidence in _TIER_A_LABELS:
        return "A" if verified else "B"
    if evidence == _DERIVED:
        return "B" if verified else "C"
    if evidence in _ASSUMED or evidence in _CONFIG:
        return "C"
    return "D"


def _verification_status(metric: str, evidence: str, corroborated_by: int) -> str:
    if evidence in _ABSENT:
        return "unavailable"
    return "verified" if _fact_verified(metric, corroborated_by) else "not_verified"


def _fact(
    fid,
    workload,
    metric,
    value,
    unit,
    artifact,
    phase,
    evidence,
    *,
    region="",
    op_id="",
    abstraction="",
    level="",
    caveat="",
    implication="",
    corroborated_by=1,
) -> dict:
    tier = evidence_tier(evidence, metric, corroborated_by)
    return {
        "fact_id": fid,
        "workload": workload,
        "region": region,
        "phase": phase,
        "op_id": op_id,
        "abstraction": abstraction,
        "boundary_level": level,
        "metric_name": metric,
        "metric_value": value,
        "metric_unit": unit,
        "source_artifact": artifact,
        "source_phase": phase,
        "evidence_type": evidence,
        "derivation_type": _derivation(evidence),
        "evidence_tier": tier,
        "verification_status": _verification_status(metric, evidence, corroborated_by),
        "metric_class": metric_class(metric, _derivation(evidence)),
        "dse_question": _METRIC_QUESTION.get(metric, ""),
        "verifying_check": _METRIC_CHECK.get(metric, ""),
        "corroborated_by": corroborated_by,
        "caveat": caveat,
        "dse_implication": implication,
        # presentation-worthy = a SIGNAL metric (decision-relevant, not a row-count) that is tier A/B
        # with an implication. Corroboration / a harness check are reported as STRENGTH (see
        # corroborated_by + verifying_check), not the candidacy gate -- so real single-source
        # recovered signal is kept and count-padding is excluded.
        "presentation_candidate": (metric in SIGNAL_METRICS and tier in ("A", "B") and bool(implication)),
    }


_SMALL = "magnitudes are small random-init capture instances (structure real)"


# --------------------------------------------------------------------------- P13-c evidence strength


def evidence_strength(facts: list[dict]) -> dict:
    from collections import Counter

    tier = Counter(f["evidence_tier"] for f in facts)
    deriv = Counter(f["derivation_type"] for f in facts)
    klass = Counter(f["metric_class"] for f in facts)
    by_wl = Counter(f["workload"] for f in facts)
    by_phase = Counter(f["source_phase"] for f in facts)
    by_art = Counter(f["source_artifact"] for f in facts)
    verified = sum(1 for f in facts if f["verification_status"] == "verified")
    sig = klass.get("signal", 0) + klass.get("measured", 0)
    return {
        "total_facts": len(facts),
        "signal_facts": sig,
        "context_facts": klass.get("context", 0),
        "signal_to_total": round(sig / len(facts), 3) if facts else 0.0,
        "by_metric_class": dict(klass),
        "by_tier": dict(tier),
        "by_derivation": dict(deriv),
        "by_workload": dict(by_wl),
        "by_phase": dict(by_phase),
        "by_artifact": dict(by_art),
        "verified_facts": verified,
        "measured_facts": deriv.get("measured", 0),
        "derived_facts": deriv.get("derived", 0),
        "assumed_facts": deriv.get("assumed", 0),
        "unavailable_facts": deriv.get("unavailable", 0),
        "high_confidence_findings": tier.get("A", 0) + tier.get("B", 0),
        "assumption_heavy_claims": tier.get("C", 0),
        "weak_findings": tier.get("D", 0),
    }


# =========================================================================== P15 signal study
# Given ONLY workload artifacts, what changes the future DSE search space? Every output below is
# recovered/derived from the captures (or a host measurement) — no quantity for unbuilt hardware.


def canonical_signal_table(facts: list[dict]) -> list[dict]:
    """The one headline table: signal/measured metrics only, with tier, strength, DSE question.

    Context/count metrics (provenance, redundant corroboration, raw row counts) are excluded — they
    live in the full fact table for traceability but never in the headline."""
    rows = []
    seen = set()
    for f in facts:
        if f.get("metric_class") not in ("signal", "measured"):
            continue
        cb = f.get("corroborated_by", 1)
        strength = (
            "measured"
            if f.get("metric_class") == "measured"
            else (
                "verified+corroborated"
                if (f.get("verifying_check") and int(cb or 1) >= 2)
                else "verified"
                if f.get("verifying_check")
                else "corroborated x%d" % int(cb or 1)
                if int(cb or 1) >= 2
                else "single-source"
            )
        )
        # entity = the row's discriminator (per-abstraction / per-region metrics share workload=ALL,
        # so without it the headline shows uninterpretable duplicate rows)
        entity = f.get("abstraction") or f.get("region") or f.get("workload", "")
        # collapse rows that are the same metric on the same entity+value (e.g. one accuracy fact
        # surfaced by two artifacts with different implication wording) to one headline row
        key = (
            f.get("dse_question", ""),
            f["metric_name"],
            f.get("workload", ""),
            entity,
            str(f.get("metric_value", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "dse_question": f.get("dse_question", ""),
                "metric": f["metric_name"],
                "workload": f.get("workload", ""),
                "entity": entity,
                "value": f.get("metric_value", ""),
                "unit": f.get("metric_unit", ""),
                "evidence_tier": f.get("evidence_tier", ""),
                "strength": strength,
                "verification_status": f.get("verification_status", ""),
                "dse_implication": f.get("dse_implication", ""),
            }
        )
    rows.sort(key=lambda r: (r["dse_question"], r["metric"], r["workload"], r["entity"]))
    return rows


def _strength_md(s, scope) -> str:
    L = [
        f"# Evidence-strength report ({scope})\n",
        f"Total normalized facts: **{s['total_facts']}** — **{s['signal_facts']} signal/measured** "
        f"(decision-relevant; can become findings) + **{s['context_facts']} context** "
        f"(row counts / provenance / corroboration — traceability only, never a finding). "
        f"signal ratio = {s['signal_to_total']}.\n",
        "Tiers — A: IR-recovered/measured + verified; B: recovered-unverified or derived+verified; "
        "C: assumed_reference / config; D: unavailable/unknown.\n",
        "| tier | facts |",
        "|---|---|",
    ]
    for t in ("A", "B", "C", "D"):
        L.append(f"| {t} | {s['by_tier'].get(t, 0)} |")
    L.append(
        f"\n- measured: {s['measured_facts']} · derived: {s['derived_facts']} · "
        f"assumed: {s['assumed_facts']} · unavailable: {s['unavailable_facts']}"
    )
    L.append(f"- verified facts: {s['verified_facts']}/{s['total_facts']}")
    L.append(
        f"- high-confidence (A+B): {s['high_confidence_findings']} · "
        f"assumption-heavy (C): {s['assumption_heavy_claims']} · weak (D): {s['weak_findings']}"
    )
    L.append("\nBy phase: " + ", ".join(f"{k}={v}" for k, v in sorted(s["by_phase"].items())))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- P15 emitters

_CANON_COLS = [
    "dse_question",
    "metric",
    "workload",
    "entity",
    "value",
    "unit",
    "evidence_tier",
    "strength",
    "verification_status",
    "dse_implication",
]
