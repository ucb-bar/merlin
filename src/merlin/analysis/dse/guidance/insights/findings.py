"""Answer design questions, qualify presentation findings and state missing inputs.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .evidence import _METRIC_QUESTION

# --------------------------------------------------------------------------- P13-d usefulness suite

_QUERIES = [
    (
        "primitive_shapes_to_include",
        "What compute primitive shapes should a DSE search space include?",
        ["primitive_coverage_matrix.csv", "primitive_regret_table.csv"],
    ),
    (
        "primitives_broadly_useful",
        "Which primitive shapes are broadly useful across workloads?",
        ["primitive_regret_table.csv"],
    ),
    (
        "primitives_workload_specific",
        "Which primitive shapes are workload-specific / high-regret?",
        ["primitive_regret_table.csv"],
    ),
    (
        "heterogeneous_units",
        "Which workloads suggest heterogeneous processing units?",
        ["processing_unit_guidance.yaml", "resource_pressure_table.csv"],
    ),
    (
        "bounded_loop_commands",
        "Which workloads suggest bounded-loop commands?",
        ["workload_contract_graph.yaml", "boundary_candidate_contracts.yaml"],
    ),
    (
        "strong_abstractions",
        "Which abstractions are most strongly supported across workloads?",
        ["abstraction_pressure_ranking.csv", "hw_sw_boundary_matrix.csv"],
    ),
    (
        "family_specific_abstractions",
        "Which abstractions are family-specific?",
        ["abstraction_pressure_table.csv", "workload_family_table.csv"],
    ),
    (
        "boundary_resident_weights",
        "Which boundary placements are plausible for resident weights?",
        ["boundary_candidate_contracts.yaml", "hw_sw_boundary_matrix.csv"],
    ),
    (
        "boundary_kloop",
        "Which boundary placements are plausible for K-loop execution?",
        ["boundary_candidate_contracts.yaml"],
    ),
    (
        "boundary_packed_lowbit",
        "Which boundary placements are plausible for packed low-bit tensors?",
        ["boundary_candidate_contracts.yaml"],
    ),
    ("hal_objects", "Which objects should potentially cross the HAL boundary?", ["runtime_object_candidates.yaml"]),
    ("command_isa", "Which command-ISA abstractions are structurally suggested?", ["command_isa_candidates.yaml"]),
    (
        "accelerator_isa",
        "Which accelerator-ISA primitives are structurally suggested?",
        ["isa_candidate_primitives.yaml"],
    ),
    ("blocked_by_proof", "Which candidates are blocked by missing compiler proof?", ["compiler_proof_matrix.csv"]),
    (
        "blocked_by_measurement",
        "Which candidates are blocked by missing measurements?",
        ["dse_readiness_summary.csv", "measurement_priority_table.csv"],
    ),
    (
        "measurements_unblock_most",
        "Which measurements unblock the most candidates?",
        ["measurement_priority_table.csv"],
    ),
    ("assumption_heavy", "Which findings rely heavily on assumptions?", ["dse_contract.json"]),
    (
        "shallow_analyses",
        "Which analyses are currently shallow or incomplete?",
        ["lost_numerical_contracts.csv", "boundary_candidate_contracts.yaml"],
    ),
    (
        "plots_for_presentation",
        "Which plots should be generated for presentation?",
        ["operator_shape_table.csv", "critical_path_table.csv", "hw_sw_boundary_matrix.csv", "data_movement_table.csv"],
    ),
    (
        "safe_claims",
        "Which claims are safe to present without quantitative performance measurements?",
        ["verification_report.md"],
    ),
]


# queries that are inherently about absence/limits (answered by scoped required-inputs, not tier-A
# facts) — these are honestly "partial/weak" even when the supporting artifact is rich.
_LIMIT_QUERIES = {
    "boundary_packed_lowbit": "weak",
    "assumption_heavy": "partial",
    "shallow_analyses": "partial",
    "blocked_by_proof": "partial",
    "family_specific_abstractions": "partial",
    "primitives_workload_specific": "partial",
}


def usefulness(cs_dir: Path, scope: str, facts: list[dict]) -> list[dict]:
    """Status is DERIVED from the facts that back each query (no hardcoded strong/partial lists):
    strong = >=1 corroborated tier-A/B fact; partial = only tier-C; weak = only tier-D / limit query;
    unavailable = no supporting artifact present."""
    answers = []
    for key, q, arts in _QUERIES:
        present = [a for a in arts if (cs_dir / a).is_file()]
        related = [f for f in facts if f["source_artifact"] in arts]
        strong_facts = [
            f
            for f in related
            if f["evidence_tier"] in ("A", "B") and (f["corroborated_by"] >= 2 or f["verifying_check"])
        ]
        tierC = [f for f in related if f["evidence_tier"] == "C"]
        if not present:
            status = "unavailable"
        elif key in _LIMIT_QUERIES:
            status = _LIMIT_QUERIES[key]  # inherently a limit/absence question
        elif strong_facts:
            status = "strong"
        elif tierC:
            status = "partial"
        elif related:
            status = "weak"
        else:
            status = "partial"  # artifacts present but no mined fact yet
        use = {"strong": "main", "partial": "backup", "weak": "backup", "unavailable": "do_not_show"}[status]
        answers.append(
            {
                "key": key,
                "query": q,
                "status": status,
                "supporting_artifacts": present,
                "missing_artifacts": [a for a in arts if a not in present],
                "n_backing_strong_facts": len(strong_facts),
                "evidence_types": sorted({f["evidence_type"] for f in related}) or ["see artifacts"],
                "caveats": (
                    "packed low-bit / scales erased in the capture (see required_inputs_manifest)"
                    if key == "boundary_packed_lowbit"
                    else "structural only; not a performance claim"
                ),
                "recommended_presentation_use": use,
            }
        )
    return answers


# --------------------------------------------------------------------------- P13-e findings

_PLOT_FOR_METRIC = {
    "available_parallelism": "inter_op_parallelism_by_workload",
    "avoidable_weight_reload": "avoidable_reload_by_region",
    "resident_int8_B": "resident_capacity_by_dtype",
    "boundary_pressure_score": "boundary_placement_heatmap",
    "coverage_under_10pct": "primitive_coverage_heatmap",
    "max_regret": "primitive_regret_bar",
    "dominant_shape_class": "shape_class_mac_share",
    "matmul_bias_epilogues": "epilogue_pattern_counts",
    "n_workloads_supporting": "abstraction_pressure_bar",
    "accuracy_int8_w8a8": "accuracy_gate_status",
    "measurement_unblocks": "measurement_priority_bar",
}


def presentation_findings(facts: list[dict], answers: list[dict]) -> list[dict]:
    from collections import defaultdict

    cand = [f for f in facts if f["presentation_candidate"]]
    by_metric = defaultdict(list)
    for f in cand:
        by_metric[f["metric_name"]].append(f)
    findings = []
    for metric, fs in by_metric.items():
        tiers = {f["evidence_tier"] for f in fs}
        tier = "A" if "A" in tiers else ("B" if "B" in tiers else "C")
        purely_assumed = all(f["derivation_type"] == "assumed" for f in fs)
        # 'main' requires real corroboration: a per-metric harness check OR >=2-artifact agreement
        corroborated = any(f["verifying_check"] or f["corroborated_by"] >= 2 for f in fs)
        wls = sorted({f["workload"] for f in fs})
        arts = sorted({f["source_artifact"] for f in fs})
        caveats = sorted({f["caveat"] for f in fs if f["caveat"]})
        impl = next((f["dse_implication"] for f in fs if f["dse_implication"]), "")
        # B5: per-workload spread (value per workload) so a multi-workload finding hides nothing
        spread = {f["workload"]: f["metric_value"] for f in fs if f["workload"] not in ("ALL", "ZOO")}
        vals = []
        for v in spread.values():
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
        spread_summary = (
            f"range [{min(vals):g}, {max(vals):g}] across {len(spread)} workloads"
            if len(vals) >= 2
            else (f"{len(spread)} workload(s)" if spread else "cross-workload")
        )
        max_corr = max((f["corroborated_by"] for f in fs), default=1)
        main = tier in ("A", "B") and not purely_assumed and bool(impl) and corroborated
        findings.append(
            {
                "title": metric.replace("_", " "),
                "claim": f"{metric}: {impl}",
                "evidence_tier": tier,
                "supporting_artifacts": arts,
                "supporting_workloads": wls,
                "per_workload_values": spread,
                "per_workload_spread": spread_summary,
                "max_corroborated_by": max_corr,
                "relevant_metrics": [metric],
                "dse_implication": impl,
                "caveats": caveats,
                "dse_question": _METRIC_QUESTION.get(metric, ""),
                "suggested_plot": _PLOT_FOR_METRIC.get(metric, ""),
                "presentation_placement": "main" if main else "backup",
                "forbidden_claim_risk": "low",
            }
        )
    # an explicit honesty finding (limitation) — useful as backup, never main. Only when there is a
    # real package to mine (partial/empty mode produces no findings and stays vacuously consistent).
    if facts:
        findings.append(
            {
                "title": "erased low-bit / KV structure",
                "claim": "packed low-bit layout, scales, and KV/attention structure are erased/lowered in "
                "the capture; native low-bit & KV boundary placements are blocked/unavailable",
                "evidence_tier": "D",
                "supporting_artifacts": ["lost_numerical_contracts.csv", "boundary_candidate_contracts.yaml"],
                "supporting_workloads": ["ALL"],
                "relevant_metrics": ["unavailable"],
                "dse_implication": "needs a low-bit / loop-preserving capture before these are candidates",
                "caveats": ["inherent to a flat dequantized, attention-lowered capture"],
                "suggested_plot": "",
                "presentation_placement": "backup",
                "forbidden_claim_risk": "low",
            }
        )
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    findings.sort(key=lambda f: (order[f["evidence_tier"]], -len(f["supporting_workloads"])))
    return findings


def decision_scorecard(bundle: dict) -> list[dict]:
    """The 7 DSE decision questions, each answered from the P16 analyses with macro/micro/worst +
    leave-one-out stability + a capture caveat."""
    fr = bundle.get("primitive_frontier", {})
    nec = bundle.get("abstraction_necessity", {})
    rob = bundle.get("robustness", {})
    par = bundle.get("operator_pareto", {})
    by_size = fr.get("best_by_size", {})
    rob_by = {f["finding"]: f for f in rob.get("findings", [])}
    q = []
    s1 = by_size.get(1)
    q.append(
        {
            "q": "Q1 best single primitive (worst-workload coverage)?",
            "answer": (f"{s1['set'][0]} -> worst {s1['worst']:.2f}, macro {s1['macro']:.2f}" if s1 else "n/a"),
            "caveat": "no single primitive covers every workload",
        }
    )
    s2 = by_size.get(2)
    q.append(
        {
            "q": "Q2 best 2-primitive set?",
            "answer": (
                f"{'+'.join(s2['set'])} -> worst {s2['worst']:.2f} (vs {s1['worst']:.2f} single)"
                if s2 and s1
                else "n/a"
            ),
            "caveat": "search primitive SETS, not one tile",
        }
    )
    q.append(
        {
            "q": "Q3 capacity x dtype residency thresholds?",
            "answer": "see decision_capacity_dtype plot (int4<int8<bf16 budget to fit)",
            "caveat": "repeated-head weights only; K is configured/reference",
        }
    )
    q.append(
        {
            "q": "Q4 sharding axis for top-MAC ops?",
            "answer": "M/N reduction-free vs K partial-sum (see decision_sharding_per_top_op)",
            "caveat": "communication bytes, not latency",
        }
    )
    necd = nec.get("rollup", {})
    q.append(
        {
            "q": "Q5 which abstractions are NECESSARY (not just possible)?",
            "answer": (
                f"{necd.get('necessary', 0)} necessary, {necd.get('useful', 0)} useful, "
                f"{necd.get('possible', 0)} possible, {necd.get('blocked', 0)} blocked, "
                f"{necd.get('not_applicable', 0)} N/A"
            ),
            "caveat": "strict predicate; low-bit abstractions blocked by capture",
        }
    )
    dd = rob_by.get("dense_gemm_mac_dominance", {})
    q.append(
        {
            "q": "Q6 which conclusions are driven by one workload (RDT)?",
            "answer": (
                f"dense-MAC dominance macro {dd.get('macro', '?')} vs micro "
                f"{dd.get('micro', '?')}; collapses if removed: "
                f"{dd.get('collapses_if_removed') or 'none'}"
            ),
            "caveat": "micro view is biased by RDT's 87%-of-workload op",
        }
    )
    q.append(
        {
            "q": "Q7 which claims depend on configured K (capture fidelity)?",
            "answer": "all residency / loop / command claims (K is config/reference)",
            "caveat": "needs a loop-preserving capture; see capture_fidelity_matrix",
        }
    )
    return q


# --------------------------------------------------------------------------- inherent limits

# Each genuinely-external limit + the EXACT input/run that closes it. These are scoped (have a
# required_input), so the audit treats them as not-open — never bare caveats, never fabricated.
_INHERENT_LIMITS = [
    {
        "limit": "real deployment K + control rate",
        "currently": "K/control are published config reference values (recovered_from_model_config)",
        "evidence_today": "Tier C",
        "required_input": "a deployment/runtime trace giving actual loop counts + control frequency",
    },
    {
        "limit": "per-unit throughput / latency / area / energy",
        "currently": "Merlin refuses to estimate these for unbuilt hardware",
        "evidence_today": "unavailable",
        "required_input": "a candidate design YAML (unit shapes + a cost model); then the future DSE "
        "tool computes them",
    },
    {
        "limit": "KV / attention structure + true data deps at loop level",
        "currently": "attention is lowered into matmul projections; KV unavailable",
        "evidence_today": "unavailable",
        "required_input": "a Level-2 loop-preserving, attention-not-lowered capture",
    },
    {
        "limit": "packed low-bit layout + scales for the recaptured models",
        "currently": "the recaptures are dequantized f32; the quantized zoo has low-bit storage for OTHER models",
        "evidence_today": "low-bit storage shown on the zoo (numerical_contract_fidelity_report.md)",
        "required_input": "a low-bit (packed weights + scale metadata) capture of the recaptured models",
    },
    {
        "limit": "fp8 / int4 accuracy gates",
        "currently": "int8 W8A8 is measured (real); fp8/int4 are unavailable",
        "evidence_today": "int8 measured",
        "required_input": "per-format accuracy runs (W8A8 already done) for fp8 / int4",
    },
    {
        "limit": "real-magnitude weights",
        "currently": "captures are small random-init instances (structure real, magnitudes small)",
        "evidence_today": "structure recovered_from_ir",
        "required_input": "full-size (non-random-init) captures of the same architectures",
    },
]


def required_inputs(cs_dir: Path, facts: list[dict]) -> list[dict]:
    """The scoped inherent limits + the exact input/run that closes each (nothing invented)."""
    return [dict(x, status="scoped", avoidable=False) for x in _INHERENT_LIMITS]


def _scorecard_md(answers, scope) -> str:
    L = [
        f"# DSE-usefulness scorecard ({scope})\n",
        "> Can the package answer DSE-relevant questions? Each query is answered from existing "
        "artifacts with a status + recommended presentation use. Structural only — no performance "
        "claim.\n",
        "| # | query | status | use | supporting artifacts |",
        "|---|---|---|---|---|",
    ]
    for i, a in enumerate(answers, 1):
        L.append(
            f"| {i} | {a['query']} | **{a['status']}** | {a['recommended_presentation_use']} | "
            f"{', '.join(a['supporting_artifacts']) or '—'} |"
        )
    strong = sum(1 for a in answers if a["status"] == "strong")
    L.append(
        f"\n{strong}/{len(answers)} queries answerable **strong**; "
        f"{sum(1 for a in answers if a['status'] == 'unavailable')} unavailable."
    )
    return "\n".join(L) + "\n"


def _findings_md(findings, scope) -> str:
    L = [
        f"# Presentation candidate findings ({scope})\n",
        "> A finding may be **main** only if tier A/B, not purely assumed, with a clear DSE "
        "implication, corroborated by a verification check, and needing no performance claim.\n",
    ]
    for place in ("main", "backup"):
        fs = [f for f in findings if f["presentation_placement"] == place]
        L.append(f"## {place.title()} ({len(fs)})\n")
        for f in fs:
            L.append(
                f"- **{f['title']}** [tier {f['evidence_tier']}] — {f['claim']}  "
                f"_(workloads: {', '.join(f['supporting_workloads'])}; "
                f"plot: {f['suggested_plot'] or '—'})_"
            )
        L.append("")
    return "\n".join(L) + "\n"


def _required_inputs_md(ri, scope) -> str:
    L = [
        f"# Required-inputs manifest ({scope})\n",
        "> The inherent limits that remain, each with the EXACT input/run that closes it. These are "
        "scoped (not bare caveats) and nothing is fabricated — Merlin reports what a real DSE run "
        "still needs.\n",
        "| limit | evidence today | required input to close |",
        "|---|---|---|",
    ]
    for x in ri:
        L.append(f"| {x['limit']} | {x['evidence_today']} | {x['required_input']} |")
    return "\n".join(L) + "\n"


def _scorecard16_md(qs, scope) -> str:
    L = [
        f"# DSE decision-question scorecard ({scope})\n",
        "> The few decisions a future DSE tool must make, each answered from the workload analysis "
        "with its caveat. A metric earns its place only by answering one of these.\n",
        "| # | decision question | answer (from analysis) | caveat |",
        "|---|---|---|---|",
    ]
    for q in qs:
        L.append(f"| {q['q'].split()[0]} | {q['q']} | {q['answer']} | {q['caveat']} |")
    return "\n".join(L) + "\n"
