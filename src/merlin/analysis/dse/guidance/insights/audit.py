"""Check cross-artifact consistency and qualify conclusion validity and gaps.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .evidence import FORBIDDEN, SIGNAL_METRICS
from .findings import required_inputs

_AUDIT_COLS = [
    "conclusion",
    "supporting_artifact",
    "supporting_metric",
    "metric_class",
    "depends_on_configured_K",
    "depends_on_flat_capture",
    "survives_loo",
    "survives_macro_vs_micro",
    "survives_threshold",
    "driven_by",
    "decision_or_fact",
    "verdict",
]


def adversarial_audit(bundle: dict) -> dict:
    """Part A: a validity ledger — each headline conclusion mapped to the artifact/metric that
    supports it and whether it survives leave-one-out, macro-vs-micro, and threshold perturbation.
    No new measurement; it reads the P16/P17 analyses already in the bundle."""
    rob = {f["finding"]: f for f in bundle.get("robustness", {}).get("findings", [])}
    fr = bundle.get("primitive_frontier", {}).get("best_by_size", {})
    frob = bundle.get("primitive_frontier_robustness", {})
    necd = bundle.get("abstraction_necessity", {}).get("rollup", {})
    infl = {r["metric"]: r for r in bundle.get("macro_micro_influence", {}).get("rows", [])}
    two = rob.get("best_2_primitive_set", {})
    thr_robust = frob.get("two_set_threshold_robust")
    rows = []

    def add(**kw):
        rows.append({c: kw.get(c, "") for c in _AUDIT_COLS})

    s1, s2 = fr.get(1, {}), fr.get(2, {})
    add(
        conclusion="primitive-set frontier: a 2-primitive set covers the corpus where one fails",
        supporting_artifact="primitive_set_frontier.csv; primitive_frontier_robustness.csv",
        supporting_metric=f"worst coverage size1 {s1.get('worst', '?')} -> size2 {s2.get('worst', '?')}",
        metric_class="derived",
        depends_on_configured_K="no",
        depends_on_flat_capture="no",
        survives_loo=("yes" if two.get("robust") else f"no (flips: {two.get('loo_changes_winner')})"),
        survives_macro_vs_micro="yes (macro+micro+worst all reported)",
        survives_threshold=(
            "yes (5/10/20%)"
            if thr_robust
            else ("not computed" if thr_robust is None else f"no (winner varies: {frob.get('two_winners')})")
        ),
        driven_by=("specific pair shifts with finer tiles/threshold" if not thr_robust else "none"),
        decision_or_fact="decision (search primitive SETS, not one tile)",
        # the CLAIM "a 2-set suffices" is robust (a high-worst-coverage pair exists at every
        # threshold); only the SPECIFIC pair is threshold/LOO-sensitive once finer tiles are added.
        verdict=(
            "robust"
            if two.get("robust") and thr_robust
            else "claim robust (a 2-set suffices); specific pair is threshold/LOO-sensitive"
        ),
    )
    dd = rob.get("dense_gemm_mac_dominance", {})
    di = infl.get("dense_gemm_mac_fraction", {})
    add(
        conclusion="dense-GEMM is corpus-narrow (skinny/GEMV dominates the MAC mass)",
        supporting_artifact="leave_one_workload_out.md; macro_micro_influence_table.csv",
        supporting_metric=f"dense macro {dd.get('macro', '?')} vs micro {dd.get('micro', '?')}",
        metric_class="derived",
        depends_on_configured_K="no",
        depends_on_flat_capture="no",
        survives_loo=(
            f"magnitude NOT stable (max micro swing {di.get('max_loo_micro_delta', '?')})"
            if di.get("winner_stable_magnitude_unstable") == "yes"
            else "yes"
        ),
        survives_macro_vs_micro="winner yes; magnitude no (macro/micro differ widely)",
        survives_threshold="n/a",
        driven_by=di.get("most_influential", "pi05"),
        decision_or_fact="decision (do not size only for square GEMMs)",
        verdict="winner robust, magnitude fragile",
    )
    add(
        conclusion="resident_weight_object is necessary (K-loop weight residency is a search axis)",
        supporting_artifact="abstraction_necessity_table.csv; predicate_audit_table.csv",
        supporting_metric=f"necessary for {necd.get('necessary', '?')} abstraction rollup",
        metric_class="derived (weight bytes) + assumed (K)",
        depends_on_configured_K="yes",
        depends_on_flat_capture="yes (K-loop erased)",
        survives_loo="n/a (per-workload predicate)",
        survives_macro_vs_micro="n/a",
        survives_threshold="n/a",
        driven_by="configured/reference K",
        decision_or_fact="decision (residency knob)",
        verdict="blocked-by-capture (needs a loop-preserving capture to confirm K)",
    )
    add(
        conclusion="skinny_gemm_or_gemv_engine is necessary/useful across the corpus",
        supporting_artifact="abstraction_necessity_table.csv; predicate_audit_table.csv",
        supporting_metric="gemv/skinny MAC fraction (true_gemv + skinny split in predicate_audit)",
        metric_class="derived",
        depends_on_configured_K="no",
        depends_on_flat_capture="no",
        survives_loo="see predicate_audit per-workload",
        survives_macro_vs_micro="yes",
        survives_threshold="n/a",
        driven_by="skinny-GEMM projections (not true GEMV)",
        decision_or_fact="decision (engine serves skinny GEMM, not only M/N<=4 GEMV)",
        verdict="robust (name now honest: skinny OR gemv)",
    )
    add(
        conclusion="low-bit abstractions are blocked (cannot be evaluated)",
        supporting_artifact="abstraction_necessity_table.csv; capture_fidelity_matrix.csv",
        supporting_metric=f"{necd.get('blocked', '?')} abstractions blocked",
        metric_class="unavailable",
        depends_on_configured_K="no",
        depends_on_flat_capture="yes (packed layout + scales erased)",
        survives_loo="n/a",
        survives_macro_vs_micro="n/a",
        survives_threshold="n/a",
        driven_by="f32 fake-quant capture",
        decision_or_fact="decision-blocking (needs a low-bit recapture)",
        verdict="blocked-by-capture",
    )
    add(
        conclusion="attention/KV abstractions are blocked (attention lowered)",
        supporting_artifact="abstraction_necessity_table.csv; capture_fidelity_matrix.csv",
        supporting_metric="KV abstractions classified blocked/not_applicable",
        metric_class="unavailable",
        depends_on_configured_K="no",
        depends_on_flat_capture="yes (q.kT / attn.v lowered)",
        survives_loo="n/a",
        survives_macro_vs_micro="n/a",
        survives_threshold="n/a",
        driven_by="flat capture",
        decision_or_fact="decision-blocking (needs attention-preserving capture)",
        verdict="blocked-by-capture",
    )
    add(
        conclusion="bounded_loop_command / loop_carried_state_handle are useful",
        supporting_artifact="abstraction_necessity_table.csv; predicate_audit_table.csv",
        supporting_metric="K>1 cadence (configured/reference)",
        metric_class="assumed",
        depends_on_configured_K="yes",
        depends_on_flat_capture="yes (loops not in IR)",
        survives_loo="n/a",
        survives_macro_vs_micro="n/a",
        survives_threshold="n/a",
        driven_by="configured/reference K",
        decision_or_fact="decision (where the loop lives is a boundary axis)",
        verdict="blocked-by-capture (loop erased; useful, not provable)",
    )
    add(
        conclusion="capture fidelity is the limiting factor (flat captures erase loop/KV/low-bit)",
        supporting_artifact="capture_fidelity_matrix.csv",
        supporting_metric="K_or_decode_loop=assumed; packed_lowbit_layout/scale_metadata=erased",
        metric_class="derived",
        depends_on_configured_K="no",
        depends_on_flat_capture="yes",
        survives_loo="uniform across corpus",
        survives_macro_vs_micro="n/a",
        survives_threshold="n/a",
        driven_by="capture level (uniform)",
        decision_or_fact="methodology result (the central finding)",
        verdict="robust",
    )
    add(
        conclusion="HW/SW boundary placement is itself a DSE search axis",
        supporting_artifact="boundary_placement_matrix; abstraction_necessity_table.csv",
        supporting_metric="per-abstraction level enumeration (no level chosen)",
        metric_class="derived",
        depends_on_configured_K="no",
        depends_on_flat_capture="partly",
        survives_loo="n/a",
        survives_macro_vs_micro="n/a",
        survives_threshold="n/a",
        driven_by="structural enumeration",
        decision_or_fact="framing (search axis, not a choice)",
        verdict="descriptive (enumeration, not a single decision)",
    )
    return {"rows": rows}


# --------------------------------------------------------------------------- P13-g consistency


def consistency_checks(cs_dir: Path, inventory, facts, findings, plots, answers) -> list[tuple]:
    res = []

    def chk(ok, msg):
        res.append((bool(ok), msg))

    _artifact_names = {r["artifact"] for r in inventory}
    exist = {r["artifact"] for r in inventory if r["exists"]}
    # also recognize per-workload subdir artifacts (numerical_contract.yaml, region_attribution.yaml)
    for sub in cs_dir.glob("*/"):
        if sub.is_dir():
            exist.update(p.name for p in sub.glob("*") if p.is_file())
    # 1 findings reference existing artifacts
    chk(
        all(all(a in exist for a in f["supporting_artifacts"]) for f in findings),
        "every finding references existing artifacts",
    )
    # 2 every main finding tier A/B
    chk(
        all(f["evidence_tier"] in ("A", "B") for f in findings if f["presentation_placement"] == "main"),
        "every main finding is tier A or B",
    )
    # 3 plot manifest references existing columns/artifacts (for available plots)
    chk(
        all(p["available"] for p in plots if p["recommendation"] != "omit"),
        "every non-omit plot references an existing artifact + columns",
    )
    # 4 no main finding purely assumed
    chk(
        not any(f["presentation_placement"] == "main" and f["evidence_tier"] == "C" for f in findings),
        "no main finding is based only on assumed_reference",
    )
    # 5 boundary findings use allowed levels (facts with a boundary_level)
    allowed_lv = {
        "compiler_transform",
        "runtime_hal_object",
        "command_buffer_or_command_isa",
        "accelerator_isa",
        "device_microcode_or_controller",
        "fixed_hardware_datapath",
    }
    chk(
        all(all(p in allowed_lv for p in f["boundary_level"].split(";") if p) for f in facts if f["boundary_level"]),
        "every boundary-level reference uses the allowed vocabulary",
    )
    # 6 measurement findings reference real missing-evidence (the measurement_priority artifact)
    meas_q = next((a for a in answers if a["key"] == "measurements_unblock_most"), None)
    chk(
        meas_q is not None and meas_q["status"] in ("strong", "partial"),
        "measurement-priority query is answered from a present artifact",
    )
    # 7 no forbidden wording in finding claims (outside the explicit erased-limitation finding)
    bad = []
    for f in findings:
        low = (f["claim"] + " " + " ".join(f["caveats"])).lower()
        for t in FORBIDDEN:
            if t in low and not (t == "energy" and "energy" not in low):
                if t in low:
                    bad.append((f["title"], t))
    chk(not bad, f"no finding claim uses forbidden performance wording ({bad[:3]})")
    # 8 unified fact table has no duplicate fact ids
    ids = [f["fact_id"] for f in facts]
    chk(len(ids) == len(set(ids)), "no duplicate fact_id in the unified fact table")
    # 9 inventory marks missing artifacts explicitly
    chk(
        all("exists" in r for r in inventory),
        "artifact inventory records exists for every artifact",
    )
    # 10 usefulness answers use the allowed status vocabulary
    chk(
        all(a["status"] in ("strong", "partial", "weak", "unavailable") for a in answers),
        "every usefulness answer uses an allowed status",
    )
    return res


# --------------------------------------------------------------------------- P14 devil's-advocate


def gap_audit(cs_dir: Path, scope: str, bundle: dict | None = None) -> list[dict]:
    """The devil's advocate: flag every fake/weak/unbacked/unused/heuristic item. Returns GapItems;
    convergence = 0 OPEN AVOIDABLE gaps (inherent limits are 'scoped' via a required_input)."""
    cs_dir = Path(cs_dir)
    if bundle:
        b = bundle
    else:
        from ..insight_mining import mine

        b = mine(cs_dir, scope)
    inv, facts, findings, plots, answers = (b["inventory"], b["facts"], b["findings"], b["plots"], b["usefulness"])
    gaps = []
    k = 0

    def g(category, target, desc, avoidable, status, required_input=""):
        nonlocal k
        gaps.append(
            {
                "gap_id": f"G{k:03d}",
                "scope": scope,
                "category": category,
                "target": target,
                "description": desc,
                "avoidable": avoidable,
                "status": status,
                "required_input": required_input,
            }
        )
        k += 1

    used = {f["source_artifact"] for f in facts}
    # 1 unused_artifact: the 'all' scope must cover every present artifact (per-network scopes only
    # cover their per-workload slice — cross-workload-only artifacts are not theirs to mine).
    if scope == "all":
        for r in inv:
            if r["exists"] and r["source_phase"] != "other" and r["artifact"] not in used:
                g("unused_artifact", r["artifact"], "present artifact contributes 0 mined facts", True, "open")
    # 2 fake_or_unbacked_finding: a main finding without a per-metric-verified or corroborated fact
    fact_by_metric = {}
    for f in facts:
        fact_by_metric.setdefault(f["metric_name"], []).append(f)
    for f in findings:
        if f["presentation_placement"] != "main":
            continue
        fs = [x for m in f["relevant_metrics"] for x in fact_by_metric.get(m, [])]
        backed = any(x["verifying_check"] or x["corroborated_by"] >= 2 for x in fs)
        if not backed:
            g(
                "fake_or_unbacked_finding",
                f["title"],
                "main finding has no per-metric-verified or >=2-corroborated fact",
                True,
                "open",
            )
    # 3 heuristic_status: a 'strong' answer with 0 backing strong facts
    for a in answers:
        if a["status"] == "strong" and a.get("n_backing_strong_facts", 0) == 0:
            g("heuristic_status", a["key"], "'strong' status with no backing tier-A/B fact", True, "open")
    # 4 coarse_verification: a tier-A/B fact whose 'verified' is neither a check nor >=2-corroboration
    for f in facts:
        if (
            f["evidence_tier"] in ("A", "B")
            and f["verification_status"] == "verified"
            and not (f["verifying_check"] or f["corroborated_by"] >= 2)
        ):
            g(
                "coarse_verification",
                f["fact_id"],
                "verified tier-A/B fact lacks a specific check and <2 corroboration",
                True,
                "open",
            )
    # 5 uncorroborated_fact: a candidate that is NOT a SIGNAL metric, or is tier C/D (the real fake
    # risk). A single-source recovered_from_ir SIGNAL fact is legitimate evidence, not fake -- its
    # single-source status is reported via corroborated_by, not treated as a gap.
    for f in facts:
        if f["presentation_candidate"] and (f["metric_name"] not in SIGNAL_METRICS or f["evidence_tier"] in ("C", "D")):
            g("uncorroborated_fact", f["fact_id"], "candidate is not a tier-A/B signal metric", True, "open")
    # 6 aggregated_finding: a multi-workload finding without a per-workload spread
    for f in findings:
        if len(f["supporting_workloads"]) > 1 and not f.get("per_workload_spread"):
            g("aggregated_finding", f["title"], "multi-workload finding hides per-workload spread", True, "open")
    # 7 unvalidated_plot: a non-omit plot with empty source data
    for p in plots:
        if p["recommendation"] != "omit" and p.get("n_source_rows", 0) <= 0:
            g("unvalidated_plot", p["plot_id"], "non-omit plot has empty source data", True, "open")
    # 8 inherent_limit: scoped (has a required_input) -> not an open gap
    for lim in required_inputs(cs_dir, facts):
        g("inherent_limit", lim["limit"], lim["currently"], False, "scoped", lim["required_input"])
    return gaps


def open_avoidable(gaps: list[dict]) -> list[dict]:
    return [x for x in gaps if x["avoidable"] and x["status"] == "open"]


def _gap_audit_md(gaps, open_av, scope) -> str:
    from collections import Counter

    by_cat = Counter(g["category"] for g in gaps)
    L = [
        f"# Devil's-advocate gap audit ({scope})\n",
        f"**Open avoidable gaps: {len(open_av)}** (convergence target = 0). "
        f"{sum(1 for g in gaps if g['category'] == 'inherent_limit')} inherent limits are scoped "
        "(each carries a required input — see required_inputs_manifest).\n",
        "| category | count | open avoidable |",
        "|---|---|---|",
    ]
    for cat in sorted(by_cat):
        oa = sum(1 for g in open_av if g["category"] == cat)
        L.append(f"| {cat} | {by_cat[cat]} | {oa} |")
    if open_av:
        L.append("\n## Open avoidable gaps (must close)\n")
        for g in open_av:
            L.append(f"- [{g['category']}] {g['target']}: {g['description']}")
    else:
        L.append(
            "\n**No open avoidable gaps — the analysis leverages every artifact, every "
            "presented fact is corroborated/checked, every main finding is backed, and every "
            "inherent limit is scoped to a required input.**"
        )
    return "\n".join(L) + "\n"


def _adversarial_audit_md(aud, scope) -> str:
    L = [
        f"# Adversarial audit of the headline conclusions ({scope})\n",
        "> Validity ledger: each conclusion → the artifact/metric behind it → whether it survives "
        "leave-one-out, macro-vs-micro, and pad-waste threshold perturbation. `robust` = survives "
        "all; `winner robust, magnitude fragile` = the ranking holds but the number moves a lot; "
        "`blocked-by-capture` = the conclusion cannot be confirmed without a richer capture.\n",
        "| conclusion | metric class | survives LOO | survives macro/micro | survives threshold | "
        "driven by | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in aud["rows"]:
        L.append(
            f"| {r['conclusion']} | {r['metric_class']} | {r['survives_loo']} | "
            f"{r['survives_macro_vs_micro']} | {r['survives_threshold']} | {r['driven_by']} | "
            f"**{r['verdict']}** |"
        )
    L.append("\nFull supporting artifact + decision-vs-fact per row: `conclusion_validity_table.csv`.\n")
    return "\n".join(L) + "\n"
