"""Write the existing ordered mining artifacts without changing their schemas or text.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .artifacts import _inventory_md
from .audit import _AUDIT_COLS, _adversarial_audit_md, _gap_audit_md
from .capture import _ABLATION_COLS, _ablation_md, _capture_fidelity_md
from .evidence import _CANON_COLS, FACT_COLUMNS, _strength_md
from .findings import _findings_md, _required_inputs_md, _scorecard16_md, _scorecard_md
from .influence import _INFLUENCE_COLS, _LOO_DELTA_COLS, _PER_FAMILY_COLS, _influence_md, _per_family_md
from .necessity import _PRED_COLS, _necessity_cols, _necessity_md, _predicate_audit_md
from .omissions import _ERASE_COLS, _OMIT_COLS, _erasure_evidence_md, _omitted_op_md
from .plots import _plot_md, _plots_index_md
from .primitives import (
    _FROB_COLS,
    _UNCOV_COLS,
    _frontier_md,
    _frontier_robustness_md,
    _frontier_rows,
    _pareto_md,
    _robustness_md,
)
from .requirements import _ENV_COLS, _deadline_sensitivity_md, _residency_tradeoff_md
from .summaries import _findings_digest_md, _p17_findings_md, _readme_md, _signal_report_md, _slides_md
from .workloads import _COVER_COLS, _HOTSPOT_COLS, _corpus_md, _coverage_md, _family_md, _hotspots_md

# --------------------------------------------------------------------------- emitters (run folder)


def _csv_text(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
    return buf.getvalue()


def emit_run(bundle: dict, run_dir, rendered: list[str]) -> None:
    """Write all P13 deliverables for one mined scope into the run folder (non-committed)."""
    from merlin.common.artifacts import Artifact, yaml_artifact

    run_dir = Path(run_dir)
    scope = bundle["scope"]
    inv, facts = bundle["inventory"], bundle["facts"]
    Artifact(
        "artifact_inventory.csv",
        _csv_text(
            inv,
            [
                "artifact",
                "path",
                "exists",
                "source_phase",
                "file_type",
                "row_count",
                "schema_valid",
                "has_evidence_fields",
                "checked_by_verify",
                "unavailable",
                "unknown",
                "assumed_reference",
                "derived_requirement",
                "measured",
            ],
        ),
    ).write(run_dir)
    Artifact("artifact_inventory.md", _inventory_md(inv)).write(run_dir)
    Artifact("unified_fact_table.csv", _csv_text(facts, FACT_COLUMNS)).write(run_dir)
    yaml_artifact(
        "unified_fact_table.yaml",
        {"unified_fact_table": {"scope": scope, "facts": facts}},
        header=f"unified_fact_table: {scope} (no performance claim)",
    ).write(run_dir)
    s = bundle["evidence_strength"]
    Artifact(
        "evidence_strength_table.csv",
        _csv_text([{"metric": k, "value": v} for k, v in s.items() if not isinstance(v, dict)], ["metric", "value"]),
    ).write(run_dir)
    Artifact("evidence_strength_report.md", _strength_md(s, scope)).write(run_dir)
    Artifact("dse_usefulness_scorecard.md", _scorecard_md(bundle["usefulness"], scope)).write(run_dir)
    yaml_artifact(
        "dse_usefulness_answers.yaml",
        {"dse_usefulness_answers": {"scope": scope, "answers": bundle["usefulness"]}},
        header=f"dse_usefulness_answers: {scope}",
    ).write(run_dir)
    Artifact("presentation_candidate_findings.md", _findings_md(bundle["findings"], scope)).write(run_dir)
    Artifact(
        "presentation_candidate_findings.csv",
        _csv_text(
            [
                {
                    **f,
                    "supporting_artifacts": "; ".join(f["supporting_artifacts"]),
                    "supporting_workloads": "; ".join(f["supporting_workloads"]),
                    "relevant_metrics": "; ".join(f["relevant_metrics"]),
                    "caveats": "; ".join(f["caveats"]),
                }
                for f in bundle["findings"]
            ],
            [
                "title",
                "claim",
                "evidence_tier",
                "presentation_placement",
                "max_corroborated_by",
                "per_workload_spread",
                "supporting_artifacts",
                "supporting_workloads",
                "relevant_metrics",
                "dse_implication",
                "caveats",
                "suggested_plot",
                "forbidden_claim_risk",
            ],
        ),
    ).write(run_dir)
    # P14: gap-audit report + required-inputs manifest
    gaps = bundle.get("gaps", [])
    oa = bundle.get("open_avoidable_gaps", [])
    Artifact(
        "gap_audit_report.csv",
        _csv_text(
            gaps, ["gap_id", "scope", "category", "target", "description", "avoidable", "status", "required_input"]
        ),
    ).write(run_dir)
    Artifact("gap_audit_report.md", _gap_audit_md(gaps, oa, scope)).write(run_dir)
    ri = bundle.get("required_inputs", [])
    yaml_artifact(
        "required_inputs_manifest.yaml",
        {"required_inputs_manifest": {"scope": scope, "inherent_limits": ri}},
        header=f"required_inputs_manifest: {scope} (exact inputs to close each limit)",
    ).write(run_dir)
    Artifact("required_inputs_manifest.md", _required_inputs_md(ri, scope)).write(run_dir)
    yaml_artifact(
        "plot_manifest.yaml",
        {"plot_manifest": {"scope": scope, "plots": bundle["plots"], "rendered": rendered}},
        header=f"plot_manifest: {scope}",
    ).write(run_dir)
    Artifact("plot_manifest.md", _plot_md(bundle["plots"], scope, rendered)).write(run_dir)
    checks = bundle["consistency_checks"]
    npass = sum(1 for ok, _ in checks if ok)
    cm = [f"# Cross-artifact consistency checks ({scope})\n", f"**{npass}/{len(checks)} passed.**\n"]
    cm += [f"- [{'PASS' if ok else 'FAIL'}] {msg}" for ok, msg in checks]
    Artifact("consistency_checks.md", "\n".join(cm) + "\n").write(run_dir)
    Artifact("insight_mining_README.md", _readme_md(bundle, rendered)).write(run_dir)
    # P15 signal-first study deliverables (signal-only; every output answers a DSE question)
    Artifact("canonical_signal_table.csv", _csv_text(bundle["canonical_signal"], _CANON_COLS)).write(run_dir)
    h = bundle["hotspots"]
    Artifact("per_operator_hotspots.csv", _csv_text(h["rows"], _HOTSPOT_COLS)).write(run_dir)
    Artifact("per_operator_hotspots.md", _hotspots_md(h, scope)).write(run_dir)
    # cross-workload artifacts are corpus-level — only meaningful (and only emitted) in 'all'
    if bundle["abstraction_coverage"]:
        Artifact("abstraction_coverage_table.csv", _csv_text(bundle["abstraction_coverage"], _COVER_COLS)).write(
            run_dir
        )
        Artifact("abstraction_coverage.md", _coverage_md(bundle["abstraction_coverage"], scope)).write(run_dir)
    if bundle["family_summary"]["families"]:
        Artifact("workload_family_summary.md", _family_md(bundle["family_summary"], scope)).write(run_dir)
    if bundle["corpus_plan"]:
        Artifact("corpus_expansion_plan.md", _corpus_md(bundle["corpus_plan"], scope)).write(run_dir)
    Artifact("signal_findings_report.md", _signal_report_md(bundle, scope)).write(run_dir)
    # P16 decision-frontier & robustness deliverables
    nec = bundle["abstraction_necessity"]
    Artifact("abstraction_necessity_table.csv", _csv_text(nec["rows"], _necessity_cols(nec["workloads"]))).write(
        run_dir
    )
    Artifact("abstraction_necessity.md", _necessity_md(nec, scope)).write(run_dir)
    Artifact(
        "primitive_set_frontier.csv",
        _csv_text(
            _frontier_rows(bundle["primitive_frontier"]),
            ["set_size", "primitive_set", "worst_workload_coverage", "macro_coverage", "micro_coverage", "max_regret"],
        ),
    ).write(run_dir)
    Artifact("primitive_set_frontier.md", _frontier_md(bundle["primitive_frontier"], scope)).write(run_dir)
    par = bundle["operator_pareto"]
    _pcols = (
        ["workload", "n_ops"]
        + [f"k_{m}_{int(t * 100)}" for t in par["thresholds"] for m in ("macs", "wbytes")]
        + ["top_op", "top_op_mac_share", "n_distinct_shapes", "top_shape_multiplicity"]
    )
    Artifact("operator_pareto_hotspots.csv", _csv_text(par["rows"], _pcols)).write(run_dir)
    Artifact("operator_pareto_hotspots.md", _pareto_md(par, scope)).write(run_dir)
    cf = bundle["capture_fidelity"]
    Artifact("capture_fidelity_matrix.csv", _csv_text(cf["matrix"], ["feature"] + cf["workloads"])).write(run_dir)
    Artifact("capture_fidelity_matrix.md", _capture_fidelity_md(cf, scope)).write(run_dir)
    Artifact("decision_question_scorecard.md", _scorecard16_md(bundle["decision_scorecard"], scope)).write(run_dir)
    Artifact("presentation_slide_candidates.md", _slides_md(bundle, scope)).write(run_dir)
    if bundle["robustness"]["findings"]:
        Artifact("leave_one_workload_out.md", _robustness_md(bundle["robustness"], scope)).write(run_dir)
    # P17 adversarial audit & predicate audit (predicate audit always; corpus analyses at 'all')
    pa = bundle["predicate_audit"]
    if pa["rows"]:
        Artifact("predicate_audit_table.csv", _csv_text(pa["rows"], _PRED_COLS)).write(run_dir)
        Artifact("predicate_audit_table.md", _predicate_audit_md(pa, scope)).write(run_dir)
    aud = bundle["adversarial_audit"]
    Artifact("conclusion_validity_table.csv", _csv_text(aud["rows"], _AUDIT_COLS)).write(run_dir)
    Artifact("adversarial_audit_report.md", _adversarial_audit_md(aud, scope)).write(run_dir)
    fro = bundle.get("primitive_frontier_robustness", {})
    if fro.get("rows"):
        Artifact("primitive_frontier_robustness.csv", _csv_text(fro["rows"], _FROB_COLS)).write(run_dir)
        Artifact("primitive_frontier_robustness.md", _frontier_robustness_md(fro, scope)).write(run_dir)
        Artifact("uncovered_ops_by_primitive_set.csv", _csv_text(fro["uncovered_rows"], _UNCOV_COLS)).write(run_dir)
    inf = bundle.get("macro_micro_influence", {})
    if inf.get("rows"):
        Artifact("macro_micro_influence_table.csv", _csv_text(inf["rows"], _INFLUENCE_COLS)).write(run_dir)
        Artifact("leave_one_out_delta_table.csv", _csv_text(inf["loo_rows"], _LOO_DELTA_COLS)).write(run_dir)
        Artifact("workload_influence_report.md", _influence_md(inf, scope)).write(run_dir)
    # P17 requirements envelope (Part E) + omitted-operator accounting (Part F)
    env = bundle.get("timing_envelope", {})
    if env.get("rows"):
        Artifact("timing_requirement_envelope.csv", _csv_text(env["rows"], _ENV_COLS)).write(run_dir)
        Artifact("deadline_sensitivity_report.md", _deadline_sensitivity_md(env, scope)).write(run_dir)
        Artifact("residency_vs_deadline_tradeoff.md", _residency_tradeoff_md(env, scope)).write(run_dir)
    om = bundle.get("omitted_ops", {})
    if om.get("rows"):
        Artifact("visible_vs_erased_work_table.csv", _csv_text(om["rows"], _OMIT_COLS)).write(run_dir)
        Artifact("omitted_operator_accounting.md", _omitted_op_md(om, scope)).write(run_dir)
    # Stage C: capture-erasure evidence, per-family decomposition, and the critique's exact-name aliases
    ce = bundle.get("capture_erasure", {})
    if ce.get("rows"):
        Artifact("capture_erasure_evidence.csv", _csv_text(ce["rows"], _ERASE_COLS)).write(run_dir)
        Artifact("capture_erasure_evidence.md", _erasure_evidence_md(ce, scope)).write(run_dir)
    pf = bundle.get("per_family", {})
    if pf.get("rows"):
        Artifact("per_family_summary.csv", _csv_text(pf["rows"], _PER_FAMILY_COLS)).write(run_dir)
        Artifact("per_family_summary.md", _per_family_md(pf, scope)).write(run_dir)
    if inf.get("rows"):  # alias: Part A's `metric_stability_table.csv` (= macro/micro + LOO delta)
        Artifact("metric_stability_table.csv", _csv_text(inf["rows"], _INFLUENCE_COLS)).write(run_dir)
    if om.get("rows"):  # alias: Part A's `known_omissions_table.csv` (= recovered/erased work)
        Artifact("known_omissions_table.csv", _csv_text(om["rows"], _OMIT_COLS)).write(run_dir)
    ab = bundle.get("capture_ablation", {})  # Stage B: real capture-level ablation
    if ab.get("rows"):
        Artifact("capture_level_ablation.csv", _csv_text(ab["rows"], _ABLATION_COLS)).write(run_dir)
        Artifact("capture_level_ablation.md", _ablation_md(ab, scope)).write(run_dir)
    if env.get("rows"):  # P17 final report (corpus-level, when the requirements envelope exists)
        Artifact("P17_FINDINGS.md", _p17_findings_md(bundle, scope)).write(run_dir)
    Artifact("DSE_FINDINGS.md", _findings_digest_md(bundle, bundle.get("cs_dir", run_dir), rendered)).write(run_dir)
    Artifact("presentation_plots_index.md", _plots_index_md(bundle["plots"], scope, rendered, "generated_plots")).write(
        run_dir
    )
