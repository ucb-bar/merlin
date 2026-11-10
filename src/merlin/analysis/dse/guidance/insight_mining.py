"""Compose evidence-mining analyses over explicit case-study inputs.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .insights.artifacts import artifact_inventory
from .insights.audit import adversarial_audit, consistency_checks, gap_audit, open_avoidable
from .insights.capture import capture_fidelity, capture_level_ablation
from .insights.evidence import canonical_signal_table, evidence_strength
from .insights.facts import unified_facts
from .insights.findings import decision_scorecard, presentation_findings, required_inputs, usefulness
from .insights.influence import macro_micro_influence, per_family_summary
from .insights.necessity import abstraction_necessity, predicate_audit
from .insights.omissions import capture_erasure_evidence, operator_recovery_accounting
from .insights.plots import plot_manifest
from .insights.primitives import operator_pareto, primitive_frontier_robustness, primitive_set_frontier, robustness
from .insights.requirements import timing_requirement_envelope
from .insights.workloads import (
    abstraction_coverage,
    corpus_expansion_plan,
    per_operator_hotspots,
    workload_family_summary,
)

# --------------------------------------------------------------------------- bundle


def mine(cs_dir, scope: str) -> dict:
    cs_dir = Path(cs_dir)
    inv = artifact_inventory(cs_dir)
    facts = unified_facts(cs_dir, scope)
    strength = evidence_strength(facts)
    answers = usefulness(cs_dir, scope, facts)
    findings = presentation_findings(facts, answers)
    plots = plot_manifest(cs_dir, scope)
    checks = consistency_checks(cs_dir, inv, facts, findings, plots, answers)
    bundle = {
        "scope": scope,
        "inventory": inv,
        "facts": facts,
        "evidence_strength": strength,
        "usefulness": answers,
        "findings": findings,
        "plots": plots,
        "consistency_checks": checks,
        "required_inputs": required_inputs(cs_dir, facts),
    }
    bundle["gaps"] = gap_audit(cs_dir, scope, bundle)
    bundle["open_avoidable_gaps"] = open_avoidable(bundle["gaps"])
    # P15 signal-first study layer (signal-only, every output answers a DSE question). Per-network
    # scopes get their OWN canonical table + hotspots; the cross-workload artifacts (coverage /
    # family / corpus) are corpus-level by nature and are emitted only in the 'all' scope (a single
    # workload's coverage/regret/family is degenerate, so dumping the corpus-wide table into a
    # per-network folder would misrepresent it).
    bundle["canonical_signal"] = canonical_signal_table(facts)
    bundle["hotspots"] = per_operator_hotspots(cs_dir, scope)
    is_all = scope == "all"
    bundle["abstraction_coverage"] = abstraction_coverage(cs_dir) if is_all else []
    bundle["family_summary"] = (
        workload_family_summary(cs_dir, findings)
        if is_all
        else {"families": {}, "family_specific_findings": [], "cross_family_findings": []}
    )
    bundle["corpus_plan"] = corpus_expansion_plan(cs_dir) if is_all else {}
    bundle["cs_dir"] = str(cs_dir)
    # P16 decision-frontier & robustness (corpus-level analyses computed at 'all'; the per-network
    # necessity/fidelity are still meaningful per workload but the frontier/robustness need the corpus)
    bundle["abstraction_necessity"] = abstraction_necessity(cs_dir)
    bundle["primitive_frontier"] = primitive_set_frontier(cs_dir)
    bundle["operator_pareto"] = operator_pareto(cs_dir)
    bundle["robustness"] = robustness(cs_dir) if is_all else {"workloads": [], "findings": []}
    bundle["capture_fidelity"] = capture_fidelity(cs_dir)
    bundle["decision_scorecard"] = decision_scorecard(bundle)
    # P17 adversarial audit: predicate audit is per-workload meaningful; the frontier-robustness and
    # influence analyses are corpus-level (computed only at 'all'); the conclusion ledger reads them.
    bundle["predicate_audit"] = predicate_audit(cs_dir)
    bundle["primitive_frontier_robustness"] = (
        primitive_frontier_robustness(cs_dir) if is_all else {"workloads": [], "rows": [], "uncovered_rows": []}
    )
    bundle["macro_micro_influence"] = (
        macro_micro_influence(cs_dir) if is_all else {"workloads": [], "rows": [], "loo_rows": []}
    )
    bundle["adversarial_audit"] = adversarial_audit(bundle)
    # P17 requirements envelope + omitted-op accounting (Parts E/F; corpus-level at 'all')
    bundle["timing_envelope"] = timing_requirement_envelope(cs_dir) if is_all else {"workloads": [], "rows": []}
    bundle["omitted_ops"] = operator_recovery_accounting(cs_dir)
    bundle["capture_erasure"] = capture_erasure_evidence(cs_dir) if is_all else {"rows": []}
    bundle["per_family"] = per_family_summary(cs_dir) if is_all else {"rows": []}
    bundle["capture_ablation"] = capture_level_ablation(cs_dir) if is_all else {"rows": [], "unlock": []}
    return bundle
