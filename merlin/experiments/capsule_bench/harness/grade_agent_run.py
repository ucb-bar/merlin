#!/usr/bin/env python3
"""Grade a frozen agent run through capsule_bench_v0 (post-agent, full-access phase).

Public/dev phase (repairable in a real run) -> freeze -> hidden phase (after freeze only). Writes
score_capsule.json per phase, an iteration_000 snapshot, run_manifest.yaml, and final_report.md.
Runs OUTSIDE the agent sandbox (needs spike/verilator + the hidden goldens).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

import _common as C

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.targetgen import capsule_grade as CG  # noqa: E402
from merlin.targetgen import capsule_runner as CR  # noqa: E402
import freeze_run  # noqa: E402


# This is the certification tier for the Arm-4 Gemmini functional experiment.  A cheaper-tier pass is
# useful iteration feedback, but is not a completed formal run.  Keep the requirement next to the
# post-freeze grader: this is the only process allowed to read the hidden capsules.
FORMAL_REQUIRED_TIER = "L3"


def phase_completion(score: Mapping | None, *, required_tier: str = FORMAL_REQUIRED_TIER) -> tuple[bool, list[str]]:
    """Fail-closed completion predicate for one official grading phase.

    ``functional_pass`` alone is insufficient: an empty/mis-rooted suite used to look like ``0/0`` and
    the headline can omit which tier supplied the evidence.  A formal phase therefore has to be
    non-vacuous, fully passing, gradeable, complete at the predeclared L3 tier, and RTL-backed for every
    passing capsule.  Malformed or absent fields are refusals, never inferred successes.
    """
    if not isinstance(score, Mapping):
        return False, ["score_missing_or_malformed"]

    reasons: list[str] = []
    n_capsules = score.get("n_capsules")
    n_passed = score.get("n_passed")
    if not isinstance(n_capsules, int) or isinstance(n_capsules, bool) or n_capsules <= 0:
        reasons.append("capsule_set_empty_or_malformed")
    if not isinstance(n_passed, int) or isinstance(n_passed, bool):
        reasons.append("pass_count_missing_or_malformed")
    elif isinstance(n_capsules, int) and not isinstance(n_capsules, bool) and n_passed != n_capsules:
        reasons.append("not_all_capsules_passed")
    if score.get("functional_pass") != 1:
        reasons.append("functional_pass_not_true")
    if score.get("gradeable") is not True:
        reasons.append("numeric_grade_not_gradeable")
    if score.get("integrity_status") != "clean":
        reasons.append("submission_integrity_not_clean")
    if score.get("numeric_all_exact") is not True:
        reasons.append("numeric_exactness_not_complete")
    # Operator capsules emit one kernel and must pass decoded instruction-trace conformance.  Whole
    # models are multi-kernel host/accelerator compositions and have no honest single model-level trace;
    # they instead carry a distinct, fail-closed accelerator-execution check.  Require both applicable
    # scopes, and require their counts to cover the entire admitted denominator -- never stamp a model
    # trace pass for an artifact that does not exist.
    scope = score.get("structural_evidence_scope")
    if not isinstance(scope, Mapping):
        reasons.append("structural_evidence_scope_missing_or_malformed")
    else:
        n_trace = scope.get("n_instruction_trace_capsules")
        n_model = scope.get("n_model_execution_capsules")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (n_trace, n_model)):
            reasons.append("structural_evidence_scope_counts_malformed")
        else:
            if isinstance(n_capsules, int) and not isinstance(n_capsules, bool) \
                    and n_trace + n_model != n_capsules:
                reasons.append("structural_evidence_scope_denominator_mismatch")
            if n_trace > 0 and score.get("trace_all_pass") is not True:
                reasons.append("trace_conformance_not_complete")
            if n_model > 0 and score.get("model_execution_all_pass") is not True:
                reasons.append("model_execution_evidence_not_complete")
    if score.get("structural_evidence_all_pass") is not True:
        reasons.append("structural_evidence_not_complete")

    # Cohort selection happens before the scored denominator.  Require its arithmetic and policy in the
    # formal record so a hidden source-pool capsule cannot disappear through an undocumented filter.
    # `frozen_target_capability_operand_dtype` is code-derived after freeze and excludes only a dtype
    # absent from every frozen target capability; it does not expose held-out names to the agent.
    admission = score.get("cohort_admission")
    if not isinstance(admission, Mapping):
        reasons.append("cohort_admission_missing_or_malformed")
    else:
        policy = admission.get("policy")
        if policy not in {"all_discovered", "frozen_target_capability_operand_dtype",
                          "descriptor_capability_and_resource_v1"}:
            reasons.append("cohort_admission_policy_invalid")
        source_n = admission.get("n_source_capsules")
        admitted_n = admission.get("n_admitted_capsules")
        capability_excluded_n = admission.get("n_capability_excluded")
        resource_excluded_n = admission.get("n_resource_excluded")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
               for v in (source_n, admitted_n, capability_excluded_n, resource_excluded_n)):
            reasons.append("cohort_admission_counts_malformed")
        else:
            if source_n <= 0 or admitted_n <= 0:
                reasons.append("cohort_admission_vacuous")
            if source_n != admitted_n + capability_excluded_n + resource_excluded_n:
                reasons.append("cohort_admission_arithmetic_invalid")
            if isinstance(n_capsules, int) and not isinstance(n_capsules, bool) \
                    and admitted_n != n_capsules:
                reasons.append("cohort_admission_denominator_mismatch")
        for field in ("excluded_name_set_sha256", "admitted_name_set_sha256"):
            digest = admission.get(field)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(ch not in "0123456789abcdef" for ch in digest)):
                reasons.append(f"cohort_admission_{field}_invalid")
        if policy == "descriptor_capability_and_resource_v1":
            if admission.get("resource_policy") != "representative_l3_capstones_v1":
                reasons.append("cohort_admission_resource_policy_invalid")
            required_models = admission.get("required_admitted_models")
            if not isinstance(required_models, list) or not required_models:
                reasons.append("cohort_admission_required_models_missing")

    # The diagnostic score denominator deliberately excludes work that was never measured. Formal
    # completion must not inherit that smaller denominator: otherwise a screen budget, model timeout,
    # deferred capstone, or hidden capsule marked ineligible can disappear while the surviving subset
    # reports all-pass. Require each unmeasured count explicitly and require zero.
    unmeasured_counts = (
        "n_not_graded_ineligible", "n_gated_deferred", "n_screened_only",
        "n_budget_exhausted", "n_incomplete", "n_not_gradeable_no_oracle",
    )
    for field in unmeasured_counts:
        value = score.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            reasons.append(f"{field}_missing_or_malformed")
        elif value != 0:
            reasons.append(f"{field}_nonzero")

    tier_reached = score.get("tier_reached")
    tier_count = tier_reached.get(required_tier) if isinstance(tier_reached, Mapping) else None
    if not isinstance(tier_count, int) or isinstance(tier_count, bool):
        reasons.append(f"{required_tier}_count_missing_or_malformed")
    elif isinstance(n_capsules, int) and not isinstance(n_capsules, bool) and tier_count != n_capsules:
        reasons.append(f"not_all_capsules_reached_{required_tier}")

    evidence = score.get("pass_evidence")
    rtl_backed = evidence.get("rtl_backed") if isinstance(evidence, Mapping) else None
    if not isinstance(rtl_backed, int) or isinstance(rtl_backed, bool):
        reasons.append("rtl_backed_count_missing_or_malformed")
    elif isinstance(n_capsules, int) and not isinstance(n_capsules, bool) and rtl_backed != n_capsules:
        reasons.append("not_all_capsules_rtl_backed")
    return not reasons, reasons


def _phase_manifest(score: Mapping, *, required_tier: str) -> dict:
    complete, failures = phase_completion(score, required_tier=required_tier)
    return {
        "passed": score.get("public_passed") or score.get("hidden_passed")
        or f"{score.get('n_passed')}/{score.get('n_capsules')}",
        "functional_pass": score.get("functional_pass"),
        "n_passed": score.get("n_passed"),
        "n_capsules": score.get("n_capsules"),
        "gradeable": score.get("gradeable"),
        "integrity_status": score.get("integrity_status"),
        "numeric_all_exact": score.get("numeric_all_exact"),
        "trace_all_pass": score.get("trace_all_pass"),
        "model_execution_all_pass": score.get("model_execution_all_pass"),
        "structural_evidence_all_pass": score.get("structural_evidence_all_pass"),
        "structural_evidence_scope": score.get("structural_evidence_scope"),
        "unmeasured_counts": {
            field: score.get(field) for field in (
                "n_not_graded_ineligible", "n_gated_deferred", "n_screened_only",
                "n_budget_exhausted", "n_incomplete", "n_not_gradeable_no_oracle",
            )
        },
        "highest_tier": score.get("highest_tier"),
        "tier_reached": score.get("tier_reached"),
        "pass_evidence": score.get("pass_evidence"),
        "cohort_admission": score.get("cohort_admission"),
        "formal_complete": complete,
        "completion_failures": failures,
    }


def _roots(spec: str) -> list[str]:
    """A comma-separated root list. A target's graded suite spans sibling capsule CATEGORIES, so one
    path cannot name it; the launcher resolves them from the descriptor and passes them through here."""
    return [s for s in (x.strip() for x in str(spec).split(",")) if s]


def _score(pkg, capsules, runs_root, labels, no_oracle):
    # Resolve the TARGET'S OWN oracle ladder from its contract (external_backend->program_oracle,
    # chipyard->spike/verilator, else arc) — never pass None here, which historically fell back to the
    # gemmini spike/verilator MLIR-lowering oracle and mis-graded atlas (torch-mlir run_lowering.py crash).
    # `{}` = honest no-oracle (L0/L1/trace only). sim_via is self-resolved from the contract.
    adapters = {} if no_oracle else CR.oracle_adapters(C.TARGET)
    return CG.grade(pkg, capsules_root=_roots(capsules), runs_root=runs_root, labels=labels,
                    contract=str(C.REPO / "merlin/contract"),
                    oracle_adapters=adapters, timeout=900, target=C.TARGET, no_oracle=no_oracle,
                    # The public set was materialized from the descriptor before the run.  The hidden
                    # pool cannot reveal names there, so derive its hardware-capable cohort only here,
                    # after freeze, outside the sandbox.  The score seals the source/admitted counts.
                    capability_admission=labels == {"hidden"})



def _cost_phrase(proc: dict) -> str:
    """Render the money field so a notional figure can never be read as a spend."""
    if proc.get("estimated_cost_usd") is not None:
        return f"cost=${proc['estimated_cost_usd']}"
    notional = proc.get("subscription_notional_usd")
    if notional is not None:
        return f"cost=n/a ({proc.get('billing_mode') or 'notional'}: ${notional} notional)"
    return f"cost=n/a ({proc.get('cost_unavailable_reason') or 'no usage metadata'})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--capsules", default=str(C.REPO / "merlin/contract" / "capsules"),
                    help="comma-separated capsule roots for the PUBLIC/dev phase. The launcher resolves "
                         "these from the target descriptor (primary corpus + its sibling categories); "
                         "the default is the shared parent and is only right for a target whose corpus "
                         "IS that parent.")
    ap.add_argument("--hidden-capsules", default=None,
                    help="comma-separated capsule roots for the HIDDEN phase (defaults to --capsules). "
                         "The launcher passes the target's own hidden dir so the public-only "
                         "materialized subset does not yield an empty hidden grade.")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--skip-hidden", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    pkg = run_dir / "submission"

    # Pin whole-model tile certification to the simulator assigned to the formal RTL tier by the
    # target's own manifest.  An inherited MERLIN_MESH_SIM=spike must not silently turn an L3 claim into
    # a functional-model result; each tile also records and is checked for derived_from_rtl and
    # cycle_accurate evidence by capsule_grade.model_execution_check.
    _runner_cfg = CR._config_for_target(C.TARGET, None, "i8xi8_i32")
    _formal_mesh_sim = _runner_cfg.tier_sim.get(FORMAL_REQUIRED_TIER)
    if FORMAL_REQUIRED_TIER not in _runner_cfg.rtl_tiers or not _formal_mesh_sim:
        raise SystemExit(f"formal tier {FORMAL_REQUIRED_TIER} is not bound to an RTL simulator for "
                         f"target {C.TARGET}")
    _inherited_mesh_sim = os.environ.get("MERLIN_MESH_SIM")
    os.environ["MERLIN_MESH_SIM"] = _formal_mesh_sim

    # --- public/dev phase ---
    (run_dir / "grading_public").mkdir(parents=True, exist_ok=True)
    pub = _score(str(pkg), a.capsules, str(run_dir / "grading_public"), {"public", "dev"}, a.no_oracle)
    (run_dir / "grading_public" / "score_capsule.json").write_text(json.dumps(pub, indent=2))

    # --- iteration_000 snapshot (dummy = one-shot; a real repairing agent appends more) ---
    it = run_dir / "iterations" / "iteration_000"
    it.mkdir(parents=True, exist_ok=True)
    (it / "capsule_status_after.yaml").write_text(yaml.safe_dump(
        {c["capsule"]: c["status"] for c in pub.get("per_capsule", [])}, sort_keys=True))
    (it / "first_failure.yaml").write_text(yaml.safe_dump(pub.get("first_failure_planes", {})))
    (it / "notes.md").write_text("iteration_000: initial submission grade (public/dev).\n")

    # --- freeze, then hidden phase (immutability enforced) ---
    frozen = freeze_run.freeze(run_dir)
    hid = {"available": False}
    if not a.skip_hidden:
        # re-hash the submission right before hidden grading; refuse if it changed since freeze
        recheck = C.hash_tree(pkg)["sha256"]
        mutated = recheck != frozen["submission_sha256"]
        import datetime as _dt
        fj = json.loads((run_dir / "freeze.json").read_text())
        fj["hidden_grading_started_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        fj["submission_sha256_recheck"] = recheck
        fj["workspace_mutable_after_freeze"] = bool(mutated)
        (run_dir / "freeze.json").write_text(json.dumps(fj, indent=2))
        if mutated:
            raise SystemExit(f"FREEZE VIOLATION: submission changed after freeze "
                             f"({frozen['submission_sha256'][:12]} -> {recheck[:12]}); refusing hidden grade")
        (run_dir / "grading_hidden").mkdir(parents=True, exist_ok=True)
        hid = _score(str(pkg), (a.hidden_capsules or a.capsules),
                     str(run_dir / "grading_hidden"), {"hidden"}, a.no_oracle)
        (run_dir / "grading_hidden" / "score_capsule.json").write_text(json.dumps(hid, indent=2))

    public_phase = _phase_manifest(pub, required_tier=FORMAL_REQUIRED_TIER)
    hidden_phase = _phase_manifest(hid, required_tier=FORMAL_REQUIRED_TIER)
    public_phase.update({
        "numeric_all_exact": pub.get("numeric_all_exact"),
        "trace_all_pass": pub.get("trace_all_pass"),
        "model_execution_all_pass": pub.get("model_execution_all_pass"),
        "structural_evidence_all_pass": pub.get("structural_evidence_all_pass"),
        "structural_evidence_scope": pub.get("structural_evidence_scope"),
        "first_failure_planes": pub.get("first_failure_planes"),
    })
    if a.skip_hidden:
        hidden_phase["passed"] = "not_run"
        hidden_phase["completion_failures"] = ["hidden_grade_skipped"]
        hidden_phase["formal_complete"] = False
    formal_grade_complete = bool(public_phase["formal_complete"] and hidden_phase["formal_complete"])
    completion_failures = [
        *(f"public:{reason}" for reason in public_phase["completion_failures"]),
        *(f"hidden:{reason}" for reason in hidden_phase["completion_failures"]),
    ]

    # --- process metrics (from launcher), env, run_manifest ---
    ctt = {}
    ctt_path = run_dir / "cost_time_toolcalls.yaml"
    if ctt_path.is_file():
        ctt = yaml.safe_load(ctt_path.read_text()) or {}

    manifest = {
        "run_id": run_dir.name, "arm": a.arm, "model": a.model,
        "repo_sha": frozen["repo_sha"], "submission_sha256": frozen["submission_sha256"],
        "frozen_at": frozen["frozen_at"],
        "integrity_status": pub.get("integrity_status"),
        "integrity_exempt": pub.get("integrity_exempt"),
        "public_dev": public_phase,
        "hidden": hidden_phase,
        "completion": {
            "formal_grade_complete": formal_grade_complete,
            "required_tier": FORMAL_REQUIRED_TIER,
            "failures": completion_failures,
        },
        "cycles_diagnostic": pub.get("cycles_diagnostic", {}),
        "process": {"wall_time_seconds": ctt.get("wall_time_seconds"),
                    "tokens_total": ctt.get("tokens_total"),
                    "tokens_input": ctt.get("tokens_input"),
                    "tokens_output": ctt.get("tokens_output"),
                    "tokens_reasoning": ctt.get("tokens_reasoning"),
                    "estimated_cost_usd": ctt.get("estimated_cost_usd"),
                    # A subscription-seat run has no per-token spend; its dollars are notional and
                    # kept in their own field so nothing sums them into a money budget.
                    "billing_mode": ctt.get("billing_mode"),
                    "subscription_notional_usd": ctt.get("subscription_notional_usd"),
                    "cost_unavailable_reason": ctt.get("cost_unavailable_reason"),
                    "tool_calls": ctt.get("tool_calls"),
                    "metrics_available": ctt.get("available")},
        "iterations": len(list((run_dir / "iterations").glob("iteration_*"))),
        "oracle_mode": "no_oracle(L0/L1/trace)" if a.no_oracle
                       else f"contract-routed({'+'.join(sorted(CR.oracle_adapters(C.TARGET)))})",
        # HONEST gradeability: a --no-oracle run is a structure-only smoke — the numeric verdict is
        # withheld (capsules read back not_gradeable_no_oracle, never a numeric pass), so the run is NOT
        # gradeable. A graded run carries gradeable=true. Sourced from the grade score (gradeable flag).
        "gradeable": bool(pub.get("gradeable", not a.no_oracle)),
        "gradeable_reason": (None if not a.no_oracle else
                             "numeric oracle unavailable — structural (L0/L1/trace) tiers only"),
        "model_mesh_sim": {
            "required_tier": FORMAL_REQUIRED_TIER,
            "simulator": _formal_mesh_sim,
            "inherited_value_overridden": _inherited_mesh_sim,
        },
    }
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    fr = [f"# {run_dir.name} — final report ({a.arm})", "",
          f"- integrity: {manifest['integrity_status']} (exempt={manifest['integrity_exempt']})",
          f"- public/dev: functional_pass={manifest['public_dev']['functional_pass']} "
          f"passed={manifest['public_dev']['passed']} highest_tier={manifest['public_dev']['highest_tier']}",
          f"- hidden: {manifest['hidden']['passed']} (after freeze {frozen['submission_sha256'][:12] if frozen['submission_sha256'] else 'n/a'})",
          f"- process: wall={manifest['process']['wall_time_seconds']}s "
          f"tokens={manifest['process']['tokens_total']} {_cost_phrase(manifest['process'])} "
          f"tool_calls={manifest['process']['tool_calls']} (available={manifest['process']['metrics_available']})",
          f"- oracle_mode: {manifest['oracle_mode']}",
          f"- gradeable: {manifest['gradeable']}"
          + (f" ({manifest['gradeable_reason']})" if manifest.get('gradeable_reason') else ""), ""]
    (run_dir / "final_report.md").write_text("\n".join(fr) + "\n")
    print(f"graded {run_dir.name}: public functional_pass={pub.get('functional_pass')} "
          f"({manifest['public_dev']['passed']}), hidden={manifest['hidden']['passed']}, "
          f"integrity={manifest['integrity_status']} formal_grade_complete={formal_grade_complete}")
    return 0 if formal_grade_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
