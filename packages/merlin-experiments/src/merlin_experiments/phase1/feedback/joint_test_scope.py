"""Bounded advisory smoke selection; omitted capsules remain unmeasured."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import yaml

from .. import joint_control


def focused():
    return joint_control.selected() is not None and os.environ.get("MERLIN_JOINT_GRADE_MILESTONE") != "1"


def select(root, *, submission, curriculum=None):
    state = joint_control.selected()
    if state is None:
        return root, None
    held = joint_control.held_capsules(state)
    records = []
    for path in sorted(Path(root).rglob("capsule.yaml")):
        doc = yaml.safe_load(path.read_text())
        if doc.get("label", "public") not in {"public", "dev"} or path.parent.name in held:
            continue
        family = str((doc.get("operation") or {}).get("op") or doc.get("kind") or "unknown")
        volume = sum(math.prod(x.get("shape", ())) for x in doc.get("inputs", ())
                     if all(type(d) is int and d > 0 for d in x.get("shape", ())))
        records.append((family, volume, path.parent.name, path.parent))
    if not records:
        raise ValueError("joint automatic development selection is empty")
    required = sorted(r[2] for r in records)
    if not focused():
        return [r[3] for r in records], {"scope": "milestone", "required_development": len(records),
                                        "selected": required, "pending": []}
    families = {r[0] for r in records}
    reason = "unknown impact: representative semantic-family smoke"
    chosen = None
    hint = Path(submission) / "joint-test-scope.json"
    try:
        if hint.is_symlink() or hint.stat().st_size > 16384:
            raise ValueError("invalid impact hint")
        value = json.loads(hint.read_text())
        if not isinstance(value, dict):
            raise ValueError("impact hint must be an object")
        if "capsules" in value:
            requested = value["capsules"]
            if ("families" in value or not isinstance(requested, list) or not 1 <= len(requested) <= 16
                    or any(type(name) is not str for name in requested)
                    or len(set(requested)) != len(requested) or not set(requested) <= set(required)):
                raise ValueError("exact smoke requires unique permitted development capsule IDs, at most 16")
            selected = set(requested)
            chosen = [row for row in records if row[2] in selected]
            reason = "author-requested exact development cases; advisory selection only"
        else:
            requested = value.get("families")
            if not isinstance(requested, list) or not requested or not set(requested) <= families:
                raise ValueError("unknown impact families")
            families = set(requested)
            reason = "author-declared affected families; advisory selection only"
    except (OSError, ValueError, TypeError):
        pass
    if chosen is None:
        chosen = []
        for family in sorted(families):
            members = sorted(r for r in records if r[0] == family)
            chosen.append(members[0])
            if len(members) > 1:
                chosen.append(members[-1])
    curriculum_report = None
    if curriculum is not None:
        chosen, curriculum_report = curriculum.select(records, policy=state['policy'],
                                                       hint=[r[2] for r in chosen])
        reason = 'rolling unresolved correctness queue; admission time bounds snapshot residency'
    # Only the fallback smoke has a fixed count. Native curriculum supplies the
    # unresolved roster; the existing scheduler admits/refills within its deadline.
    if curriculum is None:
        chosen = chosen[:16]
    names = sorted(r[2] for r in chosen)
    selection = {"scope": "rolling" if curriculum is not None else "smoke", "reason": reason, "selected": names,
                                    "required_development": len(required),
                                    "pending": sorted(set(required) - set(names)),
                                    "full_coverage_required": True}
    if curriculum_report is not None:
        selection['curriculum'] = curriculum_report
        # Passing evidence remains separately attributed, not a new measured row.
        selection['pending'] = curriculum_report['pending']
    return [r[3] for r in chosen], selection


def annotate(verdict, selection):
    if selection is None:
        return verdict
    failure = verdict.get("package_failure")
    deferred = {row.get("capsule") for row in verdict.get("per_capsule", ())
                if row.get("status") == "gated"} & set(selection["selected"])
    if deferred:
        selection = dict(selection, pending=sorted(set(selection["pending"]) | deferred),
                         execution_status="selected_capsules_deferred")
    if failure and not verdict.get("per_capsule"):
        selection = dict(selection, pending=sorted(set(selection["pending"]) | set(selection["selected"])),
                         execution_status="blocked_before_capsule_execution")
        verdict.update(all_pass=False, gradeable=False)
    refusal = verdict.get("package_precheck_refusal") or {}
    if (failure and not verdict.get("per_capsule")
            and refusal.get("schema") == "merlin.package_precheck_refusal.v1"
            and refusal.get("scope") == "selected_suite"
            and refusal.get("stage") in {"load_package", "integrity_scan"}
            and refusal.get("candidate_execution_started") is False):
        # Only the native pre-candidate code path can issue this scope. No case
        # error/category or arbitrary build process failure expands to siblings.
        import hashlib
        diagnostic = {**refusal, "failure": failure}
        reference = hashlib.sha256(json.dumps(diagnostic, sort_keys=True, allow_nan=False).encode()).hexdigest()
        verdict["shared_package_diagnostic"] = {**diagnostic, "sha256": reference,
            "affected_selected": len(selection["selected"]), "reuse_authority": False}
        verdict["per_capsule"] = [{"capsule": name, "status": "gated", "tiers": {},
            "execution_status": "not_started", "correctness": "unknown",
            "shared_package_diagnostic_sha256": reference}
            for name in selection["selected"]]
        verdict.update(n_capsules=len(selection["selected"]), n_executed_capsules=0,
            n_passed=0, mandatory_tiers_complete=False,
            count_scope="selected_pending_cohort_no_execution")
    verdict["joint_test_selection"] = selection
    verdict["n_required_development"] = selection["required_development"]
    verdict["n_selected_development"] = len(selection["selected"])
    verdict["n_pending_development"] = len(selection["pending"])
    verdict["certified_complete"] = False
    first = next((row for row in verdict.get("per_capsule", ())
                  if row.get("failure_detail") or row.get("status") in {"fail", "error"}), None)
    package_failure = ({"scope": "compiler_package", "plane": failure.get("plane"),
                        "detail": str(failure.get("detail", "package rejected before capsule execution"))[:600]}
                       if failure else None)
    verdict["feedback_brief"] = {
        "checks": selection["selected"], "passed": verdict.get("n_passed"),
        "pending": len(selection["pending"]), "scope": selection["scope"],
        "first_actionable_failure": package_failure or ({"capsule": first.get("capsule"),
                                      "detail": str(first.get("failure_detail", first.get("status")))[:600]}
                                     if first else None),
        "performance_model_confidence": "consult selected versioned per-capsule feedback; unknown here",
        "execution_status": selection.get("execution_status", "see_per_capsule_results"),
        "artifact": "qa/verdict.json",
    }
    if selection.get('curriculum'):
        report = selection['curriculum']
        witnesses = report['witness_prerequisites']
        verdict['feedback_brief']['correctness_curriculum'] = {
            key: report[key] for key in ('original_development_required', 'original_unresolved',
                'next_batch', 'physical_rule', 'parallel_large_spike', 'correctness_scope')}
        verdict['feedback_brief']['correctness_curriculum'].update(
            reused_correctness=len(report['exact_correctness_reused']),
            calibration_prerequisites={'selected_witnesses': len(witnesses),
                'matched_spike_passes': sum(w['correctness'] == 'matched_spike_pass' for w in witnesses),
                'qualified_physical_coverage': 'consult existing exact-scope calibration registry',
                'next_questions': [w for w in witnesses if w['capsule'] in report['next_batch']][:3]})
    from .joint_performance_priority import summarize as performance_priority
    verdict["feedback_brief"]["performance_priority"] = performance_priority(verdict.get("per_capsule", ()))
    if selection["scope"] == "smoke":
        if verdict.get("all_pass") is True:
            verdict["all_pass"] = None
        verdict["gradeable"] = False
        verdict["count_scope"] = "selected_advisory_smoke_only"
    return verdict
