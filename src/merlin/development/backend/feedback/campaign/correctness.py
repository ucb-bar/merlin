"""Bounded advisory smoke selection; omitted capsules remain unmeasured."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import yaml

from merlin.development.backend.campaign import control as joint_control
from merlin.development.backend.feedback.campaign.curriculum import _passed, _structural_prerequisites

# ---- joint_test_scope --------------------------------------------------------------------------

PERFORMANCE_SCOPES = ("scored", "excluded")


def scope_selection(policy, declared_excluded, required):
    """Excluded cases from the capsules' declared field and the policy's derived roster."""
    roster = (policy.get("performance_scope") or {}).get("excluded", ())
    if not isinstance(roster, (list, tuple)) or any(type(name) is not str for name in roster):
        raise ValueError("policy performance_scope.excluded must be a list of case IDs")
    excluded = (set(declared_excluded) | set(roster)) & set(required)
    return {
        "excluded": sorted(excluded),
        "scored": len(set(required) - excluded),
        "sources": ["capsule.performance_scope", "policy.performance_scope.excluded"],
    }


def excluded_cases(selection):
    """The excluded names of a recorded ``joint_test_selection`` (empty for older grades)."""
    scope = (selection or {}).get("performance_scope") or {}
    return frozenset(name for name in scope.get("excluded") or () if isinstance(name, str))


def focused():
    return joint_control.selected() is not None and os.environ.get("MERLIN_JOINT_GRADE_MILESTONE") != "1"


def select(root, *, submission, curriculum=None, additional_cases=None):
    state = joint_control.selected()
    if state is None:
        return root, None
    held = joint_control.held_capsules(state)
    records = []
    additional_cases = additional_cases or {}
    if any(Path(path).name != name for name, path in additional_cases.items()):
        raise ValueError("additional development capsule path differs from its selected ID")
    paths = list(sorted(Path(root).rglob("capsule.yaml")))
    paths.extend(Path(p) / "capsule.yaml" for p in additional_cases.values())
    seen, declared_excluded = set(), set()
    search_roles = {}
    for path in paths:
        doc = yaml.safe_load(path.read_text())
        name = path.parent.name
        if name in seen:
            raise ValueError("joint development suites contain duplicate capsule IDs")
        seen.add(name)
        if name in additional_cases and (
            doc.get("name") != name or doc.get("label") not in {"public", "dev"} or name in held
        ):
            raise ValueError("additional development capsule is not its selected public case")
        if ("label" in doc and doc["label"] not in {"public", "dev"}) or path.parent.name in held:
            continue
        scope = doc.get("performance_scope")
        if "performance_scope" in doc and scope not in PERFORMANCE_SCOPES:
            raise ValueError(f"{name}: performance_scope must be one of {PERFORMANCE_SCOPES}")
        if scope == "excluded":
            declared_excluded.add(name)
        family = str((doc.get("operation") or {}).get("op") or doc.get("kind") or "unknown")
        volume = sum(
            math.prod(x.get("shape", ()))
            for x in doc.get("inputs", ())
            if all(type(d) is int and d > 0 for d in x.get("shape", ()))
        )
        from merlin.development.backend.session.inputs.capability_inventory import _digest, classify

        original = set(state["policy"].get("correctness_curriculum", {}).get("original_capsules", ()))
        witnesses = set(state["policy"].get("gsim", {}).get("witness_capsules", ()))
        member = dict(
            name=name,
            source_sha256=_digest({k: v for k, v in doc.items() if k != "__dir__"}),
            cohort=(
                "original_correctness"
                if name in original
                else "scalability_performance"
                if name in additional_cases
                else "calibration"
                if name in witnesses
                else "public_development"
            ),
            declared_operation=(doc.get("operation") or {}).get("op"),
            input_tensors=doc.get("inputs", ()),
        )
        search_roles[name] = classify(member, selection=state["policy"].get("workload_roles"))["workload_role"]
        records.append((family, volume, path.parent.name, path.parent))
    if not records:
        raise ValueError("joint automatic development selection is empty")
    required = sorted(r[2] for r in records)
    cohorts = {
        "core_and_calibration": len(required) - len(additional_cases),
        "public_scalability": len(additional_cases),
    }
    # These memberships come from the native verified suite and operator policy,
    # never from a candidate's impact hints or result labels.
    original = set(state["policy"].get("correctness_curriculum", {}).get("original_capsules", ()))
    performance_cohorts = {
        "primary": sorted(additional_cases),
        "correctness_regression": sorted(original & set(required)),
        # A witness that is also a selected scalability case is ranked there. Listing it in both lanes
        # made the whole roster "invalid_or_overlapping" and dropped it from every ranking.
        "calibration_diagnostics": sorted(
            set(state["policy"].get("gsim", {}).get("witness_capsules", ())) - original - set(additional_cases)
        ),
    }
    performance_scope = scope_selection(state["policy"], declared_excluded, required)
    if not focused():
        return [r[3] for r in records], {
            "scope": "milestone",
            "required_development": len(records),
            "selected": required,
            "pending": [],
            "cohorts": cohorts,
            "performance_cohorts": performance_cohorts,
            "performance_scope": performance_scope,
            "workload_roles": search_roles,
        }
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
            if (
                "families" in value
                or not isinstance(requested, list)
                or not 1 <= len(requested) <= 16
                or any(type(name) is not str for name in requested)
                or len(set(requested)) != len(requested)
                or not set(requested) <= set(required)
            ):
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
    except (OSError, ValueError, TypeError) as error:
        reason = f"unknown impact: hint unavailable ({type(error).__name__}); representative semantic-family smoke"
    if chosen is None:
        chosen = []
        for family in sorted(families):
            members = sorted(r for r in records if r[0] == family)
            chosen.append(members[0])
            if len(members) > 1:
                chosen.append(members[-1])
    curriculum_report = None
    if curriculum is not None:
        chosen, curriculum_report = curriculum.select(records, policy=state["policy"], hint=[r[2] for r in chosen])
        reason = "rolling unresolved correctness queue; admission time bounds snapshot residency"
    # Only the fallback smoke has a fixed count. Native curriculum supplies the
    # unresolved roster; the existing scheduler admits/refills within its deadline.
    if curriculum is None:
        chosen = chosen[:16]
    names = sorted(r[2] for r in chosen)
    selection = {
        "scope": "rolling" if curriculum is not None else "smoke",
        "reason": reason,
        "selected": names,
        "cohorts": cohorts,
        "performance_cohorts": performance_cohorts,
        "performance_scope": performance_scope,
        "workload_roles": search_roles,
        "required_development": len(required),
        "pending": sorted(set(required) - set(names)),
        "full_coverage_required": True,
    }
    if curriculum_report is not None:
        selection["curriculum"] = curriculum_report
        # Passing evidence remains separately attributed, not a new measured row.
        selection["pending"] = curriculum_report["pending"]
    return [r[3] for r in chosen], selection


def annotate(verdict, selection):
    if selection is None:
        return verdict
    failure = verdict.get("package_failure")
    deferred = {row.get("capsule") for row in verdict.get("per_capsule", ()) if row.get("status") == "gated"} & set(
        selection["selected"]
    )
    if deferred:
        selection = dict(
            selection,
            pending=sorted(set(selection["pending"]) | deferred),
            execution_status="selected_capsules_deferred",
        )
    if failure and not verdict.get("per_capsule"):
        selection = dict(
            selection,
            pending=sorted(set(selection["pending"]) | set(selection["selected"])),
            execution_status="blocked_before_capsule_execution",
        )
        verdict.update(all_pass=False, gradeable=False)
    refusal = verdict.get("package_precheck_refusal") or {}
    if (
        failure
        and not verdict.get("per_capsule")
        and refusal.get("schema") == "merlin.package_precheck_refusal.v1"
        and refusal.get("scope") == "selected_suite"
        and refusal.get("stage") in {"load_package", "integrity_scan"}
        and refusal.get("candidate_execution_started") is False
    ):
        # Only the native pre-candidate code path can issue this scope. No case
        # error/category or arbitrary build process failure expands to siblings.
        import hashlib

        diagnostic = {**refusal, "failure": failure}
        reference = hashlib.sha256(json.dumps(diagnostic, sort_keys=True, allow_nan=False).encode()).hexdigest()
        verdict["shared_package_diagnostic"] = {
            **diagnostic,
            "sha256": reference,
            "affected_selected": len(selection["selected"]),
            "reuse_authority": False,
        }
        verdict["per_capsule"] = [
            {
                "capsule": name,
                "status": "gated",
                "tiers": {},
                "execution_status": "not_started",
                "correctness": "unknown",
                "shared_package_diagnostic_sha256": reference,
            }
            for name in selection["selected"]
        ]
        verdict.update(
            n_capsules=len(selection["selected"]),
            n_executed_capsules=0,
            n_passed=0,
            mandatory_tiers_complete=False,
            count_scope="selected_pending_cohort_no_execution",
        )
    verdict["joint_test_selection"] = selection
    verdict["n_required_development"] = selection["required_development"]
    verdict["n_selected_development"] = len(selection["selected"])
    verdict["n_pending_development"] = len(selection["pending"])
    verdict["certified_complete"] = False
    first = next(
        (
            row
            for row in verdict.get("per_capsule", ())
            if row.get("failure_detail") or row.get("status") in {"fail", "error"}
        ),
        None,
    )
    package_failure = (
        {
            "scope": "compiler_package",
            "plane": failure.get("plane"),
            "detail": str(failure["detail"])[:600] if failure.get("detail") is not None else None,
        }
        if failure
        else None
    )
    verdict["feedback_brief"] = {
        "checks": selection["selected"],
        "passed": verdict.get("n_passed"),
        "pending": len(selection["pending"]),
        "scope": selection["scope"],
        "first_actionable_failure": package_failure
        or (
            {"capsule": first.get("capsule"), "detail": str(first.get("failure_detail", first.get("status")))[:600]}
            if first
            else None
        ),
        "performance_model_confidence": "consult selected versioned per-capsule feedback; unknown here",
        "execution_status": selection.get("execution_status"),
        "artifact": "qa/verdict.json",
    }
    if selection.get("curriculum"):
        report = selection["curriculum"]
        witnesses = report["witness_prerequisites"]
        verdict["feedback_brief"]["correctness_curriculum"] = {
            key: report[key]
            for key in (
                "original_development_required",
                "original_unresolved",
                "next_batch",
                "physical_rule",
                "parallel_large_spike",
                "correctness_scope",
            )
        }
        if report.get("sampling"):
            # Stable cases this grade did not re-run; each keeps its earlier, attributed result.
            sampling = report["sampling"]
            verdict["feedback_brief"]["correctness_curriculum"]["regression_sampling"] = {
                "carried_sampled": len(report.get("carried_sampled") or {}),
                "label": sampling.get("status_label"),
                "covering_sample": len(sampling.get("covering_sample") or ()),
                "full_sweep": sampling.get("full_sweep"),
                "full_sweep_reasons": sampling.get("full_sweep_reasons"),
                "scope": sampling.get("scope"),
            }
        verdict["feedback_brief"]["correctness_curriculum"].update(
            reused_correctness=len(report["exact_correctness_reused"]),
            calibration_prerequisites={
                "selected_witnesses": len(witnesses),
                "matched_spike_passes": sum(w["correctness"] == "matched_spike_pass" for w in witnesses),
                "qualified_physical_coverage": "consult existing exact-scope calibration registry",
                "next_questions": [w for w in witnesses if w["capsule"] in report["next_batch"]][:3],
            },
        )
        # The witness statuses above come from the pre-grade selection. Say what THIS grade observed
        # and how a witness actually reaches gSIM; this grants no admission or qualification.
        from merlin.development.backend.feedback.certification.tier_policy import calibration_access

        verdict["feedback_brief"]["correctness_curriculum"]["calibration_prerequisites"]["gsim_access"] = (
            calibration_access(verdict.get("per_capsule", ()), witnesses)
        )
    from merlin.development.backend.feedback.campaign.joint_performance_priority import (
        summarize as performance_priority,
    )

    verdict["feedback_brief"]["performance_priority"] = performance_priority(
        verdict.get("per_capsule", ()),
        cohorts=selection.get("performance_cohorts", {}),
        roles=selection.get("workload_roles"),
    )
    if selection["scope"] == "smoke":
        if verdict.get("all_pass") is True:
            verdict["all_pass"] = None
        verdict["gradeable"] = False
        verdict["count_scope"] = "selected_advisory_smoke_only"
    return verdict


# ---- joint_physical_correctness ----------------------------------------------------------------
# Original-corpus correctness admission into the native certificate queue.
#
# No calibration fit, timing qualification, or new simulator runner lives here.


def roster():
    state = joint_control.selected()
    return set((state or {}).get("policy", {}).get("gsim_correctness", {}).get("capsules", ()))


def _split_l3_prerequisites(row, source_sha256):
    """Recognize a completed cheap stage, not a certificate or complete result."""
    from merlin.development.targetgen.grading.trace_check import requirements_complete

    tiers = row.get("tiers") or {}
    failure = row.get("failure") or {}
    l3 = tiers.get("L3") or {}
    source = row.get("compiler_source") or {}
    numeric = row.get("numeric") or {}
    outputs = numeric.get("per_output")
    if not (
        row.get("status") == "incomplete"
        and failure.get("plane") == "oracle_unavailable"
        and failure.get("category") == "NOT_RUN_IS_NOT_PASS"
        and failure.get("tier") == "L3"
        and failure.get("tier_status") == "unavailable"
        and l3.get("status") == "unavailable"
        and l3.get("mandatory") is True
        and l3.get("reason")
        == (
            "this capsule declares required oracle tier L3 (elaborated_rtl) "
            "but this phase supplies no adapter for it; reachable tiers here: ['L2']"
        )
        and failure.get("tier_reason") == l3["reason"]
        and source.get("status") == "stable"
        and source_sha256
        and source.get("candidate_sha256") == source_sha256
        and requirements_complete(row.get("trace_check"))
        and not (row.get("trace_check") or {}).get("violations")
        and not (row.get("trace_check") or {}).get("unavailable_requirements")
        and isinstance(outputs, dict)
        and outputs
        and numeric.get("missing_outputs") == []
        and numeric.get("mismatch_count") == 0
        and all(
            isinstance(v, dict)
            and v.get("status") == "pass"
            and type(v.get("n_elements")) is int
            and v["n_elements"] > 0
            for v in outputs.values()
        )
    ):
        return False
    # L0/L1 can be genuinely inapplicable to whole-program execution. The
    # shared native checker below validates their exact paired reasons and
    # complete numerical output evidence; it does not accept arbitrary skips.
    # No other unavailable/failed tier may hide behind the expected L3 gap.
    if any(
        t not in {"L0", "L1", "L3"} and record.get("mandatory") is True and record.get("status") != "pass"
        for t, record in tiers.items()
    ):
        return False
    # The private projection is used for scheduling only. The original durable
    # result, failure, curriculum credit and certification state stay unchanged.
    return _passed(dict(row, failure=None)) == "spike_correctness"


_MODEL_NATIVE_TIER_GAP = {
    "plane": "candidate_model_execution",
    "category": "NOT_RUN_IS_NOT_PASS",
    "detail": "mandatory native tier has not completed",
}


def model_l3_prerequisites(row, source_sha256):
    """A whole-model row whose only open obligation is its native L3 tier.

    The Spike grade proves a model's L0-L2 on the exact snapshot it compiled, but never runs its L3,
    so the row stays ``incomplete`` with the grader's own native-tier gap -- and was never admitted
    here, which is why no whole-model case ever reached L3 in the grade. Admit exactly that gap: every
    other mandatory tier passes (L2 on Spike), all outputs pass, the model check names no violation,
    and the model child's compiler identity is the snapshot being promoted (``source_sha256``).
    """
    tiers = row.get("tiers") or {}
    l2, l3 = tiers.get("L2") or {}, tiers.get("L3") or {}
    numeric = row.get("numeric") or {}
    outputs = numeric.get("per_output")
    child = row.get("snapshot_external_model_child") or {}
    compiler = child.get("compiler_sha256") or (row.get("compiler_source") or {}).get("candidate_sha256")
    if not (
        row.get("kind") == "model"
        and row.get("status") == "incomplete"
        and source_sha256
        and row.get("failure") == _MODEL_NATIVE_TIER_GAP
        and l3.get("mandatory") is True
        and l3.get("status") == "unverified"
        and l2.get("status") == "pass"
        and l2.get("engine") == "spike"
        and _structural_prerequisites(row)
        and numeric.get("status") == "pass"
        and numeric.get("mismatch_count") == 0
        and numeric.get("missing_outputs") in (None, [])
        and isinstance(outputs, dict)
        and outputs
        and all(
            isinstance(v, dict)
            and v.get("status") == "pass"
            and type(v.get("n_elements")) is int
            and v["n_elements"] > 0
            for v in outputs.values()
        )
        and not (row.get("model_execution_check") or {}).get("violations")
        and compiler == source_sha256
    ):
        return False
    return not any(
        t != "L3" and isinstance(record, dict) and record.get("mandatory") is True and record.get("status") != "pass"
        for t, record in tiers.items()
    )


def eligible(rows, *, source_root=None):
    """Only closed native structural/numerical receipts authorize a physical job."""
    required = roster()
    source_sha256 = None
    if source_root is not None and any(row.get("failure") for name, row in rows.items() if name in required):
        from merlin.common.digest import hash_tree

        source_sha256 = hash_tree(Path(source_root))["sha256"]
    ready = set()
    for name, row in rows.items():
        trace = row.get("trace_check") or {}
        if name not in required or trace.get("status") == "fail" or trace.get("violations"):
            continue
        if (
            _passed(row) == "spike_correctness"
            or _split_l3_prerequisites(row, source_sha256)
            or model_l3_prerequisites(row, source_sha256)
        ):
            ready.add(name)
    return ready


def selected_request(request):
    return (
        request.get("joint_physical_correctness") is True
        and request.get("promoted") is True
        and request.get("sim") == "gsim"
        and request.get("tiers") == "L3"
        and request.get("capsules") in roster()
    )


def project(ws, state, *, source=None, instrument=None, closed_rows=None):
    """Project protected records; cross-source reuse needs a matching native L2 identity."""
    from merlin.development.backend.feedback.certification import promotion as P
    from merlin.development.backend.feedback.certification.tier_evidence import reusable_certificate

    required = roster()
    if not required:
        return
    path = Path(ws) / "qa/joint_physical_correctness.json"
    previous = json.loads(path.read_text()) if path.is_file() else {}
    source = source or previous.get("source_sha256")
    instrument = instrument or previous.get("instrument_sha256")
    if not source:
        return
    refusals = set(previous.get("expected_refusal_satisfied", ())) if previous.get("source_sha256") == source else set()
    refusals |= {
        name for name, row in (closed_rows or {}).items() if name in required and _passed(row) == "expected_refusal"
    }
    obligations = {}
    for name in sorted(required):
        loops = [
            v
            for v in P._slots_ro(state, name, "L2").values()
            if v.get("digest") == source and v.get("status") == "pass"
        ]
        observed = {
            v.get("execution_digest")
            for v in P._slots_ro(state, name, "L2").values()
            if v.get("digest") == source and v.get("status") == "pass" and P._valid_execution(v.get("execution_digest"))
        }
        records, reuse = [], {}
        for value in P._slots_ro(state, name, "L3").values():
            if not instrument or value.get("instrument") != instrument:
                continue
            boundary = next(
                (
                    matched
                    for loop in loops
                    if (matched := reusable_certificate(value, loop, name=name, source=source, instrument=instrument))
                    is not None
                ),
                None,
            )
            if value.get("digest") == source or value.get("execution_digest") in observed or boundary is not None:
                records.append(value)
                if boundary is not None:
                    reuse[value["execution_digest"]] = boundary
        # A queued duplicate is not a measured counterexample to an applicable
        # certificate. A later recorded failure still takes precedence.
        if records and records[-1].get("status") in {"pending", "joined"}:
            for record in reversed(records):
                if record.get("status") == "fail":
                    break
                if record.get("status") == "pass" and record.get("execution_digest") in reuse:
                    records.append(record)
                    break
        status = records[-1].get("status") if records else "pending"
        obligations[name] = {
            "status": "expected_refusal_satisfied"
            if name in refusals
            else status
            if status in {"pass", "fail", "pending"}
            else "pending",
            "physical_executed": None,
            "source_sha256": source,
            "certificate_source_sha256": records[-1].get("digest") if records else None,
            "reused_execution_identity": records[-1].get("execution_digest")
            if records and records[-1].get("digest") != source
            else None,
            "boundary_reuse": reuse.get(records[-1].get("execution_digest")) if records else None,
        }
    statuses = [row["status"] for row in obligations.values()]
    report = {
        "schema": "merlin.joint_physical_correctness.v1",
        "source_sha256": source,
        "instrument_sha256": instrument,
        "required": len(required),
        "obligations": obligations,
        "passed": statuses.count("pass"),
        "failed": statuses.count("fail"),
        "pending": statuses.count("pending"),
        "expected_refusal_satisfied": sorted(refusals),
        "complete": bool(statuses) and all(s in {"pass", "expected_refusal_satisfied"} for s in statuses),
        "scope": "original capsule L3 correctness; not timing calibration or full-target validation",
        "source_currentness": "exact reported source only; no claim about current author edits",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".physical-", delete=False) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def attach(path, verdict):
    """Keep the physical denominator visible beside cheap snapshot progress."""
    if not roster():
        return verdict
    report_path = Path(path).parent / "joint_physical_correctness.json"
    try:
        report = json.loads(report_path.read_text())
        summary = {
            key: report[key]
            for key in (
                "required",
                "passed",
                "failed",
                "pending",
                "complete",
                "expected_refusal_satisfied",
                "source_sha256",
                "instrument_sha256",
                "scope",
                "source_currentness",
            )
        }
    except (OSError, ValueError, KeyError):
        summary = {
            "required": len(roster()),
            "passed": 0,
            "failed": 0,
            "pending": len(roster()),
            "complete": False,
            "source_sha256": None,
            "scope": "original capsule L3 correctness pending",
        }
    summary["artifact"] = "qa/joint_physical_correctness.json"
    verdict["joint_physical_correctness"] = summary
    verdict.setdefault("feedback_brief", {})["physical_correctness"] = summary
    return verdict
