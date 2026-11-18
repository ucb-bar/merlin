"""Turn selected reference evidence into ordinary, controlled selfcheck requests.

This module plans experiments; it neither edits a compiler nor dispatches work.
The selected owner supplies public applicability, real knob domains and exact
selfcheck commands. A reference difference is a hypothesis, not a speedup or a
proof that a pass is missing. Results reconcile only through receipts whose full
content hashes the independent evaluation owner has admitted in the context.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.analysis.mining.reconstruction.learning import REQUIRED_CHECKS, ROUTES, learning_tasks
from merlin.analysis.mining.reconstruction.proposals import digest
from merlin.compiler.selection import knobs as tuning_knobs

# ---- reconstruction_experiments ----------------------------------------------------------------


def _goal_bindings(task, context):
    """Explain exact declared joins; neither names nor nearby shapes prove applicability."""
    domains = context.get("tuning", {}).get("domains", {})
    capsules = context.get("capsules", [])
    rows = []
    for goal in task["goals"]:
        selected = [row for row in capsules if goal in row.get("goals", ())]
        applicable = [
            row
            for row in selected
            if "region_id" not in task
            or row.get("region_id") == task["region_id"]
            or task["region_id"] in row.get("region_ids", ())
        ]
        rows.append(
            dict(
                goal=goal,
                declared_knobs=[name for name, row in sorted(domains.items()) if goal in row.get("goals", ())],
                selected_capsules=[row["id"] for row in selected],
                development_capsules=[row["id"] for row in applicable if row.get("split", "training") != "held_out"],
                held_out_capsules=[row["id"] for row in applicable if row.get("split") == "held_out"],
            )
        )
    return dict(
        goals=rows,
        next_action="Bind the same diagnosed goal to an implemented finite knob and an explicitly selected "
        "development capsule, including its region when required. Review source semantics, "
        "numerics and the actual consumer before changing declarations.",
        authority="declared bindings only; missing applicability is not inferred or granted",
    )


def _analysis_requests(context):
    """Select ordinary public emission diagnostics, without a transformation trial.

    A selected public capsule can be inspected without claiming that a reference
    mechanism applies to it. These commands establish no semantic correspondence
    or numerical/timing result and never enter the experiment outcome journal.
    """
    from merlin.analysis.mining.reconstruction.diagnoses import _command

    baseline = context.get("baseline", {}).get("submission")
    if not isinstance(baseline, str) or not baseline:
        return [], ["binding.baseline_submission"]
    diagnostics = []
    for capsule in context.get("capsules", []):
        if capsule.get("split", "training") == "held_out":
            continue
        argv = _command(capsule, baseline, baseline)
        if argv is None:
            continue
        argv = argv if "--candidate-diagnostics" in argv else [*argv, "--candidate-diagnostics"]
        command = shlex.join(argv)
        diagnostics.append(
            dict(
                capsule=capsule["id"],
                argv=argv,
                command=command if len(command) <= 600 else "Read the complete diagnostic argv in the detail.",
                scope="selected public capsule emission; reference applicability remains unproved",
                purpose="Collect current source, logical input, emitted artifact and original producer observations.",
                authority="diagnostic only; no numerical, mechanism, timing or experiment-completion credit",
            )
        )
        if len(diagnostics) == 2:
            break
    return diagnostics, (
        ["evidence.current_source_and_emission_observations"]
        if diagnostics
        else ["binding.development_diagnostic_command"]
    )


def _requests(task, context, identity, budget, remaining, variant_remaining):
    from merlin.analysis.mining.reconstruction.diagnoses import _PINS, _command, _same, _sha

    current = context.get("tuning", {}).get("current", {})
    domains = context.get("tuning", {}).get("domains", {})
    knobs = [(name, row) for name, row in sorted(domains.items()) if set(task["goals"]) & set(row.get("goals", []))]
    capsules = [row for row in context.get("capsules", []) if set(task["goals"]) & set(row.get("goals", []))]
    missing = ["execution." + key for key in _PINS if not _sha(context.get(key))]
    if "region_id" in task:
        capsules = [
            row
            for row in capsules
            if row.get("region_id") == task["region_id"] or task["region_id"] in row.get("region_ids", ())
        ]
        if not capsules:
            missing.append("binding.region_capsule_applicability")
    evaluator = context.get("evaluator", {})
    if not evaluator.get("identity") or not _sha(evaluator.get("sha256")):
        missing.append("binding.independent_evaluator")
    baseline = context.get("baseline", {}).get("submission")
    if not isinstance(baseline, str) or not baseline:
        missing.append("binding.baseline_submission")
    if not knobs:
        missing.append("binding.mechanism_to_declared_knob")
    if not capsules:
        missing.append("binding.public_capsule_applicability")
    eligible = [row for row in capsules if row.get("split", "training") != "held_out"]
    if capsules and not eligible:
        missing.append("binding.development_capsule")
    if missing:
        return [], [], missing
    requests, hypotheses = [], []
    variants = []
    for name, domain in knobs:
        for value in domain["values"]:
            if not _same(value, current[name]):
                settings = dict(current, **{name: value})
                variants.append((name, value, settings, set(task["goals"]) & set(domain.get("goals", []))))
    shared_goal = any(goals & set(capsule.get("goals", [])) for _, _, _, goals in variants for capsule in eligible)
    for capsule in eligible:
        if len(requests) >= remaining:
            break
        if _command(capsule, baseline, baseline) is None:
            missing.append("binding.exact_selfcheck_argv:" + capsule["id"])
            continue
        base = _request(task, capsule, context, identity, current, [], baseline, "baseline")
        pairs = []
        for name, value, settings, goals in variants:
            if not goals & set(capsule.get("goals", [])):
                continue
            if len(requests) + len(pairs) + 2 > remaining or len(hypotheses) >= variant_remaining:
                break
            request = _request(task, capsule, context, identity, settings, [name], baseline, "variant")
            request["baseline_request_id"] = base["request_id"]
            pairs.append(request)
            hypotheses.append(
                dict(
                    hypothesis_id=digest(dict(task=task["task_id"], knob=name, value=value))[:24],
                    summary=f"change declared {name}; hold every other setting fixed",
                    basis="inferred",
                    changed_factors=[name],
                    value=value,
                    request_id=request["request_id"],
                )
            )
        if pairs:
            requests.extend([base, *pairs])
    if not variants:
        missing.append("binding.nonbaseline_declared_alternative")
    elif not shared_goal:
        missing.append("binding.shared_goal_applicability")
    elif not requests and not missing:
        missing.append("budget.controlled_requests_exhausted")
    return requests, hypotheses, sorted(set(missing))


def _request(task, capsule, context, identity, settings, changed, submission, role):
    from merlin.analysis.mining.reconstruction.diagnoses import _command, _json

    overrides = {name: settings[name] for name in changed}
    argv = _command(capsule, submission, context["baseline"]["submission"])
    payload = dict(
        submission=submission,
        submission_sha256=context["compiler_sha256"],
        capsules=capsule["id"],
        tuning_overrides=overrides,
    )
    for flag in ("--suite", "--sim", "--tiers", "--timeout", "--workers"):
        if flag in argv:
            at = argv.index(flag)
            if at + 1 < len(argv):
                payload[flag[2:]] = argv[at + 1]
    request = dict(
        task_id=task["task_id"],
        role=role,
        submission=submission,
        argv=argv,
        dispatch="existing_selfcheck_tuning_broker",
        tuning_overrides=overrides,
        broker_payload=payload,
        baseline_compiler_sha256=context["compiler_sha256"],
        baseline_settings_sha256=digest(context["tuning"]["current"]),
        sim=payload.get("sim"),
        tiers=capsule.get("tiers"),
        settings=_json(settings),
        settings_sha256=digest(settings),
        changed_factors=changed,
        capsule=capsule["id"],
        split=capsule.get("split", "training"),
        context_identity=identity,
        goals=task["goals"],
        status="ready",
    )
    request["request_id"] = digest(request)[:32]
    # Display arguments carry the already computed semantic identity. Computing
    # it first avoids a request-id/argv cycle, and uses the existing CLI/broker.
    if overrides:
        argv.extend(
            ["--tuning-overrides", json.dumps(overrides, sort_keys=True, separators=(",", ":"), allow_nan=False)]
        )
    argv.extend(["--reference-request", request["request_id"]])
    payload["reference_experiment_request_id"] = request["request_id"]
    return request


def _trusted(receipt, request, context, identity):
    from merlin.analysis.mining.reconstruction.diagnoses import _sha

    evaluator = context.get("evaluator", {})
    if digest(receipt) not in evaluator.get("trusted_result_sha256", []):
        return False
    expected = dict(identity=evaluator.get("identity"), sha256=evaluator.get("sha256"))
    if receipt.get("evaluator") != expected or receipt.get("context_identity") != identity:
        return False
    if any(receipt.get(key) != context.get(key) for key in ("source_sha256", "environment_sha256", "reference_sha256")):
        return False
    if receipt.get("parent_compiler_sha256") != context.get("compiler_sha256"):
        return False
    if (
        receipt.get("request_id") != request["request_id"]
        or receipt.get("settings_sha256") != request["settings_sha256"]
    ):
        return False
    pins = receipt.get("artifact_pins", [])
    return (
        _sha(receipt.get("compiler_sha256"))
        and _sha(receipt.get("program_sha256"))
        and bool(pins)
        and all(isinstance(row, dict) and row.get("path") and _sha(row.get("sha256")) for row in pins)
    )


def _complete(receipt):
    gates = receipt.get("gates", {})
    return all(gates.get(name) == "pass" for name in REQUIRED_CHECKS)


def _cycles(receipt):
    from merlin.analysis.mining.reconstruction.diagnoses import _sha

    measurement = receipt.get("measurement", {})
    cycles = measurement.get("cycles")
    if (
        measurement.get("kind") == "measured"
        and measurement.get("complete_region") is True
        and _sha(measurement.get("domain_sha256"))
        and type(cycles) in (int, float)
        and cycles > 0
    ):
        return cycles, measurement["domain_sha256"]
    return None


def _reconcile(task, context, identity, history):
    receipts = {}
    for request in task["requests"]:
        matched = [row for row in history if isinstance(row, dict) and _trusted(row, request, context, identity)]
        if not matched:
            continue
        # Two different results for the same immutable request need owner
        # adjudication. Ordering an input history cannot choose the better one.
        if len({digest(row) for row in matched}) != 1:
            task["next_action"]["gaps"].append("evidence.conflicting_owner_receipts")
            continue
        receipt = matched[0]
        receipts[request["request_id"]] = receipt
        gates = receipt.get("gates", {})
        request["status"] = (
            "completed"
            if _complete(receipt)
            else "rejected"
            if any(gates.get(name) == "fail" for name in REQUIRED_CHECKS)
            else "unranked"
        )
        task["prior_attempts"].append(
            dict(
                authority="selected_evaluation_owner",
                result_sha256=digest(receipt),
                request_id=request["request_id"],
                outcome=request["status"],
                artifact_pins=receipt["artifact_pins"],
            )
        )
    for request in task["requests"]:
        if request["role"] != "variant" or request["request_id"] not in receipts:
            continue
        receipt = receipts[request["request_id"]]
        baseline = receipts.get(request["baseline_request_id"])
        if not _complete(receipt):
            continue
        if baseline is not None and receipt["program_sha256"] == baseline["program_sha256"]:
            request["status"] = "rejected"
            task["next_action"]["gaps"].append("evidence.unchanged_emitted_program")
            continue
        established = set(receipt.get("established", []))
        admitted = set(task["goals"]) <= established
        if admitted:
            task["mechanism_status"] = "independently_admitted"
            task["status"] = "unranked"
        if baseline is None or not _complete(baseline):
            continue
        before, after = _cycles(baseline), _cycles(receipt)
        if before is None or after is None or before[1] != after[1]:
            request["status"] = "unranked"
            continue
        request["measured_comparison"] = dict(before_cycles=before[0], after_cycles=after[0], domain_sha256=before[1])
        task["performance_status"] = "measured_improvement" if after[0] < before[0] else "measured_nonimprovement"
        if after[0] >= before[0]:
            request["status"] = "rejected"
        if task["kind"] == "measurement_gap":
            task["status"] = "resolved"
            task["summary"] = "independent owner collected matched complete-region timing"
    active = [request for request in task["requests"] if request["status"] == "ready"]
    variants = [request for request in task["requests"] if request["role"] == "variant"]
    if variants and not active and all(row["status"] == "rejected" for row in variants):
        task["status"] = "rejected"
        task["next_action"]["action"] = "retain_incumbent_and_change_hypothesis_or_context"


def _diagnosis_identity(row):
    values = dict(kind=row["kind"], goals=row["goals"], summary=row["summary"])
    if "region_id" in row:
        values["region_id"] = row["region_id"]
    return digest(values)


def build_experiment_packet(reconstruction_index, *, context, history=(), budget=None, diagnosis_offset=0):
    """Build reproducible tasks; evidence owner, commands and legality stay explicit.

    ``history`` may include normalized pinned prior experiments and raw execution
    receipts. Only full receipt hashes in ``context.evaluator.trusted_result_sha256``
    can reconcile requests. That allowlist must come from the independent owner,
    never from a proposed compiler. It is not an independence proof by itself.
    """
    from merlin.analysis.mining.reconstruction.diagnoses import (
        _PUBLIC,
        SCHEMA,
        ExperimentBudget,
        _context,
        _diagnoses,
        _json,
        execution_context_identity,
        packet_identity,
    )

    context, index, history = _context(context), _json(reconstruction_index), _json(list(history))
    budget = ExperimentBudget(**budget) if isinstance(budget, dict) else budget or ExperimentBudget()
    if not isinstance(budget, ExperimentBudget):
        raise ValueError("use an ExperimentBudget or its bounded field mapping")
    references = index.get("inventory", index).get("references", [])
    if context.get("visibility") in _PUBLIC and any(row.get("visibility") not in _PUBLIC for row in references):
        raise ValueError("selected public visibility cannot expose private reference evidence")
    identity = execution_context_identity(context)
    if type(diagnosis_offset) is not int or diagnosis_offset < 0:
        raise ValueError("diagnosis offset must be a nonnegative integer")
    tasks, remaining, variant_remaining = [], budget.max_requests, budget.max_variants
    diagnoses = list(_diagnoses(index, context, history))
    unique = {}
    for diagnosis in diagnoses:
        key = _diagnosis_identity(diagnosis)
        if key in unique:
            unique[key]["evidence"].extend(diagnosis["evidence"])
        else:
            unique[key] = diagnosis
    selected = list(unique.items())[diagnosis_offset : diagnosis_offset + budget.max_tasks]
    for key, diagnosis in selected:
        task = dict(
            task_id=digest(dict(context=identity, diagnosis=key))[:24],
            diagnosis_identity=key,
            kind=diagnosis["kind"],
            goals=diagnosis["goals"],
            status="ready",
            summary=diagnosis["summary"],
            evidence=diagnosis["evidence"],
            prior_attempts=[],
            hypotheses=[],
            requests=[],
            mechanism_status="unknown",
            performance_status="unknown",
            acceptance=dict(
                mechanism_checks=list(REQUIRED_CHECKS),
                goal_observation_required=True,
                performance=dict(
                    required="paired measured complete-region cycles",
                    same_domain=True,
                    static_counts_are_scores=False,
                    strict_improvement_for_promotion=True,
                ),
                policy_consumer="merlin.compiler.transforms.exploration.shape_assessment",
                held_out_required=True,
                policy_promotion="frozen validation/held-out assessment; mechanism admission grants no speedup",
            ),
            next_action=dict(route=diagnosis["route"], action=diagnosis["action"], depends_on=[], gaps=[]),
        )
        if "region_id" in diagnosis:
            task["region_id"] = diagnosis["region_id"]
        if "learning_task" in diagnosis:
            task["learning_task"] = diagnosis["learning_task"]
        if diagnosis.get("evidence_collection"):
            diagnostics, missing = _analysis_requests(context)
            task["next_action"].update(phase="evidence_collection", diagnostics=diagnostics)
        else:
            task["requests"], task["hypotheses"], missing = _requests(
                task, context, identity, budget, remaining, variant_remaining
            )
        remaining -= len(task["requests"])
        variant_remaining -= len(task["hypotheses"])
        if not diagnosis.get("evidence_collection"):
            task["hypotheses"].insert(
                0,
                dict(
                    hypothesis_id=digest(dict(task=task["task_id"], control=True))[:24],
                    summary="retain the incumbent; the proposed mechanism may not improve complete-region cycles",
                    basis="inferred",
                    changed_factors=[],
                ),
            )
        task["next_action"]["gaps"] = missing
        if any(
            gap in missing
            for gap in (
                "binding.mechanism_to_declared_knob",
                "binding.public_capsule_applicability",
                "binding.region_capsule_applicability",
                "binding.development_capsule",
                "binding.shared_goal_applicability",
            )
        ):
            task["next_action"]["binding_requirements"] = _goal_bindings(task, context)
        if diagnosis.get("library_routes"):
            task["next_action"]["library_routes"] = diagnosis["library_routes"]
            if "binding.mechanism_to_declared_knob" in missing:
                task["next_action"]["gaps"].append("construction.declare_goal_bound_alternative")
        if missing:
            task["status"] = "blocked"
        for i, row in enumerate(history):
            if not isinstance(row, dict):
                continue
            changed = [
                item.get("name") if isinstance(item, dict) else item for item in row.get("changed_factors") or []
            ]
            domains = context.get("tuning", {}).get("domains", {})
            goals = row.get("goals") or [g for name in changed for g in domains.get(name, {}).get("goals", [])]
            related = any(evidence["pointer"] == f"/history/{i}" for evidence in task["evidence"])
            if row.get("outcome") and (related or set(goals) & set(task["goals"])):
                task["prior_attempts"].append(
                    dict(
                        authority="prior_artifact_record; not current-context qualification",
                        record_sha256=digest(row),
                        outcome=row["outcome"],
                        artifact_pins=row.get("artifact_pins", []),
                        context_match=False,
                    )
                )
        _reconcile(task, context, identity, history)
        tasks.append(task)
    for task in tasks:
        if task["kind"] in ("scoped_mechanism_difference", "prior_regression"):
            task["next_action"]["depends_on"] = [
                row["task_id"]
                for row in tasks
                if row["kind"] in ("semantic_mismatch", "missing_evidence")
                and row["status"] != "resolved"
                and row.get("region_id") == task.get("region_id")
            ]
    packet = dict(
        schema=SCHEMA,
        execution_context_identity=identity,
        context=context,
        tasks=tasks,
        budget=asdict(budget),
        search=dict(
            diagnoses=len(diagnoses),
            truncated=diagnosis_offset + len(selected) < len(unique),
        ),
        coverage=dict(
            total_diagnoses=len(unique),
            selected_start=diagnosis_offset,
            selected_end=diagnosis_offset + len(selected),
            next_offset=(diagnosis_offset + len(selected) if diagnosis_offset + len(selected) < len(unique) else None),
        ),
    )
    packet["identity"] = packet_identity(packet)
    return _json(packet)
