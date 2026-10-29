"""Bounded author hypotheses, reviewed by the existing trusted diagnostic owner.

Proposals name public witnesses and a discriminator, never runtimes, commands,
source groups, labels or fit parameters. The owner reopens completed native
numerical/trace products before creating an existing diagnostic collection
request. Its measurements remain diagnostic-only and use the campaign ledger.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from merlin.runtime import _native_files as F
from . import joint_control
from .feedback import joint_performance as PERF
from ..phase2.contracts import document_sha256

SCHEMA = "merlin.mechanism_question_proposal.v1"


def _proposal(value, state):
    from . import joint_diagnostic_collection as D
    if (not isinstance(value, dict) or set(value) != {"schema", "question", "discriminator", "capsules", "purpose"}
            or value.get("schema") != SCHEMA or value.get("purpose") not in {D.UNKNOWN_COST, D.VALIDATION}):
        raise ValueError("mechanism proposal has unsupported fields/schema/purpose")
    for key in ("question", "discriminator"):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 2048:
            raise ValueError("mechanism question and discriminator must be bounded nonempty text")
    names = value["capsules"]
    if (not isinstance(names, list) or not 1 <= len(names) <= 4 or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
            or any(n not in state["policy"]["gsim"]["witness_capsules"] for n in names)):
        raise ValueError("mechanism proposal exceeds the original public witness roster")
    if not state["policy"].get("native_calibration"):
        raise ValueError("mechanism questions require the selected native calibration owner")
    return value


def submit(proposal, *, request_id, environment=None):
    """Broker-only enqueue; returns no measurement grant or numerical credit."""
    from . import joint_calibration as C
    env = os.environ if environment is None else environment
    state_path = env.get(joint_control.ENV)
    if not state_path:
        raise ValueError("no joint campaign selected")
    state = PERF._read(F.pin(Path(state_path), max_bytes=16 << 20))
    value = _proposal(proposal, state)
    selection = json.loads(env[PERF.ENV]) if env.get(PERF.ENV) else state.get("performance_selection")
    if not selection:
        raise ValueError("mechanism question lacks its selected evaluator")
    PERF._read(selection)
    record = dict(schema="merlin.mechanism_question_intake.v1", proposal=value,
        performance_selection=selection, profile=state["policy"]["native_calibration"],
        policy_sha256=state["policy_sha256"])
    key = document_sha256(record)
    root = Path(state_path).parent / "mechanism_questions"
    if len(list(root.glob("*/proposal.json"))) >= 64 and not (root / key / "proposal.json").exists():
        raise ValueError("bounded mechanism question intake requires host review")
    path = root / key / "proposal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if PERF._read(F.pin(path, max_bytes=16 << 20)) != record:
            raise ValueError("mechanism question identity changed")
    else:
        C._write(path, record)
    return dict(status="queued_for_trusted_diagnostic_review", question_id=key,
        request_id=request_id, all_pass=None, numeric_assessment="not_run",
        fit_eligible=False, dispatch_authorized=False, changes_authoring=False)


def step(*, state_path, context):
    """One question per polling boundary; original observer/producer checks remain."""
    from . import joint_calibration as C, joint_diagnostic_collection as D
    from .joint_model_activation import current
    state = joint_control.selected()
    if state is None or not state["policy"].get("native_calibration"):
        return dict(status="skipped_no_profile", fit_eligible=False)
    root = Path(state_path).parent / "mechanism_questions"
    paths = sorted(root.glob("*/proposal.json"))[:65]
    if len(paths) > 64:
        return dict(status="blocked_question_intake_bound", fit_eligible=False)
    selected_pin = current(state_path)
    profile = PERF._read(state["policy"]["native_calibration"])
    selection = PERF._read(selected_pin)
    waiting = None
    for path in paths:
        directory = path.parent
        terminal = directory / "review.json"
        if terminal.exists():
            continue
        try:
            intake = PERF._read(F.pin(path, max_bytes=16 << 20))
            value = _proposal(intake["proposal"], state)
            if (intake.get("schema") != "merlin.mechanism_question_intake.v1"
                    or intake.get("policy_sha256") != state["policy_sha256"]
                    or intake.get("profile") != state["policy"]["native_calibration"]
                    or intake.get("performance_selection") != selected_pin):
                raise ValueError("mechanism question is stale or belongs to a different campaign/profile")
            questions = {name:value["question"] + "\nDiscriminator: " + value["discriminator"]
                         for name in value["capsules"]}
            # Existing diagnostic schema bounds the combined question to 2048.
            if any(len(q) > 2048 for q in questions.values()):
                raise ValueError("combined question/discriminator exceeds diagnostic review bound")
            registry = Path(state_path).parent / "calibration_registry"
            entries = sorted(registry.glob("*/entry.json"))[:2049]
            if len(entries) > 2048:
                raise ValueError("mechanism review exceeds original bounded product registry")
            matched = set()
            for entry in entries:
                match = D.eligible(PERF._read(F.pin(entry, max_bytes=16 << 20)), profile=profile,
                    selection=selection, questions=questions, observation_purpose=value["purpose"])
                if match is not None:
                    matched.add(match[0])
            if matched != set(questions):
                waiting = waiting or dict(status="waiting_for_original_numerical_trace_products", question_id=directory.name,
                    missing_capsules=sorted(set(questions) - matched), fit_eligible=False)
                continue
            request_path = directory / "diagnostic-request.json"
            request = dict(schema=D.SCHEMA, policy_sha256=state["policy_sha256"],
                profile=state["policy"]["native_calibration"], performance_selection=selected_pin,
                purpose=value["purpose"], questions=questions, fit_eligible=False,
                max_observations=len(questions), review_id="trusted-public-mechanism-" + directory.name)
            if not request_path.exists():
                C._write(request_path, request)
            request_pin = F.pin(request_path, max_bytes=16 << 20)
            D.request(request_pin, {**state, "performance_selection": selected_pin})
            response = D.step(request_pin=request_pin, state_path=state_path, context=context,
                              output=directory / "diagnostics")
            # A waiting owner keeps polling; terminal results are retained once.
            if response["status"] in {"reviewed_diagnostic_limit_reached", "diagnostic_attempt_review_required"}:
                C._write(terminal, dict(response, question_id=directory.name))
            return dict(response, question_id=directory.name, fit_eligible=False)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            result = dict(status="mechanism_question_denied", reason=str(error),
                          question_id=directory.name, fit_eligible=False, changes_authoring=False)
            C._write(terminal, result)
            return result
    return waiting or dict(status="no_pending_mechanism_questions", fit_eligible=False)
