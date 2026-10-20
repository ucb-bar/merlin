"""Explicit capsule-qualified, Spike-only ordinary development scope.

The original qualification verifier and complete V4 input snapshot remain the
admission owner. This scope never supplies full-model, RTL or final authority.
"""

from __future__ import annotations

import copy
import weakref
from pathlib import Path
from types import MethodType

from . import capsule_iteration as CI
from .contracts import StageGateError, canonical_json, sha256_file

_ISSUED = weakref.WeakKeyDictionary()
_POLICIES = weakref.WeakKeyDictionary()
_ISSUER = object()


class CapsuleOrdinaryScope:
    def __init__(self, owner, baseline, binding, *, cohort_declaration, _issuer):
        if _issuer is not _ISSUER or type(owner) is not CI.FrozenQualifiedCapsulePhase1:
            raise StageGateError("capsule development requires its original qualified capsule owner")
        methods = tuple(
            (
                name,
                getattr(CI.FrozenQualifiedCapsulePhase1, name),
                getattr(CI.FrozenQualifiedCapsulePhase1, name).__code__,
            )
            for name in ("verify", "_qualification")
        )
        from . import evaluation_pool as EP

        pool_status = (EP.ComponentEvaluationPool.status, EP.ComponentEvaluationPool.status.__code__)
        state = (
            owner,
            Path(baseline),
            canonical_json(binding),
            methods,
            tuple(
                (p, sha256_file(p))
                for p in (Path(__file__).resolve(), Path(CI.__file__).resolve(), Path(EP.__file__).resolve())
            ),
            copy.deepcopy(cohort_declaration),
            pool_status,
        )
        _ISSUED[self] = state
        try:
            self.record()
        except BaseException:
            _ISSUED.pop(self, None)
            raise

    def record(self):
        if type(self) is not CapsuleOrdinaryScope or self not in _ISSUED:
            raise StageGateError("capsule development scope was not originally issued")
        from .evaluation_pool import ComponentEvaluationPool

        owner, baseline, expected, methods, sources, cohort, pool_status = _ISSUED[self]
        for boundary in (0, 1):
            if type(owner) is not CI.FrozenQualifiedCapsulePhase1:
                raise StageGateError("capsule development changed its original owner")
            for name, function, code in methods:
                method = getattr(owner, name)
                if (
                    type(method) is not MethodType
                    or method.__self__ is not owner
                    or method.__func__ is not function
                    or function.__code__ is not code
                ):
                    raise StageGateError("capsule development qualification callback changed")
            function, code = pool_status
            if ComponentEvaluationPool.status is not function or function.__code__ is not code:
                raise StageGateError("capsule development original pool verifier changed")
            if any(sha256_file(p) != digest for p, digest in sources):
                raise StageGateError("capsule development verifier source changed")
            cohort_value = read_cohort(cohort)
            if boundary == 0:
                binding = owner.verify(baseline)
                if canonical_json(binding) != expected:
                    raise StageGateError("capsule development original qualification changed")
        return dict(
            schema="merlin.capsule_qualified_ordinary_development.v1",
            phase1=copy.deepcopy(binding),
            required_cohort_declaration=copy.deepcopy(cohort),
            required_cohort=cohort_value,
            full_models_pending=True,
            formal_grade_complete=False,
            final_results_qualified=False,
            promotion_authority="NONE",
            allowed_execution="compile_spike_only",
            scope="unscored capsule-qualified source development; every frozen cohort obligation retained",
        )


def select(
    owner,
    *,
    functional_runs_root,
    functional_run_id,
    functional_submission_sha256,
    target_experiment,
    required_cohort_declaration,
):
    read_cohort(required_cohort_declaration)
    if type(owner) is not CI.FrozenQualifiedCapsulePhase1:
        raise StageGateError("capsule development requires the exact original qualified capsule owner")
    if (
        owner.run_dir != Path(functional_runs_root).absolute() / functional_run_id
        or owner.submission_sha256 != functional_submission_sha256
        or sha256_file(owner.descriptor) != target_experiment.descriptor_sha256
    ):
        raise StageGateError("capsule development selects another run, compiler or descriptor")
    config = dict(
        phase1_validation_scope="capsules",
        functional_run_id=functional_run_id,
        functional_submission_sha256=functional_submission_sha256,
        waive_functional_gate=[],
        capsule_qualification=dict(path=str(owner.qualification_path), sha256=owner.qualification_sha256),
    )
    functional, checked = CI.selected_functional_run(Path(functional_runs_root), config, target_experiment.path)
    binding = owner.verify(functional.submission_dir)
    if checked.verify(functional.submission_dir) != binding:
        raise StageGateError("capsule development selection changed its original qualification")
    return functional, CapsuleOrdinaryScope(
        owner, functional.submission_dir, binding, cohort_declaration=required_cohort_declaration, _issuer=_ISSUER
    )


def attach(policy, scope):
    if policy in _POLICIES or type(scope) is not CapsuleOrdinaryScope:
        raise StageGateError("capsule development policy scope must be selected once")
    verify(scope)
    if policy.feedback_evaluator._capsule_development_scope is not scope:
        raise StageGateError("capsule policy changed its original feedback scope")
    _POLICIES[policy] = (scope, [])


def enabled(policy):
    return policy in _POLICIES


def binding(policy):
    if policy not in _POLICIES:
        return {}
    scope, _ = _POLICIES[policy]
    if policy.feedback_evaluator._capsule_development_scope is not scope:
        raise StageGateError("capsule development feedback lost its original scope")
    return {"capsule_development": verify(scope)}


def require_kind(policy, kind):
    if enabled(policy) and kind not in allowed_kinds():
        raise StageGateError("capsule development admits compile/Spike feedback only; RTL is disabled")


def allowed_kinds():
    from .ordinary_fast_feedback import KIND, SELECTED_KIND

    return KIND, SELECTED_KIND


def actions():
    from . import ordinary_impact_selection as I
    from . import ordinary_measurement_selection as M
    from . import ordinary_queue_calibration as C
    from . import queued_search_actions as Q

    denied = {M.ACTION, I.ACTION, C.ACTION}
    from dataclasses import replace

    from .component_workflow import EVALUATION_SUBMIT_ACTION

    return tuple(
        replace(
            a,
            purpose="Queue retained revision with compile/Spike feedback only; all "
            "unselected obligations remain pending",
        )
        if a.name == EVALUATION_SUBMIT_ACTION
        else a
        for a in Q.actions()
        if a.name not in denied
    )


_ORIGINAL_RECORD = CapsuleOrdinaryScope.record
_ORIGINAL_RECORD_CODE = _ORIGINAL_RECORD.__code__


def verify(scope):
    method = getattr(scope, "record", None)
    if (
        type(scope) is not CapsuleOrdinaryScope
        or type(method) is not MethodType
        or method.__self__ is not scope
        or method.__func__ is not _ORIGINAL_RECORD
        or _ORIGINAL_RECORD.__code__ is not _ORIGINAL_RECORD_CODE
    ):
        raise StageGateError("capsule development lost its original scope verifier")
    return method()


def note_submission(policy, action, response):
    if not enabled(policy):
        return
    import json

    from . import component_workflow as C
    from . import ordinary_fast_feedback as F

    if action in {C.EVALUATION_SUBMIT_ACTION, F.SELECTED_ACTION} and response.get("returncode") == 0:
        pool = live_pool(policy)
        value = json.loads(response["stdout"])
        job = pool.status(value["job_id"])
        require_kind(policy, job["kind"])
        import hashlib

        _POLICIES[policy][1].append((action, hashlib.sha256(response["stdout"].encode()).hexdigest(), value["job_id"]))


def verify_receipts(policy, path, actions, audit, *, candidate_sha256=None):
    from . import corpus_feedback as CF

    binding(policy)
    rows, workflow, successes, required = CF.verify_broker_action_receipts(
        path, actions, audit, candidate_sha256=candidate_sha256
    )
    pool = live_pool(policy)
    observed = {(r["action"], r["stdout_sha256"]) for r in rows if r.get("returncode") == 0}
    jobs = []
    for action, digest, identity in _POLICIES[policy][1]:
        if (action, digest) not in observed:
            continue
        job = pool.status(identity)
        if job["kind"] in allowed_kinds() and job["state"] == "completed" and job["source_sha256"] == candidate_sha256:
            feedback = Path(pool.root) / identity / "feedback.json"
            jobs.append(dict(path=str(feedback), sha256=sha256_file(feedback), job_id=identity))
    if not jobs:
        raise StageGateError("capsule development needs completed original compile/Spike feedback for the final source")
    binding(policy)
    return dict(
        path=str(path),
        sha256=sha256_file(path),
        count=len(rows),
        successful_actions=sorted(successes),
        required_actions=sorted(required),
        feedback_successes=len(jobs),
        workflow_policy=workflow,
        feedback_receipts=jobs,
        candidate_sha256=candidate_sha256,
        final_candidate_feedback_verified=True,
        all_required_succeeded=True,
        scope="capsule_development",
        promotion_authority="NONE",
    )


def read_cohort(pin):
    from merlin.common.strict_json import loads

    from .public_diagnostic_launch import read_pin

    # Operator-selected original public planning declaration, not qualification.
    value = loads(read_pin(pin, max_bytes=8 << 20), max_bytes=8 << 20)
    if not isinstance(value, dict) or not value:
        raise StageGateError("capsule development requires its complete original pending cohort declaration")
    return value


def stage_selection(owner, run_root, run_id, digest, target, cohort, waive, search, families, capsules):
    from . import functional_inputs as FI

    if owner is None:
        if cohort is not None:
            raise StageGateError("pending cohort selection requires its explicit capsule owner")
        return FI.inspect_stage_functional_run(run_root, run_id, digest, waive=frozenset(waive or ())), None
    if waive or search is None or families != "all" or capsules != "all":
        raise StageGateError(
            "capsule development requires unwaived qualification, queued search and the complete declared corpus"
        )
    return select(
        owner,
        functional_runs_root=run_root,
        functional_run_id=run_id,
        functional_submission_sha256=digest,
        target_experiment=target,
        required_cohort_declaration=cohort,
    )


def final_guard(scope, guard, baseline, candidate, target, boundary, stage_root, frozen, sandbox_inputs):
    from . import normal_resource_launch as NR

    try:
        with NR.selected_boundary(boundary, writable_roots=(stage_root,)):
            if scope is not None:
                verify(scope)
                return dict(
                    status="pending",
                    reason="capsule development: final functional/full-model qualification remains pending",
                    offenders=[],
                    rows=[],
                )
            return guard(
                baseline,
                candidate,
                target,
                frozen_functional=frozen,
                grade_root=stage_root / "functional_guard_grade",
                sandbox_inputs=sandbox_inputs,
            )
    except Exception as exc:
        return dict(status="unavailable", reason=f"{type(exc).__name__}: {str(exc)[:200]}", offenders=[], rows=[])


def sentinel_record(scope, sentinel):
    if scope is not None:
        return {}
    return dict(
        capsule=sentinel.capsule,
        capsule_path=sentinel.capsule_path,
        frozen_source_path=sentinel.frozen_source_path,
        capsule_sha256=sentinel.capsule_sha256,
        required_lanes=list(sentinel.required_lanes),
        required_tiers=list(sentinel.required_tiers),
    )


def development_record(scope, record, complete):
    scoped = verify(scope)
    record.update(
        phase1_qualification=scoped["phase1"],
        **CI.checkpoint_scope(scoped["phase1"]),
        capsule_development=scoped,
        schema_version="capsule_qualified_source_development_v1",
        kind="capsule_source_development_checkpoint",
    )
    record["admission"].update(
        consumable=False, development_checkpoint_complete=complete, consumer="capsule_source_development_only"
    )
    record["development_feedback"].update(
        action="evaluation-submit-functional-selected",
        required_per_round=True,
        engine="spike",
        scope="complete frozen cohort; selected observations keep other obligations pending",
        redaction="original compile/Spike feedback and domain-qualified screening only; no measured promotion",
    )


def verify_checkpoint(record):
    CI.verify_checkpoint_scope(record)
    if (
        record.get("kind") != "capsule_source_development_checkpoint"
        or record.get("schema_version") != "capsule_qualified_source_development_v1"
        or record.get("admission", {}).get("consumable") is not False
    ):
        raise StageGateError("capsule development cannot become a consumable performance candidate")


def initial_analysis(scope, feedback, baseline, candidate, sentinel, contract_root, timeout_s, target):
    if scope is not None:
        return {"status": "pending", "scope": "capsule development; full-model analysis not admitted"}
    from . import emission_analysis as EA

    return EA.analyze_whole_model_emission(
        baseline,
        candidate,
        sentinel,
        contract_root=contract_root,
        timeout_s=timeout_s,
        peak_macs_per_cycle=getattr(feedback, "peak_macs_per_cycle", None),
        achievable_macs_per_cycle=getattr(feedback, "achievable_macs_per_cycle", None),
        target=str(target or ""),
    )


def stage_actions(scope, original):
    return actions() if scope is not None else original


def record_path(stage_root, scope):
    return stage_root / ("capsule_development_checkpoint.json" if scope is not None else "performance_candidate.json")


def live_pool(policy):
    from .evaluation_pool import ComponentEvaluationPool

    pool = policy.evaluation_pool
    method = getattr(pool, "status", None)
    if (
        type(pool) is not ComponentEvaluationPool
        or pool.policy is not policy
        or type(method) is not MethodType
        or method.__self__ is not pool
        or method.__func__ is not ComponentEvaluationPool.status
    ):
        raise StageGateError("capsule source submissions require the original live pool and verifier")
    return pool
