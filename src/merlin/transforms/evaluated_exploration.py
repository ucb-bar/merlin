"""Bounded real pass exploration joined to externally completed cycle feedback.

The ordinary caller supplies source/correctness/provider custody and target pass
factories. This module neither runs a simulator nor invents a schedule, score or
verification grant. Selection returns the already emitted evaluated product.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import Executor, as_completed
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass, fields, replace
from time import monotonic_ns

from .pass_exploration import (
    EmittedProgram,
    ObservedMetrics,
    explore_candidates,
    generate_candidates,
    rank_performance_variants,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def exploration_context_identity(context):
    """Bind the actual shared PipelineContext without inspecting a target name."""
    from .pipeline_state import PipelineContext

    if type(context) is not PipelineContext:
        raise ValueError("explicit ordinary pipeline context required")
    return _sha(json.dumps(asdict(context), sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True)
class ExplorationIdentity:
    source_sha256: str
    abi_sha256: str
    inputs_sha256: str
    reference_sha256: str
    facts_sha256: str
    library_sha256: str
    provider_sha256: str
    scope_sha256: str
    roi_sha256: str
    context_sha256: str
    artifact_format: str
    basis: str

    def __post_init__(self):
        if (any(not _pin(getattr(self, field.name)) for field in fields(self) if field.name.endswith("_sha256"))
                or self.basis not in ("modeled_roi", "measured_roi")
                or type(self.artifact_format) is not str or not self.artifact_format or len(self.artifact_format) > 64):
            raise ValueError("exploration needs complete source/ABI/input/reference/context/provider/ROI pins")

    def digest(self):
        return _sha(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True)
class CompletedObservation:
    identity: ExplorationIdentity
    program_sha256: str
    feedback_sha256: str
    numerical_status: str
    timing_status: str
    complete_roi: bool
    cycles: int | None

    def __post_init__(self):
        if (type(self.identity) is not ExplorationIdentity or not _pin(self.program_sha256)
                or not _pin(self.feedback_sha256) or self.numerical_status not in ("pass", "fail", "unknown")
                or self.timing_status not in ("complete", "unknown", "failed") or type(self.complete_roi) is not bool
                or self.cycles is not None and (type(self.cycles) is not int or self.cycles <= 0)):
            raise ValueError("external observation requires exact identities and explicit numerical/timing status")
        self.identity.__post_init__()

    @property
    def eligible(self):
        return (self.numerical_status == "pass" and self.timing_status == "complete"
                and self.complete_roi and self.cycles is not None)


@dataclass(frozen=True, init=False)
class EvaluationRequest:
    """Immutable emitted input, without mutable IR or compiler-directory copies.

    The ordinary owner pins any external artifacts named by its private receipt
    with evaluate_emitted's hold_inputs callback. Canonical receipt bytes prevent
    caller mutation while an external evaluation is in flight.
    """
    program: EmittedProgram
    identity: ExplorationIdentity
    receipt_json: str
    program_sha256: str
    evaluation_identity: str

    def __init__(self, program, receipt, identity):
        if type(program) is not EmittedProgram or type(identity) is not ExplorationIdentity:
            raise ValueError("evaluation request needs actual emitted bytes and exact domain")
        program.__post_init__()
        identity.__post_init__()
        if type(receipt) is not dict:
            raise ValueError("evaluation request needs its actual plain pipeline receipt")
        payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        context = receipt.get("after", {}).get("context")
        if (_sha(json.dumps(context, sort_keys=True, separators=(",", ":")).encode())
                != identity.context_sha256 or program.format != identity.artifact_format):
            raise ValueError("actual emitted pipeline context/format differs from the selected evaluation domain")
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "receipt_json", payload)
        object.__setattr__(self, "program_sha256", _sha(program.data))
        object.__setattr__(self, "evaluation_identity", identity.digest())

    @property
    def receipt(self):
        return json.loads(self.receipt_json)

    @property
    def key(self):
        return self.evaluation_identity, self.program_sha256


@dataclass(frozen=True)
class EvaluationCompletion:
    """Small per-completion result; no retained compiler tree or artifact copy."""
    identity: ExplorationIdentity
    program_sha256: str
    observation: CompletedObservation | None
    origin: str
    submitted_ns: int
    completed_ns: int
    error: tuple[str, str] | None = None

    def record(self):
        observed = self.observation
        return dict(schema="merlin.emitted_evaluation_completion.v1",
            evaluation_identity=self.identity.digest(), program_sha256=self.program_sha256,
            status="evaluation_error" if self.error else "pending" if observed is None else
                "rankable" if observed.eligible else "unranked", origin=self.origin,
            numerical_status=None if observed is None else observed.numerical_status,
            timing_status=None if observed is None else observed.timing_status,
            complete_roi=None if observed is None else observed.complete_roi,
            cycles=None if observed is None else observed.cycles,
            feedback_sha256=None if observed is None else observed.feedback_sha256,
            submitted_ns=self.submitted_ns, completed_ns=self.completed_ns,
            external_wall_ns=self.completed_ns-self.submitted_ns,
            wall_scope="submission through owner collection, including external admission and queue wait",
            error=self.error, performance_qualification=False)


def evaluate_emitted(requests, evaluate, *, executor=None, max_in_flight=1,
                     max_requests=4096, on_completion=None, hold_inputs=None, reuse_observations=()):
    """Stream unique external completions across variant AND shape domains.

    Emit requests with existing explore_candidates(..., feedback=None). Caller
    supplies one Executor and ordinary resource-leased evaluation adapter; this
    function creates no worker/CPU pool or simulator. A process executor receives
    only immutable program bytes, a fresh plain receipt and domain, never IR.
    executor=None preserves serial execution. max_in_flight bounds submissions;
    it does not reserve cores or override the ordinary tool owner's CPU leases.

    Dedup is (exact domain, emitted bytes). Different shapes/provider/ROI pins
    never share evidence. Only eligible exact reuse_observations avoid a new
    callback. UNKNOWN remains unranked; it is never a static-count score.

    hold_inputs(request) is an optional owner context manager pinning external
    blobs/overlays BEFORE submit through the end of on_completion(request,
    completion). It closes before yielding the small completion record. Caller
    owns retention/publication/deletion: no path is inferred or removed here.
    Infrastructure errors publish an error completion then propagate; owned
    submissions not yet running are cancelled, running jobs keep their pins until
    their callbacks finish. The caller retains ownership of the executor.
    """
    if (not callable(evaluate) or executor is not None and not isinstance(executor, Executor)
            or type(max_in_flight) is not int or not 1 <= max_in_flight <= 256
            or executor is None and max_in_flight != 1
            or type(max_requests) is not int or not 1 <= max_requests <= 65536
            or on_completion is not None and not callable(on_completion)
            or hold_inputs is not None and not callable(hold_inputs)):
        raise ValueError("explicit evaluator, caller-owned executor and bounded in-flight/request budget required")
    cache = {}
    for observed in reuse_observations:
        if type(observed) is not CompletedObservation:
            raise ValueError("reuse requires exact completed observations")
        observed.__post_init__()
        if observed.eligible:
            key = observed.identity.digest(), observed.program_sha256
            if key in cache and cache[key] != observed:
                raise ValueError("conflicting same-domain observations cannot be silently reused")
            cache[key] = observed
    seen, pending = set(), {}

    def complete(request, observed, origin, started, error=None):
        if observed is not None:
            if type(observed) is not CompletedObservation:
                raise ValueError("evaluation must return an explicit completed observation or pending None")
            observed.__post_init__()
            if observed.identity != request.identity or observed.program_sha256 != request.key[1]:
                raise ValueError("evaluation does not match this exact source/program/provider/ROI")
        result = EvaluationCompletion(request.identity, request.key[1], observed, origin,
                                      started, monotonic_ns(), error)
        if on_completion is not None:
            on_completion(request, result)
        return result

    def collect(future):
        request, hold, started = pending.pop(future)
        with hold:
            try:
                observed = future.result()
            except BaseException as error:
                complete(request, None, "external_evaluation", started,
                         (type(error).__name__, str(error)[:512]))
                raise
            return complete(request, observed, "external_evaluation", started)

    try:
        for count, request in enumerate(requests, 1):
            if count > max_requests:
                raise ValueError("emitted evaluation request budget exhausted")
            if type(request) is not EvaluationRequest:
                raise ValueError("external evaluation requires immutable EvaluationRequest objects")
            if request.key in seen:
                continue
            seen.add(request.key)
            started = monotonic_ns()
            # Enter on the owner thread before submitting. ExitStack preserves
            # that same-thread lifecycle through callback publication/collection.
            hold = ExitStack()
            try:
                hold.enter_context(hold_inputs(request) if hold_inputs else nullcontext())
                if request.key in cache:
                    with hold:
                        result = complete(request, cache[request.key], "existing_observation", started)
                    yield result
                elif executor is None:
                    with hold:
                        try:
                            observed = evaluate(request.program, request.receipt, request.identity)
                        except BaseException as error:
                            complete(request, None, "external_evaluation", started,
                                     (type(error).__name__, str(error)[:512]))
                            raise
                        result = complete(request, observed, "external_evaluation", started)
                    yield result
                else:
                    future = executor.submit(evaluate, request.program, request.receipt, request.identity)
                    pending[future] = request, hold, started
            except BaseException:
                hold.close()
                raise
            if len(pending) >= max_in_flight:
                yield collect(next(as_completed(tuple(pending))))
        while pending:
            yield collect(next(as_completed(tuple(pending))))
    finally:
        # Pins cover callbacks still running even if the consumer stops early.
        # Only this invocation's queued submissions are cancelled, never peers.
        for future in tuple(pending):
            future.cancel()
        for future, (_, hold, _) in tuple(pending.items()):
            while not future.done():
                try:
                    future.result()
                except KeyboardInterrupt:
                    continue  # The still-running owner must retain its input pins.
                except BaseException:
                    break  # Preserve the original error; no ranking issued.
            hold.close()
        pending.clear()


@dataclass(frozen=True)
class EvaluatedExploration:
    candidates: tuple
    observations: tuple[CompletedObservation, ...]
    ranking: dict
    winner: object | None
    outcomes: tuple[dict, ...] = ()
    incumbent: CompletedObservation | None = None
    construction: dict | None = None
    identity: ExplorationIdentity | None = None

    def summary(self):
        """Compact actual outcomes; no inferred failure owner or speedup."""
        return dict(schema="merlin.evaluated_exploration.outcomes.v1", outcomes=self.outcomes,
                    ranking=self.ranking, construction=self.construction,
                    evaluation_identity=self.identity.digest() if self.identity else None,
                    incumbent_program_sha256=self.incumbent.program_sha256 if self.incumbent else None,
                    winner_variant=self.winner.evaluation.variant if self.winner else None,
                    performance_qualification=False)


def _outcome(candidate, observations, incumbent):
    record = dict(variant=candidate.evaluation.variant, **candidate.outcome())
    if candidate.emission is None:
        return record
    digest = _sha(candidate.emission.data)
    same = digest == incumbent.program_sha256 if incumbent else None
    record.update(program_sha256=digest, same_as_incumbent=same)
    observed = observations.get(digest)
    if observed is None:
        record.update(evaluation=dict(status="pending"), status="evaluation_pending", layer="evaluation",
                      next_action="collect_exact_artifact_observation_or_try_another_legal_alternative")
        return record
    record["evaluation"] = dict(numerical_status=observed.numerical_status,
        timing_status=observed.timing_status, complete_roi=observed.complete_roi, cycles=observed.cycles,
        eligible=observed.eligible, feedback_sha256=observed.feedback_sha256)
    if observed.numerical_status != "pass":
        status, layer = "numerical_" + observed.numerical_status, "numerical"
        action = "inspect_original_output_comparison_before_any_performance_claim"
    elif observed.timing_status != "complete":
        status, layer = "timing_" + observed.timing_status, "timing"
        action = "inspect_provider_feedback_or_explore_legal_structural_changes_unranked"
    elif not observed.complete_roi:
        status, layer, action = "timing_partial_roi", "timing", "request_same_complete_roi_without_extrapolating"
    elif observed.cycles is None:
        status, layer, action = "timing_cycles_missing", "timing", "obtain_cycles_no_static_count_substitution"
    else:
        layer = "performance"
        if incumbent is None:
            status, action = "cycle_ranked", "compare_explicit_same_domain_incumbent_for_gain_or_regression"
        elif not incumbent.eligible:
            status, action = "cycle_ranked_incumbent_unranked", "retain_alternative_no_incumbent_speedup_claim"
        else:
            delta = observed.cycles - incumbent.cycles
            status = ("same_as_incumbent" if same else "cycle_improvement" if delta < 0
                      else "cycle_regression" if delta > 0 else "cycle_tie")
            action = "retain_faster_evaluated_product" if delta < 0 else "retain_incumbent_try_distinct_emission"
            record["comparison"] = dict(incumbent_program_sha256=incumbent.program_sha256,
                incumbent_cycles=incumbent.cycles, delta_cycles=delta,
                cycles_ratio=observed.cycles / incumbent.cycles, basis=observed.identity.basis)
    record.update(status=status, layer=layer, next_action=action)
    return record


def explore_evaluated(module, families, facts, *, output, emission, evaluate, identity,
                      before=(), max_candidates=16, max_search_steps=4096, incumbent=None,
                      executor=None, max_in_flight=1, on_completion=None, hold_inputs=None, **session_options):
    """Run actual registered passes, deduplicate emitted bytes, consume feedback.

    evaluate(EmittedProgram, pipeline_receipt, ExplorationIdentity) returns a
    CompletedObservation or None for pending feedback. The callback belongs to
    the ordinary tool owner; supplied hashes detect mismatched joins, not forged
    evidence. No missing cycles, partial ROI or numerical failure can win.

    An optional exact same-domain incumbent is an existing observation, not a
    guessed first variant. Its identical emitted artifact reuses that observation.
    With it, winner names only a strictly faster eligible candidate; None retains
    the caller's incumbent (or makes no speedup decision if it is unranked).
    Outcome hints explain missing evidence without blocking further authoring.
    Optional executor/max_in_flight use evaluate_emitted's ordinary caller-owned
    budget; on_completion and hold_inputs share its publication/retention seam.
    """
    if type(identity) is not ExplorationIdentity or not callable(evaluate):
        raise ValueError("explicit exploration identity and external evaluation adapter required")
    identity.__post_init__()
    if incumbent is not None:
        if type(incumbent) is not CompletedObservation:
            raise ValueError("incumbent must be an exact completed observation")
        incumbent.__post_init__()
        if incumbent.identity != identity:
            raise ValueError("incumbent differs from the exact source/provider/ROI exploration domain")
    space = generate_candidates(families, output=output, before=before,
                                max_candidates=max_candidates, max_search_steps=max_search_steps)
    observations, candidates, feedback_records = {}, [], {}

    def requests():
        for candidate in explore_candidates(module, space, facts, output=output,
                emission=emission, **session_options):
            candidates.append(candidate)
            if candidate.emission is not None and candidate.duplicate_of is None:
                yield EvaluationRequest(candidate.emission, candidate.evaluation.result.receipt, identity)

    for completion in evaluate_emitted(requests(), evaluate, executor=executor,
            max_in_flight=max_in_flight, on_completion=on_completion, hold_inputs=hold_inputs,
            reuse_observations=() if incumbent is None else (incumbent,)):
        observed = completion.observation
        if observed is not None:
            observations[completion.program_sha256] = observed
        feedback_records[completion.program_sha256] = dict(
            program_sha256=completion.program_sha256, evaluation_identity=identity.digest(),
            status="pending" if observed is None else "rankable" if observed.eligible else "unranked",
            feedback_sha256=None if observed is None else observed.feedback_sha256,
            observation_origin="explicit_incumbent" if completion.origin == "existing_observation"
                else "external_evaluation")
    candidates = tuple(replace(candidate, feedback=feedback_records[_sha(candidate.emission.data)])
        if candidate.emission is not None and candidate.duplicate_of is None else candidate
        for candidate in candidates)
    unique = {_sha(item.emission.data): item for item in candidates
              if item.emission is not None and item.duplicate_of is None}
    metrics = tuple((digest, ObservedMetrics(identity.basis, identity.digest(), (("cycles", observation.cycles),)))
                    for digest, observation in observations.items() if observation.eligible)
    ranking = rank_performance_variants(metrics) if metrics else {
        "ranked": (), "decision_metric": "complete_same_roi_cycles", "performance_qualification": False}
    ranking = {**ranking, "unranked_programs": tuple(digest for digest in unique
               if digest not in observations or not observations[digest].eligible),
               "policy": "no static-count, partial-ROI or UNKNOWN fallback"}
    winner = unique[ranking["ranked"][0][0]] if ranking["ranked"] else None
    if incumbent is not None:
        if not incumbent.eligible or not ranking["ranked"] or ranking["ranked"][0][1][0] >= incumbent.cycles:
            winner = None
        ranking["incumbent_program_sha256"] = incumbent.program_sha256
        ranking["selection"] = ("candidate_improvement" if winner is not None else "incumbent_retained"
                                if incumbent.eligible else "incumbent_unranked_no_comparison")
    outcomes = tuple(_outcome(candidate, observations, incumbent) for candidate in candidates)
    return EvaluatedExploration(candidates, tuple(observations.values()), ranking, winner, outcomes, incumbent,
                                dict(search_steps=space.search_steps, truncated=space.truncated,
                                     refusals=space.refusals), identity)
