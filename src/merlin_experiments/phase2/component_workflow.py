"""Explicit generated-component feedback; no complete-model training authority.

The launcher admits its minimal view and fresh session isolation separately.
Comparisons use the Phase 1 compiler. Estimates confer no measurement or accuracy;
RTL uses its exact workload gate.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

from merlin.benchharness import hash_tree
from merlin.perf.phase2_calibration_bundle import prepare_phase2_calibration
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from . import component_cca as CC
from . import corpus as C
from . import corpus_feedback as CF
from .broker import BrokerAction
from .broker_evidence import _is_sha256, workflow_binding
from .broker_policy import (
    ANALYSIS_ACTION,
    COMPONENT_ONLY_V1,
    INVENTORY_ACTION,
    ActionAdmission,
    BrokerServices,
    WorkflowPolicy,
    _build_action_registry,
    _record_host_refusal,
)
from .contracts import StageGateError, canonical_json, document_sha256, mapping_file, sha256_file

ANALYTICAL_ACTION = "component-analytical-feedback"
RTL_ACTION = "component-rtl-feedback"
CCA_ACTION = CC.ACTION
SCHEMA = "host-component-feedback-v1"
REVISION_FREEZE_ACTION = "revision-freeze"
EVALUATION_SUBMIT_ACTION = "evaluation-submit"
EVALUATION_STATUS_ACTION = "evaluation-status"
EVALUATION_CANCEL_ACTION = "evaluation-cancel"
POOL_ACTIONS = (REVISION_FREEZE_ACTION, EVALUATION_SUBMIT_ACTION, EVALUATION_STATUS_ACTION, EVALUATION_CANCEL_ACTION)
SEARCH_STATUS_ACTION = "search-status"
SEARCH_FINALIZE_ACTION = "search-finalize"
SEARCH_SELECT_ACTION = "search-select"
SEARCH_ACTIONS = (SEARCH_STATUS_ACTION, SEARCH_FINALIZE_ACTION, SEARCH_SELECT_ACTION)


def evaluation_pool_actions(*, search=False):
    annotations = ("hypothesis", "edit_family") if search else ()
    return tuple(
        BrokerAction(name, ("__host_component_pool__", name), bindings, purpose, False)
        for name, bindings, purpose in (
            (
                REVISION_FREEZE_ACTION,
                ("parent_revision", *annotations),
                "Freeze authorized source; parent_revision=none starts a branch",
            ),
            (
                EVALUATION_SUBMIT_ACTION,
                ("revision_id", "kind"),
                "Queue fixed component feedback for a retained revision",
            ),
            (
                EVALUATION_STATUS_ACTION,
                ("job_id",),
                "Read queue state and verified completed feedback; no final acceptance",
            ),
            (EVALUATION_CANCEL_ACTION, ("job_id",), "Cancel queued work or terminate and reap an active evaluation"),
        )
    )


def search_actions():
    return tuple(
        BrokerAction(name, ("__host_component_search__", name), (), purpose, False)
        for name, purpose in (
            (SEARCH_STATUS_ACTION, "Read conservative measured ranking and bounded search stopping status"),
            (SEARCH_FINALIZE_ACTION, "Stop submissions and finish or reap queued work within the tool budget"),
            (SEARCH_SELECT_ACTION, "Restore the current measured incumbent after finalization; join all authors first"),
        )
    )


_GLOBAL_INPUTS = (
    "functional_base",
    "e2e_sentinel",
    "global_experiment",
    "global_probe_provider",
    "global_semantic_provider",
    "global_context_provider",
    "global_paired_context_provider",
    "global_source_pair_provider",
    "feedback_evaluator",
)


def unavailable_component_actions(*, analytical=False, rtl=False, structural=False, cca=False):
    """Unknown providers remain unavailable during registry inspection."""
    choices = (
        (ANALYTICAL_ACTION, analytical, "explicit calibrated component provider is unavailable"),
        (RTL_ACTION, rtl, "explicit certified component RTL provider is unavailable"),
        (ANALYSIS_ACTION, structural, "component structural analysis service is unavailable"),
        (CCA_ACTION, cca, "explicit component CCA provider is unavailable"),
    )
    return {name: reason for name, installed, reason in choices if not installed}


def component_actions():
    return (
        BrokerAction(
            CCA_ACTION,
            ("__host_component_cca__",),
            (),
            "All-facet generated-component CCA gaps and unknowns; no cycle or acceptance claim",
            False,
        ),
        BrokerAction(
            ANALYTICAL_ACTION,
            ("__host_component_analytical__",),
            (),
            "Generated components only: calibrated intervals, unknown costs explicit; no measured cycles",
            False,
        ),
        BrokerAction(
            RTL_ACTION,
            ("__host_component_rtl__",),
            (),
            "Bounded generated components through the existing exact-workload GSIM gate; never Spike timing",
            False,
        ),
        BrokerAction(
            ANALYSIS_ACTION,
            ("__host_owned_command_buffer_analysis__", "{baseline_json}", "{candidate_json}"),
            ("baseline_json", "candidate_json"),
            "Structural component command-buffer comparison; no speedup claim",
            False,
        ),
        BrokerAction(
            INVENTORY_ACTION,
            ("__host_owned_optimization_inventory__",),
            (),
            "Current compiler edit-owner inventory; no workload graph access",
            False,
        ),
    )


def _pin(path, digest):
    path = Path(path)
    if not _is_sha256(digest) or path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise StageGateError("component provider source/config identity changed")
    return path.resolve()


def _callable_source(callback):
    while isinstance(callback, partial):
        callback = callback.func
    source = inspect.getsourcefile(callback) if callable(callback) else None
    if source is None:
        raise StageGateError("component provider lacks an inspectable source owner")
    path = Path(source).resolve()
    return path, sha256_file(path)


def _callable_code(callback):
    while isinstance(callback, partial):
        callback = callback.func
    if inspect.ismethod(callback):
        callback = callback.__func__
    return getattr(callback, "__code__", None)


class ComponentAnalyticalEvaluation(Protocol):
    def __call__(
        self,
        *,
        candidate: Path,
        corpus: C.FrozenPerformanceCorpus,
        timeout_s: float,
    ) -> Mapping[tuple[str, str], tuple[CycleInterval, CycleInterval]]: ...


@dataclass(frozen=True)
class ComponentAnalyticalProvider:
    """One exact component cohort and calibration; estimates confer no acceptance."""

    evaluate: ComponentAnalyticalEvaluation
    implementation: Path
    implementation_sha256: str
    calibration_adapter: Path
    calibration_adapter_sha256: str
    calibration_sha256: str
    binding: object | None = None

    def validate(self):
        implementation = _pin(self.implementation, self.implementation_sha256)
        if _callable_source(self.evaluate)[0] != implementation:
            raise StageGateError("component callback does not execute its pinned implementation")
        adapter = _pin(self.calibration_adapter, self.calibration_adapter_sha256)
        prepared = prepare_phase2_calibration(adapter)
        calibration = prepared.get("calibration")
        if prepared.get("status") != "ready" or not isinstance(calibration, Mapping):
            raise StageGateError("component analytical calibration is unavailable")
        if not _is_sha256(self.calibration_sha256) or document_sha256(calibration) != self.calibration_sha256:
            raise StageGateError("component analytical calibration changed")
        if self.binding is not None:
            from .component_analytical import ComponentAnalyticalBinding

            if type(self.binding) is not ComponentAnalyticalBinding:
                raise StageGateError("component analytical binding is not host-owned")
            self.binding.validate(calibration)
        return calibration

    @property
    def configuration_sha256(self):
        return (
            document_sha256({"binding_sha256": self.binding.sha256, "calibration_sha256": self.calibration_sha256})
            if self.binding is not None
            else self.calibration_sha256
        )


def _interval(value):
    if type(value) is not CycleInterval:
        raise StageGateError("component analytical endpoints require typed CycleInterval")
    if value.resolved and (
        not math.isfinite(value.lo) or not math.isfinite(value.hi) or not value.provenance or value.missing
    ):
        raise StageGateError("resolved component estimate lacks finite qualified provenance")
    if not value.resolved and (
        not value.missing or any(not isinstance(reason, str) or not reason.strip() for reason in value.missing)
    ):
        raise StageGateError("unknown component cost lacks its reason")
    if any(not isinstance(item, str) or not item.strip() for item in value.provenance):
        raise StageGateError("component cost provenance is malformed")
    return value.to_dict()


def _corpus(corpus):
    if type(corpus) is not C.FrozenPerformanceCorpus or not corpus.capsules:
        raise StageGateError("component profile requires a frozen generated corpus")
    C.verify_frozen_performance_corpus(corpus)
    document = json.loads(corpus.manifest_path.read_bytes())
    source = document.get("source")
    if not isinstance(source, Mapping) or any(
        not _is_sha256(source.get(key)) for key in ("provenance_sha256", "performance_generation_sha256")
    ):
        raise StageGateError("component corpus lacks generated provenance")
    identities = {(member.family, member.capsule) for member in corpus.capsules}
    if len(identities) != len(corpus.capsules):
        raise StageGateError("component corpus has duplicate members")
    for member in corpus.capsules:
        descriptor = mapping_file(member.source_dir / "capsule.yaml", yaml_file=True)
        if (
            descriptor != member.descriptor
            or descriptor.get("name") != member.capsule
            or descriptor.get("label") != "dev"
            or descriptor.get("source_role") != "derived_sweep"
            or descriptor.get("kind") == "model"
        ):
            raise StageGateError("component corpus must contain generated development components, not models")
    return identities


def validate_component_feedback(document):
    """Strict answer-free envelope; estimates and existing RTL evidence stay distinct."""
    fields = {
        "schema",
        "workflow_id",
        "tier",
        "candidate_sha256",
        "corpus_sha256",
        "manifest_sha256",
        "provider_sha256",
        "configuration_sha256",
        "evidence",
        "promotion",
    }
    if (
        not isinstance(document, Mapping)
        or set(document) not in (fields, fields | {"screening"})
        or document.get("schema") != SCHEMA
    ):
        raise StageGateError("component feedback schema is invalid")
    if "screening" in document and document.get("tier") != "calibrated_component_analytical":
        raise StageGateError("component screening requires complete analytical cost feedback")
    if document["workflow_id"] != COMPONENT_ONLY_V1 or document["promotion"] != "NO_FINAL_ACCEPTANCE":
        raise StageGateError("component feedback cannot grant final acceptance")
    for key in ("candidate_sha256", "corpus_sha256", "manifest_sha256", "provider_sha256", "configuration_sha256"):
        if not _is_sha256(document[key]):
            raise StageGateError("component feedback identity is invalid")
    if document["tier"] == "certified_component_rtl":
        evidence = CF.validate_redacted_feedback(document["evidence"])
        if (
            evidence["candidate_sha256"] != document["candidate_sha256"]
            or evidence["tuning_corpus_sha256"] != document["corpus_sha256"]
            or evidence["certificate_sha256"] != document["configuration_sha256"]
        ):
            raise StageGateError("component RTL envelope differs from certified evidence")
    elif document["tier"] == CC.TIER:
        CC.validate_evidence(document)
    elif document["tier"] == "calibrated_component_analytical":
        evidence = document["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise StageGateError("component analytical evidence is empty")
        seen = set()
        for row in evidence:
            if not isinstance(row, Mapping) or set(row) not in (
                {"family", "capsule", "baseline", "candidate"},
                {"family", "capsule", "baseline", "candidate", "complete_cost"},
            ):
                raise StageGateError("component analytical member schema is invalid")
            if "complete_cost" in row:
                from merlin.perf.component_cost import validate_complete_cost_report

                detailed = row["complete_cost"]
                if not isinstance(detailed, Mapping) or set(detailed) != {"baseline", "candidate", "objective"}:
                    raise StageGateError("component complete-cost projection is invalid")
                if detailed["objective"] not in ("cold", "warm"):
                    raise StageGateError("component complete-cost regime is invalid")
                for arm in ("baseline", "candidate"):
                    validate_complete_cost_report(detailed[arm])
                    if detailed[arm]["regimes"][detailed["objective"]]["total"] != row[arm]:
                        raise StageGateError("component complete-cost total differs from analytical interval")
            identity = row["family"], row["capsule"]
            if any(not isinstance(item, str) or not item for item in identity) or identity in seen:
                raise StageGateError("component analytical member identity is invalid")
            seen.add(identity)
            for arm in ("baseline", "candidate"):
                interval = row[arm]
                if not isinstance(interval, Mapping) or set(interval) != {
                    "lo",
                    "hi",
                    "resolved",
                    "provenance",
                    "missing",
                }:
                    raise StageGateError("component analytical interval schema is invalid")
                if not isinstance(interval["provenance"], list) or not isinstance(interval["missing"], list):
                    raise StageGateError("component analytical interval evidence is invalid")
                if type(interval["resolved"]) is not bool:
                    raise StageGateError("component analytical resolved status must be boolean")
                parsed = CycleInterval(
                    interval["lo"], interval["hi"], tuple(interval["provenance"]), tuple(interval["missing"])
                )
                if _interval(parsed) != dict(interval):
                    raise StageGateError("component analytical interval is inconsistent")
        if "screening" in document:
            from .component_screening import validate_screening_projection

            validate_screening_projection(document["screening"], evidence)
    else:
        raise StageGateError("component tier is unsupported; Spike is not RTL timing")
    return dict(document)


class ComponentOnlyPolicy(WorkflowPolicy):
    @property
    def workflow_id(self):
        return COMPONENT_ONLY_V1

    def __init__(
        self,
        *,
        component_corpus,
        component_analytical=None,
        component_rtl=None,
        component_cca=None,
        baseline_admission=None,
        independent_runtime=None,
        **inputs,
    ):
        if any(inputs.get(key) is not None for key in _GLOBAL_INPUTS):
            raise StageGateError("component profile refuses inherited whole-model/legacy feedback capabilities")
        services = inputs.get("services", BrokerServices())
        if type(services) is not BrokerServices:
            raise StageGateError("component services require the exact host-owned service contract")
        if services.whole_model_analysis is not None or services.global_analysis_view is not None:
            raise StageGateError("component profile refuses whole-model services")
        super().__init__(**inputs)
        self.component_corpus = component_corpus
        self.component_analytical = component_analytical
        self.component_rtl = component_rtl
        self.component_cca = component_cca
        self.baseline_admission = baseline_admission
        self.independent_runtime = independent_runtime
        self.evaluation_pool = None
        self.search_coordinator = None
        if component_analytical is not None and type(component_analytical) is not ComponentAnalyticalProvider:
            raise StageGateError("explicit typed component analytical provider required")
        self._component_bindings = (component_corpus.manifest_sha256, component_corpus.capsules_sha256)
        self._target_binding = sha256_file(self.target_experiment.path)
        if component_cca is not None and type(component_cca) is not CC.ComponentCCAProvider:
            raise StageGateError("explicit typed component CCA provider required")
        self._provider_selections = (
            component_analytical,
            component_rtl,
            component_cca,
            services,
            baseline_admission,
            independent_runtime,
        )
        self._cca_callable = component_cca.evaluate if component_cca is not None else None
        self._cca_code = _callable_code(self._cca_callable)
        self._analytical_callable = component_analytical.evaluate if component_analytical is not None else None
        self._analytical_code = _callable_code(self._analytical_callable)
        self._structural_source = (
            _callable_source(services.command_buffer_analysis) if services.command_buffer_analysis is not None else None
        )
        self._rtl_configuration = None
        if component_rtl is not None:
            from .development_feedback import DevelopmentGsimFeedback

            if type(component_rtl) is not DevelopmentGsimFeedback:
                raise StageGateError("component RTL must use the existing certified GSIM evaluator")
            rtl = component_rtl
            self._rtl_configuration = (
                _callable_source(rtl.executor),
                rtl.executor,
                rtl.certificate.path,
                rtl.certificate.sha256,
                rtl.baseline,
                rtl.baseline_sha256,
                document_sha256(rtl.rtl_identity),
                _callable_source(type(rtl)),
            )
        self._validate()

    def _validate(self):
        from .component_baseline import verify_policy_baselines

        verify_policy_baselines(self)
        if sha256_file(self.target_experiment.path) != self._target_binding:
            raise StageGateError("component target configuration changed")
        if (
            self.component_analytical,
            self.component_rtl,
            self.component_cca,
            self.services,
            self.baseline_admission,
            self.independent_runtime,
        ) != self._provider_selections:
            raise StageGateError("component provider selection changed")
        if self._structural_source is not None:
            _pin(*self._structural_source)
        if any(getattr(self, key, None) is not None for key in _GLOBAL_INPUTS):
            raise StageGateError("component profile gained a forbidden legacy capability")
        identities = _corpus(self.component_corpus)
        if mapping_file(self.component_corpus.manifest_path).get("target") != self.target_experiment.target:
            raise StageGateError("component corpus targets another descriptor")
        if self._component_bindings != (self.component_corpus.manifest_sha256, self.component_corpus.capsules_sha256):
            raise StageGateError("component corpus selection changed")
        if self.component_cca is not None:
            if (
                self.component_cca.evaluate is not self._cca_callable
                or _callable_code(self.component_cca.evaluate) is not self._cca_code
            ):
                raise StageGateError("component CCA callable binding changed")
            self.component_cca.validate()
        if self.component_analytical is not None:
            if type(self.component_analytical) is not ComponentAnalyticalProvider:
                raise StageGateError("explicit typed component analytical provider required")
            if (
                self.component_analytical.evaluate is not self._analytical_callable
                or _callable_code(self.component_analytical.evaluate) is not self._analytical_code
            ):
                raise StageGateError("component analytical callable binding changed")
            calibration = self.component_analytical.validate()
            if calibration.get("target_sha256") != sha256_file(self.target_experiment.path):
                raise StageGateError("component calibration targets another descriptor")
        if self.component_rtl is not None:
            from .development_feedback import DevelopmentGsimFeedback

            if type(self.component_rtl) is not DevelopmentGsimFeedback:
                raise StageGateError("component RTL must use the existing certified GSIM evaluator")
            rtl = self.component_rtl
            from .gsim_gate import load_certificate

            if (
                rtl.executor is not self._rtl_configuration[1]
                or (
                    rtl.certificate.path,
                    rtl.certificate.sha256,
                    rtl.baseline,
                    rtl.baseline_sha256,
                    document_sha256(rtl.rtl_identity),
                )
                != self._rtl_configuration[2:7]
            ):
                raise StageGateError("component RTL configuration changed")
            _pin(*self._rtl_configuration[0])
            _pin(*self._rtl_configuration[7])
            if str(hash_tree(rtl.baseline)["sha256"]) != rtl.baseline_sha256:
                raise StageGateError("component RTL baseline compiler changed")
            certificate = load_certificate(rtl.certificate.path, expected_sha256=rtl.certificate.sha256)
            if certificate.target != self.target_experiment.target:
                raise StageGateError("component RTL certificate targets another descriptor")
            if (
                not isinstance(self.feedback_round, int)
                or isinstance(self.feedback_round, bool)
                or self.feedback_round < 0
            ):
                raise StageGateError("component RTL requires a nonnegative round identity")
            if (rtl.corpus.manifest_sha256, rtl.corpus.capsules_sha256) != self._component_bindings:
                raise StageGateError("component RTL evaluator uses another corpus")
            if rtl.target_experiment is not self.target_experiment:
                raise StageGateError("component RTL evaluator uses another target selection")
            from . import gsim_gate, paired_measurement

            if set(rtl.decisions) != identities:
                raise StageGateError("component RTL lacks exact workload decisions")
            for member in self.component_corpus.capsules:
                expected = gsim_gate.plan_evaluation(
                    certificate,
                    paired_measurement.gsim_workload(member),
                    phase="development_correctness",
                    gsim_available=True,
                )
                if (
                    not expected.admitted
                    or not expected.eligible
                    or not expected.use_gsim
                    or expected.selected_engine != "gsim"
                    or expected.to_dict() != rtl.decisions[(member.family, member.capsule)].to_dict()
                ):
                    raise StageGateError("component RTL workload is outside the exact certified envelope")
        return identities

    @property
    def unavailable(self):
        return unavailable_component_actions(
            analytical=self.component_analytical is not None,
            rtl=self.component_rtl is not None,
            structural=self.services.command_buffer_analysis is not None,
            cca=self.component_cca is not None,
        )

    def build_registry(self):
        actions = _build_action_registry(
            self.candidate, self.target_experiment, component_only=True, unavailable=self.unavailable
        )
        if self.evaluation_pool is not None:
            actions = (*actions, *evaluation_pool_actions(search=self.search_coordinator is not None))
        return (*actions, *search_actions()) if self.search_coordinator is not None else actions

    def enable_evaluation_pool(self, *, revisions, resources, job_resources, root, beam=3, max_pending=6, max_jobs=256):
        """Explicit host opt-in, before constructing the broker's sealed action registry."""
        from .evaluation_pool import ComponentEvaluationPool

        if self.evaluation_pool is not None or revisions.authority.initial_source != self.candidate:
            raise StageGateError("component evaluation pool requires the exact original candidate authority once")
        self.evaluation_pool = ComponentEvaluationPool(
            policy=self,
            revisions=revisions,
            resources=resources,
            job_resources=job_resources,
            root=root,
            beam=beam,
            max_pending=max_pending,
            max_jobs=max_jobs,
        )
        return self.evaluation_pool

    def validate_actions(self, actions):
        expected = {action.name: action for action in self.build_registry()}
        if set(actions) != set(expected) or any(
            actions[name].advertised() != expected[name].advertised() for name in expected
        ):
            raise StageGateError("component capability profile action contract differs")

    def admission(self, name):
        if name not in {action.name for action in self.build_registry()}:
            raise StageGateError("action is outside the component capability profile")
        return ActionAdmission(
            2 if name == RTL_ACTION else None, False, "component RTL action is limited to two invocations per round", ""
        )

    def execute(self, request, action_name, rendered, call_index, timeout_s, started):
        if self.search_coordinator is not None and action_name in SEARCH_ACTIONS:
            coordinator = self.search_coordinator
            self._validate()
            if action_name == SEARCH_STATUS_ACTION:
                answer = coordinator.status()
            elif action_name == SEARCH_FINALIZE_ACTION:
                answer = coordinator.finalize(timeout_s=timeout_s)
            else:
                answer = coordinator.select()
            if time.monotonic() - started > timeout_s:
                raise StageGateError("search action exceeded its original tool budget")
            return (
                {
                    "returncode": 0,
                    "stdout": canonical_json(answer).decode(),
                    "stderr": "",
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                answer if action_name == SEARCH_SELECT_ACTION else None,
            )
        if self.evaluation_pool is not None and action_name in POOL_ACTIONS:
            pool = self.evaluation_pool
            if action_name == REVISION_FREEZE_ACTION:
                self._validate()
                parent = rendered["parent_revision"]
                options = (
                    {key: rendered[key] for key in ("hypothesis", "edit_family")}
                    if self.search_coordinator is not None
                    else {}
                )
                answer = pool.revisions.freeze(
                    self.candidate, parent_revision=None if parent == "none" else parent, options=options
                ).public_record()
            elif action_name == EVALUATION_SUBMIT_ACTION:
                if self.search_coordinator is not None:
                    answer = self.search_coordinator.submit(
                        rendered["revision_id"], kind=rendered["kind"], timeout_s=timeout_s
                    )
                else:
                    answer = pool.submit(rendered["revision_id"], kind=rendered["kind"], timeout_s=timeout_s)
            elif action_name == EVALUATION_STATUS_ACTION:
                answer = pool.status(rendered["job_id"])
            else:
                answer = pool.cancel(rendered["job_id"])
            return (
                {
                    "returncode": 0,
                    "stdout": canonical_json(answer).decode(),
                    "stderr": "",
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                None,
            )
        self._validate()
        if action_name == INVENTORY_ACTION:
            return self._inventory(request, action_name, rendered, call_index, timeout_s, started)
        if action_name == ANALYSIS_ACTION:
            return self._command_buffers(request, action_name, rendered, call_index, timeout_s, started)
        if action_name not in (ANALYTICAL_ACTION, RTL_ACTION, CCA_ACTION):
            if action_name in {action.name for action in self.build_registry()}:
                return None  # Only the sealed candidate command contract reaches the sandbox.
            raise StageGateError("action is outside the component capability profile")
        try:
            screening = None
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise StageGateError("component evaluation exhausted its wall budget before invocation")
            candidate_sha = str(hash_tree(self.candidate)["sha256"])
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise StageGateError("component evaluation exhausted its wall budget while sealing inputs")
            if action_name == CCA_ACTION:
                provider = self.component_cca
                if provider is None:
                    raise StageGateError("component CCA provider is unavailable")
                from .supervised_feedback import bounded_feedback

                evidence = bounded_feedback(
                    CC.evaluate,
                    timeout_s=remaining,
                    output=self.receipt_path.parent / "component_workers",
                    kwargs=dict(
                        provider=provider,
                        candidate=self.candidate,
                        candidate_sha256=candidate_sha,
                        corpus=self.component_corpus,
                        target_descriptor=self.target_experiment.path,
                        timeout_s=remaining,
                    ),
                )
                tier, provider_sha, config_sha = (
                    CC.TIER,
                    provider.implementation_sha256,
                    provider.configuration_sha256(self._target_binding),
                )
            elif action_name == ANALYTICAL_ACTION:
                provider = self.component_analytical
                if provider is None:
                    raise StageGateError("component analytical provider is unavailable")
                from .supervised_feedback import bounded_feedback

                raw = bounded_feedback(
                    provider.evaluate,
                    timeout_s=remaining,
                    output=self.receipt_path.parent / "component_workers",
                    kwargs=dict(candidate=self.candidate, corpus=self.component_corpus, timeout_s=remaining),
                    lease_path=None if provider.binding is None else provider.binding.lease_path,
                )
                if set(raw) != self._validate():
                    raise StageGateError("analytical feedback does not cover the exact generated components")
                evidence = [
                    {
                        "family": identity[0],
                        "capsule": identity[1],
                        "baseline": _interval(pair[0]),
                        "candidate": _interval(pair[1]),
                        **({"complete_cost": raw.reports[identity]} if hasattr(raw, "reports") else {}),
                    }
                    for identity, pair in sorted(raw.items())
                ]
                if getattr(raw, "screening", None) is not None:
                    from .component_screening import project_screening

                    screening = project_screening(raw.screening, evidence)
                tier, provider_sha, config_sha = (
                    "calibrated_component_analytical",
                    provider.implementation_sha256,
                    provider.configuration_sha256,
                )
            else:
                rtl = self.component_rtl
                if rtl is None:
                    raise StageGateError("component RTL provider is unavailable")
                rtl_budget = int(remaining)
                if rtl_budget <= 0:
                    raise StageGateError("component RTL has no complete execution second remaining")
                evidence = CF.validate_redacted_feedback(
                    rtl.evaluate(
                        self.candidate, round_index=self.feedback_round, call_index=call_index, timeout_s=rtl_budget
                    )
                )
                if (
                    evidence["candidate_sha256"] != candidate_sha
                    or evidence["tuning_corpus_sha256"] != self.component_corpus.capsules_sha256
                    or evidence["certificate_sha256"] != rtl.certificate.sha256
                    or {(row["family"], row["capsule"]) for row in evidence["cells"]} != self._validate()
                ):
                    raise StageGateError("component RTL evidence identity differs")
                tier = "certified_component_rtl"
                provider_sha = self._rtl_configuration[0][1]
                config_sha = rtl.certificate.sha256
            if action_name == CCA_ACTION:
                self._validate()
            if time.monotonic() - started > timeout_s or str(hash_tree(self.candidate)["sha256"]) != candidate_sha:
                raise StageGateError("component evaluation timed out or changed compiler bytes")
            document = validate_component_feedback(
                {
                    "schema": SCHEMA,
                    "workflow_id": COMPONENT_ONLY_V1,
                    "tier": tier,
                    "candidate_sha256": candidate_sha,
                    "corpus_sha256": self.component_corpus.capsules_sha256,
                    "manifest_sha256": self.component_corpus.manifest_sha256,
                    "provider_sha256": provider_sha,
                    "configuration_sha256": config_sha,
                    "evidence": evidence,
                    **({"screening": screening} if screening is not None else {}),
                    "promotion": "NO_FINAL_ACCEPTANCE",
                }
            )
            return (
                {
                    "returncode": 0,
                    "stdout": canonical_json(document).decode(),
                    "stderr": "",
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                document,
            )
        except Exception as exc:
            try:
                from .supervised_feedback import record_feedback_refusal

                record_feedback_refusal(
                    output=self.receipt_path.parent / "component_workers",
                    action=action_name,
                    call_index=call_index,
                    candidate=self.candidate,
                    expected_candidate_sha256=locals().get("candidate_sha"),
                    error=exc,
                )
            except Exception:  # noqa: BLE001 - private evidence failure cannot replace the primary refusal
                pass
            _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
            return (
                {
                    "returncode": 125,
                    "stdout": "",
                    "stderr": f"component feedback refused ({type(exc).__name__})",
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                None,
            )

    def _inventory_document(self, request, rendered, call_index, timeout_s):
        from merlin.perf.agent_guidance import inspect_compiler_package

        return inspect_compiler_package(self.candidate).to_dict()

    def _command_buffers(self, request, action_name, rendered, call_index, timeout_s, started):
        """Use the explicit component target, without a legacy feedback evaluator."""
        try:
            document = self.services.command_buffer_analysis(
                Path(rendered["baseline_json"]),
                Path(rendered["candidate_json"]),
                candidate_root=Path(self.candidate),
                peak_macs_per_cycle=None,
                achievable_macs_per_cycle=None,
                target=self.target_experiment.target,
            )
            self._validate()
            if time.monotonic() - started > timeout_s:
                raise StageGateError("component structural analysis exceeded its wall budget")
            return (
                {
                    "returncode": 0,
                    "stdout": canonical_json(document).decode(),
                    "stderr": "",
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                None,
            )
        except Exception as exc:
            return (
                {
                    "returncode": 125,
                    "stdout": "",
                    "stderr": f"component structural analysis refused ({type(exc).__name__})",
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                None,
            )

    @staticmethod
    def verify_receipts(path, actions, audit, *, candidate_sha256=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise StageGateError("component broker receipt stream is absent or linked")
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        except (ValueError, UnicodeError) as exc:
            raise StageGateError("component broker receipt stream is malformed") from exc
        expected = {action.name for action in actions}
        observed = []
        feedback = []
        for index, row in enumerate(rows):
            if (
                not isinstance(row, Mapping)
                or type(row.get("index")) is not int
                or row["index"] != index
                or row.get("receipt_schema_version") != 1
                or type(row.get("returncode")) is not int
                or row.get("action") not in expected
                or row.get("state") not in ("complete", "rejected")
                or row.get("state") == "rejected"
                and row["returncode"] == 0
            ):
                raise StageGateError("component broker receipt is invalid")
            if any(
                not _is_sha256(row.get(key))
                for key in ("bindings_command_sha256", "argv_sha256", "stdout_sha256", "stderr_sha256")
            ):
                raise StageGateError("component broker receipt identity is invalid")
            observed.append({"action": row["action"], "bindings_sha256": row["bindings_command_sha256"]})
            if (
                row["action"] in (ANALYTICAL_ACTION, RTL_ACTION, CCA_ACTION, SEARCH_SELECT_ACTION)
                and row.get("returncode") == 0
            ):
                evidence = _pin(row.get("feedback_receipt_path", ""), row.get("feedback_receipt_sha256"))
                if not evidence.is_relative_to(Path(path).resolve().parent):
                    raise StageGateError("component feedback receipt escaped its private owner")
                envelope = json.loads(evidence.read_bytes())
                if hashlib.sha256(canonical_json(envelope)).hexdigest() != row["feedback_receipt_sha256"]:
                    raise StageGateError("component feedback receipt is not canonical")
                if row["action"] == SEARCH_SELECT_ACTION:
                    if (
                        not isinstance(envelope, Mapping)
                        or set(envelope) != {"schema", "selected", "feedback", "promotion"}
                        or envelope["schema"] != "merlin.component_search_selection.v1"
                        or envelope["promotion"] != "NO_FINAL_ACCEPTANCE"
                        or not isinstance(envelope["selected"], Mapping)
                    ):
                        raise StageGateError("component search selection receipt is invalid")
                    document = validate_component_feedback(envelope["feedback"])
                    if envelope["selected"].get("source_sha256") != document["candidate_sha256"]:
                        raise StageGateError("component search selected another measured source")
                else:
                    document = validate_component_feedback(envelope)
                if row["stdout_sha256"] != row["feedback_receipt_sha256"]:
                    raise StageGateError("component feedback differs from the observed stdout")
                tier = {
                    ANALYTICAL_ACTION: "calibrated_component_analytical",
                    RTL_ACTION: "certified_component_rtl",
                    CCA_ACTION: CC.TIER,
                    SEARCH_SELECT_ACTION: "certified_component_rtl",
                }[row["action"]]
                if document["tier"] != tier:
                    raise StageGateError("component feedback tier differs from its action")
                feedback.append(document)
        try:
            binding = workflow_binding(rows, COMPONENT_ONLY_V1)
        except ValueError as exc:
            raise StageGateError(str(exc)) from exc
        if binding != {"status": "bound", "id": COMPONENT_ONLY_V1}:
            raise StageGateError("new component receipts must bind their explicit profile")
        if audit.get("broker_invocations") != observed:
            raise StageGateError("component transcript does not match broker receipts")
        successes = {row["action"] for row in rows if row.get("returncode") == 0}
        if {action.name for action in actions if action.required} - successes:
            raise StageGateError("required component command receipts are missing")
        if candidate_sha256 is not None and (
            not _is_sha256(candidate_sha256)
            or not any(document["candidate_sha256"] == candidate_sha256 for document in feedback)
        ):
            raise StageGateError("final component feedback lacks the sealed compiler identity")
        return {
            "workflow_policy": binding,
            "all_required_succeeded": True,
            "feedback_successes": len(feedback),
            "final_acceptance": "NOT_ESTABLISHED",
        }
