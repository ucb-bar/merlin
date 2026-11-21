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

from merlin.analysis.optimization.evaluation.calibration_bundle import prepare_phase2_calibration
from merlin.common.digest import hash_tree
from merlin.compiler.partition.global_plan import CycleInterval
from merlin.development.backend.optimization.author_interface.broker_policy import (
    ANALYSIS_ACTION,
    COMPONENT_ONLY_V1,
    INVENTORY_ACTION,
    ActionAdmission,
    BrokerServices,
    WorkflowPolicy,
    _build_action_registry,
    _record_host_refusal,
)
from merlin.development.backend.optimization.author_interface.broker_service import BrokerAction
from merlin.development.backend.optimization.components import cca_reports as CC
from merlin.development.backend.optimization.components.feedback_actions import (
    _GLOBAL_INPUTS,
    ANALYTICAL_ACTION,
    CCA_ACTION,
    EVALUATION_STATUS_ACTION,
    EVALUATION_SUBMIT_ACTION,
    POOL_ACTIONS,
    REVISION_FREEZE_ACTION,
    RTL_ACTION,
    SCHEMA,
    SEARCH_ACTIONS,
    SEARCH_FINALIZE_ACTION,
    SEARCH_SELECT_ACTION,
    SEARCH_STATUS_ACTION,
    ComponentAnalyticalProvider,
    _callable_code,
    _callable_source,
    _corpus,
    _interval,
    _pin,
    evaluation_pool_actions,
    search_actions,
    unavailable_component_actions,
    validate_component_feedback,
)
from merlin.development.backend.optimization.contracts import (
    StageGateError,
    _is_sha256,
    canonical_json,
    document_sha256,
    mapping_file,
    sha256_file,
    workflow_binding,
)
from merlin.development.backend.optimization.search import corpus as C
from merlin.development.backend.optimization.search import corpus_feedback as CF

# ---- component_workflow ------------------------------------------------------------------------


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
            from merlin.development.backend.optimization.measurement.development_feedback import (
                DevelopmentGsimFeedback,
            )

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
        from merlin.development.backend.optimization.components.baseline import verify_policy_baselines

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
            from merlin.development.backend.optimization.measurement.development_feedback import (
                DevelopmentGsimFeedback,
            )

            if type(self.component_rtl) is not DevelopmentGsimFeedback:
                raise StageGateError("component RTL must use the existing certified GSIM evaluator")
            rtl = self.component_rtl
            from merlin.development.backend.optimization.measurement.gsim_gate import load_certificate

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
            from merlin.development.backend.optimization.measurement import (
                gsim_gate,
            )
            from merlin.development.backend.optimization.measurement import (
                paired_plan as paired_measurement,
            )

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
        from merlin.development.backend.optimization.search.pool import ComponentEvaluationPool

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
                from merlin.development.backend.optimization.author_interface.broker_transports import (
                    bounded_feedback,
                )

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
                from merlin.development.backend.optimization.author_interface.broker_transports import (
                    bounded_feedback,
                )

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
                    from merlin.development.backend.optimization.components.baseline import project_screening

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
                from merlin.development.backend.optimization.author_interface.broker_transports import (
                    record_feedback_refusal,
                )

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
        from merlin.analysis.optimization.guidance.agent_brief import inspect_compiler_package

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
