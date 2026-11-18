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
            from merlin.development.backend.optimization.components.analytical_provider import (
                ComponentAnalyticalBinding,
            )

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
                from merlin.analysis.optimization.evaluation.component_cost import (
                    validate_complete_cost_report,
                )

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
            from merlin.development.backend.optimization.components.baseline import (
                validate_screening_projection,
            )

            validate_screening_projection(document["screening"], evidence)
    else:
        raise StageGateError("component tier is unsupported; Spike is not RTL timing")
    return dict(document)
