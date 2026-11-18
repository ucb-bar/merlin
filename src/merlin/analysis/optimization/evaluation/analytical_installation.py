"""Production host adapter for the fast Phase-2 portfolio gate.

The adapter is deliberately data driven.  Target plugins bind arbitrary emitted-feature JSON
pointers to resource service models; this module contains no target, model, opcode, or shape table.
It joins those models to the already host-verified whole-program analysis, derives source/task
coverage, composes resource occupancy with an explicitly measured operator, and returns the mapping
consumed by :mod:`merlin.analysis.optimization.evaluation.portfolio_quality`.

An absent input remains absent.  In particular, command-buffer bytes are not physical traffic,
static instruction counts are not cycles, and a placement declaration is not executed accelerator
work.  Physical movement requires a counter-calibrated feature coefficient, cycles require a
resource model, and accelerator placement requires source-bound task instruction roles.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.analysis.optimization.evaluation.fast_portfolio import (
    AnalyticalMetrics,
    FastEvaluationPolicy,
    unavailable_fast_evaluation,
)
from merlin.analysis.optimization.evaluation.global_planner import OccupancySummary
from merlin.analysis.optimization.evaluation.lane_cost import dtype_bits
from merlin.analysis.optimization.evaluation.portfolio_quality import (
    FourModelQualitySchema,
    QualityObservation,
    standard_four_model_quality_schema,
)
from merlin.analysis.perf.bounds.decomposition import ResourceKind, compose_resource_times
from merlin.analysis.perf.bounds.headroom import Composition
from merlin.common import digest as _mdigest
from merlin.common import digest as _mjson
from merlin.compiler.partition.global_plan import CycleInterval

# ---- phase2_analytical_provider ----------------------------------------------------------------


def _metrics(
    analysis: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    arm: str,
    calibration: _Calibration,
    provider_sha256: str,
    portfolio_sha256: str,
) -> AnalyticalMetrics:
    from merlin.analysis.optimization.evaluation.analytical_calibration import (
        _arm_artifacts,
        _Calibration,
        _compose,
        _context,
        _coverage,
        _digest,
        _feature_cycles,
        _json_pointer,
        _mapping,
        _number,
    )

    command, plan, task_evidence = _arm_artifacts(analysis, artifacts, arm)
    lowered_sha = str(plan["candidate_lowered_sha256"])
    command_sha = str(plan["candidate_command_buffer_sha256"])
    plan_sha = str(plan["plan_digest"])
    target_activity = _mapping(_mapping(_mapping(analysis.get("diagnostics")).get("target_artifact_activity")).get(arm))
    provenance = (
        f"lowered sha256:{lowered_sha}",
        f"command-buffer sha256:{command_sha}",
        f"declared-plan sha256:{plan_sha}",
        f"verified-plan-evidence sha256:{_digest(plan)}",
        f"task-instruction-evidence sha256:{_digest(task_evidence)}",
        f"target-activity-evidence sha256:{_digest(target_activity)}",
        f"calibration sha256:{calibration.document_sha256}",
        f"provider sha256:{provider_sha256}",
        f"target sha256:{calibration.target_sha256}",
        f"portfolio sha256:{portfolio_sha256}",
    )
    context = _context(analysis, command, plan, task_evidence, arm)
    resource_lo: dict[str, tuple[ResourceKind, float]] = {}
    resource_hi: dict[str, tuple[ResourceKind, float]] = {}
    physical_bytes = 0.0
    physical_complete = True
    movement_commands = 0.0
    encoding_count = 0.0
    encoding_bytes = 0.0
    encoding_complete = True
    encoding_lo = encoding_hi = 0.0
    floor_resources: dict[str, tuple[ResourceKind, float]] = {}
    floor_complete = True
    effects_by_resource: dict[str, set[str]] = {}

    def unknown(reason: str) -> AnalyticalMetrics:
        return AnalyticalMetrics.from_mapping(
            {
                "cycles": CycleInterval.unknown(reason).to_dict(),
                "movement_bytes": None,
                "movement_scope": "unavailable",
                "occupancy": None,
                "coverage": _coverage(command, plan, task_evidence, calibration.accelerator_compute_roles, provenance),
                "roofline": None,
                "encoding_conversions": {
                    "count": None,
                    "bytes": None,
                    "cycles": CycleInterval.unknown(reason).to_dict(),
                },
                "risk_score": calibration.risk_score,
                "provenance": list(provenance),
            }
        )

    for feature in calibration.features:
        try:
            count = float(_number(_json_pointer(context, feature.pointer), f"feature {feature.id} count"))
        except (KeyError, TypeError, ValueError):
            count = math.nan
        if not math.isfinite(count):
            return unknown(f"feature {feature.id} is UNKNOWN")
        cycles = _feature_cycles(feature, count, calibration.movement)
        if cycles is None or not cycles.resolved:
            return unknown(f"feature {feature.id} cycles are UNKNOWN")
        resource_kind = (
            ResourceKind.COMPUTE
            if feature.kind == "compute"
            else ResourceKind.FIXED
            if feature.kind == "fixed"
            else ResourceKind.MOVEMENT
        )
        for destination, endpoint in ((resource_lo, float(cycles.lo)), (resource_hi, float(cycles.hi))):
            previous = destination.get(feature.resource, (resource_kind, 0.0))
            if previous[0] is not resource_kind:
                raise ValueError("one resource cannot have multiple analytical kinds")
            destination[feature.resource] = (resource_kind, previous[1] + endpoint)
        effects_by_resource.setdefault(feature.resource, set()).update(feature.effects)
        if feature.kind in {"movement", "encoding"}:
            if feature.physical_bytes_per_unit is None:
                if count:
                    physical_complete = False
            else:
                physical_bytes += feature.physical_bytes_per_unit * count
            if feature.commands_per_unit is not None:
                movement_commands += feature.commands_per_unit * count
        if feature.kind == "encoding":
            if feature.transitions_per_unit is None or feature.physical_bytes_per_unit is None:
                if count:
                    encoding_complete = False
            else:
                encoding_count += feature.transitions_per_unit * count
                encoding_bytes += feature.physical_bytes_per_unit * count
            encoding_lo += float(cycles.lo)
            encoding_hi += float(cycles.hi)
        if feature.floor_cycles_per_unit is None:
            if count:
                floor_complete = False
        else:
            previous = floor_resources.get(feature.resource, (resource_kind, 0.0))
            floor_resources[feature.resource] = (resource_kind, previous[1] + feature.floor_cycles_per_unit * count)

    lo, hi = _compose(resource_lo, calibration), _compose(resource_hi, calibration)
    if hi < lo or lo <= 0:
        raise ValueError("analytical resource model produced an invalid cycle interval")
    engine_sum = sum(value for kind, value in resource_hi.values() if kind.is_engine)
    fixed = sum(value for kind, value in resource_hi.values() if not kind.is_engine)
    compute_resources = tuple(sorted(name for name, (kind, _) in resource_hi.items() if kind is ResourceKind.COMPUTE))
    movement_resources = tuple(sorted(name for name, (kind, _) in resource_hi.items() if kind is ResourceKind.MOVEMENT))
    # A composition operator over more than one engine on either side establishes aggregate saved
    # service time, but cannot say which part was compute/movement overlap.  Only the two-resource
    # case licenses the latency-hiding metric.  Fixed service is serial and cancels algebraically.
    two_resource_overlap = len(compute_resources) == len(movement_resources) == 1
    overlap = max(0.0, engine_sum + fixed - hi) if two_resource_overlap else None
    available = (
        min(resource_hi[compute_resources[0]][1], resource_hi[movement_resources[0]][1])
        if two_resource_overlap
        else None
    )
    movement_elapsed = resource_hi[movement_resources[0]][1] if len(movement_resources) == 1 else None
    occupancy = OccupancySummary(
        total_cycles=hi,
        busy_cycles=tuple(sorted((name, value) for name, (_, value) in resource_hi.items())),
        compute_resources=compute_resources,
        movement_resources=movement_resources,
        movement_elapsed_cycles=movement_elapsed,
        overlap_cycles=overlap,
        overlap_available_cycles=available,
        idle_cycles=None,
        critical_path_cycles=hi,
        movement_bytes=physical_bytes if physical_complete else None,
        movement_commands=(int(movement_commands) if movement_commands.is_integer() else None),
        encoding_transitions=(int(encoding_count) if encoding_complete and encoding_count.is_integer() else None),
        provenance=provenance,
        missing=tuple(
            filter(
                None,
                (
                    "a resource timeline for idle cycles",
                    ("a compute/movement partition of aggregate overlap" if not two_resource_overlap else ""),
                    ("a movement elapsed union" if len(movement_resources) > 1 else ""),
                ),
            )
        ),
    )

    roofline = None
    if floor_complete and floor_resources:
        lower_bound = _compose(floor_resources, calibration)
        if lower_bound <= lo:
            maximum = max(value for _, value in floor_resources.values())
            limiters = tuple(
                sorted(
                    name
                    for name, (_, value) in floor_resources.items()
                    if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-12)
                )
            )
            effects = tuple(sorted({effect for name in limiters for effect in effects_by_resource.get(name, ())}))
            if effects:
                roofline = {
                    "lower_bound_cycles": lower_bound,
                    "resource_floors": {name: value for name, (_, value) in floor_resources.items()},
                    "limiting_resources": list(limiters),
                    "optimization_effects": list(effects),
                    "composition": (
                        f"{calibration.composition.value} with measured eta={calibration.composition_eta:g}"
                    ),
                    "provenance": [*provenance, "composition sha256:" + calibration.composition_provenance_sha256],
                }
    coverage = _coverage(command, plan, task_evidence, calibration.accelerator_compute_roles, provenance)
    encoding_cycles = (
        CycleInterval(encoding_lo, encoding_hi, provenance=provenance)
        if encoding_complete
        else CycleInterval.unknown("encoding conversion cycle evidence is incomplete")
    )
    return AnalyticalMetrics.from_mapping(
        {
            "cycles": CycleInterval(lo, hi, provenance=provenance).to_dict(),
            "movement_bytes": physical_bytes if physical_complete else None,
            "movement_scope": "physical" if physical_complete else "unavailable",
            "occupancy": occupancy.to_dict(),
            "coverage": coverage,
            "roofline": roofline,
            "encoding_conversions": {
                "count": int(encoding_count) if encoding_complete and encoding_count.is_integer() else None,
                "bytes": encoding_bytes if encoding_complete else None,
                "cycles": encoding_cycles.to_dict(),
            },
            "risk_score": calibration.risk_score,
            "provenance": list(provenance),
        }
    )


def _quality_observations(
    raw: Mapping[str, Any], *, model_sha256: str, corpus_sha256: str, arm_sha256s: Mapping[str, str]
) -> dict[str, QualityObservation]:
    from merlin.analysis.optimization.evaluation.analytical_calibration import _finite, _mapping, _sha256

    result: dict[str, QualityObservation] = {}
    for arm in ("baseline", "candidate"):
        row = _mapping(raw.get(arm))
        evidence_sha = row.get("evidence_sha256")
        if (
            row.get("model_sha256") != model_sha256
            or row.get("corpus_sha256") != corpus_sha256
            or row.get("artifact_sha256") != arm_sha256s[arm]
            or not _sha256(evidence_sha)
        ):
            raise ValueError(f"{arm} quality observation is not bound to model/corpus/artifact bytes")
        values = _mapping(row.get("values"))
        result[arm] = QualityObservation(
            tuple(sorted((str(name), _finite(value, f"quality {name}")) for name, value in values.items())),
            (
                f"quality evidence sha256:{evidence_sha}",
                f"corpus sha256:{corpus_sha256}",
                f"artifact sha256:{arm_sha256s[arm]}",
            ),
            row.get("complete") is True,
        )
    return result


@dataclass(frozen=True)
class FastEvaluatorInstallation:
    """Frozen provider installation or a fail-closed exact-only fallback."""

    quality_schema: FourModelQualitySchema
    provider: Callable[..., Mapping[str, Any]] | None
    policy: FastEvaluationPolicy | None
    provider_binding: Mapping[str, Any] | None
    fallback: Mapping[str, Any] | None

    def experiment_kwargs(self) -> dict[str, Any]:
        """Keyword arguments accepted directly by ``GlobalPerfExperiment``."""
        if self.provider is None:
            return {}
        return {
            "fast_evaluation_provider": self.provider,
            "fast_evaluation_policy": self.policy,
            "quality_budgets": self.quality_schema.budget_map,
            "fast_evaluation_provider_binding": copy.deepcopy(dict(self.provider_binding or {})),
        }


def build_fast_evaluator_installation(
    ordered_member_sha256s: Sequence[str],
    *,
    classification_member_sha256: str,
    corpus_sha256_by_member: Mapping[str, str] | None,
    calibration: Mapping[str, Any] | Path | None,
    calibration_sha256: str | None = None,
    quality_observer: Callable[..., Mapping[str, Any]] | None = None,
    quality_observer_sha256: str | None = None,
    policy: FastEvaluationPolicy | None = None,
    maximum_model_seconds: float = 60.0,
) -> FastEvaluatorInstallation:
    """Build an immutable adapter/binding for the global experiment.

    Without every held-out corpus, a quality observer, and a calibration artifact, this returns an
    exact-only installation with no provider kwargs.  The global experiment can therefore continue
    exact transformations while approximate promotion remains impossible.
    """
    from merlin.analysis.optimization.evaluation.analytical_calibration import (
        PROVIDER_BINDING_SCHEMA,
        _digest,
        _implementation_sha256,
        _load_document,
        _mapping,
        _parse_calibration,
        _sha256,
    )

    quality = standard_four_model_quality_schema(
        ordered_member_sha256s,
        classification_member_sha256=classification_member_sha256,
        corpus_sha256_by_member=corpus_sha256_by_member,
    )
    absent = []
    if quality.mode != "accuracy_bounded":
        absent.append("complete held-out quality corpora")
    if quality_observer is None or not _sha256(quality_observer_sha256):
        absent.append("content-addressed host quality observer")
    if calibration is None:
        absent.append("content-addressed analytical calibration")
    if absent:
        reason = "missing " + ", ".join(absent) + "; approximate transformations are disabled"
        exact = (
            quality
            if quality.mode == "exact_only"
            else type(quality)("exact_only", quality.ordered_member_sha256s, (), reason)
        )
        return FastEvaluatorInstallation(exact, None, None, None, unavailable_fast_evaluation(reason=reason))
    if (
        isinstance(maximum_model_seconds, bool)
        or not isinstance(maximum_model_seconds, (int, float))
        or not 0 < float(maximum_model_seconds) <= 60
    ):
        raise ValueError("host analytical evaluation must have a per-model budget in (0, 60]")
    document, document_sha, source_kind = _load_document(calibration, calibration_sha256)
    parsed = _parse_calibration(document, document_sha, source_kind)
    implementation_sha = _implementation_sha256()
    frozen_document = copy.deepcopy(document)
    evidence_sha256s = tuple(dict.fromkeys((document_sha, *parsed.evidence_sha256s, str(quality_observer_sha256))))
    binding = {
        "schema": PROVIDER_BINDING_SCHEMA,
        "implementation_sha256": implementation_sha,
        "execution": "host_analytical_only",
        "full_model_simulation_allowed": False,
        "resource_admission": "serialized_one_model_at_a_time",
        "maximum_model_seconds": float(maximum_model_seconds),
        "calibration_sha256s": list(evidence_sha256s),
        "calibration_document_sha256": document_sha,
        "calibration_document_source": source_kind,
        "target_sha256": parsed.target_sha256,
        "quality_observer_sha256": quality_observer_sha256,
        "ordered_member_sha256s": list(quality.ordered_member_sha256s),
        "corpus_sha256_by_member": dict(corpus_sha256_by_member or {}),
        "provider_contract": "no target execution, layer simulation, or complete-model simulation",
    }
    frozen_binding = copy.deepcopy(binding)

    def provider(**kwargs: Any) -> Mapping[str, Any]:
        supplied_binding = kwargs.get("provider_binding")
        if (
            supplied_binding != frozen_binding
            or _digest(frozen_document) != _digest(document)
            or _implementation_sha256() != implementation_sha
        ):
            raise ValueError("host analytical provider binding or calibration changed")
        sentinel = kwargs.get("sentinel")
        model_sha = getattr(sentinel, "capsule_sha256", None)
        if model_sha not in quality.ordered_member_sha256s:
            raise ValueError("host analytical provider received an unbound portfolio member")
        portfolio_sha = kwargs.get("portfolio_sha256")
        if not _sha256(portfolio_sha) or kwargs.get("target_sha256") != parsed.target_sha256:
            raise ValueError("host analytical provider requires target and portfolio identities")
        descriptor = kwargs.get("target_descriptor")
        if descriptor is not None:
            descriptor_path = Path(descriptor)
            if (
                descriptor_path.is_symlink()
                or not descriptor_path.is_file()
                or hashlib.sha256(descriptor_path.read_bytes()).hexdigest() != parsed.target_sha256
            ):
                raise ValueError("host analytical provider target descriptor changed")
        analysis = _mapping(kwargs.get("analysis"))
        artifacts = _mapping(kwargs.get("artifacts"))
        if _mapping(analysis.get("workload")).get("capsule_sha256") != model_sha:
            raise ValueError("whole-model analysis does not match the requested portfolio member")
        baseline = _metrics(analysis, artifacts, "baseline", parsed, implementation_sha, portfolio_sha)
        candidate = _metrics(analysis, artifacts, "candidate", parsed, implementation_sha, portfolio_sha)
        diagnostics = _mapping(analysis.get("diagnostics"))
        baseline_sha = str(
            _mapping(diagnostics.get("verified_baseline_global_plan_emission")).get("candidate_lowered_sha256") or ""
        )
        candidate_sha = str(
            _mapping(diagnostics.get("verified_global_plan_emission")).get("candidate_lowered_sha256") or ""
        )
        corpus_sha = str((corpus_sha256_by_member or {})[model_sha])
        quality_raw = quality_observer(
            analysis=analysis,
            artifacts=artifacts,
            sentinel=sentinel,
            corpus_sha256=corpus_sha,
            target_descriptor=kwargs.get("target_descriptor"),
            target_sha256=kwargs.get("target_sha256"),
            execution="host_reference_only_no_target_or_model_simulator",
        )
        if not isinstance(quality_raw, Mapping):
            raise TypeError("host quality observer must return a mapping")
        observations = _quality_observations(
            quality_raw,
            model_sha256=model_sha,
            corpus_sha256=corpus_sha,
            arm_sha256s={"baseline": baseline_sha, "candidate": candidate_sha},
        )
        return {
            "baseline": baseline,
            "candidate": candidate,
            "baseline_quality": observations["baseline"],
            "candidate_quality": observations["candidate"],
            "provider_provenance": {
                "binding_sha256": _digest(frozen_binding),
                "calibration_sha256": parsed.document_sha256,
                "target_sha256": parsed.target_sha256,
                "portfolio_sha256": portfolio_sha,
                "baseline_lowered_sha256": baseline_sha,
                "candidate_lowered_sha256": candidate_sha,
            },
        }

    return FastEvaluatorInstallation(quality, provider, policy or FastEvaluationPolicy(), binding, None)


__all__ = [
    "CALIBRATION_SCHEMA",
    "FastEvaluatorInstallation",
    "PROVIDER_BINDING_SCHEMA",
    "build_fast_evaluator_installation",
]
