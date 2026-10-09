"""Installed independent-component analytical provider construction.

The target callback extracts actual emitted/executed features. It receives only
one generated member at a time and never receives a complete-model sentinel.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import pickle
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.component_cost import ComponentCostScope, ComponentFeatureObservation, complete_component_cost
from merlin.perf.component_screen import (
    ComponentOpportunity,
    ComponentScreenPolicy,
    rank_component_opportunities,
    validate_component_screen_report,
)
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS
from merlin.perf.phase2_calibration_bundle import prepare_phase2_calibration

from . import corpus as C
from .component_baseline import verify_baseline_admission
from .component_feature_arms import IndependentArmFeatures
from .component_runtime import require_independent_runtime
from .component_workflow import ComponentAnalyticalProvider, _callable_code, _callable_source, _corpus, _pin
from .contracts import StageGateError, document_sha256, mapping_file, sha256_file
from .portfolio_launch import acquire_host_resource_lease


class ComponentAnalyticalResults(dict):
    """Legacy interval-pair mapping plus answer-free complete-cost reports."""

    def __init__(self, intervals, reports, screening=None):
        super().__init__(intervals)
        self.reports = reports
        self.screening = screening


def _verify_artifacts(observation):
    if type(observation) is not ComponentFeatureObservation or not observation.artifact_files:
        raise StageGateError("component features need typed exact emitted/executed artifact evidence")
    for path, digest in observation.artifact_files:
        _pin(path, digest)


def _independent_calibration(adapter, runtime, qualification):
    pins = dict(runtime.source_pins)
    for path in (adapter, qualification):
        if path is not None:
            if path not in pins:
                raise StageGateError("component calibration/held inputs are outside independent runtime membership")
            _pin(path, pins[path])
    prepared = prepare_phase2_calibration(adapter)
    for row in prepared.get("evidence_files", []):
        path = Path(row["path"])
        if pins.get(path) != row["sha256"]:
            raise StageGateError("component calibration evidence is outside independently qualified membership")
        _pin(path, row["sha256"])
    return prepared.get("calibration")


def _qualified_domains(path, digest, calibration_sha256):
    if path is None:
        return ()
    _pin(path, digest)
    report = mapping_file(path)
    validate_component_screen_report(report)
    policy = ComponentScreenPolicy()
    coverage = report.get("interval_coverage", {})
    if (
        report.get("schema") != "component_cycle_screen_validation_v1"
        or report.get("promotion") != "SCREENING_ONLY"
        or report.get("exposable") is not True
        or report.get("reasons") != []
        or report.get("calibration_sha256") != calibration_sha256
        or report.get("policy_sha256") != policy.sha256
        or report.get("policy") != policy.to_dict()
        or coverage.get("n", 0) < policy.minimum_predictions
        or coverage.get("rate", 0) < policy.minimum_interval_coverage
    ):
        raise StageGateError("component held validation does not qualify the selected calibration and policy")
    ranking = report.get("ranking", {})
    overall, slices = ranking.get("overall", {}), ranking.get("slices", {})
    qualifying = [row for row in slices.values() if row.get("decided", 0) >= policy.minimum_slice_decided]
    if (
        ranking.get("exposable") is not True
        or overall.get("decided", 0) < policy.minimum_decided
        or overall.get("rate", 0) < policy.minimum_rank_rate
        or len(qualifying) < policy.minimum_slices
        or any(row.get("rate", 0) < policy.minimum_rank_rate for row in qualifying)
        or report.get("absolute_error", {}).get("maximum_relative", math.inf) > policy.maximum_relative_error
    ):
        raise StageGateError("component held ranking/error evidence does not meet screening thresholds")
    return (report["domain_sha256"],)


@dataclass(frozen=True)
class ComponentAnalyticalBinding:
    baseline: Path
    baseline_sha256: str
    corpus: C.FrozenPerformanceCorpus
    target_descriptor: Path
    target_sha256: str
    scope: ComponentCostScope
    calibration_adapter: Path
    feature_provider: Callable[..., Any]
    feature_implementation: Path
    feature_sha256: str
    feature_code: object
    dependencies: tuple[tuple[Path, str], ...]
    feature_context_pins: tuple[tuple[Path, str], ...]
    qualification: Path | None
    qualification_sha256: str | None
    output: Path
    lease_path: Path
    max_workers: int
    memory_per_worker_bytes: int
    engine_slots: int
    objective: str
    baseline_admission: object
    independent_runtime: object
    independent_arms: IndependentArmFeatures | None = None

    @property
    def sha256(self):
        return document_sha256(
            {
                "baseline_sha256": self.baseline_sha256,
                "baseline_admission_sha256": self.baseline_admission.sha256,
                "independent_runtime_sha256": self.independent_runtime.sha256,
                "corpus_sha256": self.corpus.capsules_sha256,
                "manifest_sha256": self.corpus.manifest_sha256,
                "target_sha256": self.target_sha256,
                "scope_sha256": self.scope.sha256,
                "feature_sha256": self.feature_sha256,
                "dependencies": [(str(p), d) for p, d in self.dependencies],
                "qualification_sha256": self.qualification_sha256,
                "max_workers": self.max_workers,
                "memory_per_worker_bytes": self.memory_per_worker_bytes,
                "engine_slots": self.engine_slots,
                "objective": self.objective,
                **(
                    {"independent_arms_sha256": self.independent_arms.verify()}
                    if self.independent_arms is not None
                    else {}
                ),
            }
        )

    def validate(self, calibration):
        verify_baseline_admission(
            self.baseline_admission,
            baseline=self.baseline,
            corpus=self.corpus,
            target_descriptor=self.target_descriptor,
        )
        runtime = require_independent_runtime(
            self.independent_runtime, required_roles=("feature_provider",), target_descriptor=self.target_descriptor
        )
        if self.feature_provider is not runtime.services.feature_provider:
            raise StageGateError("component features differ from independently evaluated observation support")
        if self.independent_arms is not None:
            if type(self.independent_arms) is not IndependentArmFeatures:
                raise StageGateError("per-arm feature selection requires its fixed input/process/cache owner")
            self.independent_arms.require_feedback_owner(runtime, self.feature_provider)
        verify_binding = getattr(runtime.qualification, "verify_feedback_binding", None)
        if not callable(verify_binding):
            raise StageGateError("component features lack independent complete scope/held measurement qualification")
        verify_binding(
            baseline_admission=self.baseline_admission,
            scope=self.scope,
            calibration_adapter=self.calibration_adapter,
            qualification=self.qualification,
            objective=self.objective,
        )
        _corpus(self.corpus)
        _pin(self.target_descriptor, self.target_sha256)
        _pin(self.feature_implementation, self.feature_sha256)
        if (
            _callable_source(self.feature_provider)[0] != self.feature_implementation
            or _callable_code(self.feature_provider) is not self.feature_code
        ):
            raise StageGateError("component feature callback differs from its pinned implementation")
        if self.baseline.is_symlink() or str(hash_tree(self.baseline)["sha256"]) != self.baseline_sha256:
            raise StageGateError("component analytical baseline freeze changed")
        for path, digest in self.dependencies:
            if dict(runtime.source_pins).get(path) != digest:
                raise StageGateError("component model dependency is outside independently qualified runtime membership")
            _pin(path, digest)
        if _feature_context_pins(self.feature_provider) != self.feature_context_pins:
            raise StageGateError("component feature provider's selected execution context changed")
        if calibration.get("target_sha256") != self.target_sha256:
            raise StageGateError("component analytical calibration targets another descriptor")
        if (
            self.qualification is not None
            and dict(runtime.source_pins).get(self.qualification) != self.qualification_sha256
        ):
            raise StageGateError("component held validation is outside independently qualified runtime membership")
        return _qualified_domains(self.qualification, self.qualification_sha256, document_sha256(calibration))


def _feature_context_pins(feature_provider):
    owner = feature_provider
    while isinstance(owner, partial):
        owner = owner.func
    pins = getattr(feature_provider, "component_source_pins", None)
    if pins is None:
        pins = getattr(getattr(owner, "__self__", None), "component_source_pins", {})
    if not isinstance(pins, Mapping):
        raise StageGateError("component prepared feature context requires exact private dependency pins")
    return tuple(sorted(((Path(p).resolve(), digest) for p, digest in pins.items()), key=lambda row: str(row[0])))


def _paired_status(observations, name):
    statuses = tuple(getattr(row, name) for row in observations)
    if any(status in ("FAIL", "REFUSAL") for status in statuses):
        return False
    return True if all(status == "PASS" for status in statuses) else None


def _applicability(binding, observation, workspace):
    from .component_measurement_qualification import IndependentMeasurementQualification

    qualification = binding.independent_runtime.qualification
    if type(qualification) is not IndependentMeasurementQualification:
        raise StageGateError("component cost needs independently qualified semantic applicability")
    coordinates, _lookup = qualification.observe_applicability(observation, workspace)
    return qualification.applicability_domain, coordinates


def _available_workers(binding):
    cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, raw = line.partition(":")
        if name == "MemAvailable":
            available = int(raw.split()[0]) * 1024
            break
    if available is None:
        raise StageGateError("component feedback cannot derive available worker memory")
    workers = min(cpu, binding.max_workers, binding.engine_slots, available // binding.memory_per_worker_bytes)
    if workers < 1:
        raise StageGateError("component feedback has no admitted CPU/memory/engine capacity")
    return workers


def _evaluate(binding, adapter, *, candidate, corpus, timeout_s):
    started = time.monotonic()
    # Forward the remaining development budget and reject late completion.
    # The thread coordinator cannot preempt a callback that ignores its timeout;
    # hard process termination belongs to separately supervised execution.
    deadline = started + min(float(timeout_s), ITERATION_MAX_SECONDS)
    calibration = _independent_calibration(adapter, binding.independent_runtime, binding.qualification)
    if not isinstance(calibration, Mapping):
        raise StageGateError("component controlled calibration is unavailable")
    domains = binding.validate(calibration)
    if (corpus.manifest_sha256, corpus.capsules_sha256) != (
        binding.corpus.manifest_sha256,
        binding.corpus.capsules_sha256,
    ):
        raise StageGateError("component analytical evaluation selected another exact corpus")
    candidate = Path(candidate).resolve()
    if candidate.is_relative_to(binding.baseline) or binding.baseline.is_relative_to(candidate):
        raise StageGateError("component analytical compiler arms overlap")
    binding.output.mkdir(parents=True, exist_ok=True)
    call_root = Path(tempfile.mkdtemp(prefix="component_cost_", dir=binding.output))
    measured = call_root / "candidate"
    shutil.copytree(candidate, measured, symlinks=True)
    candidate_sha = str(hash_tree(measured)["sha256"])
    workers = _available_workers(binding)
    from .feedback_guardian import has_parent_feedback_lease

    delegated_lease = has_parent_feedback_lease(binding.lease_path)
    lease = None if delegated_lease else acquire_host_resource_lease(call_root, lease_path=binding.lease_path)
    if lease is None and not delegated_lease:
        raise StageGateError("component analytical shared engine lease is busy")
    intervals, reports, opportunities = {}, {}, []
    try:

        def one(member):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StageGateError("component feedback exhausted development simulation budget")
            extraction_started = time.monotonic()
            workspace = call_root / (document_sha256([member.family, member.capsule]))
            workspace.mkdir()
            cache_key = document_sha256(
                [binding.sha256, document_sha256(calibration), candidate_sha, member.source_sha256]
            )
            cache = binding.output / "cache" / (cache_key + ".pickle")
            cache.parent.mkdir(exist_ok=True)
            pair = None
            if binding.independent_arms is None and cache.is_file() and not cache.is_symlink():
                try:
                    with cache.open("rb") as stream:
                        pair = pickle.load(stream)
                    for observation in pair:
                        _verify_artifacts(observation)
                except (OSError, ValueError, StageGateError, EOFError):
                    pair = None
            if pair is None:
                extra = {}
                if binding.independent_arms is not None:
                    extra["cache_context"] = {
                        "binding_sha256": binding.sha256,
                        "runtime_sha256": binding.independent_runtime.sha256,
                        "calibration_sha256": document_sha256(calibration),
                        "applicability_sha256": binding.independent_runtime.qualification.applicability_sha256,
                        "qualified_domains": list(domains),
                    }
                pair = binding.feature_provider(
                    baseline=binding.baseline,
                    candidate=measured,
                    member=member,
                    corpus=corpus,
                    target_descriptor=binding.target_descriptor,
                    scope=binding.scope,
                    workspace=workspace,
                    timeout_s=remaining,
                    **extra,
                )
            if type(pair) is not tuple or len(pair) != 2:
                raise StageGateError("component features require exactly baseline and candidate observations")
            if any(type(observation) is not ComponentFeatureObservation for observation in pair):
                raise StageGateError("component features require typed exact execution observations")
            if pair[0].inputs_sha256 != pair[1].inputs_sha256:
                raise StageGateError("component compiler arms must execute identical admitted inputs")
            costs, details = [], []
            for arm, (observation, compiler_sha) in enumerate(
                zip(
                    pair,
                    (binding.baseline_sha256, candidate_sha),
                    strict=True,
                )
            ):
                _verify_artifacts(observation)
                if (
                    observation.compiler_sha256,
                    observation.member_sha256,
                    observation.corpus_sha256,
                    observation.target_sha256,
                    observation.scope_sha256,
                ) != (
                    compiler_sha,
                    member.source_sha256,
                    corpus.capsules_sha256,
                    binding.target_sha256,
                    binding.scope.sha256,
                ):
                    raise StageGateError("component feature evidence belongs to different exact inputs")
                domain, coordinates = _applicability(binding, observation, workspace / (str(arm) + "_" + compiler_sha))
                totals, report = complete_component_cost(
                    observation,
                    calibration,
                    scope=binding.scope,
                    qualified_domains=domains,
                    applicability_domain=domain,
                    applicability_coordinates=coordinates,
                )
                costs.append(totals[binding.objective])
                details.append(report)
            if binding.independent_arms is None:
                temporary = cache.with_suffix(".partial." + workspace.name)
                with temporary.open("wb") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    pickle.dump(pair, stream, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(cache)
            return (
                (member.family, member.capsule),
                tuple(costs),
                {"baseline": details[0], "candidate": details[1], "objective": binding.objective},
                ComponentOpportunity(
                    document_sha256([member.family, member.capsule]),
                    member.family,
                    binding.objective,
                    costs[0],
                    costs[1],
                    time.monotonic() - extraction_started,
                    _paired_status(pair, "legality_status"),
                    _paired_status(pair, "functional_status"),
                ),
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for identity, pair, detail, opportunity in pool.map(one, corpus.capsules):
                intervals[identity], reports[identity] = pair, detail
                opportunities.append(opportunity)
        if time.monotonic() > deadline or str(hash_tree(measured)["sha256"]) != candidate_sha:
            raise StageGateError("component analytical evaluation timed out or changed frozen compiler bytes")
        binding.validate(calibration)
        family_counts = {}
        for member in corpus.capsules:
            family_counts[member.family] = family_counts.get(member.family, 0) + 1
        shares = {
            document_sha256([member.family, member.capsule]): 1 / len(family_counts) / family_counts[member.family]
            for member in corpus.capsules
        }
        screening = rank_component_opportunities(opportunities, work_shares=shares)
        (call_root / "complete_cost.json").write_text(
            json.dumps(
                {
                    "schema": "component_analytical_execution_v1",
                    "configuration_sha256": binding.sha256,
                    "candidate_sha256": candidate_sha,
                    "workers": workers,
                    "elapsed_seconds": time.monotonic() - started,
                    "screening": screening,
                    "work_share_basis": "equal generated families and equal members within family",
                    "members": [
                        {"family": identity[0], "capsule": identity[1], **reports[identity]}
                        for identity in sorted(reports)
                    ],
                }
            )
        )
        return ComponentAnalyticalResults(intervals, reports, screening)
    finally:
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()


def build_component_analytical_provider(
    *,
    baseline: Path,
    corpus: C.FrozenPerformanceCorpus,
    target_descriptor: Path,
    feature_provider: Callable[..., Any],
    calibration_adapter: Path,
    scope: ComponentCostScope,
    output: Path,
    lease_path: Path,
    dependencies: Mapping[Path, str],
    qualification: Path | None = None,
    max_workers: int = 1,
    memory_per_worker_bytes: int,
    engine_slots: int = 1,
    objective: str = "warm",
    baseline_admission=None,
    independent_runtime=None,
    independent_arms: IndependentArmFeatures | None = None,
) -> ComponentAnalyticalProvider:
    """Bind an exact independent corpus to calibrated, complete, bounded feedback.

    ``feature_provider`` receives the keyword arguments in :func:`_evaluate` and
    returns two ``ComponentFeatureObservation`` values. Target-owned native/ISA
    mechanisms remain callbacks under the admitted sandbox. Missing held validation
    leaves complete totals UNKNOWN while retaining per-region calibrated evidence.
    """
    if type(scope) is not ComponentCostScope or objective not in ("cold", "warm"):
        raise StageGateError("component analytical provider requires complete scope and explicit regime")
    if any(type(x) is not int or x < 1 for x in (max_workers, memory_per_worker_bytes, engine_slots)):
        raise StageGateError("component analytical worker resource grants must be positive")
    baseline, target_descriptor, adapter = (
        Path(p).resolve() for p in (baseline, target_descriptor, calibration_adapter)
    )
    verify_baseline_admission(baseline_admission, baseline=baseline, corpus=corpus, target_descriptor=target_descriptor)
    runtime = require_independent_runtime(
        independent_runtime, required_roles=("feature_provider",), target_descriptor=target_descriptor
    )
    qualification = Path(qualification).resolve() if qualification else None
    output, lease_path = Path(output).resolve(), Path(lease_path).resolve()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (baseline, corpus.root)):
        raise StageGateError("component analytical private output overlaps frozen inputs")
    calibration = _independent_calibration(adapter, runtime, qualification)
    if not isinstance(calibration, Mapping):
        raise StageGateError("component analytical controlled calibration is unavailable")
    feature_path, feature_sha = _callable_source(feature_provider)
    context_pins = _feature_context_pins(feature_provider)
    dependency_pins = {Path(path).resolve(): digest for path, digest in dependencies.items()}
    for path, digest in context_pins:
        if path in dependency_pins and dependency_pins[path] != digest:
            raise StageGateError("component execution context conflicts with explicit dependency identity")
        dependency_pins[path] = digest
    owner = Path(__file__).resolve()
    binding = ComponentAnalyticalBinding(
        baseline,
        str(hash_tree(baseline)["sha256"]),
        corpus,
        target_descriptor,
        sha256_file(target_descriptor),
        scope,
        adapter,
        feature_provider,
        feature_path,
        feature_sha,
        _callable_code(feature_provider),
        tuple(sorted(dependency_pins.items(), key=lambda row: str(row[0]))),
        context_pins,
        qualification,
        sha256_file(qualification) if qualification else None,
        output,
        lease_path,
        max_workers,
        memory_per_worker_bytes,
        engine_slots,
        objective,
        baseline_admission,
        independent_runtime,
        independent_arms,
    )
    binding.validate(calibration)
    return ComponentAnalyticalProvider(
        partial(_evaluate, binding, adapter),
        owner,
        sha256_file(owner),
        adapter,
        sha256_file(adapter),
        document_sha256(calibration),
        binding,
    )
