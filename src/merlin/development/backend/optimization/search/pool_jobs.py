"""Host-selected component jobs with bounded grants and actual descendant cleanup.

Only the fixed analytical and certified RTL routes can run. Queue observations
are control data; verified returned feedback retains its existing scientific tier.
"""

from __future__ import annotations

import json
import math
import os
import resource
import socket
import stat
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from merlin.common.storage import retention as storage
from merlin.development.backend.optimization.author_interface.broker_transports import (
    FeedbackValueLimits,
    _close_guardian,
    _complete_cleanup,
    _packet,
    receive_feedback_value,
    run_feedback_guardian,
)
from merlin.development.backend.optimization.campaign import worker_resources as WORKER_RESOURCES
from merlin.development.backend.optimization.campaign.external_resources import (
    ExternalResourceBoundary,
    executing_resource_boundary,
)
from merlin.development.backend.optimization.campaign.portfolio_launch import acquire_host_resource_lease
from merlin.development.backend.optimization.contracts import (
    StageGateError,
    canonical_json,
    document_sha256,
    sha256_file,
    write_json,
)
from merlin.development.backend.optimization.ordinary import ordinary_measurement_selection as SELECTION
from merlin.development.backend.optimization.ordinary.ordinary_compile_only import KIND as COMPILE_ONLY_KIND
from merlin.development.backend.optimization.search import evaluation_costs as COSTS
from merlin.development.backend.optimization.search.retained_revisions import (
    RetainedRevisionStore,
    executing_revisions,
)
from merlin.development.backend.optimization.search.search_inputs import (
    ORDINARY_SELECTED_FAST_KIND,
    SearchInputs,
)
from merlin.experiments.execution import _protocol as protocol

# ---- evaluation_pool ---------------------------------------------------------------------------

KINDS = ("component-analytical-feedback", "component-rtl-feedback")

SELECTED_KINDS = (SELECTION.KIND, ORDINARY_SELECTED_FAST_KIND)

TERMINAL = frozenset({"completed", "refused", "cancelled"})


def _engine_slots(job, grant, *, ordinary):
    """Only the fixed no-execution route may release simulator capacity."""
    return 0 if ordinary and job.kind == COMPILE_ONLY_KIND else grant.engine_slots


@dataclass(frozen=True)
class AssignedResources:
    cpu_ids: tuple[int, ...]
    ram_bytes: int
    disk_bytes: int
    engine_slots: int
    author_cores: int = 4
    reserve_fraction: float = 0.2
    require_hard_limits: bool = False

    def verify(self, *, external_boundary=None):
        if (
            type(self.cpu_ids) is not tuple
            or not self.cpu_ids
            or any(type(cpu) is not int or cpu < 0 for cpu in self.cpu_ids)
            or len(set(self.cpu_ids)) != len(self.cpu_ids)
            or not set(self.cpu_ids).issubset(os.sched_getaffinity(0))
            or any(
                type(value) is not int or value < 1 for value in (self.ram_bytes, self.disk_bytes, self.engine_slots)
            )
            or type(self.author_cores) is not int
            or self.author_cores < 0
            or self.author_cores >= len(self.cpu_ids)
            or type(self.reserve_fraction) not in (int, float)
            or not 0 <= self.reserve_fraction < 1
        ):
            raise StageGateError("evaluation pool requires explicit usable assigned CPU/RAM/disk/engine capacity")
        if type(self.require_hard_limits) is not bool:
            raise StageGateError("hard resource selection must be an explicit bool")
        if external_boundary is not None:
            if type(external_boundary) is not ExternalResourceBoundary:
                raise StageGateError("resource enforcement requires the exact live-issued external boundary")
            external_boundary.verify(cpu_ids=self.cpu_ids, ram_bytes=self.ram_bytes, disk_bytes=self.disk_bytes)
        elif self.require_hard_limits:
            raise StageGateError(
                "aggregate hard limits require a separately selected enforcing envelope; unavailable here"
            )


@dataclass(frozen=True)
class JobResources:
    cores: int
    ram_bytes: int
    disk_bytes: int
    engine_slots: int = 1
    files: int = 100_000

    def verify(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.cores, self.ram_bytes, self.disk_bytes, self.engine_slots, self.files)
        ):
            raise StageGateError("evaluation jobs require positive CPU/RAM/disk/engine grants")


@dataclass
class _Job:
    job_id: str
    revision: object
    kind: str
    timeout_s: float
    closure: str
    environment: str
    root: Path
    state: str = "queued"
    deadline: float | None = None
    control: object = None
    guardian_pid: int | None = None
    guardian_fd: int | None = None
    pins: list = field(default_factory=list)
    lease: object = None
    result: object = None
    refusal: str | None = None
    cpus: tuple = ()
    cancel_requested: bool = False
    selected_members: tuple = ()
    member_selection_sha256: str = field(default_factory=lambda: document_sha256(()))
    completion_deadline: float | None = None


def _environment():
    # Bind absence as well as selected values without retaining secrets in receipts.
    return document_sha256(dict(os.environ))


def _fixed_evaluator(
    *,
    policy,
    revision,
    baseline_revision,
    kind,
    timeout_s,
    root,
    grant,
    cpus,
    index,
    resource_boundary=None,
    selected_members=(),
):
    """Fixed private worker entry; no callback, command or import selection in requests."""
    with executing_resource_boundary(resource_boundary):
        return _evaluate_fixed_routes(
            policy=policy,
            revision=revision,
            baseline_revision=baseline_revision,
            kind=kind,
            timeout_s=timeout_s,
            root=root,
            grant=grant,
            cpus=cpus,
            index=index,
            selected_members=selected_members,
        )


def _evaluate_fixed_routes(
    *, policy, revision, baseline_revision, kind, timeout_s, root, grant, cpus, index, selected_members=()
):
    deadline = time.monotonic() + timeout_s
    from merlin.development.backend.optimization.components.baseline import project_screening
    from merlin.development.backend.optimization.components.feedback_actions import (
        SCHEMA,
        _interval,
        validate_component_feedback,
    )
    from merlin.development.backend.optimization.search import corpus_feedback as CF

    revision.verify()
    resource.setrlimit(resource.RLIMIT_AS, (grant.ram_bytes, grant.ram_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (grant.disk_bytes, grant.disk_bytes))
    temporary = root / "tmp"
    WORKER_RESOURCES.configure(cpus, temporary=temporary)
    os.environ["MERLIN_PERF_SWEEP_WORKERS"] = "1"
    inputs = SearchInputs(policy)
    if inputs.ordinary:
        workers = min(grant.cores, grant.engine_slots)
        os.environ["MERLIN_PERF_SWEEP_WORKERS"] = str(workers)
        with executing_revisions(revision, baseline_revision):
            raw = policy.evaluate_retained(
                revision,
                baseline_revision,
                work_root=root / "rtl",
                call_index=index,
                timeout_s=max(0, deadline - time.monotonic()) if inputs.diagnostic else timeout_s,
                workers=workers,
                kind=kind,
                **({"selected_members": selected_members} if kind in SELECTED_KINDS else {}),
            )
            result = inputs.validate_result(
                raw, revision, kind, **({"deadline": deadline} if inputs.diagnostic else {})
            )
            if kind == ORDINARY_SELECTED_FAST_KIND:
                from merlin.development.backend.optimization.ordinary.ordinary_fast_feedback import (
                    validate_selected_feedback,
                )

                return validate_selected_feedback(result, members=selected_members)
            return SELECTION.validate_feedback(result, members=selected_members) if kind == SELECTION.KIND else result
    with executing_revisions(revision, baseline_revision):
        policy._validate()
    if kind == KINDS[0]:
        from merlin.development.backend.optimization.components.analytical_provider import _evaluate

        provider = policy.component_analytical
        binding = replace(provider.binding, output=root / "analytical", max_workers=1, engine_slots=1)
        with executing_revisions(revision, baseline_revision):
            raw = _evaluate(
                binding,
                provider.calibration_adapter,
                candidate=revision.source,
                corpus=policy.component_corpus,
                timeout_s=timeout_s,
                retained_revision=revision,
                retained_baseline=baseline_revision,
            )
        with executing_revisions(revision, baseline_revision):
            roster = policy._validate()
            configuration_sha = provider.configuration_sha256
        if set(raw) != roster:
            raise StageGateError("queued analytical result omitted the exact generated member roster")
        evidence = [
            {
                "family": identity[0],
                "capsule": identity[1],
                "baseline": _interval(pair[0]),
                "candidate": _interval(pair[1]),
                "complete_cost": raw.reports[identity],
            }
            for identity, pair in sorted(raw.items())
        ]
        extra = {"screening": project_screening(raw.screening, evidence)} if raw.screening is not None else {}
        tier, provider_sha, config_sha = (
            "calibrated_component_analytical",
            provider.implementation_sha256,
            configuration_sha,
        )
    elif kind == KINDS[1]:
        rtl = replace(
            policy.component_rtl,
            work_root=root / "rtl",
            _baseline_cache=None,
            _profile_baseline_cache=None,
            _prefetched={},
            _totals=None,
            _spend=None,
        )
        with executing_revisions(revision, baseline_revision):
            evidence = CF.validate_redacted_feedback(
                rtl.evaluate(
                    revision.source,
                    round_index=policy.feedback_round,
                    call_index=index,
                    timeout_s=int(timeout_s),
                    retained_revision=revision,
                    complete_roster=True,
                )
            )
        with executing_revisions(revision, baseline_revision):
            if (
                {(row["family"], row["capsule"]) for row in evidence["cells"]} != policy._validate()
                or evidence["candidate_sha256"] != revision.source_sha256
                or evidence["tuning_corpus_sha256"] != policy.component_corpus.capsules_sha256
                or evidence["certificate_sha256"] != rtl.certificate.sha256
            ):
                raise StageGateError("queued RTL result differs from its exact certified roster or identity")
            tier, provider_sha, config_sha, extra = (
                "certified_component_rtl",
                policy._rtl_configuration[0][1],
                rtl.certificate.sha256,
                {},
            )
    else:
        raise StageGateError("evaluation kind is outside the fixed component routes")
    revision.verify()
    return validate_component_feedback(
        {
            "schema": SCHEMA,
            "workflow_id": policy.workflow_id,
            "tier": tier,
            "candidate_sha256": revision.source_sha256,
            "corpus_sha256": policy.component_corpus.capsules_sha256,
            "manifest_sha256": policy.component_corpus.manifest_sha256,
            "provider_sha256": provider_sha,
            "configuration_sha256": config_sha,
            "evidence": evidence,
            "promotion": "NO_FINAL_ACCEPTANCE",
            **extra,
        }
    )
