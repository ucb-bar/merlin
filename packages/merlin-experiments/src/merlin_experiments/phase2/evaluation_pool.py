"""Host-selected component jobs with bounded grants and actual descendant cleanup.

Only the fixed analytical and certified RTL routes can run. Queue observations
are control data; verified returned feedback retains its existing scientific tier.
"""

from __future__ import annotations

import json
import math
import os
import resource
import select
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from merlin.common import storage_lifecycle as storage
from merlin_experiments.execution import _protocol as protocol

from . import ordinary_measurement_selection as SELECTION
from .contracts import StageGateError, canonical_json, document_sha256, sha256_file, write_json
from .external_resources import ExternalResourceBoundary, executing_resource_boundary
from .feedback_guardian import run_feedback_guardian
from .feedback_protocol import FeedbackValueLimits, receive_feedback_value
from .ordinary_compile_only import KIND as COMPILE_ONLY_KIND
from .portfolio_launch import acquire_host_resource_lease
from .retained_revisions import RetainedRevisionStore, executing_revisions
from .search_inputs import SearchInputs
from .supervised_feedback import _close_guardian, _complete_cleanup, _packet

KINDS = ("component-analytical-feedback", "component-rtl-feedback")
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
    from . import corpus_feedback as CF
    from .component_screening import project_screening
    from .component_workflow import SCHEMA, _interval, validate_component_feedback

    revision.verify()
    os.sched_setaffinity(0, cpus)
    resource.setrlimit(resource.RLIMIT_AS, (grant.ram_bytes, grant.ram_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (grant.disk_bytes, grant.disk_bytes))
    temporary = root / "tmp"
    temporary.mkdir(mode=0o700)
    os.environ["TMPDIR"] = str(temporary)
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
                timeout_s=timeout_s,
                workers=workers,
                kind=kind,
                **({"selected_members": selected_members} if kind == SELECTION.KIND else {}),
            )
            result = inputs.validate_result(raw, revision, kind)
            return SELECTION.validate_feedback(result, members=selected_members) if kind == SELECTION.KIND else result
    with executing_revisions(revision, baseline_revision):
        policy._validate()
    if kind == KINDS[0]:
        from .component_analytical import _evaluate

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


class ComponentEvaluationPool:
    def __init__(
        self,
        *,
        policy,
        revisions,
        resources,
        job_resources,
        root,
        beam=3,
        max_pending=6,
        max_jobs=256,
        external_boundary=None,
    ):
        from .component_analytical import ComponentAnalyticalBinding

        self.inputs = SearchInputs(policy)

        if (
            type(revisions) is not RetainedRevisionStore
            or type(resources) is not AssignedResources
            or type(job_resources) is not JobResources
            or type(beam) is not int
            or not 1 <= beam <= 3
            or type(max_pending) is not int
            or not beam <= max_pending <= 6
            or type(max_jobs) is not int
            or not max_pending <= max_jobs <= 1024
        ):
            raise StageGateError("evaluation pool requires its fixed component policy and bounded host grants")
        resources.verify(external_boundary=external_boundary)
        job_resources.verify()
        self.inputs.validate()
        if (
            self.inputs.analytical is not None
            and type(self.inputs.analytical.binding) is not ComponentAnalyticalBinding
        ):
            raise StageGateError("queued analytical route requires its issued independent execution binding")
        self.policy, self.revisions = policy, revisions
        self._external_boundary = self._selected_boundary = external_boundary
        self.resources, self.grant, self.beam, self.max_pending = resources, job_resources, beam, max_pending
        self.max_jobs = max_jobs
        self._feedback_bytes = 0
        self._feedback_budget = resources.ram_bytes // 16
        self.enforcement = {
            "admission": "assigned CPU/RAM/disk/weighted engine tokens",
            "cpu": "worker affinity inherited by descendants",
            "ram": "per-process RLIMIT_AS; aggregate descendants UNKNOWN",
            "disk": "per-file RLIMIT_FSIZE and sampled aggregate cancellation; kernel quota UNKNOWN",
            "hard_aggregate_limits": "UNKNOWN",
            "feedback_history_bytes": self._feedback_budget,
        }
        self.root = Path(root)
        if self.root.resolve() != self.root or self.root.is_symlink() or self.root.exists():
            raise StageGateError("evaluation pool requires a fresh canonical private root")
        protected = (
            policy.candidate,
            self.inputs.baseline,
            self.inputs.corpus.root,
            revisions.root,
            revisions.authority.output,
        )
        if any(self.root.is_relative_to(path) or path.is_relative_to(self.root) for path in protected):
            raise StageGateError("evaluation output overlaps a compiler, corpus or authority input owner")
        self.capacity = JobResources(
            len(resources.cpu_ids) - resources.author_cores,
            int(resources.ram_bytes * (1 - resources.reserve_fraction)),
            int(resources.disk_bytes * (1 - resources.reserve_fraction)) - revisions.max_total_bytes,
            resources.engine_slots,
        )
        if any(
            getattr(job_resources, field) > getattr(self.capacity, field)
            for field in ("cores", "ram_bytes", "disk_bytes", "engine_slots")
        ):
            raise StageGateError("one evaluation exceeds its assigned capacity after author and source reserves")
        provider = self.inputs.analytical
        lease_path = (
            provider.binding.lease_path if provider is not None else self.root.parent / "component-engine.lease"
        )
        self._lease_path = lease_path
        self._writable_roots = self._resource_writable_roots(lease_path)
        self._verify_resource_boundary()
        if external_boundary is not None:
            self.enforcement.update(external_boundary.receipt())
        self.root.mkdir(parents=True, mode=0o700)
        self._engine_lease = acquire_host_resource_lease(self.root, lease_path=lease_path)
        if self._engine_lease is None:
            raise StageGateError("evaluation outer coordinator engine lease is busy")
        self._lease_path = lease_path
        from merlin.common import owned_children as core_owned_children
        from merlin_experiments.execution import owned_children

        from . import (
            _kernel_resources,
            component_analytical,
            component_execution,
            component_workflow,
            development_feedback,
            external_resources,
            feedback_guardian,
            feedback_protocol,
            retained_revisions,
            search_inputs,
            supervised_feedback,
        )

        owners = (
            Path(__file__),
            component_analytical,
            component_execution,
            component_workflow,
            development_feedback,
            feedback_guardian,
            feedback_protocol,
            retained_revisions,
            search_inputs,
            external_resources,
            _kernel_resources,
            supervised_feedback,
            owned_children,
            core_owned_children,
            protocol,
        )
        self._implementation_pins = tuple(
            (
                Path(owner if isinstance(owner, Path) else owner.__file__).resolve(),
                sha256_file(Path(owner if isinstance(owner, Path) else owner.__file__).resolve()),
            )
            for owner in owners
        )
        self._worker_code = _fixed_evaluator.__code__
        try:
            self.baseline_revision = revisions.freeze(self.inputs.baseline)
        except BaseException:
            self._engine_lease.close()
            raise
        self._jobs, self._lock, self._wake = {}, threading.RLock(), threading.Event()
        self._closed = False
        self._submissions_stopped = False
        self._resource_selection = (
            self.resources,
            self.grant,
            self.capacity,
            self.beam,
            self.max_pending,
            self.max_jobs,
        )
        self._thread = threading.Thread(target=self._serve, name="component-evaluation-pool", daemon=True)
        self._thread.start()

    def _resource_writable_roots(self, lease_path):
        from merlin.common.paths import out_dir

        roots = [
            self.root,
            self.revisions.root,
            self.policy.candidate,
            self.policy.receipt_path.parent,
            Path(lease_path).parent,
            out_dir(),
        ]
        provider = self.inputs.analytical
        if provider is not None:
            roots.append(provider.binding.output)
            if provider.binding.independent_arms is not None:
                roots.append(provider.binding.independent_arms.cache_root)
            compiled = getattr(provider.binding, "compiled_arms", None)
            if compiled is not None:
                roots.append(compiled.execution.output)
        if self.inputs.rtl is not None:
            roots.append(self.inputs.rtl.work_root)
        if self._external_boundary is not None:
            temporary = os.environ.get("TMPDIR")
            if not temporary:
                raise StageGateError("hard resource launch needs explicit TMPDIR on capped backing storage")
            roots.append(Path(temporary))
        return tuple(sorted(set(map(Path, roots))))

    def _verify_resource_boundary(self):
        if self._external_boundary is not self._selected_boundary:
            raise StageGateError("evaluation external resource owner changed")
        if self._external_boundary is None:
            return
        if self._resource_writable_roots(self._lease_path) != self._writable_roots:
            raise StageGateError("evaluation writable resource owners changed")
        context = getattr(
            getattr(getattr(self.policy, "independent_runtime", None), "qualification", None), "context", None
        )
        if getattr(context, "container_transport", None) is not None:
            raise StageGateError("hard resource boundary cannot admit an unbound external container executor")
        self._external_boundary.verify(writable_roots=self._writable_roots)

    def _closure(self):
        self.resources.verify(external_boundary=self._external_boundary)
        self._verify_resource_boundary()
        if (
            self.resources,
            self.grant,
            self.capacity,
            self.beam,
            self.max_pending,
            self.max_jobs,
        ) != self._resource_selection:
            raise StageGateError("evaluation assigned resource selection changed")
        if _fixed_evaluator.__code__ is not self._worker_code:
            raise StageGateError("fixed evaluation worker selection changed")
        for path, digest in self._implementation_pins:
            if sha256_file(path) != digest:
                raise StageGateError("fixed evaluation implementation source changed")
        self.inputs.validate()
        if self.inputs.ordinary:
            return document_sha256(
                {
                    "normal_policy": self.policy.search_closure_sha256,
                    "implementation": [(str(path), digest) for path, digest in self._implementation_pins],
                    "resources": asdict(self.resources),
                    "external_boundary": None if self._external_boundary is None else self._external_boundary.verify(),
                    "grant": asdict(self.grant),
                }
            )
        provider = self.inputs.analytical
        return document_sha256(
            {
                "corpus": self.policy._component_bindings,
                "target": self.policy._target_binding,
                "analytical": None if provider is None else provider.configuration_sha256,
                "rtl": None
                if self.policy.component_rtl is None
                else {
                    "certificate_path": str(self.policy.component_rtl.certificate.path),
                    "certificate_sha256": self.policy.component_rtl.certificate.sha256,
                    "baseline_sha256": self.policy.component_rtl.baseline_sha256,
                    "rtl_identity": self.policy.component_rtl.rtl_identity,
                },
                "runtime": self.policy.independent_runtime.sha256,
                "baseline": self.policy.baseline_admission.sha256,
                "implementation": [(str(path), digest) for path, digest in self._implementation_pins],
                "resources": asdict(self.resources),
                "external_boundary": None if self._external_boundary is None else self._external_boundary.verify(),
                "grant": asdict(self.grant),
            }
        )

    def _verify(self, job):
        if job.member_selection_sha256 != document_sha256(job.selected_members):
            raise StageGateError("queued original member selection changed")
        if job.kind == SELECTION.KIND:
            SELECTION.select_members(self.policy, job.selected_members)
        elif job.selected_members:
            raise StageGateError("queued complete evaluation cannot select a partial original roster")
        job.revision.verify()
        self.baseline_revision.verify()
        with executing_revisions(job.revision, self.baseline_revision):
            closure = self._closure()
        if closure != job.closure or _environment() != job.environment:
            raise StageGateError("queued evaluation source/input/corpus/tool/environment/runtime selection drifted")

    def submit(self, revision_id, *, kind=None, timeout_s=600, selected_members=()):
        with self._lock:
            kind = self.inputs.kinds[0] if kind is None else kind
            if self._closed or self._submissions_stopped:
                raise StageGateError("evaluation submissions stopped")
            selected_route = self.inputs.ordinary and kind == SELECTION.KIND
            if (
                kind not in self.inputs.kinds
                and not selected_route
                or type(timeout_s) not in (int, float)
                or not 1 <= timeout_s <= 2400
            ):
                raise StageGateError("evaluation submission has an unsupported kind or deadline")
            if selected_route:
                selected_members = SELECTION.select_members(self.policy, selected_members)
            elif selected_members:
                raise StageGateError("member selection is diagnostic-only on the normal selected RTL route")
            if (self.inputs.analytical if kind == KINDS[0] else self.inputs.rtl) is None:
                raise StageGateError("selected queued component evaluation provider is unavailable")
            if sum(job.state not in TERMINAL for job in self._jobs.values()) >= self.max_pending:
                raise StageGateError("evaluation queue backpressure: pending grant is full")
            if len(self._jobs) >= self.max_jobs:
                raise StageGateError("evaluation retained job history budget is exhausted")
            if self._retained_output_bytes() + self.grant.disk_bytes > self.capacity.disk_bytes:
                raise StageGateError("evaluation queue backpressure: retained outputs consume the disk grant")
            revision = self.revisions.get(revision_id)
            if set(revision.options) - {"hypothesis", "edit_family"} or any(
                type(value) is not str or not value or len(value) > 512 for value in revision.options.values()
            ):
                raise StageGateError(
                    "fixed component evaluators accept only bounded hypothesis/edit-family annotations"
                )
            identity = uuid.uuid4().hex
            job = _Job(
                identity, revision, kind, float(timeout_s), self._closure(), _environment(), self.root / identity
            )
            job.selected_members = selected_members
            job.member_selection_sha256 = document_sha256(selected_members)
            job.lease = storage.acquire(job.root, owner="phase2 queued component evaluation")
            # Protect caches and immutable inputs for the entire queued/active lifetime.
            paths = {
                self.root,
                revision.source,
                self.baseline_revision.source,
                self.inputs.corpus.root,
                self.policy.target_experiment.path,
            }
            if self.inputs.ordinary:
                paths.update(Path(row["path"]) for row in self.policy.binding()["runtime_source_pins"])
            else:
                paths.update(Path(path) for path in dict(self.policy.independent_runtime.source_pins))
            if self.inputs.analytical is not None:
                arms = self.inputs.analytical.binding.independent_arms
                if arms is not None:
                    paths.add(arms.cache_root)
            try:
                for path in paths:
                    job.pins.append((path, storage.pin(path, reason="queued/active component evaluation")))
            except BaseException:
                for path, token in job.pins:
                    storage.unpin(path, token)
                job.lease.close("failed")
                raise
            self._jobs[identity] = job
            self._save(job)
            self._wake.set()
            return self._status(job)

    def _disk_bytes(self, root):
        total, entries = 0, 0
        if not root.exists():
            return 0
        directories = [root]
        while directories:
            directory = directories.pop()
            with os.scandir(directory) as children:
                for child in children:
                    entries += 1
                    if entries > self.grant.files:
                        raise StageGateError("evaluation output exceeded its file/entry grant")
                    if child.is_symlink():
                        raise StageGateError("evaluation output contains a linked product")
                    if child.is_dir(follow_symlinks=False):
                        directories.append(Path(child.path))
                    elif child.is_file(follow_symlinks=False):
                        total += child.stat(follow_symlinks=False).st_size
                    else:
                        raise StageGateError("evaluation output contains a special product")
        return total

    def _retained_output_bytes(self):
        # Keep completed artifacts priced after their CPU/RAM/engine grants close.
        total = sum(self._disk_bytes(job.root) for job in self._jobs.values() if job.state in TERMINAL)
        provider = self.inputs.analytical
        if provider is not None and provider.binding.independent_arms is not None:
            total += self._disk_bytes(provider.binding.independent_arms.cache_root)
        total += sum(path.stat().st_size for path in self.root.glob("*.json"))
        calibration_root = self.root / "calibrations"
        if calibration_root.exists():
            total += self._disk_bytes(calibration_root)
        return total

    def _status(self, job):
        enforcement = dict(self.enforcement)
        if self._external_boundary is not None:
            try:
                self._verify_resource_boundary()
            except StageGateError:
                enforcement["hard_aggregate_limits"] = "UNKNOWN"
        return {
            "job_id": job.job_id,
            **job.revision.public_record(),
            "kind": job.kind,
            "state": job.state,
            "terminal": job.state in TERMINAL,
            "result": json.loads(canonical_json(job.result)) if job.state == "completed" else None,
            "refusal": job.refusal,
            "promotion": "NO_FINAL_ACCEPTANCE",
            "grant": asdict(self.grant),
            "resource_enforcement": enforcement,
            "selected_members": [list(key) for key in job.selected_members],
        }

    def status(self, job_id):
        with self._lock:
            job = self._get(job_id)
            if job.state == "completed":
                try:
                    self._verify(job)
                    path = job.root / "feedback.json"
                    if path.is_symlink() or path.read_bytes() != canonical_json(job.result):
                        raise StageGateError("queued evaluation completed feedback changed")
                except Exception as error:
                    job.state, job.result, job.refusal = "refused", None, f"evaluation refused ({type(error).__name__})"
                    self._save(job)
            return self._status(job)

    def _get(self, job_id):
        if job_id not in self._jobs:
            raise StageGateError("evaluation job was not issued by this controller")
        return self._jobs[job_id]

    def observations(self):
        """Host search consumes only reopened statuses from this issued job roster."""
        with self._lock:
            return tuple(self.status(identity) for identity in self._jobs)

    def cancel(self, job_id):
        with self._lock:
            job = self._get(job_id)
            if job.state == "queued":
                job.state = "cancelled"
                self._release(job)
            elif job.state not in TERMINAL and job.state != "cleanup_unknown":
                job.cancel_requested = True
                job.state = "cancelling"
            self._save(job)
            self._wake.set()
            return self._status(job)

    def _save(self, job):
        write_json(self.root / (job.job_id + ".json"), self._status(job))

    def _release(self, job, *, state=None):
        for path, token in job.pins:
            storage.unpin(path, token)
        job.pins.clear()
        job.lease.close("completed" if (state or job.state) == "completed" else "failed")

    def _start(self, job, cpus):
        self._verify(job)
        job.root.mkdir(mode=0o700)
        job.cpus, job.deadline = cpus, time.monotonic() + job.timeout_s
        write_json(
            job.root / "admission.json",
            {
                "revision": job.revision.public_record(),
                "kind": job.kind,
                "closure_sha256": job.closure,
                "environment_sha256": job.environment,
                "cpus": cpus,
                "grant": asdict(self.grant),
                "resource_enforcement": self.enforcement,
            },
        )
        destination = job.root / "result.json"
        owner_fd = protocol.self_pidfd()
        control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        job.control = control
        try:
            job.guardian_pid = os.fork()
            if job.guardian_pid == 0:
                try:
                    # No unrelated job control or inherited lock enters this owner.
                    self.revisions._lock = threading.RLock()
                    for other in self._jobs.values():
                        if other is not job:
                            if other.control is not None and other.control.fileno() >= 0:
                                other.control.close()
                            if other.guardian_fd is not None:
                                os.close(other.guardian_fd)
                    run_feedback_guardian(
                        _fixed_evaluator,
                        dict(
                            policy=self.policy,
                            revision=job.revision,
                            baseline_revision=self.baseline_revision,
                            kind=job.kind,
                            timeout_s=job.timeout_s,
                            root=job.root,
                            grant=self.grant,
                            cpus=cpus,
                            index=list(self._jobs).index(job.job_id),
                            resource_boundary=self._external_boundary,
                            **({"selected_members": job.selected_members} if job.kind == SELECTION.KIND else {}),
                        ),
                        destination,
                        FeedbackValueLimits(),
                        job.deadline,
                        child_control,
                        control,
                        owner_fd,
                        self._engine_lease.fileno(),
                        self._lease_path,
                    )
                finally:
                    os._exit(1)
            job.guardian_fd = protocol.pidfd_open(job.guardian_pid)
            child_control.close()
            os.close(owner_fd)
            owner_fd = None
            if _packet(control, job.guardian_fd, job.deadline) != {"state": "ready", "guardian_pid": job.guardian_pid}:
                raise StageGateError("evaluation guardian did not establish its child owner")
            protocol.send(control, {"operation": "start"})
            packet = _packet(control, job.guardian_fd, job.deadline)
            if packet.get("state") != "started":
                raise StageGateError("evaluation guardian did not start its fixed worker")
            job.state = "running"
            self._save(job)
        finally:
            child_control.close()
            if owner_fd is not None:
                os.close(owner_fd)

    def _finish(self, job, *, error=None, cleanup_receipt=None):
        result = None
        if error is None:
            try:
                self._verify(job)
                frame, _size = receive_feedback_value(
                    job.root / "result.json", limits=FeedbackValueLimits(), deadline=job.deadline
                )
                if type(frame) is not dict or set(frame) != {"result"}:
                    raise StageGateError(
                        "evaluation worker refused its fixed feedback envelope: " + str(frame.get("failure"))[:512]
                    )
                with executing_revisions(job.revision, self.baseline_revision):
                    result = self.inputs.validate_result(frame["result"], job.revision, job.kind)
            except Exception as failure:
                error = failure
        try:
            receipt, reaped, exitcode = _close_guardian(job.control, job.guardian_pid, job.guardian_fd, cleanup_receipt)
        except (ChildProcessError, OSError):
            receipt, reaped, exitcode = None, False, None
        if not reaped or not _complete_cleanup(receipt, job.guardian_pid, job.root):
            job.state, job.refusal = "cleanup_unknown", "evaluation descendant cleanup UNKNOWN; grants retained"
            try:
                self._save(job)
            except OSError:
                pass
            return
        os.close(job.guardian_fd)
        job.guardian_fd = None
        job.control.close()
        if not job.cancel_requested and error is None:
            try:
                self._verify(job)
                if time.monotonic() >= job.deadline:
                    raise StageGateError("completed evaluation arrived after its original deadline")
                if exitcode != 0 or receipt.get("error") is not None:
                    raise StageGateError("evaluation guardian lifecycle refused its result")
                if self._feedback_bytes + len(canonical_json(result)) > self._feedback_budget:
                    raise StageGateError("completed evaluation exceeded its retained feedback byte grant")
            except Exception as failure:
                error = failure
        state = "cancelled" if job.cancel_requested else "refused" if error else "completed"
        job.state = "finalizing"
        job.refusal = None if error is None else f"evaluation refused ({type(error).__name__})"
        job.result = result if state == "completed" else None
        try:
            if error is not None:
                write_json(job.root / "refusal.json", {"type": type(error).__name__, "detail": str(error)[:512]})
            if job.result is not None:
                write_json(job.root / "feedback.json", job.result)
                (job.root / "feedback.json").chmod(0o400)
            write_json(job.root / "cleanup.json", receipt)
            self._save(job)
            self._release(job, state=state)
        except OSError:
            job.state, job.result = "cleanup_unknown", None
            job.refusal = "evaluation storage finalization UNKNOWN; grants retained"
            try:
                self._save(job)
            except OSError:
                pass
            return
        if job.result is not None:
            self._feedback_bytes += len(canonical_json(job.result))
        job.state = state
        if state == "completed" and self.inputs.ordinary:
            # Only this parent path has proved guardian reaping, validated the
            # result and durably released the original ownership. Saved cleanup
            # reports and still-running workers cannot publish reusable entries.
            self.policy.publish_completed_artifacts(job.root)
        elif self.inputs.ordinary:
            self.policy.discard_cleaned_artifacts(job.root)
        # Exact feedback/cleanup and closed ownership are already durable. This
        # control index cannot undo them if the capped filesystem becomes full.
        try:
            self._save(job)
        except OSError:
            pass

    def _tick(self):
        active = [job for job in self._jobs.values() if job.state not in TERMINAL and job.state != "queued"]
        for job in active:
            if job.state == "cleanup_unknown":
                continue
            try:
                self._verify_resource_boundary()
                total = self._disk_bytes(job.root)
                error = None
                if total > self.grant.disk_bytes:
                    error = StageGateError("evaluation output exceeded its disk grant")
                elif time.monotonic() >= job.deadline:
                    error = StageGateError("evaluation worker exceeded its deadline")
                if job.cancel_requested or error or (job.root / "result.json").exists():
                    self._finish(
                        job, error=error or (StageGateError("evaluation cancelled") if job.cancel_requested else None)
                    )
                elif select.select([job.control, job.guardian_fd], [], [], 0)[0]:
                    packet = _packet(job.control, job.guardian_fd, job.deadline)
                    if packet.get("state") == "finished":
                        self._finish(
                            job,
                            error=StageGateError("evaluation worker exited before returning feedback"),
                            cleanup_receipt=packet.get("receipt"),
                        )
            except Exception as error:
                self._finish(job, error=error)
        active = [job for job in self._jobs.values() if job.state not in TERMINAL and job.state != "queued"]
        free = [
            cpu
            for cpu in self.resources.cpu_ids[self.resources.author_cores :]
            if all(cpu not in job.cpus for job in active)
        ]
        count = min(
            self.beam,
            self.capacity.cores // self.grant.cores,
            self.capacity.ram_bytes // self.grant.ram_bytes,
            max(0, self.capacity.disk_bytes - self._retained_output_bytes()) // self.grant.disk_bytes,
        )
        engine_used = sum(_engine_slots(job, self.grant, ordinary=self.inputs.ordinary) for job in active)
        for job in self._jobs.values():
            if job.state != "queued" or len(active) >= count:
                continue
            demand = _engine_slots(job, self.grant, ordinary=self.inputs.ordinary)
            if engine_used + demand > self.capacity.engine_slots:
                continue
            try:
                self._start(job, tuple(free[: self.grant.cores]))
            except Exception as error:
                if job.guardian_fd is not None:
                    self._finish(job, error=error)
                elif job.guardian_pid is not None:
                    job.state, job.refusal = "cleanup_unknown", "guardian ownership UNKNOWN; grants retained"
                else:
                    job.state, job.refusal = "refused", f"evaluation refused ({type(error).__name__})"
                    self._release(job)
                    self._save(job)
            if job.state not in TERMINAL:
                active.append(job)
                engine_used += demand
                free = free[self.grant.cores :]

    def _serve(self):
        while True:
            self._wake.wait(0.05)
            self._wake.clear()
            with self._lock:
                self._tick()
                if self._closed and all(
                    job.state in TERMINAL or job.state == "cleanup_unknown" for job in self._jobs.values()
                ):
                    return

    def close(self):
        with self._lock:
            self._submissions_stopped = True
            self._closed = True
            for job in self._jobs.values():
                self.cancel(job.job_id)
            self._wake.set()
        self._thread.join(timeout=15)
        with self._lock:
            if self._thread.is_alive() or any(job.state not in TERMINAL for job in self._jobs.values()):
                raise StageGateError("evaluation pool cleanup UNKNOWN; outer engine lease retained")
            if self._engine_lease is not None:
                self._engine_lease.close()
                self._engine_lease = None

    def stop_submissions(self):
        """Stop admission without cancelling the already issued queue."""
        with self._lock:
            self._submissions_stopped = True
            self._wake.set()

    def drain(self, *, timeout_s):
        """Finish issued work during a bounded wait, then cancel/reap the remainder.

        Cleanup uses the existing guardian proof; local I/O and reaping do not
        establish an OS hard deadline. Unknown cleanup keeps ownership and raises.
        """
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 <= timeout_s <= 2400:
            raise StageGateError("evaluation drain requires a finite bounded wait")
        deadline = time.monotonic() + timeout_s
        self.stop_submissions()
        while time.monotonic() < deadline:
            with self._lock:
                if all(job.state in TERMINAL for job in self._jobs.values()) or any(
                    job.state == "cleanup_unknown" for job in self._jobs.values()
                ):
                    break
            self._wake.wait(min(0.05, max(0, deadline - time.monotonic())))
        self.close()
        return self.observations()
