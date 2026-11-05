"""Original frozen-profile import admission and isolated package execution.

Numerical graders keep their original sources, oracles and verdicts. The
compiler sees one input, its declared output and the same public source grants
as its author. Byte-identical grader copies may move without changing compiler
authority; mutable source, callback-selected policies and foreign owners refuse.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
import weakref
from pathlib import Path

import yaml

from merlin.targetgen import package_runtime as P
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService

from . import setup_timing

_INVOCATION_PUBLICATION = contextvars.ContextVar("original_snapshot_publication", default=None)
_EMISSION_OBSERVATION = contextvars.ContextVar("original_profile_emission_observation", default=None)
_CONCURRENT_EXECUTORS = weakref.WeakSet()


@contextlib.contextmanager
def _verification_compute():
    """Let ready execution use shared CPUs before background custody work.

    No held lease is preempted and no check is skipped. This context ends before
    compiler execution, restoring its original author/grade priority.
    """
    from merlin.common import cpu_pool

    with cpu_pool.cpu_priority(cpu_pool.VERIFICATION), cpu_pool.shared_compute_pool():
        yield


def finish_snapshot_invocation():
    """Complete the request's custody boundary before output/receipt publication."""
    close = _INVOCATION_PUBLICATION.get()
    if close is not None:
        close()



def _checkpoint_command_admission():
    """Defer new background commands after a trusted epoch/final checkpoint.

    Ordinary source edits do not request this stop. Never call this from final
    custody or while an owned child is running. Final frozen-compiler checks do
    not carry the background snapshot admission scope and remain eligible.
    """
    from merlin.targetgen.capsule_scheduling import background_admission_selected

    if not background_admission_selected():
        return
    from .. import joint_control

    reason = joint_control.stop_reason()
    if reason is not None:
        raise P.NotMeasuredFailure(
            "preparation_admission", "BACKGROUND_DEFERRED", reason, status="gated",
            execution_status="deferred_before_command", correctness="unknown",
        )


def snapshot_scoped_entrypoint(function):
    """Only exact original execution owners can batch one compiler invocation."""

    @functools.wraps(function)
    def call(self, *args, **kwargs):
        from merlin.common.bundle_snapshot import verified_snapshot_scope
        from merlin_experiments.phase1.component_package_execution import ComponentPackageExecutor

        if type(self) is _Executor:
            selected = self
        elif type(self) is ComponentPackageExecutor:
            selected = self.original_execution
            if selected is None:
                return function(self, *args, **kwargs)
            if type(selected) is not _Executor or P.active_package_executor() is not self:
                raise ValueError("snapshot batching lost its exact original component executor")
        else:
            raise ValueError("snapshot batching requires an exact original executor")
        if type(selected.owner) is not CommonTransformService:
            raise ValueError("snapshot batching requires its original profile owner")
        _checkpoint_command_admission()
        with setup_timing.selected(
            selected.work_root.parent, selected.source_sha256,
            operation=args[1] if len(args) > 1 and isinstance(args[1], str) else function.__name__,
        ), contextlib.ExitStack() as stack:
            owner = selected.owner
            stack.enter_context(setup_timing.timed_context(
                "invocation_snapshot",
                verified_snapshot_scope(
                    owner.workspace, owner.bundle, repo=owner.repo,
                    entry=selected.snapshot_entry_for_call(), guarded_exit=True,
                ),
            ))
            token = _INVOCATION_PUBLICATION.set(stack.close)
            try:
                return function(self, *args, **kwargs)
            finally:
                _INVOCATION_PUBLICATION.reset(token)

    return call


def selected_service(*, workspace, run_dir, repo):
    """Read only the original archived manifest; historical runs keep no scope."""
    manifest = Path(run_dir) / "input_bundle_manifest.yaml"
    if not manifest.is_file():
        return None
    bundle = yaml.safe_load(manifest.read_bytes())
    if bundle.get("compiler_tool_profile") is None:
        return None
    from .snapshot_execution import selected
    from .snapshot_inputs import SnapshotInputs
    owner_type = SnapshotInputs if selected() else CommonTransformService
    return owner_type(workspace=Path(workspace), bundle_manifest=manifest, repo=Path(repo))


def functional_service(run_dir):
    """Recover the same original owner from admitted Phase 1 runtime provenance."""
    from merlin_experiments.phase2 import campaign as PC
    from merlin_experiments.phase2.contracts import mapping_file

    environment_path = Path(run_dir) / "environment.yaml"
    if not environment_path.is_file():
        archive = Path(run_dir) / "input_bundle_manifest.yaml"
        if archive.is_file() and mapping_file(archive, yaml_file=True).get("compiler_tool_profile") is not None:
            raise ValueError("original compiler profile has no admitted environment")
        return None
    environment = mapping_file(environment_path, yaml_file=True)
    if environment.get("compiler_tool_profile") is None:
        return None
    original = PC.verify_private_input_snapshot(Path(run_dir), environment)
    from .snapshot_execution import selected
    from .snapshot_inputs import SnapshotInputs
    owner_type = SnapshotInputs if selected() else CommonTransformService
    service = owner_type(
        workspace=original["workspace"], bundle_manifest=original["bundle_path"], repo=original["repo"]
    )
    if service.profile != environment["compiler_tool_profile"]:
        raise ValueError("compiler execution differs from the original Phase 1 tool profile")
    return service


def compiler_profile_binding(service, *, run_dir):
    """Close ordinary retained-job identity over its original Phase 1 owner."""
    if service is None:
        if functional_service(run_dir) is not None:
            raise ValueError("ordinary compiler lost its admitted original profile")
        return None
    from .snapshot_execution import selected
    from .snapshot_inputs import SnapshotInputs
    if selected():
        if type(service) is not SnapshotInputs or service.bundle_manifest != Path(run_dir) / "input_bundle_manifest.yaml":
            raise ValueError("external compiler selected another snapshot input owner")
        service._verify()
        return service.record()
    if (
        type(service) is not CommonTransformService
        or service.bundle_manifest != Path(run_dir) / "input_bundle_manifest.yaml"
    ):
        raise ValueError("ordinary compiler selected another original profile owner")
    service._verify()
    from merlin_experiments.phase2.contracts import mapping_file

    if service.profile != mapping_file(Path(run_dir) / "environment.yaml", yaml_file=True)["compiler_tool_profile"]:
        raise ValueError("ordinary compiler changed its original profile")
    return {
        "archive": str(service.bundle_manifest),
        "profile_sha256": service.profile["sha256"],
        "snapshot_sha256": service.snapshot_digest,
    }


@contextlib.contextmanager
def original_compiler_imports(service):
    if service is None:
        yield
        return
    if type(service) is not CommonTransformService:
        raise ValueError("compiler imports require the exact original owner")
    from merlin.verify.lean_emission_context import original_emission_checks

    with service.compiler_imports(), original_emission_checks(service):
        yield


@contextlib.contextmanager
def selfcheck_execution(bundle_manifest, *, context, submission, work_root):
    """Close setup custody before grading; keep original execution scopes alive."""
    from merlin.common.bundle_snapshot import verified_snapshot_scope
    from merlin.targetgen.target_experiment import load_target_experiment

    from . import snapshot_execution
    if bundle_manifest is None:
        if snapshot_execution.selected():
            raise ValueError("snapshot execution requires its selected input bundle")
        yield
        return
    if snapshot_execution.selected():
        from .snapshot_inputs import SnapshotInputs
        owner = SnapshotInputs(workspace=Path.cwd(), bundle_manifest=bundle_manifest, repo=context.repo)
        with snapshot_execution.execution(owner, target_experiment=load_target_experiment(context.descriptor),
                candidate=Path(submission), work_root=Path(work_root)):
            yield
        return
    workspace = Path.cwd()
    bundle = yaml.safe_load(Path(bundle_manifest).read_bytes())
    with contextlib.ExitStack() as execution:
        with verified_snapshot_scope(workspace, bundle, repo=context.repo):
            owner = CommonTransformService(workspace=workspace, bundle_manifest=bundle_manifest, repo=context.repo)
            execution.enter_context(
                _original_execution_lifetime(
                    owner,
                    target_experiment=load_target_experiment(context.descriptor),
                    candidate=Path(submission),
                    work_root=Path(work_root),
                )
            )
        # The full final setup read has completed. Actual grading is outside reuse.
        yield


@contextlib.contextmanager
def selfcheck_diagnostics(bundle_manifest, *, context, arguments):
    """Keep historical diagnostics intact; fresh ones copy one bounded compiler."""
    if bundle_manifest is None:
        yield
        return
    from merlin_experiments.phase1 import run_inputs

    from .compile_only import snapshot_submission

    original = arguments.submission
    with tempfile.TemporaryDirectory(prefix="selfcheck-diagnostic-") as temporary:
        source = Path(temporary) / "submission"
        snapshot_submission(Path(original).resolve(), source)
        source.chmod(0o700)  # Declared builds may create an excluded build/ tree.
        manifest = source / "manifest.yaml"
        if manifest.is_file():
            manifest.chmod(0o600)
        run_inputs.strip_build_state(source)
        if manifest.is_file():
            manifest.chmod(0o444)
        arguments.submission = str(source)
        try:
            with selfcheck_execution(
                bundle_manifest, context=context, submission=source, work_root=Path(temporary) / "profile-package-work"
            ):
                yield
        finally:
            arguments.submission = original


def selfcheck_compile_only(a, *, context):
    try:
        import tempfile

        from merlin.targetgen.target_experiment import load_target_experiment
        from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService

        from .compile_only import run

        if a.compiler_bundle_manifest is None or a.rtl_facts is None or a.contract is None:
            raise ValueError("compile-only requires the original frozen bundle, facts and contract")
        owner = CommonTransformService(
            workspace=Path.cwd(), bundle_manifest=a.compiler_bundle_manifest, repo=context.repo
        )
        owner._verify()
        from .public_scalability_gate import selected_budget

        scale_probe = bool(getattr(a, "scale_probe", False))
        ir_budget = selected_budget(owner) if scale_probe else None
        temporary_root = a.compiler_bundle_manifest.parent / "compile-only-work"
        temporary_root.mkdir(mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="probe-", dir=temporary_root) as temporary:
            from .compile_only import snapshot_submission

            copied = snapshot_submission(Path(a.submission).absolute(), Path(temporary) / "compiler")
            result = run(
                submission=copied,
                target_experiment=load_target_experiment(context.descriptor),
                capsules_root=a.capsules_root,
                contract=a.contract,
                facts_path=a.rtl_facts,
                common_service=owner,
                work_root=Path(temporary) / "products",
                timeout_s=min(a.timeout, 600),
                selected=a.capsules,
                scale_probe=scale_probe,
                ir_budget=ir_budget,
                public_artifact_root=Path(getattr(a, "artifact_root", None) or "selfcheck_out/compile-only").absolute(),
            )
        owner._verify()
    except Exception as error:
        result = {
            "schema": "merlin.public_compile_only_feedback.v1",
            "status": "unavailable",
            "reason": type(error).__name__,
            "all_pass": False,
            "numeric": "not_attempted",
            "rtl": "not_attempted",
            "timing_cycles": None,
        }
    txt = json.dumps(result, indent=2, sort_keys=True)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0 if result.get("cases") else 2


@setup_timing.measured("package_policy.public_python_roots")
def _public_python_roots(owner):
    """Derive import roots from pinned public destinations, including split packages."""
    from merlin.targetgen import tool_registry
    from merlin.targetgen.sandbox import bwrap as BW

    sources = list(owner.profile["common_sources"])
    for name in owner.profile["treatment_tools"]:
        if name in {"isa_dialect_generation", "lean_proof_authoring"}:
            sources.extend(
                {"path": path, "sha256": None} for path in tool_registry.spec(name).bundle_paths if path.endswith(".py")
            )
    from merlin.common.bundle_snapshot import guarded_projection

    def derive():
        # Import roots depend only on watched frozen declarations. The mount
        # builder still checks live host locations for every composed policy.
        _, grants = BW._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
        roots = set()
        for row in sources:
            tail = row["path"].removeprefix("merlin/python/")
            matches = [
                (destination, source)
                for declared, destination, source in grants
                if declared == row["path"] or declared.endswith("/" + tail)
            ]
            if len(matches) != 1:
                raise ValueError("original compiler API has no unique pinned public destination")
            destination, source = matches[0]
            if source.is_symlink() or not source.is_file():
                raise ValueError("original compiler API must be an exact frozen source leaf")
            if row["sha256"] is not None and hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("original compiler API bytes differ from its source profile")
            relative = Path(tail)
            if not relative.parts or relative.parts[0] != "merlin" or not str(destination).endswith("/" + tail):
                raise ValueError("original compiler API destination cannot define a Python package root")
            root = destination.parents[len(relative.parts) - 1]
            if ":" in str(root) or "\0" in str(root):
                raise ValueError("original compiler API root cannot form an exact Python path")
            roots.add(root)
        return tuple(sorted(roots))

    return guarded_projection(owner.workspace, owner.bundle, owner.repo,
                              ["original.public_python_roots.v1", sources], derive)



@setup_timing.measured("package_policy.public_grants")
def _public_package_parts(argv, env_prefix, owner):
    """Compose exact original grants; internal callers must close masks last."""
    from merlin.targetgen.sandbox import bwrap as BW

    from ..author_profiles import resolve, verify_bundle

    author = owner.bundle.get("author_profile")
    if author is not None:
        verify_bundle(resolve(author.get("name")), owner.bundle)
        roots = ()
    else:
        roots = _public_python_roots(owner)
    prefix = [*argv, *BW._bundle_mount_args(owner.workspace, owner.bundle, owner.repo, visibility_prefix=list(argv))]
    root = shlex.quote(":".join(map(str, roots)))
    return prefix, env_prefix + f"export PYTHONPATH={root}:$PYTHONPATH; "


def _original_package_composition(target, inputs, scope):
    """A private raw-parts hook, issued only by the exact live original executor."""
    executor = active_original_execution()
    if (
        type(executor) is not _Executor
        or executor.target is not target
        or executor.inputs is not inputs
        or executor.policy_scope is not scope
        or not executor.concurrent_entrypoints_v1()
    ):
        return None
    from merlin.targetgen.compiler_profile_admission import _issued_compiler_imports, active_compiler_imports

    owner = active_compiler_imports()
    if owner is None:
        raise ValueError("original package composition lost its import admission")

    def compose(argv, env_prefix, *, target, scope, selected_surfaces):
        from merlin.common.cpu_pool import shared_compute_pool
        from merlin.targetgen.sandbox.host_surfaces import _final_answer_policy

        expected_surfaces = () if inputs is None else inputs.surfaces
        if (
            active_original_execution() is not executor
            or _issued_compiler_imports() is not owner
            or target is not executor.target
            or scope is not executor.policy_scope
            or tuple(selected_surfaces) != tuple(expected_surfaces)
        ):
            raise ValueError("original package composition changed its owner")
        with shared_compute_pool():
            if not executor.concurrent_entrypoints_v1():
                raise ValueError("original package composition expired")
            argv, env_prefix = _public_package_parts(argv, env_prefix, owner)
            argv, answers = _final_answer_policy(
                argv,
                target,
                owner.workspace,
                owner.bundle,
                repo=owner.repo,
                capsules=None if scope is None else scope.capsules,
                selected_surfaces=selected_surfaces,
            )
            if (
                active_original_execution() is not executor
                or _issued_compiler_imports() is not owner
                or target is not executor.target
                or scope is not executor.policy_scope
            ):
                raise ValueError("original package composition changed its owner")
            if not executor.concurrent_entrypoints_v1():
                raise ValueError("original package composition expired")
            return argv, env_prefix, answers

    return compose


def public_package_policy(policy, *, target, owner, scope=None):
    """Reapply exact original public grants without the original workspace mount.

    ``scope`` (a ``PackagePolicyScope``) shares capsule metadata parsed from
    byte-identical files across one execution scope. Reopen the actual mount
    prefix before omitting a host mask whose input is already hidden.
    """
    from dataclasses import replace

    from merlin.common.cpu_pool import shared_compute_pool
    from merlin.targetgen.sandbox import bwrap as BW

    # Alias discovery walks real runtime trees. Schedule this synchronous CPU
    # work with compilation/simulation, and release its slot before any RPC or
    # child wait. The original source and final privacy checks still run fresh;
    # a scope shares only capsule metadata parsed from byte-identical files.
    with shared_compute_pool():
        prefix, env_prefix = _public_package_parts(policy.argv, policy.env_prefix, owner)
        prefix = BW.apply_final_answer_masks(
            prefix,
            target,
            owner.workspace,
            owner.bundle,
            repo=owner.repo,
            capsules=None if scope is None else scope.capsules,
        )
        return replace(policy, argv=tuple(prefix), env_prefix=env_prefix)


def active_public_package_policy(policy, *, target, scope=None):
    """The normal package box consumes only a live original core admission."""
    from merlin.targetgen.compiler_profile_admission import active_compiler_imports

    owner = active_compiler_imports()
    return policy if owner is None else public_package_policy(policy, target=target, owner=owner, scope=scope)


def active_profile_service():
    """Recover only the original live scope's service for a normal compiler call."""
    original = active_original_execution()
    if original is not None:
        original.owner._verify()
        return original.owner
    from merlin.targetgen.compiler_profile_admission import active_compiler_imports

    admission = active_compiler_imports()
    if admission is None:
        return None
    service = CommonTransformService(
        workspace=admission.workspace, bundle_manifest=admission.bundle_manifest, repo=admission.repo
    )
    if service.profile != admission.profile or service.snapshot_digest != admission.snapshot_sha256:
        raise ValueError("normal compiler call changed its live original profile")
    return service


class _Executor:
    def __init__(self, owner, target, candidate, work_root, *, sandbox_inputs=None):
        self.owner, self.target, self.work_root, self.inputs = owner, target, Path(work_root), sandbox_inputs
        self.candidate = Path(candidate)
        from merlin_experiments.phase2.campaign import PackagePolicyScope

        # One executor is one original execution scope: its entrypoint calls
        # share capsule metadata parsed from byte-identical files.
        self.policy_scope = PackagePolicyScope()
        from merlin_experiments.phase2.retained_revisions import source_inventory

        self.source_sha256 = source_inventory(
            Path(candidate), max_bytes=32 << 20, max_files=8192, exclude_generated=True
        )["sha256"]
        if not self.source_sha256 or self.work_root.resolve() != self.work_root or self.work_root.is_symlink():
            raise ValueError("original compiler execution requires actual source and an ordinary private owner")
        if self.work_root.is_relative_to(Path(candidate).resolve()):
            raise ValueError("original compiler staging cannot be candidate-owned")
        self.work_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._snapshot_entry = self._snapshot_binding = None
        self._concurrent_binding = None
        self._snapshot_thread = None
        self._snapshot_worker_lock = threading.Lock()
        self._snapshot_workers = 0
        self._snapshot_refresh_lock = threading.Lock()
        self._snapshot_retired_entries = []
        self._snapshot_refresh_attempts = 0
        self._snapshot_resource_retry_after = 0.0
        self._snapshot_anchor = None
        self._policy_lock = threading.RLock()
        self._policy_prepared = False

    def _snapshot_identity(self):
        owner = self.owner
        return (
            os.getpid(),
            owner,
            self.candidate,
            self.source_sha256,
            self.work_root,
            owner.workspace,
            owner.repo,
            getattr(owner, "bundle_manifest", None),
            json.dumps(owner.bundle, sort_keys=True, allow_nan=False),
            json.dumps(getattr(owner, "profile", None), sort_keys=True, allow_nan=False),
        )

    @contextlib.contextmanager
    def shared_snapshot_entry(self):
        """Own full cold custody and watched command-boundary payload checks."""
        from merlin.common.bundle_snapshot import (
            MARKER, bundle_snapshot_root, shared_entry_scope, shared_entry_verification,
        )

        entry, binding = None, self._snapshot_identity()
        try:
            marker = bundle_snapshot_root(self.owner.workspace) / MARKER
            self._snapshot_anchor = marker.read_bytes()
            entry = shared_entry_verification(self.owner.workspace, self.owner.bundle, repo=self.owner.repo)
            if self._snapshot_identity() != binding or marker.read_bytes() != self._snapshot_anchor:
                raise ValueError("original executor changed during shared snapshot entry setup")
            self._snapshot_entry, self._snapshot_binding = entry, binding
            self._snapshot_thread = threading.current_thread()
            with shared_entry_scope(entry):
                yield
        finally:
            current = self._snapshot_entry
            self._snapshot_entry = self._snapshot_binding = self._snapshot_thread = None
            # A fork must not operate the original parent's inherited guard locks.
            if os.getpid() == binding[0]:
                for selected in (entry, current, *self._snapshot_retired_entries):
                    if selected is not None:
                        selected.close()
            self._snapshot_retired_entries.clear()
            self._snapshot_anchor = None

    @contextlib.contextmanager
    def refreshed_operation_snapshot_v1(self, service):
        """Re-admit a stale guarded entry once for concurrent original consumers.

        The exact original lifetime and frozen marker remain authoritative. A
        refresh performs ordinary full verification; it cannot refresh a verdict.
        Old scopes retain their original guard and full-read fallback. Four
        failed/invalidated generations exhaust this optional optimization and
        restore ordinary fresh reads. Resource exhaustion is different: one
        serialized retry per thirty seconds may establish a NEW fully verified
        guard after capacity becomes available. No old digest is reauthorized
        across the unwatched interval.
        """
        from merlin.common import bundle_snapshot as B, read_lease_digest
        from merlin.common.cpu_pool import current_thread_owns_cpu_lease

        if service is not self.owner or self._concurrent_selection() is None:
            raise ValueError("snapshot refresh requires its exact concurrent original owner")
        binding = self._snapshot_binding
        if binding is None or self._snapshot_anchor is None:
            raise ValueError("snapshot refresh lacks its original entry lifetime")
        # Never wait on a refresh that needs a CPU while holding a CPU lease.
        # Such consumers keep their ordinary independent custody path.
        can_refresh = not current_thread_owns_cpu_lease()
        _checkpoint_command_admission()
        with self._snapshot_refresh_lock if can_refresh else contextlib.nullcontext():
            _checkpoint_command_admission()
            entry = self.snapshot_entry_for_call()
            root = B.bundle_snapshot_root(self.owner.workspace)
            marker = root / B.MARKER
            if marker.read_bytes() != self._snapshot_anchor:
                raise ValueError("snapshot refresh changed its original marker")
            reusable = entry is not None and entry.reusable(root, self.owner.repo, B._policy_bytes(self.owner.bundle))
            if (can_refresh and not reusable and read_lease_digest.enabled()
                    and self._snapshot_refresh_attempts < 4
                    and time.monotonic() >= self._snapshot_resource_retry_after):
                self._snapshot_refresh_attempts += 1
                replacement = None
                diagnostics = {}
                try:
                    with setup_timing.stage("snapshot_entry_refresh"):
                        replacement = B.shared_entry_verification(
                            self.owner.workspace, self.owner.bundle, repo=self.owner.repo,
                            diagnostics=diagnostics,
                        )
                    if (self._snapshot_binding != binding or self._snapshot_identity() != binding
                            or self._concurrent_selection() is None
                            or marker.read_bytes() != self._snapshot_anchor):
                        raise ValueError("snapshot refresh changed its original selection")
                    if replacement is not None:
                        if (replacement.marker_bytes != self._snapshot_anchor
                                or replacement.document != json.loads(self._snapshot_anchor)
                                or replacement.repo != self.owner.repo or replacement.root != root
                                or replacement.policy_bytes != B._policy_bytes(self.owner.bundle)):
                            raise ValueError("snapshot refresh changed its frozen input identity")
                        if entry is not None:
                            self._snapshot_retired_entries.append(entry)
                        self._snapshot_entry = entry = replacement
                    elif diagnostics.get("resource_errno") is not None:
                        # Capacity failures confer no reuse. Bound expensive
                        # re-admission to one caller per interval; waiting workers
                        # retain ordinary verification. Successful replacement
                        # still consumes the normal four-generation bound.
                        self._snapshot_refresh_attempts -= 1
                        self._snapshot_resource_retry_after = time.monotonic() + 30.0
                    else:
                        # Unsupported/unstable watches do not justify retrying
                        # once per waiting worker. Keep the ordinary fallback.
                        self._snapshot_refresh_attempts = 4
                except BaseException:
                    if replacement is not None:
                        replacement.close()
                    raise
        with B.shared_entry_scope(entry):
            try:
                # Explicitly lend this original request's live admission to the
                # framework operation. An implicit context alone cannot select
                # guarded publication; stale or absent entries still hash fully.
                yield entry
            finally:
                if self._snapshot_identity() != binding or self._concurrent_selection() is None:
                    raise ValueError("snapshot refresh lifetime expired during its operation")

    @contextlib.contextmanager
    def staged_suite_snapshot_scope_v1(self):
        """Re-admit after native model staging, before this suite starts workers.

        CAS placement can change link metadata on a frozen input inode. Such an
        event MUST invalidate the old guard. Only the original owner thread may
        replace it, using the ordinary full verification and a new kernel guard.
        Worker admission is excluded during replacement; command final reads and
        all payload/roster/policy checks remain unchanged.
        """
        from merlin.common.bundle_snapshot import shared_entry_scope, shared_entry_verification

        if (type(self) is not _Executor or self._snapshot_binding is None
                or os.getpid() != self._snapshot_binding[0]
                or threading.current_thread() is not self._snapshot_thread
                or active_original_execution() is not self
                or self._snapshot_identity() != self._snapshot_binding):
            raise ValueError("staged snapshot refresh requires its exact original owner thread")
        with self._snapshot_worker_lock:
            if self._snapshot_workers:
                raise ValueError("staged snapshot refresh cannot overlap original workers")
            old = self.snapshot_entry_for_call()
            # None is the original safe no-sharing route, not an invitation to
            # manufacture authority. Stable entries need no new admission.
            if old is not None and not old.reusable(old.root, old.repo, old.policy_bytes):
                replacement = shared_entry_verification(
                    self.owner.workspace, self.owner.bundle, repo=self.owner.repo
                )
                if self._snapshot_identity() != self._snapshot_binding:
                    if replacement is not None:
                        replacement.close()
                    raise ValueError("original executor changed during staged snapshot refresh")
                self._snapshot_entry = replacement
                old.close()
            entry = self.snapshot_entry_for_call()
        with shared_entry_scope(entry):
            try:
                yield
            finally:
                self.snapshot_entry_for_call()

    def snapshot_entry_for_call(self):
        entry = self._snapshot_entry
        if entry is not None and (
            self._snapshot_identity() != self._snapshot_binding or active_original_execution() is not self
        ):
            raise ValueError("shared snapshot entry lost its original executor/selection/process")
        return entry

    def _concurrent_selection(self):
        """Check issuer identity only; each real call still verifies all input bytes."""
        if type(self) is not _Executor or self not in _CONCURRENT_EXECUTORS:
            return None
        from merlin.targetgen.compiler_profile_admission import _issued_compiler_imports
        from merlin.verify import lean_emission_context as emission

        binding = self._concurrent_binding
        if binding is None:
            raise ValueError("concurrent original entrypoint selection was cleared during its lifetime")
        # Refuse before touching a guard or admission inherited across fork.
        if os.getpid() != binding[0][0]:
            raise ValueError("concurrent original entrypoints changed their process owner")
        admission, scope = _issued_compiler_imports(), emission._ACTIVE.get()
        if (
            self._snapshot_identity() != binding[0]
            or active_original_execution() is not self
            or (id(self.target), id(self.inputs)) != binding[1]
            or admission is not binding[2]
            or scope is not binding[3]
            or scope not in emission._ISSUED
            or not scope.live
            or scope.service is not self.owner
            or scope.admission is not admission
        ):
            raise ValueError("concurrent entrypoints lost their original executor/import/emission selection")
        return binding

    def concurrent_entrypoints_v1(self):
        """Explicit capability only while this exact original lifetime remains live."""
        return self._concurrent_selection() is not None

    def process_lower_interface_v1(self, pkg, interface, generated, *, timeout, artifact_name):
        """Explicit original-host process selection; unsupported callers stay serial."""
        from merlin.common import cpu_pool

        from . import compiler_profile_process as child

        if not child.enabled() or cpu_pool.current_thread_owns_cpu_lease():
            return NotImplemented
        if not self.concurrent_entrypoints_v1():
            raise ValueError("interface process requires its live exact original executor")
        return child.lower(self, pkg, interface, generated, timeout=timeout, artifact_name=artifact_name)

    @contextlib.contextmanager
    def concurrent_entrypoint_scope_v1(self):
        """A copied worker selects its own thread-local existing snapshot guard.

        The caller must copy context separately for every job and join all jobs
        before the original lifetime exits. No import or execution lifetime is
        entered here; ordinary command scopes keep their fresh final reads.
        """
        from merlin.common.bundle_snapshot import shared_entry_scope

        if not self.concurrent_entrypoints_v1():
            raise ValueError("concurrent entrypoints require their live original lifetime")
        with self._snapshot_worker_lock:
            entry = self.snapshot_entry_for_call()
            self._snapshot_workers += 1
        try:
            with shared_entry_scope(entry):
                try:
                    yield
                finally:
                    if not self.concurrent_entrypoints_v1():
                        raise ValueError("concurrent original entrypoint lifetime expired during work")
        finally:
            with self._snapshot_worker_lock:
                self._snapshot_workers -= 1

    @contextlib.contextmanager
    def _concurrent_lifetime(self):
        from merlin.common.access import ModuleLocationPlan
        from merlin.targetgen.compiler_profile_admission import _issued_compiler_imports
        from merlin.verify import lean_emission_context as emission

        admission, scope = _issued_compiler_imports(), emission._ACTIVE.get()
        # Historical profiles and deliberately unadmitted direct constructors
        # keep serial behavior; a dictionary or flag cannot issue this lifetime.
        if admission is None or scope is None:
            yield
            return
        plan = ModuleLocationPlan()
        try:
            self._concurrent_binding = (self._snapshot_identity(), (id(self.target), id(self.inputs)), admission, scope)
            _CONCURRENT_EXECUTORS.add(self)
            self.policy_scope.capsules.module_location_plan = plan
            self._concurrent_selection()
            yield
            self._concurrent_selection()
        finally:
            self.policy_scope.capsules.module_location_plan = None
            self.policy_scope.capsules.inventory.close()
            plan.close()
            _CONCURRENT_EXECUTORS.discard(self)
            self._concurrent_binding = None

    @setup_timing.measured("compiler_source_verify")
    def _verify(self, pkg):
        # Verify bounded source inventory without retaining another compiler copy.
        from merlin_experiments.phase2.retained_revisions import source_inventory

        source = source_inventory(pkg.directory, max_bytes=32 << 20, max_files=8192, exclude_generated=True)
        if source["sha256"] != self.source_sha256:
            raise ValueError("original compiler execution received another source revision")
        if pkg.integrity_exempt:
            raise ValueError("original profile compiler cannot claim an integrity exemption")

    @contextlib.contextmanager
    def _policy_preparation(self):
        from merlin.common.cpu_pool import current_thread_owns_cpu_lease

        if self._policy_prepared or current_thread_owns_cpu_lease():
            yield
            return
        with self._policy_lock:
            if not self._policy_prepared:
                yield
                # Failure leaves this false so the next call prepares normally.
                self._policy_prepared = True
                return
        # A caller queued behind the successful cold preparation must release
        # admission before doing its own fresh policy work.
        yield

    def _policy(self, workspace, package):
        from merlin_experiments.phase2 import campaign as PC

        inventory = self.policy_scope.capsules.inventory
        # Cold classification is single-flight per original request. Waiters
        # acquire no compute lease. Prepared calls run concurrently; individual
        # watched projections/inventories synchronize their own cache access.
        # Every call still constructs and validates its current masks.
        # A legacy caller already holding a slot cannot wait on this lock.
        with setup_timing.stage("package_policy"), self._policy_preparation(), _verification_compute():
            inventory.adopt_snapshot(self.snapshot_entry_for_call())
            with inventory.temporary_root(workspace):
                policy = PC.package_sandbox_policy(self.target, workspace, package, inputs=self.inputs, scope=self.policy_scope)
        return policy

    def _invoke(self, policy, package, argv, *, timeout, evidence=None, inputs=(), outputs=(), dependencies=()):
        from merlin.common import bwrap_args
        from merlin_experiments.phase2 import normal_resource_launch as resources

        from merlin_experiments.phase2.campaign import _verify_frozen_policy
        with setup_timing.stage("frozen_policy_verify"):
            _verify_frozen_policy(policy)
        owner = evidence or policy.workspace
        with resources.bounded_normal_scratch(
            policy.argv, owner=owner, boundary=resources.resources.active_resource_boundary()
        ) as prefix:
            command = [
                *prefix,
                "--chdir",
                str(package),
                "bash",
                "-c",
                policy.env_prefix + f"export TMPDIR={shlex.quote(str(policy.workspace))}; " + 'exec "$@"',
                "original-package",
                *argv,
            ]
            with setup_timing.stage("compiler_command"):
                if evidence is None:
                    return bwrap_args.run(
                        command, policy_end=len(prefix) + 2, capture_output=True, text=True, timeout=timeout
                    )
                return bwrap_args.run(
                    command,
                    policy_end=len(prefix) + 2,
                    directory=evidence,
                    stage="original_profile_package",
                    inputs=inputs,
                    outputs=outputs,
                    dependencies=(Path(__file__), *dependencies),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

    def build_package(self, pkg, *, timeout=1800):
        self._verify(pkg)
        build = pkg.manifest.get("build") or {}
        if not any(build.get(name) for name in ("configure", "command")):
            return
        from dataclasses import replace

        from merlin.targetgen.contract import toolchain

        with tempfile.TemporaryDirectory(prefix="build-", dir=self.work_root) as temporary:
            policy = self._policy(Path(temporary), pkg.directory)
            # Builds operate on the grader's private compiler copy. Reopen its
            # source identity afterward; only excluded build products may move.
            policy = replace(policy, argv=(*policy.argv, "--bind", str(pkg.directory), str(pkg.directory)))
            mlir, llvm = str(toolchain.mlir_cmake_dir()), str(toolchain.mlir_install() / "lib/cmake/llvm")
            subst = {"{package}": str(pkg.directory), "{mlir_dir}": mlir, "{llvm_dir}": llvm}
            policy = replace(
                policy,
                env_prefix=policy.env_prefix
                + f"export MLIR_DIR={shlex.quote(mlir)} LLVM_DIR={shlex.quote(llvm)} CM=cmake CMAKE=cmake; ",
            )
            deadline = time.monotonic() + timeout
            for name in ("configure", "command"):
                step = build.get(name)
                if not step:
                    continue

                def text(value):
                    for source, destination in subst.items():
                        value = value.replace(source, destination)
                    return value

                argv = ["bash", "-c", text(step)] if isinstance(step, str) else [text(value) for value in step]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                result = self._invoke(policy, pkg.directory, argv, timeout=remaining)
                if result.returncode:
                    raise P.CertFailure("build", "tool_crash", "boxed declared build failed: " + result.stderr[-500:])
        self._verify(pkg)

    @snapshot_scoped_entrypoint
    def run_entrypoint(
        self,
        pkg,
        name,
        input_mlir,
        output_json=None,
        *,
        timeout=600,
        invocation_directory=None,
        write_bytecode=False,
        artifact_profile=None,
    ):
        with _verification_compute():
            self._verify(pkg)
            if write_bytecode:
                raise ValueError(
                    "original profile execution cannot write Python bytecode into immutable compiler source"
                )
            # The caller owns the structural profile. It is never a compiler-selected
            # runtime adapter, namespace grant, LLVM artifact or numerical admission.
            P._admit_artifact_protocol(pkg, artifact_profile)
            dependencies = ()
            if artifact_profile is not None:
                from merlin.targetgen.artifact_bundle import require_profile

                require_profile(artifact_profile)
                dependencies = (artifact_profile.path,)
        import sys
        from dataclasses import replace

        from merlin_experiments.phase2.functional_guard_execution import copy_plain_file

        with tempfile.TemporaryDirectory(prefix="call-", dir=self.work_root) as temporary:
            workspace = Path(temporary)
            incoming = workspace / "input.mlir"
            outgoing = workspace / "output.json" if output_json is not None else None
            _checkpoint_command_admission()
            # Schedule source/input/policy preparation before borrowing the command
            # CPU. Release this loan before its separate broker/command lifetime.
            with _verification_compute():
                copy_plain_file(input_mlir, incoming)
            policy = self._policy(workspace, pkg.directory)
            with _verification_compute():
                policy = replace(policy, argv=(*policy.argv, "--ro-bind", str(incoming), str(incoming)))
                protocol = {} if artifact_profile is None else {"artifact_profile": artifact_profile}
                argv = P._resolve_argv(pkg, name, incoming, outgoing, **protocol)
                if P._needs_interpreter(pkg, argv):
                    argv = [sys.executable, *argv]
            from merlin_experiments.phase1.brokers.compiler_call_channel import compiler_call_transport

            _checkpoint_command_admission()
            with compiler_call_transport(
                self.owner,
                workspace=workspace,
                package=pkg.directory,
                input_path=incoming,
                command_name=name,
                timeout_s=timeout,
                borrow_compute=True,
            ) as transport:
                policy = replace(policy, env_prefix=transport.env_prefix(policy.env_prefix))
                result = self._invoke(
                    policy,
                    pkg.directory,
                    argv,
                    timeout=transport.remaining_timeout(),
                    evidence=invocation_directory,
                    inputs=(Path(input_mlir),),
                    dependencies=dependencies,
                )
            # Reopen both selections before publishing any staged output. A
            # profile changed during the process cannot authorize its stdout.
            with _verification_compute():
                self._verify(pkg)
                P._admit_artifact_protocol(pkg, artifact_profile)
                if artifact_profile is not None:
                    require_profile(artifact_profile)
                finish_snapshot_invocation()
                observation = _EMISSION_OBSERVATION.get()
                if observation is not None:
                    from .compiler_profile_emission import retain_original_products

                    retain_original_products(
                        observation,
                        self,
                        pkg,
                        name,
                        Path(input_mlir),
                        incoming,
                        outgoing,
                        result,
                        invocation_directory,
                    )
                if outgoing is not None and (outgoing.exists() or outgoing.is_symlink()):
                    copy_plain_file(outgoing, output_json, output=True)
        with _verification_compute():
            self._verify(pkg)
        return result


def active_original_execution():
    """Identify the live original executor without admitting another callback."""
    executor = P.active_package_executor()
    if type(executor) is _Executor:
        return executor
    if executor is None:
        return None
    from merlin_experiments.phase1.component_package_execution import ComponentPackageExecutor

    if type(executor) is ComponentPackageExecutor:
        method = getattr(executor, "original_profile_execution", None)
        original = None if method is None else method()
        if original is not None and type(original) is not _Executor:
            raise ValueError("composed compiler owner did not preserve its exact original execution")
        return original
    return None


@contextlib.contextmanager
def _request_snapshot_admission(service, candidate):
    """All joint evaluator routes share two exact immutable compiler identities."""
    from .. import joint_control
    from . import snapshot_admission

    if joint_control.selected() is None:
        yield
        return
    raw = os.environ.get('MERLIN_COMPUTE_CPU_SLOTS')
    if not raw:
        raise ValueError('joint snapshot admission requires the original shared compute root')
    root = Path(raw) / 'immutable-snapshots'
    key = snapshot_admission.identity(service, candidate)
    from merlin.targetgen.capsule_scheduling import background_admission_selected
    # Explicit final frozen checks are allowed after editing stops. Ordinary
    # background queue work observes the same trusted stop as command admission.
    stop = joint_control.stop_reason if background_admission_selected() else None
    with snapshot_admission.admitted(root, key, stop_reason=stop):
        if snapshot_admission.identity(service, candidate) != key:
            raise ValueError('compiler snapshot changed while awaiting request admission')
        try:
            yield
        finally:
            if snapshot_admission.identity(service, candidate) != key:
                raise ValueError('compiler snapshot changed during its admitted request')


@contextlib.contextmanager
def original_compiler_execution(service, *, target_experiment, candidate, work_root, sandbox_inputs=None):
    """Batch only setup for an exact owner, never the grading lifetime."""
    from merlin.common.bundle_snapshot import verified_snapshot_scope
    from merlin.targetgen.compiler_library import active_compiler_library, guarded_compiler_library_verification

    from . import snapshot_execution
    if snapshot_execution.selected():
        with snapshot_execution.execution(service, target_experiment=target_experiment,
                candidate=candidate, work_root=work_root, sandbox_inputs=sandbox_inputs):
            yield
        return
    if service is None:
        yield
        return
    if type(service) is not CommonTransformService or P.active_package_executor() is not None:
        raise ValueError("original profile execution requires its exact owner and no foreign executor")
    with _request_snapshot_admission(service, candidate), contextlib.ExitStack() as execution:
        # This issued, watched selection lives through all joined workers and
        # executor cleanup. Every library access still checks its exact guard;
        # scope exit independently verifies the reviewed bytes, even on failure.
        execution.enter_context(guarded_compiler_library_verification(active_compiler_library()))
        with verified_snapshot_scope(service.workspace, service.bundle, repo=service.repo):
            execution.enter_context(
                _original_execution_lifetime(
                    service,
                    target_experiment=target_experiment,
                    candidate=candidate,
                    work_root=work_root,
                    sandbox_inputs=sandbox_inputs,
                )
            )
        yield


@contextlib.contextmanager
def _original_execution_lifetime(service, *, target_experiment, candidate, work_root, sandbox_inputs=None):
    """One original host scope for all actual package checks and child mounts."""
    if service is None:
        yield
        return
    if type(service) is not CommonTransformService or P.active_package_executor() is not None:
        raise ValueError("original profile execution requires its exact owner and no foreign executor")
    root = Path(work_root)
    if root.resolve() != root or root.is_symlink() or root.is_relative_to(Path(candidate).resolve()):
        raise ValueError("original compiler execution requires an independent canonical private work root")
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scope-", dir=root) as temporary:
        executor = _Executor(service, target_experiment, candidate, Path(temporary), sandbox_inputs=sandbox_inputs)
        with executor.shared_snapshot_entry(), original_compiler_imports(service), P.scoped_package_executor(executor):
            with executor._concurrent_lifetime():
                yield
