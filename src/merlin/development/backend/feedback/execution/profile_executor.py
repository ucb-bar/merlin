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

from merlin.development.backend.feedback.execution.profile_runtime import (
    _CONCURRENT_EXECUTORS,
    _EMISSION_OBSERVATION,
    _checkpoint_command_admission,
    _verification_compute,
    active_original_execution,
    finish_snapshot_invocation,
    snapshot_scoped_entrypoint,
)
from merlin.development.targetgen.execution.host.common_transforms import CommonTransformService
from merlin.development.targetgen.packaging import loading as P
from merlin.evaluation.capsule.operations import setup_timing

# ---- compiler_profile_runtime ------------------------------------------------------------------


class _Executor:
    def __init__(self, owner, target, candidate, work_root, *, sandbox_inputs=None):
        self.owner, self.target, self.work_root, self.inputs = owner, target, Path(work_root), sandbox_inputs
        self.candidate = Path(candidate)
        from merlin.development.backend.optimization.campaign.functional_boundary import PackagePolicyScope

        # One executor is one original execution scope: its entrypoint calls
        # share capsule metadata parsed from byte-identical files.
        self.policy_scope = PackagePolicyScope()
        from merlin.development.backend.optimization.search.retained_revisions import source_inventory

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
        from merlin.common.storage.bundle_snapshots import (
            MARKER,
            bundle_snapshot_root,
            shared_entry_scope,
            shared_entry_verification,
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
        from merlin.common.process.cpu_pool import current_thread_owns_cpu_lease
        from merlin.common.storage import bundle_snapshots as B
        from merlin.common.storage import read_leases as read_lease_digest

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
            if (
                can_refresh
                and not reusable
                and read_lease_digest.enabled()
                and self._snapshot_refresh_attempts < 4
                and time.monotonic() >= self._snapshot_resource_retry_after
            ):
                self._snapshot_refresh_attempts += 1
                replacement = None
                diagnostics = {}
                try:
                    with setup_timing.stage("snapshot_entry_refresh"):
                        replacement = B.shared_entry_verification(
                            self.owner.workspace,
                            self.owner.bundle,
                            repo=self.owner.repo,
                            diagnostics=diagnostics,
                        )
                    if (
                        self._snapshot_binding != binding
                        or self._snapshot_identity() != binding
                        or self._concurrent_selection() is None
                        or marker.read_bytes() != self._snapshot_anchor
                    ):
                        raise ValueError("snapshot refresh changed its original selection")
                    if replacement is not None:
                        if (
                            replacement.marker_bytes != self._snapshot_anchor
                            or replacement.document != json.loads(self._snapshot_anchor)
                            or replacement.repo != self.owner.repo
                            or replacement.root != root
                            or replacement.policy_bytes != B._policy_bytes(self.owner.bundle)
                        ):
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
        from merlin.common.storage.bundle_snapshots import shared_entry_scope, shared_entry_verification

        if (
            type(self) is not _Executor
            or self._snapshot_binding is None
            or os.getpid() != self._snapshot_binding[0]
            or threading.current_thread() is not self._snapshot_thread
            or active_original_execution() is not self
            or self._snapshot_identity() != self._snapshot_binding
        ):
            raise ValueError("staged snapshot refresh requires its exact original owner thread")
        with self._snapshot_worker_lock:
            if self._snapshot_workers:
                raise ValueError("staged snapshot refresh cannot overlap original workers")
            old = self.snapshot_entry_for_call()
            # None is the original safe no-sharing route, not an invitation to
            # manufacture authority. Stable entries need no new admission.
            if old is not None and not old.reusable(old.root, old.repo, old.policy_bytes):
                replacement = shared_entry_verification(self.owner.workspace, self.owner.bundle, repo=self.owner.repo)
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
        from merlin.development.targetgen.authoring.profile_admission import _issued_compiler_imports
        from merlin.verification.lean import emission_custody as emission

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
        from merlin.common.process import cpu_pool
        from merlin.development.backend.feedback.execution import compiler_profile_process as child

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
        from merlin.common.storage.bundle_snapshots import shared_entry_scope

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
        from merlin.development.targetgen.authoring.profile_admission import _issued_compiler_imports
        from merlin.verification.lean import emission_custody as emission

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
        from merlin.development.backend.optimization.search.retained_revisions import source_inventory

        source = source_inventory(pkg.directory, max_bytes=32 << 20, max_files=8192, exclude_generated=True)
        if source["sha256"] != self.source_sha256:
            raise ValueError("original compiler execution received another source revision")
        if pkg.integrity_exempt:
            raise ValueError("original profile compiler cannot claim an integrity exemption")

    @contextlib.contextmanager
    def _policy_preparation(self):
        from merlin.common.process.cpu_pool import current_thread_owns_cpu_lease

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
        from merlin.development.backend.optimization.campaign import functional_boundary as PC
        from merlin.development.backend.optimization.campaign import (
            package_sandbox as _merlin_package_sandbox_1,
        )

        inventory = self.policy_scope.capsules.inventory
        # Cold classification is single-flight per original request. Waiters
        # acquire no compute lease. Prepared calls run concurrently; individual
        # watched projections/inventories synchronize their own cache access.
        # Every call still constructs and validates its current masks.
        # A legacy caller already holding a slot cannot wait on this lock.
        with setup_timing.stage("package_policy"), self._policy_preparation(), _verification_compute():
            inventory.adopt_snapshot(self.snapshot_entry_for_call())
            with inventory.temporary_root(workspace):
                policy = _merlin_package_sandbox_1.package_sandbox_policy(
                    self.target, workspace, package, inputs=self.inputs, scope=self.policy_scope
                )
        return policy

    def _invoke(self, policy, package, argv, *, timeout, evidence=None, inputs=(), outputs=(), dependencies=()):
        from merlin.common.process import invocation_records as bwrap_args
        from merlin.development.backend.optimization.campaign import normal_resource_launch as resources
        from merlin.development.backend.optimization.campaign.package_sandbox import _verify_frozen_policy

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
                    return bwrap_args.run_bwrap(
                        command, policy_end=len(prefix) + 2, capture_output=True, text=True, timeout=timeout
                    )
                return bwrap_args.run_bwrap(
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

        from merlin.target.contract.interfaces import abi as toolchain

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
                from merlin.development.targetgen.packaging.records import require_profile

                require_profile(artifact_profile)
                dependencies = (artifact_profile.path,)
        import sys
        from dataclasses import replace

        from merlin.development.backend.optimization.campaign.functional_guard_execution import (
            copy_plain_file,
        )

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
            from merlin.development.backend.brokers.compiler_call_channel import compiler_call_transport

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
                    from merlin.development.backend.feedback.execution.compiler_profile_emission import (
                        retain_original_products,
                    )

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
