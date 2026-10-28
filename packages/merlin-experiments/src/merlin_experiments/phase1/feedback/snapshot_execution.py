"""Explicit immutable compiler execution with an independent external evaluator.

Admission verifies the selected public inputs once. Compiler invocations receive
only the admitted compiler, public tools and their own interface/scratch. This
mode issues subprocess receipts, never original-profile Lean/pass certificates.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading

from merlin.targetgen import package_runtime as P
from merlin.common.guarded_file_digest import GuardedDigests
from . import sandbox_pool, setup_timing

ENV = "MERLIN_COMPILER_EXECUTION_MODE"
STRICT = "profile_strict_v1"
MODE = "snapshot_external_v1"


def selected_mode():
    value = os.environ.get(ENV, STRICT)
    if value not in {STRICT, MODE}:
        raise ValueError("unknown compiler execution mode")
    return value


def selected():
    return selected_mode() == MODE


def active_snapshot():
    owner = P.active_package_executor()
    return owner if type(owner) is SnapshotExecutor and owner.live and owner.pid == os.getpid() else None


def source_identity(candidate):
    from merlin_experiments.phase2.retained_revisions import source_inventory
    return source_inventory(Path(candidate), max_bytes=128 << 20, max_files=16384,
                            exclude_generated=True)["sha256"]


class SnapshotExecutor:
    """Host-issued original snapshot owner, not candidate-selectable policy."""
    def __init__(self, service, target, candidate, work_root, *, sandbox_inputs=None):
        from merlin_experiments.phase2 import campaign as C, compiler_libraries
        from . import compiler_profile_runtime as R
        self.owner, self.target, self.execution_mode = service, target, MODE
        self.inputs = sandbox_inputs
        self.candidate, self.work_root = Path(candidate), Path(work_root)
        if (self.candidate.resolve() != self.candidate or self.candidate.is_symlink()
                or self.work_root.resolve() != self.work_root or self.work_root.is_relative_to(self.candidate)):
            raise ValueError("snapshot evaluator requires separate canonical compiler and private work roots")
        self.pid, self.live = os.getpid(), True
        self.source_sha256 = source_identity(self.candidate)
        self._lock, self._integrity, self._built = threading.RLock(), False, False
        self._workers = 0
        self.source_evidence = None
        self.template = self.work_root / "io"
        self.template.mkdir(mode=0o700, parents=True)
        # One trusted policy construction, including all final answer masks.
        # Later calls replace only the private IO mount, never the grants.
        policy = C.package_sandbox_policy(target, self.template, self.candidate, inputs=sandbox_inputs)
        policy = R.public_package_policy(policy, target=target, owner=service)
        self.policy = compiler_libraries.apply(policy)
        self.receipt = self.work_root / "snapshot-admission.json"
        self.receipt.write_text(json.dumps({"schema": "merlin.snapshot_external_admission.v1",
            "execution_mode": MODE, "compiler_sha256": self.source_sha256,
            "bundle_snapshot_sha256": service.snapshot_digest,
            "profile_sha256": service.profile["sha256"],
            "input_owner": service.record() if hasattr(service, "record") else None,
            "scope": "isolated compiler execution; external grading required; no pass-proof authority"}, sort_keys=True))
        self.selected_source_digests = GuardedDigests()
        self.sandboxes = None
        if sandbox_pool.selected():
            self.sandboxes = sandbox_pool.SandboxPool(self.work_root / "sandboxes", self.template, self._launch)

    def _package(self, pkg):
        if not self.live or self.pid != os.getpid() or Path(pkg.directory) != self.candidate:
            raise ValueError("compiler invocation escaped its admitted snapshot owner")
        if pkg.integrity_exempt:
            raise ValueError("snapshot compiler cannot claim an integrity exemption")

    def concurrent_entrypoints_v1(self):
        return active_snapshot() is self

    @contextlib.contextmanager
    def concurrent_entrypoint_scope_v1(self):
        # The scheduler copies the trusted executor context. This is only its
        # lifetime check; it does not install an original-profile proof scope.
        if active_snapshot() is not self:
            raise ValueError("snapshot worker requires its live original owner")
        with self._lock:
            self._workers += 1
        try:
            with self.selected_source_digests.scope():
                yield
            if active_snapshot() is not self:
                raise ValueError("snapshot worker owner changed during evaluation")
        finally:
            with self._lock:
                self._workers -= 1

    def check_integrity_once(self, pkg, callback):
        self._package(pkg)
        with self._lock:
            if not self._integrity:
                callback()
                self._integrity = True

    def prepare_result_source(self, results):
        from merlin.targetgen.capsule_source_identity import SharedCompilerSource
        if active_snapshot() is not self:
            raise ValueError("retained source requires its original live evaluator")
        with self._lock:
            if self.source_evidence is None:
                self.source_evidence = SharedCompilerSource(self.candidate, Path(results),
                    expected_source_sha256=self.source_sha256)
            elif self.source_evidence.results != Path(results):
                raise ValueError("one evaluator cannot redirect its retained source root")

    def close(self):
        try:
            with self._lock:
                if self._workers:
                    raise ValueError("cannot close snapshot while workers are active")
            if source_identity(self.candidate) != self.source_sha256:
                raise ValueError("admitted compiler snapshot changed during evaluation")
        finally:
            if self.sandboxes is not None:
                self.sandboxes.close()
            if self.source_evidence is not None:
                self.source_evidence.close()
            self.live = False
            self.selected_source_digests.close()

    def _payload(self):
        return ["--chdir", str(self.candidate), "bash", "-c",
                self.policy.env_prefix + f"export TMPDIR={shlex.quote(str(self.template))}; " + 'exec "$@"',
                "snapshot-external-package"]

    def _launch(self, io, boundary, stack):
        from merlin_experiments.phase2 import normal_resource_launch as resources
        prefix = [*self.policy.argv, "--bind", str(io), str(self.template)]
        bounded = stack.enter_context(resources.bounded_normal_scratch(prefix, owner=io.parent, boundary=boundary))
        payload = self._payload()
        return [*bounded, *payload[:2]], payload[2:]

    def _invoke_reused(self, argv, scratch, *, timeout, evidence, inputs, outputs):
        from merlin.common import invocation_record
        from merlin_experiments.phase2 import normal_resource_launch as resources
        dependencies = (Path(__file__), Path(sandbox_pool.__file__))
        if evidence is not None:
            retained = scratch / "snapshot-admission.json"
            retained.write_bytes(self.receipt.read_bytes())
            dependencies = (retained, *dependencies)
        with self.sandboxes.box(resources.resources.active_resource_boundary()) as box:
            observation = contextlib.nullcontext(None) if evidence is None else invocation_record.observe(
                evidence, stage="snapshot_external_package", argv=box.logical_argv(argv), env=dict(os.environ),
                inputs=inputs, outputs=outputs, dependencies=dependencies)
            with observation as record, setup_timing.stage("compiler_command"):
                if record is not None:
                    record.document.update(argument_transport=box.transport, sandbox_reuse=box.record())
                result = box.run(argv, scratch, cwd=self.candidate, timeout=timeout)
                if record is not None:
                    record.complete(result)
                return result

    def _invoke(self, argv, scratch, *, timeout, evidence=None, inputs=(), outputs=(), build=False):
        from merlin.common import bwrap_args
        if self.sandboxes is not None and not build and not self.sandboxes.disabled:
            with contextlib.suppress(sandbox_pool.Unavailable):
                return self._invoke_reused(argv, scratch, timeout=timeout, evidence=evidence,
                                           inputs=inputs, outputs=outputs)
        from merlin_experiments.phase2 import normal_resource_launch as resources
        prefix = [*self.policy.argv, "--bind", str(scratch), str(self.template)]
        if build:
            build_root = self.candidate / "build"
            if build_root.is_symlink():
                raise ValueError("compiler build output is indirect")
            build_root.mkdir(exist_ok=True)
            prefix += ["--bind", str(build_root), str(build_root)]
        with resources.bounded_normal_scratch(prefix, owner=evidence or scratch,
                boundary=resources.resources.active_resource_boundary()) as bounded:
            command = [*bounded, "--chdir", str(self.candidate), "bash", "-c",
                self.policy.env_prefix + f"export TMPDIR={shlex.quote(str(self.template))}; " + 'exec "$@"',
                "snapshot-external-package", *argv]
            options = dict(capture_output=True, text=True, timeout=timeout)
            if evidence is not None:
                retained = scratch / "snapshot-admission.json"
                retained.write_bytes(self.receipt.read_bytes())
                options.update(directory=evidence, stage="snapshot_external_package", inputs=inputs,
                               outputs=outputs, dependencies=(retained, Path(__file__)))
            with setup_timing.stage("compiler_command"):
                return bwrap_args.run(command, policy_end=len(bounded) + 2, **options)

    def build_package(self, pkg, *, timeout=1800):
        from merlin.targetgen.contract import toolchain
        import time
        self._package(pkg)
        with self._lock:
            if self._built:
                return
            build = pkg.manifest.get("build") or {}
            substitutions = {"{package}": str(self.candidate),
                "{mlir_dir}": str(toolchain.mlir_cmake_dir()),
                "{llvm_dir}": str(toolchain.mlir_install() / "lib/cmake/llvm")}
            def substitute(value):
                for key, replacement in substitutions.items():
                    value = value.replace(key, replacement)
                return value
            deadline = time.monotonic() + timeout
            for name in ("configure", "command"):
                step = build.get(name)
                if not step:
                    continue
                argv = ["bash", "-c", substitute(step)] if isinstance(step, str) else list(map(substitute, step))
                argv = ["env", "MLIR_DIR=" + substitutions["{mlir_dir}"],
                        "LLVM_DIR=" + substitutions["{llvm_dir}"], "CM=cmake", "CMAKE=cmake", *argv]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                with tempfile.TemporaryDirectory(prefix="build-", dir=self.work_root) as temporary:
                    result = self._invoke(argv, Path(temporary), timeout=remaining, build=True)
                if result.returncode:
                    raise P.CertFailure("build", "tool_crash", "isolated compiler build failed: " + result.stderr[-500:])
            self._built = True

    def run_entrypoint(self, pkg, name, input_mlir, output_json=None, *, timeout=600,
                       invocation_directory=None, write_bytecode=False, artifact_profile=None):
        from merlin_experiments.phase2.functional_guard_execution import copy_plain_file
        self._package(pkg)
        if write_bytecode:
            raise ValueError("immutable compiler invocation cannot write bytecode")
        P._admit_artifact_protocol(pkg, artifact_profile)
        with contextlib.ExitStack() as temporary:
            if invocation_directory is None:
                scratch = Path(temporary.enter_context(tempfile.TemporaryDirectory(prefix="call-", dir=self.work_root)))
            else:
                retained = Path(invocation_directory) / "compiler-calls"
                if retained.is_symlink() or retained.resolve() != retained or retained.is_relative_to(self.candidate):
                    raise ValueError("compiler invocation receipts require independent canonical storage")
                retained.mkdir(parents=True, exist_ok=True, mode=0o700)
                scratch = Path(tempfile.mkdtemp(prefix="call-", dir=retained))
            incoming, outgoing = scratch / "input.mlir", scratch / "output.json"
            copy_plain_file(input_mlir, incoming)
            protocol = {} if artifact_profile is None else {"artifact_profile": artifact_profile}
            argv = P._resolve_argv(pkg, name, self.template / incoming.name,
                                  self.template / outgoing.name if output_json is not None else None, **protocol)
            if P._needs_interpreter(pkg, argv):
                argv = [sys.executable, *argv]
            result = self._invoke(argv, scratch, timeout=timeout, evidence=invocation_directory,
                                  inputs=(Path(input_mlir), incoming),
                                  outputs=(outgoing,) if output_json is not None else ())
            if output_json is not None and (outgoing.exists() or outgoing.is_symlink()):
                copy_plain_file(outgoing, output_json, output=True)
            return result


@contextlib.contextmanager
def execution(service, *, target_experiment, candidate, work_root, sandbox_inputs=None):
    from merlin.targetgen.common_transforms import CommonTransformService
    from .snapshot_inputs import SnapshotInputs
    if type(service) is CommonTransformService:
        service = SnapshotInputs(workspace=service.workspace, bundle_manifest=service.bundle_manifest, repo=service.repo)
    if not selected() or type(service) is not SnapshotInputs or P.active_package_executor() is not None:
        raise ValueError("snapshot execution requires explicit selection and original input admission")
    from merlin_experiments.phase1 import full_compiler_workspace as full
    # Resolve the reviewed tree once. Context copies carry this exact selection
    # into joined workers; None also suppresses repeated environment discovery.
    with full.scoped_full_compiler(full.selected()):
        with setup_timing.stage("snapshot_external_admission"):
            service._verify()
            root = Path(work_root)
            if root.resolve() != root or root.is_relative_to(Path(candidate).resolve()):
                raise ValueError("snapshot work root must be independent and canonical")
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            private = Path(tempfile.mkdtemp(prefix="snapshot-", dir=root))
            executor = SnapshotExecutor(service, target_experiment, candidate, private, sandbox_inputs=sandbox_inputs)
        try:
            with P.scoped_package_executor(executor), executor.selected_source_digests.scope():
                yield executor
        finally:
            executor.close()
