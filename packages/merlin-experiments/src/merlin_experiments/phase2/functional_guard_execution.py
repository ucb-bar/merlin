"""Original package boxing for a bounded final functional guard.

Each compiler sees one exact interface and its own outputs. The grade's private
model sources, numerical references and sibling results are never mounted.
"""

from __future__ import annotations

import os
import stat
import tempfile
import threading
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path

from . import campaign
from . import normal_resource_launch as resources
from .contracts import StageGateError, mapping_file, sha256_file

_ACTIVE = ContextVar("functional_guard_invocations", default=None)
_EXECUTOR = ContextVar("functional_guard_package_executor", default=None)


class _PackageExecutor:
    def build_package(self, pkg, *, timeout):
        return campaign._refuse_build(pkg, timeout=timeout)

    def run_entrypoint(self, pkg, name, input_mlir, output_json, **kwargs):
        return campaign._run_boxed(pkg, name, input_mlir, output_json, **kwargs)


def owns_executor(executor):
    return executor is not None and executor is _EXECUTOR.get()


@dataclass
class _Scope:
    policy: campaign.PackageSandboxPolicy
    inputs: object
    retained: ExitStack
    lock: threading.RLock


@contextmanager
def boxed_guard_package(target, workspace, package, *, inputs=None):
    """The original grade remains unmodified when no hard owner is selected."""
    if resources.resources.active_resource_boundary() is None:
        yield
        return
    if not isinstance(inputs, (campaign.PackageSandboxInputs, campaign.FrozenPackageSandboxInputs)):
        raise StageGateError("bounded functional guard requires its original explicit package sandbox inputs")
    if campaign.oot_runner.active_package_executor() is not None:
        raise StageGateError("bounded functional guard cannot substitute a foreign package execution owner")
    policy = campaign.package_sandbox_policy(target, workspace, package, inputs=inputs)
    with ExitStack() as retained:
        scope = _Scope(policy, inputs, retained, threading.RLock())
        token = _ACTIVE.set(scope)
        executor = _PackageExecutor()
        executor_token = _EXECUTOR.set(executor)
        try:
            with campaign.oot_runner.scoped_package_executor(executor), campaign.boxed_entrypoints(policy):
                yield
        finally:
            _EXECUTOR.reset(executor_token)
            _ACTIVE.reset(token)


def active_guard():
    return _ACTIVE.get()


def emit_pair(package, interface, scratch, tag, timeout_s, target, *, inputs=None):
    """Keep the original emission evaluator and admit only its exact interface."""
    from . import emission_analysis

    if resources.resources.active_resource_boundary() is None:
        return emission_analysis.emit_pair(package, interface, scratch, tag, timeout_s)
    incoming = Path(scratch) / "guard_interface.mlir"
    copy_plain_file(interface, incoming)
    with boxed_guard_package(target, scratch, package.directory, inputs=inputs):
        return emission_analysis.emit_pair(package, incoming, scratch, tag, timeout_s)


def require_inline_models(capsules):
    """Exec-based budget children cannot inherit this live package owner."""
    if resources.resources.active_resource_boundary() is None:
        return
    from merlin.targetgen import capsule_runner

    if capsule_runner.model_budget_seconds() is not None and any(
        mapping_file(Path(path) / "capsule.yaml", yaml_file=True).get("kind") == "model" for path in capsules
    ):
        raise StageGateError(
            "bounded final guard cannot launch the exec-based whole-model budget child; "
            "the original inline whole-model route requires MERLIN_MODEL_BUDGET_S unset"
        )


def _file_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def copy_plain_file(source, destination, *, output=False):
    """Stream exact regular-file bytes; never follow a compiler-produced link."""
    source, destination = Path(source), Path(destination)
    if destination.is_symlink() or destination.exists() and not destination.is_file():
        raise StageGateError("functional guard copy destination is linked or special")
    if output and destination.exists() and destination.stat().st_nlink != 1:
        raise StageGateError("functional guard output destination has a foreign link")
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as incoming:
        before = os.fstat(incoming.fileno())
        if not stat.S_ISREG(before.st_mode) or output and before.st_nlink != 1:
            raise StageGateError("functional guard interface/output is special or has a foreign link")
        with tempfile.NamedTemporaryFile(prefix=".guard-copy-", dir=destination.parent, delete=False) as outgoing:
            temporary = Path(outgoing.name)
            try:
                while chunk := incoming.read(1024 * 1024):
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
                after = os.fstat(incoming.fileno())
                current = source.lstat()
                if (
                    _file_identity(before) != _file_identity(after)
                    or _file_identity(after) != _file_identity(current)
                    or not stat.S_ISREG(current.st_mode)
                    or sha256_file(source) != sha256_file(temporary)
                ):
                    raise StageGateError("functional guard input/output changed during its streamed copy")
                os.chmod(temporary, 0o600 if output else 0o444)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def execute_isolated(policy, pkg, name, input_mlir, output_json, *, timeout, invocation_directory):
    """Delegate to the same checked package box under one narrower workspace."""
    scope = active_guard()
    if scope is None or scope.policy is not policy:
        raise StageGateError("functional guard invocation lost its original boxed owner")
    boundary = resources.resources.active_resource_boundary()
    if boundary is None:
        raise StageGateError("functional guard invocation lost its exact resource owner")
    storage = policy.workspace.parent / "functional_compiler_invocations"
    boundary.verify(writable_roots=(storage,))
    storage.mkdir(mode=0o700, parents=True, exist_ok=True)
    with scope.lock:
        workspace = Path(scope.retained.enter_context(tempfile.TemporaryDirectory(prefix="call-", dir=storage)))
    incoming = workspace / Path(input_mlir).name
    copy_plain_file(input_mlir, incoming)
    digest = sha256_file(incoming)
    outgoing = workspace / "declared_output.json" if output_json is not None else None
    narrow = campaign.package_sandbox_policy(policy.target_experiment, workspace, policy.package, inputs=scope.inputs)
    narrow = replace(narrow, argv=(*narrow.argv, "--ro-bind", str(incoming), str(incoming)))
    token = _ACTIVE.set(None)  # The nested call uses the original checks and cannot recurse.
    try:
        with campaign.boxed_entrypoints(narrow):
            result = campaign._run_boxed(
                pkg, name, incoming, outgoing, timeout=timeout, invocation_directory=invocation_directory
            )
    finally:
        _ACTIVE.reset(token)
    if sha256_file(incoming) != digest or sha256_file(input_mlir) != digest:
        raise StageGateError("functional guard interface changed during its original compiler call")
    if outgoing is not None and (outgoing.exists() or outgoing.is_symlink()):
        copy_plain_file(outgoing, output_json, output=True)
    return result
