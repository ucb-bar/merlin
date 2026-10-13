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
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from merlin.targetgen import package_runtime as P
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService

_INVOCATION_PUBLICATION = contextvars.ContextVar("original_snapshot_publication", default=None)


def finish_snapshot_invocation():
    """Complete the final full custody read before output/receipt publication."""
    close = _INVOCATION_PUBLICATION.get()
    if close is not None:
        close()


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
        with contextlib.ExitStack() as stack:
            owner = selected.owner
            stack.enter_context(verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo))
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
    return CommonTransformService(workspace=Path(workspace), bundle_manifest=manifest, repo=Path(repo))


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
    service = CommonTransformService(workspace=original["workspace"],
        bundle_manifest=original["bundle_path"], repo=original["repo"])
    if service.profile != environment["compiler_tool_profile"]:
        raise ValueError("compiler execution differs from the original Phase 1 tool profile")
    return service


def compiler_profile_binding(service, *, run_dir):
    """Close ordinary retained-job identity over its original Phase 1 owner."""
    if service is None:
        if functional_service(run_dir) is not None:
            raise ValueError("ordinary compiler lost its admitted original profile")
        return None
    if (type(service) is not CommonTransformService
        or service.bundle_manifest != Path(run_dir) / "input_bundle_manifest.yaml"):

        raise ValueError("ordinary compiler selected another original profile owner")
    service._verify()
    from merlin_experiments.phase2.contracts import mapping_file

    if service.profile != mapping_file(Path(run_dir) / "environment.yaml", yaml_file=True)["compiler_tool_profile"]:
        raise ValueError("ordinary compiler changed its original profile")
    return {"archive": str(service.bundle_manifest), "profile_sha256": service.profile["sha256"],
            "snapshot_sha256": service.snapshot_digest}


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

    if bundle_manifest is None:
        yield
        return
    workspace = Path.cwd()
    bundle = yaml.safe_load(Path(bundle_manifest).read_bytes())
    with contextlib.ExitStack() as execution:
        with verified_snapshot_scope(workspace, bundle, repo=context.repo):
            owner = CommonTransformService(
                workspace=workspace, bundle_manifest=bundle_manifest, repo=context.repo)
            execution.enter_context(_original_execution_lifetime(
                owner, target_experiment=load_target_experiment(context.descriptor),
                candidate=Path(submission), work_root=Path(work_root)))
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
            with selfcheck_execution(bundle_manifest, context=context, submission=source,
                    work_root=Path(temporary) / "profile-package-work"):
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
        owner = CommonTransformService(workspace=Path.cwd(), bundle_manifest=a.compiler_bundle_manifest,
                                       repo=context.repo)
        owner._verify()
        temporary_root = a.compiler_bundle_manifest.parent / "compile-only-work"
        temporary_root.mkdir(mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="probe-", dir=temporary_root) as temporary:
            from .compile_only import snapshot_submission

            copied = snapshot_submission(Path(a.submission).absolute(), Path(temporary) / "compiler")
            result = run(submission=copied,
                target_experiment=load_target_experiment(context.descriptor),
                capsules_root=a.capsules_root, contract=a.contract, facts_path=a.rtl_facts,
                common_service=owner, work_root=Path(temporary) / "products",
                timeout_s=min(a.timeout, 600), selected=a.capsules,
                scale_probe=bool(getattr(a, "scale_probe", False)))
        owner._verify()
    except Exception as error:
        result = {"schema": "merlin.public_compile_only_feedback.v1", "status": "unavailable",
                  "reason": type(error).__name__, "all_pass": False, "numeric": "not_attempted",
                  "rtl": "not_attempted", "timing_cycles": None}
    txt = json.dumps(result, indent=2, sort_keys=True)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0 if result.get("cases") else 2


def _public_python_roots(owner):
    """Derive import roots from pinned public destinations, including split packages."""
    from merlin.targetgen import tool_registry
    from merlin.targetgen.sandbox import bwrap as BW

    _, grants = BW._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
    sources = list(owner.profile["common_sources"])
    for name in owner.profile["treatment_tools"]:
        if name in {"isa_dialect_generation", "lean_proof_authoring"}:
            sources.extend({"path": path, "sha256": None} for path in tool_registry.spec(name).bundle_paths
                           if path.endswith(".py"))
    roots = set()
    for row in sources:
        tail = row["path"].removeprefix("merlin/python/")
        matches = [(destination, source) for declared, destination, source in grants
                   if declared == row["path"] or declared.endswith("/" + tail)]
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


def public_package_policy(policy, *, target, owner):
    """Reapply exact original public grants without the original workspace mount."""
    from dataclasses import replace

    from merlin.targetgen.sandbox import bwrap as BW

    roots = _public_python_roots(owner)
    prefix = [*policy.argv, *BW._bundle_mount_args(owner.workspace, owner.bundle, owner.repo)]
    prefix = BW.apply_final_answer_masks(prefix, target, owner.workspace, owner.bundle, repo=owner.repo)
    root = shlex.quote(":".join(map(str, roots)))
    return replace(policy, argv=tuple(prefix),
                   env_prefix=policy.env_prefix + f"export PYTHONPATH={root}:$PYTHONPATH; ")


def active_public_package_policy(policy, *, target):
    """The normal package box consumes only a live original core admission."""
    from merlin.targetgen.compiler_profile_admission import active_compiler_imports

    owner = active_compiler_imports()
    return policy if owner is None else public_package_policy(policy, target=target, owner=owner)

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
    service = CommonTransformService(workspace=admission.workspace,
        bundle_manifest=admission.bundle_manifest, repo=admission.repo)
    if service.profile != admission.profile or service.snapshot_digest != admission.snapshot_sha256:
        raise ValueError("normal compiler call changed its live original profile")
    return service



class _Executor:
    def __init__(self, owner, target, candidate, work_root, *, sandbox_inputs=None):
        self.owner, self.target, self.work_root, self.inputs = owner, target, Path(work_root), sandbox_inputs
        from merlin_experiments.phase2.retained_revisions import source_inventory

        self.source_sha256 = source_inventory(Path(candidate), max_bytes=32 << 20,
            max_files=8192, exclude_generated=True)["sha256"]
        if not self.source_sha256 or self.work_root.resolve() != self.work_root or self.work_root.is_symlink():
            raise ValueError("original compiler execution requires actual source and an ordinary private owner")
        if self.work_root.is_relative_to(Path(candidate).resolve()):
            raise ValueError("original compiler staging cannot be candidate-owned")
        self.work_root.mkdir(parents=True, mode=0o700, exist_ok=True)

    def _verify(self, pkg):
        # Verify bounded source inventory without retaining another compiler copy.
        from merlin_experiments.phase2.retained_revisions import source_inventory

        source = source_inventory(pkg.directory, max_bytes=32 << 20, max_files=8192, exclude_generated=True)
        if source["sha256"] != self.source_sha256:
            raise ValueError("original compiler execution received another source revision")
        if pkg.integrity_exempt:
            raise ValueError("original profile compiler cannot claim an integrity exemption")

    def _policy(self, workspace, package):
        from merlin_experiments.phase2 import campaign as PC

        policy = PC.package_sandbox_policy(self.target, workspace, package, inputs=self.inputs)
        return policy

    def _invoke(self, policy, package, argv, *, timeout, evidence=None, inputs=(), outputs=(), dependencies=()):
        from merlin.common import bwrap_args
        from merlin_experiments.phase2 import normal_resource_launch as resources

        owner = evidence or policy.workspace
        with resources.bounded_normal_scratch(policy.argv, owner=owner,
                boundary=resources.resources.active_resource_boundary()) as prefix:
            command = [*prefix, "--chdir", str(package), "bash", "-c", policy.env_prefix + 'exec "$@"',
                       "original-package", *argv]
            if evidence is None:
                return bwrap_args.run(
                    command, policy_end=len(prefix)+2, capture_output=True, text=True, timeout=timeout)
            return bwrap_args.run(
                command, policy_end=len(prefix)+2, directory=evidence, stage="original_profile_package",
                inputs=inputs, outputs=outputs, dependencies=(Path(__file__), *dependencies),
                capture_output=True, text=True, timeout=timeout)

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
            policy = replace(policy, env_prefix=policy.env_prefix
                + f"export MLIR_DIR={shlex.quote(mlir)} LLVM_DIR={shlex.quote(llvm)} CM=cmake CMAKE=cmake; ")
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
    def run_entrypoint(self, pkg, name, input_mlir, output_json=None, *, timeout=600,
                      invocation_directory=None, write_bytecode=False, artifact_profile=None):
        self._verify(pkg)
        if write_bytecode:
            raise ValueError("original profile execution cannot write Python bytecode into immutable compiler source")
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
            copy_plain_file(input_mlir, incoming)
            policy = self._policy(workspace, pkg.directory)
            policy = replace(policy, argv=(*policy.argv, "--ro-bind", str(incoming), str(incoming)))
            protocol = {} if artifact_profile is None else {"artifact_profile": artifact_profile}
            argv = P._resolve_argv(pkg, name, incoming, outgoing, **protocol)
            if P._needs_interpreter(pkg, argv):
                argv = [sys.executable, *argv]
            from merlin_experiments.phase1.brokers.compiler_call_channel import compiler_call_transport

            with compiler_call_transport(self.owner, workspace=workspace, package=pkg.directory,
                    input_path=incoming, command_name=name, timeout_s=timeout) as transport:
                policy = replace(policy, env_prefix=transport.env_prefix(policy.env_prefix))
                result = self._invoke(policy, pkg.directory, argv, timeout=transport.remaining_timeout(),
                                      evidence=invocation_directory, inputs=(Path(input_mlir),),
                                      dependencies=dependencies)
            # Reopen both selections before publishing any staged output. A
            # profile changed during the process cannot authorize its stdout.
            self._verify(pkg)
            P._admit_artifact_protocol(pkg, artifact_profile)
            if artifact_profile is not None:
                require_profile(artifact_profile)
            finish_snapshot_invocation()
            if outgoing is not None and (outgoing.exists() or outgoing.is_symlink()):
                copy_plain_file(outgoing, output_json, output=True)
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
def original_compiler_execution(service, *, target_experiment, candidate, work_root, sandbox_inputs=None):
    """Batch only setup for an exact owner, never the grading lifetime."""
    from merlin.common.bundle_snapshot import verified_snapshot_scope

    if service is None:
        yield
        return
    if type(service) is not CommonTransformService or P.active_package_executor() is not None:
        raise ValueError("original profile execution requires its exact owner and no foreign executor")
    with contextlib.ExitStack() as execution:
        with verified_snapshot_scope(service.workspace, service.bundle, repo=service.repo):
            execution.enter_context(_original_execution_lifetime(
                service, target_experiment=target_experiment, candidate=candidate,
                work_root=work_root, sandbox_inputs=sandbox_inputs))
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
        with original_compiler_imports(service), P.scoped_package_executor(executor):
            yield
