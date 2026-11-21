"""Evaluator-owned entrypoint sandbox for component functional qualification.

Only the current interface input and isolated output staging enter the compiler
namespace. Goldens, private grade records and invocation observations stay out.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from merlin.common.paths import module_source_path
from merlin.common.process import invocation_records as invocation_record
from merlin.development.backend.feedback.execution.profile_runtime import (
    finish_snapshot_invocation,
    snapshot_scoped_entrypoint,
)
from merlin.development.backend.optimization import contracts as C
from merlin.development.backend.optimization.campaign.external_resources import (
    active_resource_boundary,
    bounded_tool_scratch,
)
from merlin.development.backend.optimization.components.experiment import (
    ComponentView,
    RuntimeGrant,
    strict_tool_policy,
    verify_component_view,
)
from merlin.development.targetgen.packaging import loading as P
from merlin.experiments.execution.container_policy import prepare_strict_policy
from merlin.experiments.execution.container_transport import PreparedContainerTransport, mounts_from_strict_policy

# ---- component_package_execution ---------------------------------------------------------------


def selected_compiler_transport(runtime_authority, *, view, runtime):
    """Consume the exact command selection of already qualified runtime controls.

    Serialized selections, callable import strings and available local services
    cannot choose the compiler sandbox. Runtime qualification is still mandatory.
    """
    from merlin.development.backend.optimization.controls.component_runtime_support import (
        PreparedIndependentRuntimeContext,
    )
    from merlin.development.backend.optimization.controls.runtime_authority import IndependentComponentRuntime

    if type(runtime_authority) is not IndependentComponentRuntime:
        raise C.StageGateError("compiler command selection requires independently issued runtime authority")
    runtime_authority.verify(required_roles=("grade", "stage_verifier"))
    context = runtime_authority.qualification.context
    if type(context) is not PreparedIndependentRuntimeContext:
        raise C.StageGateError("compiler command selection requires its original prepared runtime context")
    context.verify()
    transport = context.container_transport
    if transport is not None:
        if (
            type(transport) is not PreparedContainerTransport
            or context.compiler_view != view
            or context.compiler_runtime != runtime
        ):
            raise C.StageGateError("compiler command transport changes its qualified public view/runtime grants")
        transport.verify()
    return transport


@dataclass(frozen=True)
class ComponentPackageExecutor:
    candidate: Path
    view: ComponentView
    runtime: tuple[RuntimeGrant, ...]
    evidence_root: Path
    record_stage_inspection: bool = True
    container_transport: PreparedContainerTransport | None = None
    original_execution: object = None

    def original_profile_execution(self):
        """Reopen the retained original owner, never infer one from metadata."""
        if self.original_execution is None:
            return None
        from merlin.development.backend.feedback.execution import (
            profile_executor as _merlin_profile_executor_1,
        )
        from merlin.development.backend.feedback.execution import profile_runtime as original

        selected = self.original_execution
        if type(selected) is not _merlin_profile_executor_1._Executor or P.active_package_executor() is not self:
            raise C.StageGateError("component profile composition requires its exact live original execution owner")
        _merlin_profile_executor_1._Executor._verify(
            selected, SimpleNamespace(directory=self.candidate, integrity_exempt=False)
        )
        original.CommonTransformService._verify(selected.owner)
        return selected

    def _verify_original(self, pkg):
        selected = self.original_profile_execution()
        if selected is not None:
            from merlin.development.backend.feedback.execution import (
                profile_executor as _merlin_profile_executor_2,
            )
            from merlin.development.backend.feedback.execution import profile_runtime as original

            _merlin_profile_executor_2._Executor._verify(selected, pkg)

    def _original_public_policy(self, policy, *, workspace):
        """Keep qualified component mounts and add only original public leaves."""
        selected = self.original_profile_execution()
        from merlin.development.backend.optimization.campaign import compiler_libraries as libraries
        from merlin.development.backend.optimization.campaign.functional_boundary import PackageSandboxPolicy
        from merlin.development.backend.optimization.campaign.package_sandbox import _verify_frozen_policy
        from merlin.development.targetgen.authoring.library_admission import active_compiler_library

        library = active_compiler_library()
        if selected is None and library is None:
            return policy
        prepared = PackageSandboxPolicy(
            tuple(policy),
            (),
            (),
            Path(workspace),
            self.candidate,
            None if selected is None else selected.target,
            env_prefix="",
        )
        roots = []
        if selected is not None:
            from merlin.development.backend.feedback.execution import profile_runtime as original

            prepared = original.public_package_policy(prepared, target=selected.target, owner=selected.owner)
            roots.extend(original._public_python_roots(selected.owner))
        prepared = libraries.apply(prepared)
        _verify_frozen_policy(prepared)
        if library is not None:
            roots.insert(0, libraries.guest_root(library))
        # Strict compilers can use a direct interpreter without a shell. Carry
        # the same exact candidate overlay as PYTHONPATH environment data.
        result = list(prepared.argv)
        previous = _policy_environment(result).get("PYTHONPATH", "")
        value = ":".join((*map(str, roots), *([previous] if previous else [])))
        return _set_environment(result, (("PYTHONPATH", value),))

    def build_package(self, pkg, *, timeout=1800):
        self._verify_original(pkg)
        if pkg.directory.resolve() != self.candidate:
            raise C.StageGateError("component package build escaped its immutable compiler owner")
        build = pkg.manifest.get("build") or {}
        if any(build.get(name) for name in ("configure", "command")):
            raise C.StageGateError(
                "component qualification requires an independently boxed build service for declared build steps"
            )

    @snapshot_scoped_entrypoint
    def run_entrypoint(
        self,
        pkg,
        name,
        input_mlir,
        output_json=None,
        *,
        timeout=600,
        write_bytecode=False,
        artifact_profile=None,
        invocation_directory=None,
    ):
        self._verify_original(pkg)
        if pkg.directory.resolve() != self.candidate or write_bytecode or artifact_profile is not None:
            raise C.StageGateError("component package invocation differs from its exact immutable owner/protocol")
        if (
            not Path(input_mlir).resolve().is_relative_to(self.evidence_root)
            or invocation_directory is None
            or not Path(invocation_directory).resolve().is_relative_to(self.evidence_root)
            or output_json is not None
            and not Path(output_json).resolve().is_relative_to(self.evidence_root)
        ):
            raise C.StageGateError("component compiler input/output observation escaped the private grade owner")
        selected = [row for row in self.runtime if row.destination == "/usr/bin/bwrap"]
        if len(selected) != 1:
            raise C.StageGateError("component qualification runtime must pin its sandbox executable")
        selected[0].verify()
        with tempfile.TemporaryDirectory(prefix="compiler-output-", dir=self.evidence_root.parent) as staging:
            selected_profile = self.original_profile_execution()
            if selected_profile is None:
                call_scope = contextlib.nullcontext(None)
            else:
                from merlin.development.backend.brokers.compiler_call_channel import compiler_call_transport

                call_scope = compiler_call_transport(
                    selected_profile.owner,
                    workspace=Path(staging),
                    package=self.candidate,
                    input_path=Path(input_mlir),
                    command_name=name,
                    timeout_s=timeout,
                )
            with call_scope as channel:
                host_output = Path(staging) / "command_buffer.json"
                argv = P._resolve_argv(
                    pkg,
                    name,
                    Path("/evaluation-input/interface.mlir"),
                    (
                        host_output
                        if self.original_execution is not None
                        else Path("/evaluation-output/command_buffer.json")
                    )
                    if output_json is not None
                    else None,
                )
                if P._needs_interpreter(pkg, argv):
                    interpreters = [
                        row for row in self.runtime if row.source.resolve() == Path(sys.executable).resolve()
                    ]
                    if len(interpreters) != 1:
                        raise C.StageGateError("component runtime does not pin the actual selected Python interpreter")
                    argv = [interpreters[0].destination, *argv]
                # Absolute interpreter/tool commands declared by a package must
                # resolve to reviewed destination files, never host-only aliases.
                for row in self.runtime:
                    if Path(argv[0]).is_absolute() and Path(argv[0]).resolve() == row.source.resolve():
                        argv[0] = row.destination
                        break
                policy = list(
                    strict_tool_policy(
                        self.view,
                        self.candidate,
                        runtime=self.runtime,
                        candidate_destination=str(self.candidate),
                        bwrap_binary=selected[0].source,
                        candidate_writable=False,
                    )
                )
                policy += [
                    "--ro-bind",
                    str(input_mlir),
                    "/evaluation-input/interface.mlir",
                    "--bind",
                    staging,
                    staging if self.original_execution is not None else "/evaluation-output",
                ]
                policy = self._original_public_policy(policy, workspace=staging)
                if channel is not None:
                    policy = _set_environment(policy, channel.environment())
                    timeout = channel.remaining_timeout()
                dependencies = (
                    *tuple(path for path in self.candidate.rglob("*") if path.is_file()),
                    *tuple(row.source for row in self.runtime),
                    Path(__file__),
                )
                if self.container_transport is not None:
                    if active_resource_boundary() is not None:
                        raise C.StageGateError(
                            "hard resource budget cannot admit an unbound external container executor"
                        )
                    if type(self.container_transport) is not PreparedContainerTransport:
                        raise C.StageGateError(
                            "component package requires an explicitly prepared original container transport"
                        )
                    self.container_transport.verify()
                    owned_evidence = Path(invocation_directory) / ("container-command-" + uuid.uuid4().hex)
                    container_options = {}
                    if self.original_execution is not None:
                        mask_workspace = Path(invocation_directory) / ("container-masks-" + uuid.uuid4().hex)
                        mask_workspace.mkdir()
                        plan = prepare_strict_policy(
                            tuple(policy),
                            original_environment=tuple(_policy_environment(policy).items()),
                            mask_workspace=mask_workspace,
                        )
                        mounts = plan.mounts
                        container_options = {"environment": plan.environment, "namespace_masks": plan.namespace_masks}
                    else:
                        mounts = mounts_from_strict_policy(tuple(policy))
                    with invocation_record.observe_call(
                        invocation_directory,
                        stage=name,
                        function=self.container_transport.execute,
                        arguments={
                            "command": argv,
                            "cwd": str(self.candidate),
                            "timeout_s": timeout,
                            "client_sha256": self.container_transport.client_sha256,
                            "guardian_sha256": self.container_transport.guardian_sha256,
                            "image_config_sha256": self.container_transport.configuration_sha256,
                            **container_options,
                        },
                        inputs=(input_mlir,),
                        outputs=(output_json,) if output_json is not None else (),
                        dependencies=dependencies,
                    ) as record:
                        actual = self.container_transport.execute(
                            mounts=mounts,
                            command=tuple(argv),
                            cwd=str(self.candidate),
                            evidence_root=owned_evidence,
                            timeout_s=timeout,
                            **container_options,
                        )
                        result = subprocess.CompletedProcess(
                            actual.args, actual.returncode, actual.stdout.decode("utf-8"), actual.stderr.decode("utf-8")
                        )
                        if host_output.is_symlink():
                            raise C.StageGateError("component compiler output staging contains a linked product")
                        finish_snapshot_invocation()
                        if output_json is not None and host_output.is_file():
                            shutil.copyfile(host_output, output_json)
                        record.returned(stdout=result.stdout, stderr=result.stderr)
                        self._verify_original(pkg)
                        return result
                from merlin.common.process import invocation_records as bwrap_args

                with (
                    bounded_tool_scratch([*policy, "--", *argv], owner=Path(invocation_directory)) as command,
                    bwrap_args.argument_transport(command, policy_end=len(command) - len(argv) - 1) as transport,
                    invocation_record.observe(
                        invocation_directory,
                        stage=name,
                        argv=command,
                        inputs=(input_mlir,),
                        outputs=(output_json,) if output_json is not None else (),
                        dependencies=(*dependencies, Path(bwrap_args.__file__)),
                    ) as record,
                ):
                    record.document["argument_transport"] = transport.record()
                    actual = subprocess.run(
                        list(transport.command),
                        pass_fds=transport.pass_fds,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    result = subprocess.CompletedProcess(command, actual.returncode, actual.stdout, actual.stderr)
                    if host_output.is_symlink():
                        raise C.StageGateError("component compiler output staging contains a linked product")
                    finish_snapshot_invocation()
                    if output_json is not None and host_output.is_file():
                        shutil.copyfile(host_output, output_json)
                    record.complete(result)
                    self._verify_original(pkg)
                    return result


def _policy_environment(policy):
    values = {}
    for index, value in enumerate(policy):
        if value == "--setenv":
            key, selected = policy[index + 1 : index + 3]
            if key in values:
                raise C.StageGateError("component command has duplicate selected environment fields")
            values[key] = selected
    return values


def _set_environment(policy, values):
    result = list(policy)
    for key, value in values:
        positions = [
            index + 2 for index, token in enumerate(result[:-2]) if token == "--setenv" and result[index + 1] == key
        ]
        if len(positions) > 1:
            raise C.StageGateError("component command has duplicate selected environment fields")
        if positions:
            result[positions[0]] = value
        else:
            result += ["--setenv", key, value]
    return result


@contextlib.contextmanager
def qualified_package_execution(
    *,
    candidate: Path,
    view: ComponentView,
    runtime: tuple[RuntimeGrant, ...],
    evidence_root: Path,
    container_transport: PreparedContainerTransport | None = None,
):
    if type(view) is not ComponentView:
        raise C.StageGateError("component qualification needs the independently frozen public library view")
    from merlin.development.backend.feedback.execution import profile_runtime as original

    active = P.active_package_executor()
    original_execution = original.active_original_execution()
    if active is not None and original_execution is None:
        if type(active) is not ComponentPackageExecutor or (
            active.candidate != candidate.resolve()
            or active.view != view
            or active.runtime != runtime
            or active.container_transport is not container_transport
        ):
            raise C.StageGateError("component qualification cannot replace a foreign package execution owner")
        original_execution = active.original_profile_execution()
    executor = ComponentPackageExecutor(
        candidate.resolve(),
        view,
        runtime,
        evidence_root.resolve(),
        container_transport=container_transport,
        original_execution=original_execution,
    )
    with P.scoped_package_executor(executor):
        executor.original_profile_execution()
        yield
        executor.original_profile_execution()


# ---- component_tool_readiness ------------------------------------------------------------------
# Original shared-tool probes through an explicitly selected command owner.
#
# Tool readiness does not qualify compiler semantics, author isolation or runtime.
# The fresh input owner supplies its already admitted exact transport and policy.


def probe_native_author_tools(inputs):
    """Observe the same admitted commands inside the actual native tool profile.

    Called by the fresh lifecycle only after its independent input/runtime
    admission. This config-only home receives no control credential and no
    model request. Successful observations confer no origin or correctness.
    """
    from merlin.development.backend.providers import codex_launch_codex_agent as CA

    owner = inputs.output / "readiness" / "native_author"
    home = owner / "home"
    home.mkdir(mode=0o700, parents=True)
    config = home / "config.toml"
    config.write_text(
        CA._candidate_permission_config(
            home, read_paths=("/component-inputs", *(row.destination for row in inputs.control_runtime))
        )
    )
    workspace = owner / "workspace"
    # Native tools may create temporary sandbox mountpoints. Keep those in a
    # disposable copy; original scaffold files are separately read-only binds,
    # and no persistent new member or directory may survive a successful probe.
    original = C.exact_tree_record(inputs.candidate)
    original_directories = tuple(
        sorted(path.relative_to(inputs.candidate).as_posix() for path in inputs.candidate.rglob("*") if path.is_dir())
    )
    shutil.copytree(inputs.candidate, workspace)
    policy = list(
        strict_tool_policy(
            inputs.view,
            workspace,
            runtime=inputs.control_runtime,
            candidate_destination=str(inputs.candidate),
            bwrap_binary=inputs.sandbox_binary,
        )
    )
    # Match the actual author control transport. Its model client owns network
    # access; the fixed nested candidate profile denies network to every tool.
    policy.insert(policy.index("--unshare-all") + 1, "--share-net")
    policy += ["--bind", str(home), str(home), "--setenv", "CODEX_HOME", str(home)]
    members = tuple(path for path in inputs.candidate.rglob("*") if path.is_file())
    for member in members:
        policy += ["--ro-bind", str(member), str(inputs.candidate / member.relative_to(inputs.candidate))]
    dependencies = (
        Path(__file__),
        Path(CA.__file__),
        Path(invocation_record.__file__),
        Path(C.__file__),
        module_source_path("merlin.development.backend.optimization.components.experiment"),
        config,
        *members,
        *(path for path in inputs.view.root.rglob("*") if path.is_file()),
        *(row.source for row in inputs.control_runtime),
    )
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    for probe in inputs.readiness:
        command = [
            *policy,
            "--",
            inputs.codex_destination,
            "sandbox",
            "--permission-profile",
            CA._CANDIDATE_PERMISSION_PROFILE,
            "-C",
            str(inputs.candidate),
            "--",
            *probe.command,
        ]
        with invocation_record.observe(
            owner / probe.capability,
            stage="fresh_phase1_native_author_" + probe.capability,
            argv=command,
            inputs=(config, *members),
            dependencies=dependencies,
            cwd=inputs.candidate,
            env=environment,
        ) as observation:
            result = subprocess.run(
                command, cwd=inputs.candidate, env=environment, timeout=45, capture_output=True, check=False
            )
            observation.complete(result)
        if result.returncode or C.sha256_file(observation.directory / "stdout.bin") != probe.stdout_sha256:
            raise C.StageGateError("fresh Phase 1 native author tool readiness failed: " + probe.capability)
        invocation_record.require_environment(observation.path, environment=environment)
        verify_component_view(inputs.view)
        directories = tuple(
            sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_dir())
        )
        source_directories = tuple(
            sorted(
                path.relative_to(inputs.candidate).as_posix() for path in inputs.candidate.rglob("*") if path.is_dir()
            )
        )
        if (
            C.exact_tree_record(workspace)["sha256"] != original["sha256"]
            or C.exact_tree_record(inputs.candidate) != original
            or directories != original_directories
            or source_directories != original_directories
        ):
            raise C.StageGateError("fresh Phase 1 readiness changed its original inert source membership")


def probe_shared_tools(inputs, policy, *, compiler_transport):
    if compiler_transport is not None:
        if type(compiler_transport) is not PreparedContainerTransport:
            raise C.StageGateError("shared-tool readiness requires the actual prepared compiler command transport")
        compiler_transport.verify()
    probes = inputs.output / "readiness"
    probes.mkdir(mode=0o700)
    for probe in inputs.readiness:
        dependencies = tuple(row.source for row in inputs.runtime)
        if compiler_transport is None:
            with invocation_record.observe(
                probes / probe.capability,
                stage="fresh_phase1_" + probe.capability,
                argv=[*policy, "--", *probe.command],
                dependencies=dependencies,
            ) as observation:
                result = subprocess.run([*policy, "--", *probe.command], capture_output=True, timeout=45)
                observation.complete(result)
        else:
            mounts = mounts_from_strict_policy(policy)
            with invocation_record.observe_call(
                probes / probe.capability,
                stage="fresh_phase1_" + probe.capability,
                function=compiler_transport.execute,
                arguments={"command": probe.command, "transport_sha256": compiler_transport.sha256},
                dependencies=dependencies,
            ) as observation:
                result = compiler_transport.execute(
                    mounts=mounts,
                    command=probe.command,
                    cwd=str(inputs.candidate),
                    evidence_root=probes / probe.capability / "container-command",
                    timeout_s=45,
                )
                observation.returned(stdout=result.stdout, stderr=result.stderr)
        if result.returncode or C.sha256_file(observation.directory / "stdout.bin") != probe.stdout_sha256:
            raise C.StageGateError("fresh Phase 1 admitted tool readiness failed: " + probe.capability)
