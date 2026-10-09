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

from merlin.common import invocation_record
from merlin.targetgen import package_runtime as P
from merlin_experiments.execution.container_transport import PreparedContainerTransport, mounts_from_strict_policy
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.component_experiment import ComponentView, RuntimeGrant, strict_tool_policy


def selected_compiler_transport(runtime_authority, *, view, runtime):
    """Consume the exact command selection of already qualified runtime controls.

    Serialized selections, callable import strings and available local services
    cannot choose the compiler sandbox. Runtime qualification is still mandatory.
    """
    from merlin_experiments.phase2.component_runtime_authority import IndependentComponentRuntime
    from merlin_experiments.phase2.component_runtime_support import PreparedIndependentRuntimeContext

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

    def build_package(self, pkg, *, timeout=1800):
        if pkg.directory.resolve() != self.candidate:
            raise C.StageGateError("component package build escaped its immutable compiler owner")
        build = pkg.manifest.get("build") or {}
        if any(build.get(name) for name in ("configure", "command")):
            raise C.StageGateError(
                "component qualification requires an independently boxed build service for declared build steps"
            )

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
            host_output = Path(staging) / "command_buffer.json"
            argv = P._resolve_argv(
                pkg,
                name,
                Path("/evaluation-input/interface.mlir"),
                Path("/evaluation-output/command_buffer.json") if output_json is not None else None,
            )
            if P._needs_interpreter(pkg, argv):
                interpreters = [row for row in self.runtime if row.source.resolve() == Path(sys.executable).resolve()]
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
                "/evaluation-output",
            ]
            dependencies = (
                *tuple(path for path in self.candidate.rglob("*") if path.is_file()),
                *tuple(row.source for row in self.runtime),
                Path(__file__),
            )
            if self.container_transport is not None:
                if type(self.container_transport) is not PreparedContainerTransport:
                    raise C.StageGateError(
                        "component package requires an explicitly prepared original container transport"
                    )
                self.container_transport.verify()
                mounts = mounts_from_strict_policy(tuple(policy))
                owned_evidence = Path(invocation_directory) / ("container-command-" + uuid.uuid4().hex)
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
                    )
                    result = subprocess.CompletedProcess(
                        actual.args, actual.returncode, actual.stdout.decode("utf-8"), actual.stderr.decode("utf-8")
                    )
                    if host_output.is_symlink():
                        raise C.StageGateError("component compiler output staging contains a linked product")
                    if output_json is not None and host_output.is_file():
                        shutil.copyfile(host_output, output_json)
                    record.returned(stdout=result.stdout, stderr=result.stderr)
                    return result
            with invocation_record.observe(
                invocation_directory,
                stage=name,
                argv=[*policy, "--", *argv],
                inputs=(input_mlir,),
                outputs=(output_json,) if output_json is not None else (),
                dependencies=dependencies,
            ) as record:
                result = subprocess.run([*policy, "--", *argv], capture_output=True, text=True, timeout=timeout)
                if host_output.is_symlink():
                    raise C.StageGateError("component compiler output staging contains a linked product")
                if output_json is not None and host_output.is_file():
                    shutil.copyfile(host_output, output_json)
                record.complete(result)
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
    executor = ComponentPackageExecutor(
        candidate.resolve(), view, runtime, evidence_root.resolve(), container_transport=container_transport
    )
    with P.scoped_package_executor(executor):
        yield
