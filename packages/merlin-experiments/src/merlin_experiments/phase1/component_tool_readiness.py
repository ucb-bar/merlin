"""Original shared-tool probes through an explicitly selected command owner.

Tool readiness does not qualify compiler semantics, author isolation or runtime.
The fresh input owner supplies its already admitted exact transport and policy.
"""

import shutil
import subprocess
from pathlib import Path

from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin_experiments.execution.container_transport import PreparedContainerTransport, mounts_from_strict_policy
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.component_experiment import strict_tool_policy, verify_component_view


def probe_native_author_tools(inputs):
    """Observe the same admitted commands inside the actual native tool profile.

    Called by the fresh lifecycle only after its independent input/runtime
    admission. This config-only home receives no control credential and no
    model request. Successful observations confer no origin or correctness.
    """
    from .providers import codex_agent as CA

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
        module_source_path("merlin_experiments.phase2.component_experiment"),
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
