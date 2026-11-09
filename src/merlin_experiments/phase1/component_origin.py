"""Fresh compiler origin issued only around actual isolated Phase 1 authoring.

The initial OOT contains an inert CLI and a manifest, with no dialect, lowering,
schedule or instruction implementation. Receipts describe an observation; reading
one cannot manufacture its private in-process authority. Historical submissions
and the handwritten reference never qualify as this experiment's Phase 1 output.
"""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path

from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_library import CompilerLibraryContract
from merlin.targetgen.generalization_prompt import append_general_compiler_contract
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.component_experiment import (
    ComponentView,
    RuntimeGrant,
    strict_tool_policy,
    verify_component_view,
)

from .component_compile_admission import verify_compile_roster
from .component_generation_admission import verify_generation_inputs
from .component_package_execution import selected_compiler_transport
from .component_tool_readiness import probe_native_author_tools, probe_shared_tools

_ISSUED: dict[object, tuple] = {}
_DRIVER = '''"""Structure-only OOT entrypoint. Author the compiler in this package."""
import sys

def main():
    raise NotImplementedError("Fresh scaffold has no dialect or operation lowering")

if __name__ == "__main__":
    main()
'''


def _plain(path: Path) -> Path:
    path = Path(path).absolute()
    if any(member.is_symlink() for member in (path, *path.parents)) or path.resolve() != path:
        raise C.StageGateError("fresh compiler origin refuses indirect paths")
    return path


def _tree(path: Path) -> dict:
    path = _plain(path)
    if not path.is_dir():
        raise C.StageGateError("fresh compiler origin source directory is absent")
    if any(member.is_symlink() or not (member.is_file() or member.is_dir()) for member in path.rglob("*")):
        raise C.StageGateError("fresh compiler origin source contains indirect or special members")
    return C.exact_tree_record(path)


def _readonly(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        _plain(path)
        path.chmod(path.stat().st_mode & ~0o222)


def _authority_identity(authority) -> tuple:
    return tuple(getattr(authority, item.name) for item in fields(authority) if item.name != "_issuer")


def inert_scaffold(target: str) -> dict[str, str]:
    """The complete, target-independent initial implementation, as exact bytes."""
    if not isinstance(target, str) or not target or not all(c.isalnum() or c in "_-" for c in target):
        raise C.StageGateError("fresh scaffold needs a canonical descriptor target")
    commands = {
        name: {"argv": ["python3", "driver.py", name, "{input_mlir}"]}
        for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
    }
    commands["emit_command_buffer"]["argv"].append("{output_json}")
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": target,
        "language": "python",
        "authoring": {"mode": "agent_generated_from_rtl_facts"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "driver.py"},
        "commands": commands,
    }
    # JSON is valid YAML and avoids a candidate-controlled template or serializer.
    return {"manifest.yaml": json.dumps(manifest, sort_keys=True, indent=2) + "\n", "driver.py": _DRIVER}


@dataclass(frozen=True)
class FreshToolProbe:
    capability: str
    command: tuple[str, ...]
    stdout_sha256: str

    def verify(self) -> None:
        from merlin.common.digest import is_sha256

        if (
            self.capability not in {"translator", "host_compiler", "linker", "functional_simulator"}
            or not isinstance(self.command, tuple)
            or not self.command
            or any(not isinstance(value, str) or not value or "\0" in value for value in self.command)
            or not is_sha256(self.stdout_sha256)
        ):
            raise C.StageGateError("fresh Phase 1 tools need exact independent shared-tool probes")


def _verify_author_tool_containment(*, runtime, control_runtime, readiness) -> None:
    """Join admitted tool declarations to the actual author mount closure.

    Readiness in a separate compiler transport does not make a tool reachable
    inside the author control process. This membership check issues no runtime,
    author isolation, compiler correctness or physical execution authority.
    """
    indexes = []
    for rows in (runtime, control_runtime):
        if not isinstance(rows, tuple) or any(type(row) is not RuntimeGrant for row in rows):
            raise C.StageGateError("fresh author tools require exact immutable runtime grants")
        index = {}
        for row in rows:
            row.verify()
            if row.destination in index:
                raise C.StageGateError("fresh author tools contain duplicate runtime destinations")
            index[row.destination] = row
        indexes.append(index)
    tools, control = indexes
    if any(control.get(destination) != row for destination, row in tools.items()):
        raise C.StageGateError("fresh author control closure omits or changes an admitted tool grant")
    if not isinstance(readiness, tuple) or any(type(row) is not FreshToolProbe for row in readiness):
        raise C.StageGateError("fresh author readiness requires exact tool probes")
    for probe in readiness:
        probe.verify()
        if probe.command[0] not in tools:
            raise C.StageGateError("fresh author readiness executable is outside its admitted tool grants")


@dataclass(frozen=True)
class FreshPhase1Inputs:
    hardware: object
    software: object
    execution_support: object
    library: CompilerLibraryContract
    view: ComponentView
    corpus_root: Path
    target_experiment: object
    candidate: Path
    output: Path
    runtime: tuple[RuntimeGrant, ...]
    control_runtime: tuple[RuntimeGrant, ...]
    readiness: tuple
    codex_binary: Path
    codex_destination: str
    auth_source: Path
    compile_roster: object = None
    pointer_storage: object = None
    author_sandbox: RuntimeGrant | None = None
    source_preparation: object = None

    @property
    def compiler_transport(self):
        return selected_compiler_transport(self.execution_support, view=self.view, runtime=self.runtime)

    @property
    def sandbox_binary(self) -> Path:
        if type(self.author_sandbox) is not RuntimeGrant:
            raise C.StageGateError("fresh Phase 1 needs an explicitly pinned outer author sandbox")
        self.author_sandbox.verify()
        matches = [row for row in self.control_runtime if row.destination == self.author_sandbox.destination]
        if matches != [self.author_sandbox]:
            raise C.StageGateError("fresh Phase 1 outer sandbox differs from its declared control closure")
        return self.author_sandbox.source

    def verify(self) -> dict:
        from merlin.targetgen.target_experiment import TargetExperiment

        try:
            from merlin_experiments.phase0.rtl_intake import IndependentHardwareIntake
            from merlin_experiments.phase0.software_intake import IndependentSoftwareIntake
            from merlin_experiments.phase2.component_runtime import IndependentComponentRuntime
        except ImportError as error:
            raise C.StageGateError("fresh Phase 1 independent hardware/runtime authority is unavailable") from error

        if type(self.hardware) is not IndependentHardwareIntake:
            raise C.StageGateError("fresh Phase 1 requires independently issued RTL intake authority")
        self.hardware.verify()
        if type(self.software) is not IndependentSoftwareIntake or self.software.hardware is not self.hardware:
            raise C.StageGateError("fresh Phase 1 requires matching independently reviewed minimal software authority")
        self.software.verify()
        if type(self.target_experiment) is not TargetExperiment:
            raise C.StageGateError("fresh Phase 1 requires the ordinary parsed target descriptor")
        if type(self.execution_support) is not IndependentComponentRuntime:
            raise C.StageGateError("fresh Phase 1 requires independently issued target runtime support")
        self.execution_support.verify(required_roles=("grade", "stage_verifier"))
        if (
            self.execution_support.hardware_intake is not self.hardware
            or self.execution_support.target_descriptor != Path(self.target_experiment.path).resolve()
        ):
            raise C.StageGateError("fresh Phase 1 runtime differs from its independent hardware/target intake")
        compile_roster_sha = verify_compile_roster(
            self.compile_roster, hardware=self.hardware, software=self.software, descriptor=self.target_experiment.path
        )
        if self.hardware.target != self.target_experiment.target:
            raise C.StageGateError("fresh Phase 1 target differs from its independent hardware intake")
        if type(self.library) is not CompilerLibraryContract or self.library.sha256 != self.view.library_sha256:
            raise C.StageGateError("fresh Phase 1 compiler library differs from its reviewed public grant")
        self.library.verify(self.view.root / "compiler")
        context = self.execution_support.qualification.context
        if context.compiler_library is not self.library or context.compiler_library_root != self.view.root / "compiler":
            raise C.StageGateError("fresh Phase 1 public library differs from the qualified grader selection")
        manifest = verify_component_view(self.view)
        self.hardware.verify_public_fact_view(self.view)
        self.software.verify_public_fact_view(self.view)
        from .component_pointer_storage import verify_pointer_storage

        pointer_storage_sha = verify_pointer_storage(
            self.pointer_storage,
            hardware=self.hardware,
            software=self.software,
            roster=self.compile_roster,
            view=self.view,
        )
        report = verify_generation_inputs(
            self.corpus_root, preparation=self.source_preparation, hardware=self.hardware, software=self.software
        )
        if (
            report["sha256"] != self.view.generation_sha256
            or report.get("hardware_intake_sha256") != self.hardware.sha256
        ):
            raise C.StageGateError("fresh Phase 1 coverage lacks independent hardware provenance")
        if report.get("software_intake_sha256") != self.software.sha256:
            raise C.StageGateError("fresh Phase 1 coverage lacks independent minimal software provenance")
        candidate, output = _plain(self.candidate), _plain(self.output)
        if any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in (candidate, self.view.root, _plain(self.corpus_root))
        ):
            raise C.StageGateError("fresh Phase 1 private records overlap an author or corpus grant")
        for rows in (self.runtime, self.control_runtime):
            if (
                not isinstance(rows, tuple)
                or not rows
                or any(type(row) is not RuntimeGrant for row in rows)
                or len({row.destination for row in rows}) != len(rows)
            ):
                raise C.StageGateError("fresh Phase 1 needs complete exact runtime memberships")
            for row in rows:
                row.verify()
            if any(row.source == self.auth_source or row.destination.endswith("/auth.json") for row in rows):
                raise C.StageGateError("fresh Phase 1 runtime cannot grant a credential")
        _plain(self.auth_source)
        if not self.auth_source.is_file():
            raise C.StageGateError("fresh Phase 1 control credential is unavailable")
        selected = [row for row in self.control_runtime if row.destination == self.codex_destination]
        if len(selected) != 1 or selected[0].source != self.codex_binary:
            raise C.StageGateError("fresh Phase 1 control client is outside its frozen runtime")
        if (
            not isinstance(self.readiness, tuple)
            or any(type(row) is not FreshToolProbe for row in self.readiness)
            or len({row.capability for row in self.readiness}) != len(self.readiness)
            or {row.capability for row in self.readiness}
            != {"translator", "host_compiler", "linker", "functional_simulator"}
        ):
            raise C.StageGateError(
                "fresh Phase 1 requires shared translation/compiler/linker/simulator readiness probes"
            )
        for row in self.readiness:
            row.verify()
        _verify_author_tool_containment(
            runtime=self.runtime, control_runtime=self.control_runtime, readiness=self.readiness
        )
        self.sandbox_binary
        binding = {
            "hardware_intake_sha256": self.hardware.sha256,
            "software_intake_sha256": self.software.sha256,
            "runtime_authority_sha256": self.execution_support.sha256,
            "compiler_transport_sha256": self.compiler_transport.sha256
            if self.compiler_transport is not None
            else None,
            "compile_source_roster_sha256": compile_roster_sha,
            "original_pointer_storage_sha256": pointer_storage_sha,
            "view_sha256": self.view.manifest_sha256,
            "coverage_sha256": report["sha256"],
            "coverage_schema": report["schema"],
            "execution_budget_sha256": report["generation_identity"]["execution_budget_sha256"],
            "execution_admission_sha256": report["generation_identity"]["execution_admission_sha256"],
            "library_sha256": self.library.sha256,
            "target_descriptor_sha256": C.sha256_file(self.target_experiment.path),
            "public_members": manifest["members"],
            "runtime": [{"destination": row.destination, "sha256": row.sha256} for row in self.runtime],
            "control_runtime": [{"destination": row.destination, "sha256": row.sha256} for row in self.control_runtime],
            "author_sandbox": {
                "source": str(self.author_sandbox.source),
                "destination": self.author_sandbox.destination,
                "sha256": self.author_sandbox.sha256,
            },
        }
        if self.source_preparation is not None:
            binding["source_preparation"] = {
                "sha256": self.source_preparation.sha256,
                "schema": self.source_preparation.record()["schema"],
            }
        return binding


@dataclass(frozen=True)
class FreshCompilerOrigin:
    inputs: FreshPhase1Inputs
    candidate: Path
    candidate_sha256: str
    initial_scaffold: Path
    initial_scaffold_sha256: str
    receipt: Path
    receipt_sha256: str
    transcript: Path
    transcript_sha256: str
    binding_json: str
    evidence_pins: tuple[tuple[Path, str], ...]
    _issuer: object = field(repr=False, compare=False)

    def verify(self, *, candidate: Path | None = None) -> dict:
        if type(self._issuer) is not object or self._issuer not in _ISSUED:
            raise C.StageGateError("compiler origin was not issued around actual fresh Phase 1 authoring")
        if _authority_identity(self) != _ISSUED[self._issuer]:
            raise C.StageGateError("fresh Phase 1 origin authority fields changed after issuance")
        if C.canonical_json(self.inputs.verify()).decode() != self.binding_json:
            raise C.StageGateError("fresh Phase 1 input authority changed")
        for path, sha256 in (
            *self.evidence_pins,
            (self.receipt, self.receipt_sha256),
            (self.transcript, self.transcript_sha256),
        ):
            _plain(path)
            if C.sha256_file(path) != sha256:
                raise C.StageGateError("fresh Phase 1 authoring evidence changed")
            if path.name == "invocation.json":
                invocation_record.verify(path)
        if _tree(self.initial_scaffold)["sha256"] != self.initial_scaffold_sha256:
            raise C.StageGateError("fresh Phase 1 initial scaffold changed")
        expected = inert_scaffold(self.inputs.target_experiment.target)
        actual = {
            path.relative_to(self.initial_scaffold).as_posix(): path.read_text()
            for path in self.initial_scaffold.rglob("*")
            if path.is_file()
        }
        if actual != expected:
            raise C.StageGateError("fresh Phase 1 initial scaffold contained a compiler implementation")
        selected = self.candidate if candidate is None else candidate
        if _tree(selected)["sha256"] != self.candidate_sha256:
            raise C.StageGateError("compiler bytes differ from the actual fresh Phase 1 output")
        if _tree(self.candidate)["sha256"] != self.candidate_sha256:
            raise C.StageGateError("frozen fresh Phase 1 output changed")
        return C.mapping_file(self.receipt)


def _runtime_binds(inputs: FreshPhase1Inputs, home: Path) -> list[str]:
    if not _plain(home).is_relative_to(inputs.output / "codex_homes"):
        raise C.StageGateError("fresh Phase 1 control home escaped its private owner")
    return [
        "--bind",
        str(home),
        str(home),
        "--bind",
        str(inputs.auth_source),
        str(home / "auth.json"),
        "--setenv",
        "CODEX_HOME",
        str(home),
    ]


def _control_command(inputs: FreshPhase1Inputs, inner: str, workspace: Path, _bundle: dict, *, extra_binds=None) -> str:
    inputs.verify()
    if workspace != inputs.candidate or not isinstance(extra_binds, list) or len(extra_binds) != 9:
        raise C.StageGateError("fresh Phase 1 control mount differs from its exact workspace/home")
    if extra_binds != _runtime_binds(inputs, Path(extra_binds[1])):
        raise C.StageGateError("fresh Phase 1 control contains unreviewed bindings")
    policy = list(
        strict_tool_policy(
            inputs.view,
            inputs.candidate,
            runtime=inputs.control_runtime,
            candidate_destination=str(inputs.candidate),
            bwrap_binary=inputs.sandbox_binary,
        )
    )
    policy.insert(policy.index("--unshare-all") + 1, "--share-net")
    policy += extra_binds
    selected = inner.replace(shlex.quote(str(inputs.codex_binary)), shlex.quote(inputs.codex_destination))
    command = [*policy, "--", "/bin/sh", "-c", selected]
    if len(shlex.join(command).encode()) <= 60000:
        return shlex.join(command)
    payload = b"\0".join(token.encode() for token in command[1:]) + b"\0"
    owner = inputs.output / "control_argv"
    owner.mkdir(mode=0o700, exist_ok=True)
    path = owner / (C.document_sha256({"argv": command[1:]}) + ".args")
    if not path.exists():
        path.write_bytes(payload)
        path.chmod(0o400)
    elif path.is_symlink() or path.read_bytes() != payload:
        raise C.StageGateError("fresh Phase 1 control argv changed")
    return "exec 3< " + shlex.quote(str(path)) + "; exec " + shlex.join([command[0], "--args", "3"])


def _probe_shared_tools(inputs: FreshPhase1Inputs, policy: tuple[str, ...]) -> None:
    probe_shared_tools(inputs, policy, compiler_transport=inputs.compiler_transport)


def render_fresh_phase1_prompt() -> str:
    """Serve shared compiler requirements without granting extra experiment inputs."""
    return append_general_compiler_contract(
        "Build an xDSL OOT compiler from scratch in this workspace. The initial driver has no lowering. "
        "Use only the admitted independent hardware facts, software semantics, generated development inputs "
        "and reviewed generic compiler APIs under /component-inputs. Implement operation semantics, shapes, "
        "layouts, tails, complete outputs, host/device ownership and synchronization. Derive target ISA facts "
        "from the admitted hardware evidence; refuse unknown facts. Obey the admitted instruction policy. "
        "No prior compiler, reference implementation, validation graph, previous run, model-specific "
        "schedule or answer is available or permitted. Optimize later in Phase 2 after required correctness."
    )


def run_fresh_component_phase1(
    inputs: FreshPhase1Inputs, *, model: str, effort: str, wall_budget_seconds: int
) -> FreshCompilerOrigin:
    """Author from the fixed inert scaffold through the ordinary isolated provider.

    This issues origin, not correctness. The ordinary independent full-domain
    grader must still certify every required member before Phase 2 admission.
    """
    from merlin_experiments.phase1.providers import codex_agent as CA

    if type(inputs) is not FreshPhase1Inputs or type(wall_budget_seconds) is not int or wall_budget_seconds <= 0:
        raise C.StageGateError("fresh Phase 1 requires typed inputs and a positive authoring budget")
    binding = inputs.verify()
    if any(path.exists() or path.is_symlink() for path in (inputs.candidate, inputs.output)):
        raise C.StageGateError("fresh Phase 1 refuses a preexisting compiler, output or session")
    inputs.output.mkdir(parents=True, mode=0o700)
    inputs.candidate.mkdir(parents=True, mode=0o700)
    for relative, source in inert_scaffold(inputs.target_experiment.target).items():
        (inputs.candidate / relative).write_text(source, encoding="utf-8")
    initial = inputs.output / "initial_scaffold"
    shutil.copytree(inputs.candidate, initial)
    _readonly(initial)
    initial_hash = _tree(initial)["sha256"]
    policy = strict_tool_policy(
        inputs.view,
        inputs.candidate,
        runtime=inputs.runtime,
        candidate_destination=str(inputs.candidate),
        bwrap_binary=inputs.sandbox_binary,
        candidate_writable=False,
    )
    _probe_shared_tools(inputs, policy)
    probe_native_author_tools(inputs)
    if _tree(inputs.candidate)["sha256"] != initial_hash:
        raise C.StageGateError("fresh Phase 1 readiness changed the inert initial scaffold")
    prompt = render_fresh_phase1_prompt()
    prompt_path = inputs.output / "author_prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    if (
        C.canonical_json(inputs.verify()) != C.canonical_json(binding)
        or _tree(initial)["sha256"] != initial_hash
        or _tree(inputs.candidate)["sha256"] != initial_hash
    ):
        raise C.StageGateError("fresh Phase 1 inputs or inert scaffold changed before authoring")
    with invocation_record.observe_call(
        inputs.output / "author",
        stage="fresh_phase1_authoring",
        function=CA.run_round,
        arguments={"model": model, "effort": effort, "wall_budget_seconds": wall_budget_seconds},
        inputs=(prompt_path, *(initial / relative for relative in inert_scaffold(inputs.target_experiment.target))),
        dependencies=(
            Path(__file__),
            module_source_path("merlin.targetgen.generalization_prompt"),
            *tuple(row.source for row in inputs.control_runtime),
        ),
    ) as observation:
        rc, transcript = CA.run_round(
            inputs.candidate,
            inputs.output,
            model,
            {},
            inputs.target_experiment,
            "bwrap",
            0,
            wall_budget_seconds,
            effort=effort,
            prompt=prompt,
            effective_model=model,
            continue_session=True,
            continuation_prompt=prompt,
            sandbox_command=lambda inner, ws, bundle, extra_binds=None: _control_command(
                inputs, inner, ws, bundle, extra_binds=extra_binds
            ),
            codex_binary=inputs.codex_binary,
            codex_home_root=inputs.output / "codex_homes",
            candidate_read_paths=("/component-inputs", *(row.destination for row in inputs.control_runtime)),
            runtime_binds=lambda home: _runtime_binds(inputs, home),
            require_fresh_home=True,
        )
        observation.returned(stdout=C.canonical_json({"exit_code": rc, "transcript": str(transcript)}))
    summary_path = inputs.output / "rounds" / "round_00.codex_summary.json"
    summary = C.mapping_file(summary_path)
    if (
        rc not in (0, C.ROUND_DEADLINE_EXIT)
        or summary.get("exit_code") != rc
        or not isinstance(summary.get("thread_id"), str)
        or not summary["thread_id"]
        or type(summary.get("turns_started")) is not int
        or summary["turns_started"] <= 0
        or summary.get("unknown_types")
        or summary.get("unrecovered_errors")
    ):
        raise C.StageGateError("fresh Phase 1 author transport did not complete an admitted session")
    if C.canonical_json(inputs.verify()) != C.canonical_json(binding):
        raise C.StageGateError("fresh Phase 1 input membership changed during authoring")
    if _tree(inputs.candidate)["sha256"] == initial_hash:
        raise C.StageGateError("fresh Phase 1 produced no compiler changes")
    frozen = inputs.output / "compiler"
    shutil.copytree(inputs.candidate, frozen)
    _readonly(frozen)
    candidate_hash = _tree(frozen)["sha256"]
    pins = tuple((path, C.sha256_file(path)) for path in sorted(inputs.output.rglob("invocation.json")))
    pins += ((summary_path, C.sha256_file(summary_path)),)
    receipt = inputs.output / "compiler_origin.json"
    C.write_json(
        receipt,
        {
            "schema": "merlin.fresh_compiler_origin.v1",
            "status": "fresh_origin_observed",
            "candidate_sha256": candidate_hash,
            "initial_scaffold_sha256": initial_hash,
            "bindings": binding,
            "transcript_sha256": C.sha256_file(transcript),
            "author_invocations": [{"path": str(path), "sha256": sha} for path, sha in pins],
            "scope": (
                "actual fresh Phase 1 origin only; full-domain compiler correctness and performance remain unqualified"
            ),
        },
    )
    receipt.chmod(0o400)
    origin = FreshCompilerOrigin(
        inputs,
        frozen,
        candidate_hash,
        initial,
        initial_hash,
        receipt,
        C.sha256_file(receipt),
        transcript,
        C.sha256_file(transcript),
        C.canonical_json(binding).decode(),
        pins,
        object(),
    )
    _ISSUED[origin._issuer] = _authority_identity(origin)
    origin.verify(candidate=inputs.candidate)
    return origin
