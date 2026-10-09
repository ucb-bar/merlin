"""Fresh component authoring through the existing provider and broker lifecycle.

The control client has explicit runtime files and one fresh credential-isolated
home. Candidate commands have a native permission profile with no network and
reach evaluated feedback over the ordinary broker through a Unix socket.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_bytes
from merlin_experiments.phase1.component_qualification import ComponentQualification
from merlin_experiments.phase1.providers import codex_agent as CA

from . import broker as B
from . import contracts as C
from .component_experiment import ComponentView, RuntimeGrant, strict_tool_policy, verify_component_view
from .component_launch_probe import probe_private_denial
from .component_workflow import ComponentOnlyPolicy
from .edit_authority import FrozenEditAuthority

_ISSUER = object()


def public_component_manifest(corpus) -> bytes:
    """Disclose only admitted development commitments and program identities."""
    return C.canonical_json(
        {
            "schema": "merlin.component_development_view.v1",
            "manifest_sha256": corpus.manifest_sha256,
            "capsules_sha256": corpus.capsules_sha256,
            "members": [
                {"family": row.family, "capsule": row.capsule, "sha256": row.source_sha256} for row in corpus.capsules
            ],
        }
    )


@dataclass(frozen=True)
class ComponentReadinessProbe:
    capability: str
    command: tuple[str, ...]
    stdout_sha256: str

    def verify(self) -> None:
        if self.capability not in {"compiler", "linker", "simulator", "isa", "cca"}:
            raise C.StageGateError("component readiness needs an explicit declared capability")
        if (
            not isinstance(self.command, tuple)
            or not self.command
            or any(not isinstance(value, str) or not value or "\0" in value for value in self.command)
            or not is_sha256(self.stdout_sha256)
        ):
            raise C.StageGateError("component readiness requires an exact command and expected output identity")


@dataclass(frozen=True)
class ComponentLaunchInputs:
    view: ComponentView
    candidate: Path
    policy: ComponentOnlyPolicy
    edit_authority: FrozenEditAuthority
    qualification: ComponentQualification
    runtime: tuple[RuntimeGrant, ...]
    control_runtime: tuple[RuntimeGrant, ...]
    readiness: tuple[ComponentReadinessProbe, ...]
    codex_binary: Path
    codex_destination: str
    auth_source: Path
    stage_root: Path
    price_table: Path

    @property
    def public_control_dir(self) -> Path:
        return self.stage_root / ("public-" + self.policy.receipt_path.parent.name)

    @property
    def socket_path(self) -> Path:
        return self.public_control_dir / "channel.sock"

    def verify_ipc_path(self) -> None:
        # This launch uses Linux bwrap. sockaddr_un.sun_path has 108 bytes,
        # including its required terminating NUL; count actual filesystem bytes.
        if not self.socket_path.is_absolute() or len(os.fsencode(self.socket_path)) > 107:
            raise C.StageGateError(
                "component Unix IPC socket path exceeds Linux AF_UNIX budget; "
                "select an explicitly short private stage_root"
            )

    @property
    def sandbox_binary(self) -> Path:
        from merlin_experiments.phase1.component_origin import FreshCompilerOrigin

        origin = self.qualification.compiler_origin
        if type(origin) is not FreshCompilerOrigin:
            raise C.StageGateError("component outer sandbox requires its exact fresh Phase 1 origin")
        origin.verify(candidate=self.edit_authority.seed)
        selected = origin.inputs.author_sandbox
        if type(selected) is not RuntimeGrant:
            raise C.StageGateError("component launch lacks the original explicitly pinned outer sandbox")
        selected.verify()
        matches = [row for row in self.control_runtime if row.destination == selected.destination]
        if self.control_runtime != origin.inputs.control_runtime or matches != [selected]:
            raise C.StageGateError("component outer sandbox differs from its exact fresh authoring closure")
        return selected.source

    def verify(self, *, require_baseline: bool = False) -> None:
        self.verify_ipc_path()
        if type(self.policy) is not ComponentOnlyPolicy or type(self.qualification) is not ComponentQualification:
            raise C.StageGateError("component launch requires evaluated typed domain and component policy owners")
        if type(self.edit_authority) is not FrozenEditAuthority or not self.edit_authority.configured:
            raise C.StageGateError("component launch requires frozen compiler edit authority")
        if self.candidate != self.policy.candidate or self.edit_authority.initial_source != self.candidate.resolve():
            raise C.StageGateError("component launch owners disagree on the exact candidate")
        if self.policy.receipt_path.parent.parent != self.stage_root:
            raise C.StageGateError("component feedback receipts must have one stage-owned private control directory")
        if self.policy.target_experiment.path.resolve() != self.qualification.target_descriptor:
            raise C.StageGateError("component launch qualification targets another descriptor")
        if self.view != self.qualification.view or self.runtime != self.qualification.runtime:
            raise C.StageGateError("component authoring differs from its qualified public library/runtime closure")
        manifest = verify_component_view(self.view)
        if self.view.generation_sha256 != self.qualification.coverage_sha256:
            raise C.StageGateError("component public view is not bound to the independently admitted coverage")
        public_manifest = [
            row for row in manifest["members"] if row["path"] == "contract/performance_corpus_manifest.json"
        ]
        public_bytes = public_component_manifest(self.policy.component_corpus)
        if len(public_manifest) != 1 or public_manifest[0]["sha256"] != sha256_bytes(public_bytes):
            raise C.StageGateError("component public manifest differs from the exact development corpus")
        if not any(row["role"] == "contract" for row in manifest["members"]):
            raise C.StageGateError("component launch has no reviewed public contracts")
        self.edit_authority.check_integrity()
        self.edit_authority.validate_candidate(self.candidate)
        self.qualification.verify(candidate=self.edit_authority.seed)
        if require_baseline:
            self.qualification.verify(candidate=self.candidate)
        self.policy._validate()
        if not (self.policy.component_analytical or self.policy.component_cca or self.policy.component_rtl):
            raise C.StageGateError("component authoring requires an evaluated generated-component feedback provider")
        if self.stage_root.resolve().is_relative_to(
            self.candidate.resolve()
        ) or self.stage_root.resolve().is_relative_to(self.view.root):
            raise C.StageGateError("private component stage overlaps an author grant")
        if (
            not isinstance(self.runtime, tuple)
            or not self.runtime
            or not isinstance(self.control_runtime, tuple)
            or not self.control_runtime
            or not isinstance(self.readiness, tuple)
            or any(type(row) is not RuntimeGrant for row in (*self.runtime, *self.control_runtime))
        ):
            raise C.StageGateError("component launch requires exact reviewed runtime file membership")
        self.sandbox_binary
        for grant in (*self.runtime, *self.control_runtime):
            grant.verify()
        if any(
            row.source == self.auth_source or row.destination.endswith("/auth.json")
            for row in (*self.runtime, *self.control_runtime)
        ):
            raise C.StageGateError("candidate/runtime grant cannot contain a control credential")
        if self.auth_source.is_symlink() or not self.auth_source.is_file():
            raise C.StageGateError("component control credential source is absent or linked")
        selections = [row for row in self.control_runtime if row.destination == self.codex_destination]
        if len(selections) != 1 or selections[0].source.resolve() != self.codex_binary.resolve():
            raise C.StageGateError("component control executable is not in the exact runtime closure")
        capabilities = {row.capability for row in self.readiness}
        if capabilities != {"compiler", "linker", "simulator", "isa", "cca"}:
            raise C.StageGateError("all granted compiler/linker/simulator/ISA/CCA tools need actual readiness probes")
        for probe in self.readiness:
            if type(probe) is not ComponentReadinessProbe:
                raise C.StageGateError("component readiness requires typed trusted probes")
            probe.verify()


@dataclass(frozen=True)
class ComponentToolPolicy:
    inputs: ComponentLaunchInputs
    argv: tuple[str, ...]
    network: str = "isolated_networkless"
    clear_environment: bool = True
    env_prefix: str | None = None
    process_cwd: Path | None = None

    def verify_execution(self) -> None:
        self.inputs.verify()
        expected = strict_tool_policy(
            self.inputs.view,
            self.inputs.candidate,
            runtime=self.inputs.runtime,
            candidate_destination=str(self.inputs.candidate),
            bwrap_binary=self.inputs.sandbox_binary,
        )
        if (
            self.argv != expected
            or self.network != "isolated_networkless"
            or not self.clear_environment
            or self.env_prefix
        ):
            raise C.StageGateError("component execution policy changed")


@dataclass(frozen=True)
class QualifiedComponentLaunch:
    inputs: ComponentLaunchInputs
    qualification_path: Path
    qualification_sha256: str
    selected_model: str
    selected_effort: str
    _issuer: object

    def verify(self) -> None:
        if self._issuer is not _ISSUER:
            raise C.StageGateError("component launch has no independent runtime qualification")
        self.inputs.verify()
        if C.sha256_file(self.qualification_path) != self.qualification_sha256:
            raise C.StageGateError("component launch qualification changed")
        document = C.mapping_file(self.qualification_path)
        if document.get("status") != "qualified" or document.get("bindings") != _bindings(self.inputs):
            raise C.StageGateError("component launch identity differs from evaluated isolation/readiness")
        if document.get("model") != self.selected_model or document.get("effort") != self.selected_effort:
            raise C.StageGateError("component model transport selection changed")


def _bindings(inputs: ComponentLaunchInputs) -> dict:
    return {
        "view_sha256": inputs.view.manifest_sha256,
        "generation_sha256": inputs.view.generation_sha256,
        "compiler_domain_sha256": inputs.qualification.receipt_sha256,
        "edit_authority_sha256": C.document_sha256(inputs.edit_authority.binding),
        "target_sha256": C.sha256_file(inputs.policy.target_experiment.path),
        "corpus_manifest_sha256": inputs.policy.component_corpus.manifest_sha256,
        "corpus_sha256": inputs.policy.component_corpus.capsules_sha256,
        "runtime": [{"destination": row.destination, "sha256": row.sha256} for row in inputs.runtime],
        "control_runtime": [{"destination": row.destination, "sha256": row.sha256} for row in inputs.control_runtime],
        "readiness": [
            {"capability": row.capability, "command": list(row.command), "stdout_sha256": row.stdout_sha256}
            for row in inputs.readiness
        ],
        "codex_destination": inputs.codex_destination,
        "price_table_sha256": C.sha256_file(inputs.price_table),
    }


def _runtime_binds(inputs: ComponentLaunchInputs, home: Path) -> list[str]:
    if home.resolve().is_relative_to(inputs.candidate) or home.resolve().is_relative_to(inputs.view.root):
        raise C.StageGateError("component control home overlaps candidate inputs")
    # The caller's credential file is bound only to the control home. Neither
    # strict_tool_policy nor the candidate permission profile admits that home.
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


def _control_command(inputs: ComponentLaunchInputs, inner: str, ws: Path, _bundle: dict, *, extra_binds=None) -> str:
    inputs.verify()
    if ws != inputs.candidate or not isinstance(extra_binds, list):
        raise C.StageGateError("component control command differs from its frozen workspace")
    # The native provider supplies the one explicit home from its isolated
    # per-round owner. Reject broader mounts or inherited provider defaults.
    if (
        len(extra_binds) != 9
        or extra_binds[:1] != ["--bind"]
        or extra_binds[3] != "--bind"
        or extra_binds[6:] != ["--setenv", "CODEX_HOME", extra_binds[1]]
    ):
        raise C.StageGateError("component control runtime contains unreviewed binds")
    home = Path(extra_binds[1])
    if extra_binds != _runtime_binds(inputs, home):
        raise C.StageGateError("component control credential binding changed")
    argv = list(
        strict_tool_policy(
            inputs.view,
            inputs.candidate,
            runtime=inputs.control_runtime,
            candidate_destination=str(inputs.candidate),
            bwrap_binary=inputs.sandbox_binary,
        )
    )
    # Only the model client's control namespace shares network. Candidate
    # commands use the independently probed mandatory networkless native policy.
    argv.insert(argv.index("--unshare-all") + 1, "--share-net")
    # Mount four individually admitted public IPC files. Private receipts and
    # detailed evaluator evidence remain outside the control namespace too.
    for name in ("perf_tool.py", ".perf_broker.json", "channel.sock", "receipts.jsonl"):
        source = inputs.public_control_dir / name
        if source.is_symlink() or not source.exists():
            raise C.StageGateError("component public broker IPC member is absent or linked")
        argv += ["--ro-bind", str(source), "/perf-control/" + name]
    argv += extra_binds
    selected = inner.replace(shlex.quote(str(inputs.codex_binary)), shlex.quote(inputs.codex_destination))
    command = [*argv, "--", "/bin/sh", "-c", selected]
    rendered = shlex.join(command)
    if len(rendered.encode("utf-8")) <= 60000:
        return rendered
    # A complete file closure can exceed Linux's per-argument limit when put
    # in a shell -c string. Bubblewrap reads its NUL-separated argv from a
    # private inherited descriptor; this file never enters the sandbox.
    payload = b"\0".join(token.encode("utf-8") for token in command[1:]) + b"\0"
    owner = inputs.stage_root / "control_argv"
    owner.mkdir(mode=0o700, exist_ok=True)
    path = owner / (sha256_bytes(payload) + ".args")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError:
        if path.is_symlink() or C.sha256_file(path) != sha256_bytes(payload):
            raise C.StageGateError("component control argv observation changed")
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    return "exec 3< " + shlex.quote(str(path)) + "; exec " + shlex.join([command[0], "--args", "3"])


def _read_paths(inputs: ComponentLaunchInputs) -> tuple[str, ...]:
    public_ipc = tuple(
        "/perf-control/" + name for name in ("perf_tool.py", ".perf_broker.json", "channel.sock", "receipts.jsonl")
    )
    return tuple(
        dict.fromkeys(("/component-inputs", *public_ipc, *(row.destination for row in inputs.control_runtime)))
    )


def qualify_component_launch(
    inputs: ComponentLaunchInputs,
    *,
    model: str,
    effort: str,
    timeout_s: int = 45,
) -> QualifiedComponentLaunch:
    """No-model probes of the same tool namespace and native control boundary."""
    inputs.verify(require_baseline=True)
    if type(timeout_s) is not int or not 0 < timeout_s <= 60 or not model.strip() or not effort.strip():
        raise C.StageGateError("component launch needs explicit model/effort and a bounded probe")
    if inputs.stage_root.exists() or inputs.stage_root.is_symlink():
        raise C.StageGateError("fresh component launch refuses existing stage/session state")
    inputs.stage_root.mkdir(parents=True, mode=0o700)
    tool = ComponentToolPolicy(
        inputs,
        strict_tool_policy(
            inputs.view,
            inputs.candidate,
            runtime=inputs.runtime,
            candidate_destination=str(inputs.candidate),
            bwrap_binary=inputs.sandbox_binary,
        ),
    )
    results = []
    try:
        for probe in inputs.readiness:
            tool.verify_execution()
            result = subprocess.run(
                [*tool.argv, "--", *probe.command], capture_output=True, timeout=timeout_s, check=False
            )
            passed = result.returncode == 0 and sha256_bytes(result.stdout) == probe.stdout_sha256
            results.append(
                {
                    "capability": probe.capability,
                    "returncode": result.returncode,
                    "stdout_sha256": sha256_bytes(result.stdout),
                    "stderr_sha256": sha256_bytes(result.stderr),
                }
            )
            if not passed:
                raise C.StageGateError("component live readiness failed for " + probe.capability)
        # Stage a real normal broker and prove an advertised read-only roundtrip
        # through Codex's native networkless tool sandbox, before paying a model.
        deadline = time.monotonic() + timeout_s
        actions = inputs.policy.build_registry()
        broker = B.Broker(
            tool,
            inputs.policy.target_experiment,
            inputs.candidate,
            actions,
            inputs.policy.receipt_path,
            deadline=deadline,
            workflow=inputs.policy,
            max_calls=1,
            max_tool_seconds=timeout_s,
            public_control_dir=inputs.public_control_dir,
        )
        B.stage_broker_shim(
            inputs.public_control_dir,
            host="",
            port=0,
            token=broker.token,
            tool_timeout_s=timeout_s,
            actions=actions,
            socket_path="/perf-control/channel.sock",
        )
        home = inputs.stage_root / "probe_home"
        home_info = CA.prepare_codex_home(
            home, model=model, effort=effort, workspace=inputs.candidate, candidate_read_paths=_read_paths(inputs)
        )
        CA._verify_frozen_config(home, home_info["config_sha256"])
        rounds = inputs.stage_root / "probe_rounds"
        rounds.mkdir()
        private_canary = inputs.policy.receipt_path.parent / "private-golden-canary.json"
        public_sibling = inputs.public_control_dir / "unadmitted-sibling.json"
        private_canary.write_text('{"private":"must remain unreadable"}')
        public_sibling.write_text('{"private":"directory binds must not expose this sibling"}')
        with broker.serving(socket_path=inputs.socket_path):

            def command(inner, ws, bundle, extra_binds=None):
                return _control_command(inputs, inner, ws, bundle, extra_binds=extra_binds)

            CA._preflight_candidate_sandbox(
                inputs.candidate,
                home,
                str(inputs.codex_binary),
                {},
                command,
                rounds,
                0,
                runtime_binds=lambda selected_home: _runtime_binds(inputs, selected_home),
            )
            results.append(probe_private_denial(inputs, home, private_canary, timeout_s=timeout_s))
            probe_command = shlex.join(
                (
                    str(inputs.codex_binary),
                    "sandbox",
                    "--permission-profile",
                    CA._CANDIDATE_PERMISSION_PROFILE,
                    "-C",
                    str(inputs.candidate),
                    "--",
                    "/usr/bin/python3",
                    B.BROKER_NAME,
                    "inspect-optimization-surfaces",
                )
            )
            control = _control_command(
                inputs, probe_command, inputs.candidate, {}, extra_binds=_runtime_binds(inputs, home)
            )
            result = subprocess.run(["/bin/sh", "-c", control], capture_output=True, timeout=timeout_s, check=False)
            if result.returncode != 0 or not broker.calls or broker.calls[-1].get("returncode") != 0:
                raise C.StageGateError("component native sandbox/broker roundtrip failed")
        inputs.verify(require_baseline=True)
        status, refusal = "qualified", None
    except Exception as exc:  # noqa: BLE001 - failed probes remain durable and non-authorizing
        status, refusal = "refused", type(exc).__name__ + ": " + str(exc)[:500]
    receipt = inputs.stage_root / "launch_qualification.json"
    C.write_json(
        receipt,
        {
            "schema": "merlin.component_launch_qualification.v1",
            "status": status,
            "bindings": _bindings(inputs),
            "readiness_results": results,
            "model": model,
            "effort": effort,
            "refusal": refusal,
        },
    )
    receipt.chmod(0o400)
    if status != "qualified":
        raise C.StageGateError("component authoring isolation/readiness remains unavailable: " + str(refusal))
    return QualifiedComponentLaunch(inputs, receipt, C.sha256_file(receipt), model, effort, _ISSUER)


def run_component_stage(
    launch: QualifiedComponentLaunch,
    *,
    model: str,
    effort: str,
    wall_budget_seconds: int,
    max_tool_calls: int,
    tool_timeout_seconds: int,
    suite: str,
) -> Path:
    """Preserve the public launch entrypoint over the ordinary stage controller."""
    from .component_stage import run_component_stage as run

    return run(
        launch,
        model=model,
        effort=effort,
        wall_budget_seconds=wall_budget_seconds,
        max_tool_calls=max_tool_calls,
        tool_timeout_seconds=tool_timeout_seconds,
        suite=suite,
    )
