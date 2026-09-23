"""Agent input snapshots and two-plane workspace policies from explicit host inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes as _sha256
from merlin.perf.external_objective import OBJECTIVE_DIRECTORY, ExternalObjective, objective_directory
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces
from merlin.targetgen.target_experiment import TargetExperiment

from . import campaign as CAMPAIGN
from . import contracts as CONTRACTS
from . import corpus as CORPUS
from .broker import AGENT_CORPUS_MOUNT
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json
from .functional_inputs import FrozenFunctionalInputs, _private_functional_surfaces, frozen_grant_mounts

FUNCTIONAL_BASE_MOUNT = Path("/perf-functional-base")
FUNCTIONAL_INPUT_MANIFEST_MOUNT = Path("/perf-functional-inputs/snapshot.json")
PERF_CORPUS_MANIFEST_MOUNT = Path("/perf-corpus-manifest.json")


@dataclass(frozen=True)
class AgentSandboxPolicy:
    argv: tuple[str, ...]
    answer_surface_gap: tuple[str, ...]
    network: str
    clear_environment: bool
    candidate_writable: bool
    corpus_read_only: bool
    env_prefix: str | None = None
    required_tools: tuple[TC.ToolProbe, ...] = ()
    process_cwd: Path | None = None
    frozen_inputs: CAMPAIGN.FrozenPackageSandboxInputs | None = None
    _frozen_record: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frozen_inputs is not None:
            object.__setattr__(self, "_frozen_record", _canonical_json(self.frozen_inputs.record))

    def verify_execution(self) -> None:
        """Admit captured execution configuration without consulting live providers."""
        if self.network != "available_not_an_isolation_claim" or not self.clear_environment:
            raise StageGateError("inner execution requires the explicit clear-environment policy")
        if (
            not isinstance(self.env_prefix, str)
            or not isinstance(self.required_tools, tuple)
            or not self.required_tools
            or any(not isinstance(probe, TC.ToolProbe) for probe in self.required_tools)
            or not isinstance(self.process_cwd, Path)
            or not self.process_cwd.is_absolute()
        ):
            raise StageGateError("inner execution policy lacks captured environment, probes or process cwd")
        if self.frozen_inputs is not None:
            from .qualification_policy import restore

            if _canonical_json(self.frozen_inputs.record) != self._frozen_record:
                raise StageGateError("frozen execution policy record changed after capture")
            verified = restore(self.frozen_inputs.root, json.loads(self._frozen_record))
            if (
                verified != self.frozen_inputs
                or self.env_prefix != verified.env_prefix
                or self.required_tools != verified.probes
                or self.process_cwd != verified.repo
            ):
                raise StageGateError("inner execution policy differs from verified frozen inputs")
            size = len(verified.argv)
            start = next(
                (i for i in range(len(self.argv) - size + 1) if self.argv[i : i + size] == verified.argv),
                None,
            )
            if start is None:
                raise StageGateError("inner execution policy lost its frozen tool grants")
            # Frozen compiler inputs must remain disjoint from tool destinations until
            # merged-overlay support verifies both retained tool bytes and exact additions.
            protected = []
            for state, _source, destination in BW._mounts(list(verified.argv)):
                path = _frozen_mount_destination(destination)
                if state == "expose":
                    protected.append(path)
            for state, _source, destination in BW._mounts(list(self.argv[start + size :])):
                path = _frozen_mount_destination(destination)
                if state == "expose" and any(
                    path == tool or path in tool.parents or tool in path.parents for tool in protected
                ):
                    raise StageGateError("inner execution policy shadows a frozen tool destination")
            if BW.coverage_gap(list(self.argv), list(verified.surfaces)):
                raise StageGateError("inner execution policy exposes frozen answer surfaces")


def _frozen_mount_destination(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or value.startswith("//") or "\0" in value:
        raise StageGateError("inner execution policy has an unsafe frozen mount destination")
    return path


def _sandbox_inputs(
    target: TargetExperiment,
    inputs: CAMPAIGN.PackageSandboxInputs | CAMPAIGN.FrozenPackageSandboxInputs | None,
) -> CAMPAIGN.PackageSandboxInputs | CAMPAIGN.FrozenPackageSandboxInputs:
    if inputs is None:
        return CAMPAIGN.select_package_sandbox_inputs(target)
    if isinstance(inputs, CAMPAIGN.FrozenPackageSandboxInputs):
        from .qualification_policy import restore

        return restore(inputs.root, json.loads(_canonical_json(inputs.record)))
    if not isinstance(inputs, CAMPAIGN.PackageSandboxInputs):
        raise StageGateError("unsupported agent sandbox input selection")
    return inputs


@dataclass(frozen=True)
class AgentInputSnapshot:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    content_sha256: str
    n_files: int
    n_bytes: int


def _path_is_answer(path: Path, surfaces: Sequence) -> bool:
    resolved = path.resolve()
    for surface in surfaces:
        answer = Path(surface.path).resolve()
        if resolved == answer or (surface.kind == "dir" and answer in resolved.parents):
            return True
    return False


def build_answer_free_agent_inputs(
    corpus: CORPUS.FrozenPerformanceCorpus,
    target_experiment: TargetExperiment,
    destination: Path,
    *,
    external_objective: ExternalObjective | None = None,
    external_objectives: Sequence[ExternalObjective] = (),
) -> AgentInputSnapshot:
    """Copy only non-answer capsule bytes into the read-only view exposed to the agent.

    The complete frozen corpus remains host-only.  Filtering is derived from the same answer-surface
    registry used by the bwrap coverage proof; no golden filename allow/deny list is maintained here.
    """
    raw_destination = Path(destination)
    if raw_destination.exists() or raw_destination.is_symlink():
        raise StageGateError(f"agent input snapshot already exists: {raw_destination}")
    destination = raw_destination.resolve()
    if external_objective is not None and external_objectives:
        raise StageGateError("use either the legacy external objective or the ordered portfolio")
    objectives = (external_objective,) if external_objective is not None else tuple(external_objectives)
    if (
        any(type(objective) is not ExternalObjective for objective in objectives)
        or len({objective.objective_id for objective in objectives}) != len(objectives)
        or len({objective.source_sha256 for objective in objectives}) != len(objectives)
    ):
        raise StageGateError("external objectives require distinct host-loaded typed snapshots")
    surfaces = answer_surfaces(target_experiment)
    rows: list[dict[str, Any]] = []
    destination.mkdir(parents=True)
    for member in corpus.capsules:
        original = Path(target_experiment.capsule_corpus).resolve().parent / member.source_relative_path
        for frozen_file in sorted(path for path in member.source_dir.rglob("*") if path.is_file()):
            relative_in_capsule = frozen_file.relative_to(member.source_dir)
            original_file = original / relative_in_capsule
            if _path_is_answer(original_file, surfaces):
                continue
            relative = Path(member.source_relative_path) / relative_in_capsule
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = frozen_file.read_bytes()
            output.write_bytes(payload)
            rows.append({"path": relative.as_posix(), "sha256": _sha256(payload), "n_bytes": len(payload)})
    for objective in objectives:
        relative_root = (
            Path(OBJECTIVE_DIRECTORY) if external_objective is not None else objective_directory(objective.objective_id)
        )
        objective_root = destination / relative_root
        if objective_root.exists():
            raise StageGateError("external objective collides with an existing corpus path")
        objective_root.mkdir(parents=True)
        for name, payload in objective.files():
            relative = relative_root / name
            (destination / relative).write_bytes(payload)
            rows.append({"path": relative.as_posix(), "sha256": _sha256(payload), "n_bytes": len(payload)})
    if not rows:
        raise StageGateError("answer-free performance input view contains zero files")
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(row["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(row["sha256"].encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\n")
    content_sha = aggregate.hexdigest()
    manifest = {
        "schema_version": 1,
        "source_performance_manifest_sha256": corpus.manifest_sha256,
        "source_performance_corpus_sha256": corpus.capsules_sha256,
        "answer_surface_registry": "merlin.targetgen.sandbox.answer_surfaces",
        "files": rows,
        "content_sha256": content_sha,
        "n_files": len(rows),
        "n_bytes": sum(int(row["n_bytes"]) for row in rows),
    }
    if external_objective is not None:
        manifest["external_objective"] = external_objective.record()
    if external_objectives:
        manifest["external_objectives"] = [objective.record() for objective in objectives]
    manifest_path = destination / "agent_input_manifest.json"
    payload = _canonical_json(manifest)
    manifest_path.write_bytes(payload)
    for path in sorted(destination.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return AgentInputSnapshot(
        destination, manifest_path, _sha256(payload), content_sha, len(rows), int(manifest["n_bytes"])
    )


def verify_answer_free_agent_inputs(snapshot: AgentInputSnapshot) -> None:
    root = Path(snapshot.root)
    manifest = Path(snapshot.manifest_path)
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.resolve() != root
        or any(parent.is_symlink() for parent in (root, *root.parents))
    ):
        raise StageGateError("answer-free performance input root is absent or linked")
    if manifest != root / "agent_input_manifest.json" or manifest.is_symlink() or not manifest.is_file():
        raise StageGateError("answer-free performance input manifest is absent, linked or outside its root")
    raw = manifest.read_bytes()
    if _sha256(raw) != snapshot.manifest_sha256:
        raise StageGateError("answer-free performance input manifest changed")
    document = json.loads(raw)
    rows = document.get("files")
    if not isinstance(rows, list) or len(rows) != snapshot.n_files or not rows:
        raise StageGateError("answer-free performance input manifest is incomplete")
    aggregate = hashlib.sha256()
    total = 0
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise StageGateError("answer-free performance input contains a linked or special entry")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    expected = {manifest.relative_to(root).as_posix()}
    for row in rows:
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in expected:
            raise StageGateError("answer-free performance input path escapes its snapshot")
        expected.add(relative.as_posix())
        path = snapshot.root / relative
        if path.is_symlink() or not path.is_file():
            raise StageGateError(f"answer-free performance input is absent or linked: {relative}")
        payload = path.read_bytes()
        if _sha256(payload) != row.get("sha256") or len(payload) != row.get("n_bytes"):
            raise StageGateError(f"answer-free performance input changed: {relative}")
        aggregate.update(relative.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(row["sha256"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\n")
        total += len(payload)
    if observed != expected:
        raise StageGateError("answer-free performance input file membership changed")
    if aggregate.hexdigest() != snapshot.content_sha256 or total != snapshot.n_bytes:
        raise StageGateError("answer-free performance input aggregate digest changed")


def _strip_claude_home(argv: Sequence[str]) -> list[str]:
    """Remove the broad Claude credential/history bind inherited from the shared bwrap base."""
    claude_home = Path.home() / ".claude"
    output: list[str] = []
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try") and index + 2 < len(argv):
            destination = Path(argv[index + 2])
            if destination == claude_home or claude_home in destination.parents:
                index += 3
                continue
        if option == "--tmpfs" and index + 1 < len(argv):
            destination = Path(argv[index + 1])
            if destination == claude_home or claude_home in destination.parents:
                index += 2
                continue
        output.append(option)
        index += 1
    return output


def inner_execution_policy(
    target_experiment: TargetExperiment,
    candidate: Path,
    agent_inputs: AgentInputSnapshot,
    frozen_functional: FrozenFunctionalInputs | None = None,
    functional_base: Path | None = None,
    frozen_corpus_manifest: Path | None = None,
    *,
    inputs: CAMPAIGN.PackageSandboxInputs | CAMPAIGN.FrozenPackageSandboxInputs | None = None,
) -> AgentSandboxPolicy:
    """Build the credential-free, answer-masked sandbox used by the local tool broker."""
    candidate = CONTRACTS.require_real_directory(candidate, label="performance candidate")
    verify_answer_free_agent_inputs(agent_inputs)
    inputs = _sandbox_inputs(target_experiment, inputs)
    frozen = isinstance(inputs, CAMPAIGN.FrozenPackageSandboxInputs)
    repo = inputs.repo if frozen else inputs.paths.repo.resolve()
    argv = _strip_claude_home(BW.base_argv(candidate, {}, repo=repo, _policy_test_live_inputs=True))
    argv += [
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "XDG_RUNTIME_DIR",
        "/tmp/.xdg",
    ]
    if frozen:
        argv += list(inputs.argv)
        env_prefix, probes = inputs.env_prefix, inputs.probes
    else:
        selected = {"paths": inputs.paths, "sim": inputs.sim, "harness": inputs.harness}
        memory_dir = next((str(surface.path) for surface in inputs.surfaces if surface.origin == "memory"), "")
        argv += TC.toolchain_binds(target_experiment, **selected, memory_dir=memory_dir)
        env_prefix = TC.sandbox_env(target_experiment, candidate, **selected)
        probes = tuple(TC.required_tool_probes(target_experiment, paths=inputs.paths, sim=inputs.sim))
    if frozen_functional is not None:
        argv += frozen_grant_mounts(frozen_functional)
        argv += [
            "--ro-bind",
            str(frozen_functional.public_marker or frozen_functional.marker),
            str(FUNCTIONAL_INPUT_MANIFEST_MOUNT),
        ]
    if functional_base is not None:
        argv += ["--ro-bind", str(functional_base), str(FUNCTIONAL_BASE_MOUNT)]
    if frozen_corpus_manifest is not None:
        argv += ["--ro-bind", str(frozen_corpus_manifest), str(PERF_CORPUS_MANIFEST_MOUNT)]
    argv += ["--ro-bind", str(agent_inputs.root), str(AGENT_CORPUS_MOUNT)]
    surfaces = [*inputs.surfaces, *_private_functional_surfaces(argv, frozen_functional)]
    argv = BW.apply_answer_masks(argv, surfaces)
    if "--unshare-net" in argv:
        raise StageGateError("inner execution policy unexpectedly disables required network availability")
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise StageGateError(f"inner performance-tool sandbox exposes answer surfaces: {gaps}")
    joined = " ".join(argv)
    for token in (".codex/auth.json", "AWS_ACCESS_KEY_ID", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if token in joined:
            raise StageGateError(f"inner performance-tool sandbox exposes credential token {token!r}")
    return AgentSandboxPolicy(
        tuple(argv),
        gaps,
        "available_not_an_isolation_claim",
        True,
        True,
        True,
        env_prefix,
        probes,
        repo,
        inputs if frozen else None,
    )


def outer_codex_policy(
    workspace: Path,
    agent_inputs: AgentInputSnapshot,
    runtime_binds: Sequence[str],
    target_experiment: TargetExperiment,
    frozen_functional: FrozenFunctionalInputs | None = None,
    functional_base: Path | None = None,
    control_dir: Path | None = None,
    frozen_corpus_manifest: Path | None = None,
    *,
    inputs: CAMPAIGN.PackageSandboxInputs | CAMPAIGN.FrozenPackageSandboxInputs | None = None,
) -> AgentSandboxPolicy:
    """Build Codex's filesystem boundary; network/auth exceptions are explicit and recorded."""
    workspace = CONTRACTS.require_real_directory(workspace, label="Codex round workspace")
    verify_answer_free_agent_inputs(agent_inputs)
    inputs = _sandbox_inputs(target_experiment, inputs)
    frozen = isinstance(inputs, CAMPAIGN.FrozenPackageSandboxInputs)
    repo = inputs.repo if frozen else inputs.paths.repo.resolve()
    argv = _strip_claude_home(BW.base_argv(workspace, {}, repo=repo, _policy_test_live_inputs=True))
    argv += [
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "XDG_RUNTIME_DIR",
        "/tmp/.xdg",
    ]
    argv += list(runtime_binds)
    if frozen_functional is not None:
        argv += frozen_grant_mounts(frozen_functional)
        argv += [
            "--ro-bind",
            str(frozen_functional.public_marker or frozen_functional.marker),
            str(FUNCTIONAL_INPUT_MANIFEST_MOUNT),
        ]
    if functional_base is not None:
        argv += ["--ro-bind", str(functional_base), str(FUNCTIONAL_BASE_MOUNT)]
    if control_dir is not None:
        argv += ["--ro-bind", str(control_dir), "/perf-control"]
    if frozen_corpus_manifest is not None:
        argv += ["--ro-bind", str(frozen_corpus_manifest), str(PERF_CORPUS_MANIFEST_MOUNT)]
    argv += ["--ro-bind", str(agent_inputs.root), str(AGENT_CORPUS_MOUNT)]
    surfaces = [*inputs.surfaces, *_private_functional_surfaces(argv, frozen_functional)]
    argv = BW.apply_answer_masks(argv, surfaces)
    if "--unshare-net" in argv:
        raise StageGateError("outer Codex policy unexpectedly disables required network availability")
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise StageGateError(f"outer Codex sandbox exposes answer surfaces: {gaps}")
    # Codex receives only its runtime grants, not an executable inner-tool policy.
    return AgentSandboxPolicy(
        tuple(argv),
        gaps,
        "available_not_an_isolation_claim",
        True,
        True,
        True,
        frozen_inputs=inputs if frozen else None,
    )


def run_required_tool_probes(
    policy: AgentSandboxPolicy, target_experiment: TargetExperiment, candidate: Path, *, timeout_s: int = 60
) -> list[dict[str, Any]]:
    policy.verify_execution()
    probes = policy.required_tools
    rows: list[dict[str, Any]] = []
    for probe in probes:
        command = [
            *policy.argv,
            "--chdir",
            str(candidate),
            "bash",
            "-c",
            policy.env_prefix + probe.cmd,
        ]
        policy.verify_execution()
        proc = subprocess.run(command, cwd=str(policy.process_cwd), capture_output=True, text=True, timeout=timeout_s)
        row = {
            "label": probe.label,
            "returncode": proc.returncode,
            "command": probe.cmd,
            "bind": probe.bind,
            "stdout": (proc.stdout or "")[-400:],
            "stderr": (proc.stderr or "")[-400:],
        }
        rows.append(row)
        if proc.returncode != 0:
            raise StageGateError(
                f"required inner-sandbox tool probe {probe.label!r} failed with rc={proc.returncode}; "
                f"command={row['command']!r}; stdout={row['stdout']!r}; stderr={row['stderr']!r}"
            )
    return rows


def _make_writable(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        if path.is_symlink():
            raise StageGateError(f"candidate copy contains a symlink: {path}")
        path.chmod(0o755 if path.is_dir() else (path.stat().st_mode | stat.S_IWUSR))
    root.chmod(0o755)


def fresh_round_workspace(source_submission: Path, workspace: Path, expected_sha256: str) -> Path:
    """Create a new round workspace from exactly the previous candidate bytes."""
    raw_workspace = Path(workspace)
    if raw_workspace.exists() or raw_workspace.is_symlink():
        raise StageGateError(f"round workspace is not fresh: {raw_workspace}")
    workspace = raw_workspace.resolve()
    submission = workspace / "submission"
    workspace.mkdir(parents=True)
    shutil.copytree(source_submission, submission, symlinks=False)
    _make_writable(submission)
    observed = hash_tree(submission)["sha256"]
    if observed != expected_sha256:
        raise StageGateError(f"fresh round candidate digest {observed} does not match its input {expected_sha256}")
    return submission
