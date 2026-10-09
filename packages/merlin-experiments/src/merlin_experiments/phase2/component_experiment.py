"""Minimal experiment views and final comparison gates, owned by the trusted host.

An upstream repository is not an agent grant. These primitives materialize only
reviewed members and refuse launch without independently checked isolation.
They do not certify Phase 0 derivation, an authoring transport, or hardware.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from merlin.targetgen.compiler_library import CompilerLibraryContract, freeze_compiler_library

from .contracts import StageGateError, canonical_json

VIEW_SCHEMA = "merlin.component_agent_view.v1"
_EXCLUDED = frozenset({".git", ".claude", ".codex", "out", "docs", "tests", "__pycache__"})
_ROLES = frozenset({"contract", "generated_input", "toolchain"})


def _member(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or "\\" in value
        or any(part in _EXCLUDED or part in {".", ".."} for part in path.parts)
        or any(ord(char) < 32 for char in value)
    ):
        raise StageGateError("unsafe component-view member")
    return value


def _digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise StageGateError("component evidence requires a lowercase SHA-256")
    return value


def _read(path: Path, expected: str) -> bytes:
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
        raise StageGateError("approved component source is absent or linked")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _digest(expected):
        raise StageGateError("approved component source bytes changed")
    return payload


@dataclass(frozen=True)
class ApprovedInput:
    source: Path
    destination: str
    sha256: str
    role: str

    def __post_init__(self) -> None:
        _member(self.destination)
        _digest(self.sha256)
        if self.role not in _ROLES or PurePosixPath(self.destination).parts[0] != self.role:
            raise StageGateError("component inputs require a reviewed role and matching namespace")


@dataclass(frozen=True)
class ComponentView:
    root: Path
    manifest_sha256: str
    generation_sha256: str
    library_sha256: str


def _reviewed_library(library: CompilerLibraryContract, library_root: Path) -> None:
    # Preserve the existing member/dependency refusal before deriving a fresh
    # contract from exactly the host-approved selection. The new identity check
    # does not let live directory discovery expand that selection.
    library.verify(library_root)
    frozen = freeze_compiler_library(
        library_root,
        review_id=library.review_id,
        public_modules=library.public_modules,
        sources=tuple((member.path, member.module) for member in library.members),
    )
    if frozen.sha256 != library.sha256:
        raise StageGateError("approved compiler library identity changed")


def materialize_component_view(
    destination: Path,
    *,
    library: CompilerLibraryContract,
    library_root: Path,
    inputs: tuple[ApprovedInput, ...],
    generation_sha256: str,
) -> ComponentView:
    """Copy explicit members only, with no Git metadata or private source paths.

    The generation digest binds the independently admitted Phase 0 receipt;
    this function does not turn a caller-supplied hash into derivation proof.
    """
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise StageGateError("component view destination already exists")
    if not isinstance(inputs, tuple) or any(type(member) is not ApprovedInput for member in inputs):
        raise StageGateError("component inputs require immutable approved membership")
    if not any(member.role == "generated_input" for member in inputs):
        raise StageGateError("component view has no generated development inputs")
    _digest(generation_sha256)
    _reviewed_library(library, library_root)
    selections = [
        ("compiler/" + member.path, Path(library_root) / member.path, member.sha256, "compiler")
        for member in library.members
    ]
    selections += [(member.destination, member.source, member.sha256, member.role) for member in inputs]
    if len({row[0] for row in selections}) != len(selections):
        raise StageGateError("component view has duplicate destination ownership")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".component-view-", dir=destination.parent))
    rows = []
    try:
        for relative, source, expected, role in sorted(selections):
            path = staging / _member(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = _read(source, expected)
            path.write_bytes(payload)
            path.chmod(0o444)
            rows.append({"path": relative, "sha256": expected, "n_bytes": len(payload), "role": role})
        manifest = canonical_json(
            {
                "schema": VIEW_SCHEMA,
                "generation_sha256": generation_sha256,
                "library_sha256": library.sha256,
                "members": rows,
            }
        )
        (staging / "manifest.json").write_bytes(manifest)
        (staging / "manifest.json").chmod(0o444)
        _reviewed_library(library, library_root)
        if destination.exists() or destination.is_symlink():
            raise StageGateError("component view destination appeared during materialization")
        staging.rename(destination)
        view = ComponentView(
            destination.resolve(), hashlib.sha256(manifest).hexdigest(), generation_sha256, library.sha256
        )
        verify_component_view(view)
        return view
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_component_view(view: ComponentView) -> dict:
    """Exact file/directory membership and byte checks; added history is a refusal."""
    if view.root.is_symlink() or not view.root.is_dir():
        raise StageGateError("component view root is absent or linked")
    payload = _read(view.root / "manifest.json", view.manifest_sha256)
    try:
        manifest = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise StageGateError("invalid component view manifest") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != VIEW_SCHEMA
        or manifest.get("generation_sha256") != _digest(view.generation_sha256)
        or manifest.get("library_sha256") != _digest(view.library_sha256)
        or not isinstance(manifest.get("members"), list)
        or not manifest["members"]
    ):
        raise StageGateError("component view identities or membership are invalid")
    expected = {"manifest.json"}
    directories = set()
    for row in manifest["members"]:
        if not isinstance(row, dict):
            raise StageGateError("malformed component view member")
        relative = _member(row.get("path", ""))
        if relative in expected or row.get("role") not in _ROLES | {"compiler"}:
            raise StageGateError("duplicate or unreviewed component view member")
        if PurePosixPath(relative).parts[0] != row["role"]:
            raise StageGateError("component view role differs from member namespace")
        payload = _read(view.root / relative, row.get("sha256"))
        if len(payload) != row.get("n_bytes"):
            raise StageGateError("component view member size changed")
        expected.add(relative)
        directories.update(str(parent) for parent in PurePosixPath(relative).parents if str(parent) != ".")
    observed, observed_dirs = set(), set()
    for path in view.root.rglob("*"):
        if path.is_symlink():
            raise StageGateError("component view contains a linked member")
        relative = path.relative_to(view.root).as_posix()
        if path.is_dir():
            observed_dirs.add(relative)
        elif path.is_file():
            observed.add(relative)
        else:
            raise StageGateError("component view contains a special file")
    if observed != expected or observed_dirs != directories:
        raise StageGateError("component view membership changed")
    return manifest


@dataclass(frozen=True)
class RuntimeGrant:
    source: Path
    destination: str
    sha256: str

    @staticmethod
    def verify_destination(destination: str) -> None:
        if not isinstance(destination, str) or any(ord(char) < 32 for char in destination):
            raise StageGateError("runtime grant must be an explicit system-tool file")
        target = PurePosixPath(destination)
        if (
            not target.is_absolute()
            or str(target) != destination
            or ".." in target.parts
            or any(part in _EXCLUDED for part in target.parts)
            or target.parts[1:2] not in (("usr",), ("lib",), ("lib64",), ("bin",), ("etc",))
        ):
            raise StageGateError("runtime grant must be an explicit system-tool file")

    def verify(self) -> None:
        self.verify_destination(self.destination)
        _read(self.source, self.sha256)


def strict_tool_policy(
    view: ComponentView,
    candidate: Path,
    *,
    runtime: tuple[RuntimeGrant, ...],
    candidate_destination: str = "/candidate",
    bwrap_binary: Path | None = None,
    candidate_writable: bool = True,
    mount_proc: bool = True,
) -> tuple[str, ...]:
    """Networkless tool subprocess policy; no broad checkout/home/system binds.

    Runtime files are explicitly reviewed. This policy is not an authenticated
    model-client transport or evidence that the host supports namespaces.
    Read-only execution can select the same closed namespace without granting
    the executed program writes to its original source/product workspace.
    Execution that needs no process filesystem can omit it explicitly. This
    reduces guest-visible paths; it does not qualify observer integrity.
    """
    if type(candidate_writable) is not bool:
        raise StageGateError("candidate write selection must be an explicit bool")
    if type(mount_proc) is not bool:
        raise StageGateError("process filesystem selection must be an explicit bool")
    verify_component_view(view)
    candidate = Path(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise StageGateError("component candidate is absent or linked")
    candidate = candidate.resolve()
    destination = PurePosixPath(candidate_destination)
    if not destination.is_absolute() or ".." in destination.parts or str(destination) != candidate_destination:
        raise StageGateError("candidate mount destination must be canonical and absolute")
    if candidate == view.root or candidate.is_relative_to(view.root) or view.root.is_relative_to(candidate):
        raise StageGateError("writable candidate overlaps frozen component view")
    if any(
        path.is_symlink() or any(part in _EXCLUDED for part in path.relative_to(candidate).parts)
        for path in candidate.rglob("*")
    ):
        raise StageGateError("initial component candidate contains history, reports or linked files")
    if not isinstance(runtime, tuple) or not runtime or any(type(grant) is not RuntimeGrant for grant in runtime):
        raise StageGateError("component tool policy requires explicit runtime files")
    if len({grant.destination for grant in runtime}) != len(runtime):
        raise StageGateError("duplicate runtime destination")
    argv = [
        str(bwrap_binary) if bwrap_binary is not None else "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "PYTHONPATH",
        "/component-inputs/compiler",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
    ]
    if mount_proc:
        argv += ["--proc", "/proc"]
    argv += ["--dev", "/dev", "--tmpfs", "/tmp"]
    for grant in runtime:
        grant.verify()
        argv += ["--ro-bind", str(grant.source), grant.destination]
    argv += [
        "--ro-bind",
        str(view.root),
        "/component-inputs",
        "--bind" if candidate_writable else "--ro-bind",
        str(candidate),
        candidate_destination,
        "--chdir",
        candidate_destination,
    ]
    return tuple(argv)


def run_isolation_probe(argv: tuple[str, ...], command: tuple[str, ...], *, timeout_s: float = 10) -> dict:
    """Execute a trusted sandbox probe; unavailable namespaces never count as pass.

    The caller owns probe semantics and keeps its output private. A successful
    exit alone is not a complete authoring isolation certificate.
    """
    if not argv or argv[0] != "bwrap" or "--unshare-all" not in argv or "--share-net" in argv:
        raise StageGateError("isolation probe requires a strict tool policy")
    if not command or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise StageGateError("isolation probe needs a bounded explicit command")
    try:
        result = subprocess.run([*argv, "--", *command], capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StageGateError("component isolation probe could not execute") from exc
    if result.returncode != 0:
        raise StageGateError("component isolation probe failed; experiment launch remains unavailable")
    return {
        "status": "probe_passed_not_authoring_qualified",
        "argv_sha256": hashlib.sha256(canonical_json(argv)).hexdigest(),
        "command_sha256": hashlib.sha256(canonical_json(command)).hexdigest(),
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
    }


@dataclass(frozen=True)
class FinalMemberComparison:
    member: str
    reference_cycles: int | None
    candidate_cycles: int | None
    reference_execution_sha256: str
    candidate_execution_sha256: str
    comparison_identity_sha256: str
    accuracy_passed: bool
    final_executable_passed: bool
    hardware_verified: bool


def final_component_campaign_gate(
    comparisons: tuple[FinalMemberComparison, ...],
    *,
    expected_members: tuple[str, ...],
    phase12_wall_s: float | None,
    handwritten_wall_s: float | None,
) -> dict:
    """Trusted final gate; never an agent-visible whole-model tuning objective.

    Comparison identity binds hardware, inputs, accuracy and timer boundaries
    at the independent receipt verifier. This function cannot authenticate
    arbitrary caller-supplied hardware claims or historical telemetry.
    """
    if not expected_members or len(set(expected_members)) != len(expected_members):
        raise StageGateError("final gate requires an explicit nonempty unique member set")
    if any(type(row) is not FinalMemberComparison for row in comparisons):
        raise StageGateError("final gate requires typed comparison evidence")
    if len(comparisons) != len(expected_members) or {row.member for row in comparisons} != set(expected_members):
        raise StageGateError("final comparison membership is incomplete or duplicated")
    failures, unknowns = [], []
    for row in comparisons:
        for digest in (row.reference_execution_sha256, row.candidate_execution_sha256, row.comparison_identity_sha256):
            _digest(digest)
        if any(
            type(value) is not bool
            for value in (row.accuracy_passed, row.final_executable_passed, row.hardware_verified)
        ):
            raise StageGateError("final evidence booleans must be explicit")
        if not row.accuracy_passed or not row.final_executable_passed:
            failures.append(row.member + ": correctness or executable gate failed")
        if not row.hardware_verified or row.reference_cycles is None or row.candidate_cycles is None:
            unknowns.append(row.member + ": matched hardware cycles unavailable")
            continue
        if any(type(value) is not int or value <= 0 for value in (row.reference_cycles, row.candidate_cycles)):
            raise StageGateError("final hardware cycles must be positive integers")
        if row.candidate_cycles * 100 > row.reference_cycles * 105:
            failures.append(row.member + ": exceeds 5% handwritten parity allowance")
    ratio = None
    if phase12_wall_s is None or handwritten_wall_s is None:
        unknowns.append("historical or phase 1/2 wall time unavailable")
    else:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in (phase12_wall_s, handwritten_wall_s)
        ):
            raise StageGateError("convergence comparison requires positive finite matched wall times")
        ratio = handwritten_wall_s / phase12_wall_s
        if phase12_wall_s * 20 > handwritten_wall_s:
            failures.append("convergence is less than 20 times faster")
    return {
        "schema": "merlin.component_final_gate.v1",
        "status": "fail" if failures else "unknown" if unknowns else "pass",
        "failures": failures,
        "unknowns": unknowns,
        "convergence_speedup": ratio,
        "parity_allowance": 0.05,
        "required_convergence_speedup": 20,
        "members": list(expected_members),
        "scope": "final frozen held-out comparison only",
    }
