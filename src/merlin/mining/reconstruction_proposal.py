"""Sandboxed multi-file compiler proposals for the explicit reference-guided mode.

This does not change the existing single-module pass slot or campaign custody.
The proposer sees selected source and an owner-curated task card, never evaluator
inputs. Only replacement/addition sources under declared compiler surfaces enter
an isolated candidate. Evaluator code and test/oracle paths are never editable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _relative(value):
    path = PurePosixPath(value)
    if (
        type(value) is not str
        or not value
        or path.is_absolute()
        or "\\" in value
        or any(part in (".", "..") for part in value.split("/"))
    ):
        raise ValueError("compiler paths must be canonical relative paths")
    return path


@dataclass(frozen=True)
class CompilerSurface:
    path: str
    kind: str

    def __post_init__(self):
        path = _relative(self.path)
        if self.kind not in ("pass", "representation", "adapter", "consumer") or any(
            part
            in ("tests", "test", "goldens", "capture", "mining", "verify", "validation", "targetgen", "perf", "kernels")
            for part in path.parts
        ):
            raise ValueError("only owner-selected compiler surfaces may be proposed")


@dataclass(frozen=True)
class CompilerPatch:
    files: tuple[tuple[str, str], ...]
    rationale: str = ""

    @property
    def sha256(self):
        return digest(dict(files=self.files))

    def validate(self, surfaces):
        if (
            type(self.files) is not tuple
            or not 1 <= len(self.files) <= 16
            or len({path for path, _ in self.files}) != len(self.files)
        ):
            raise ValueError("patch requires one to sixteen distinct compiler sources")
        for path, source in self.files:
            relative = _relative(path)
            if any(
                part
                in (
                    "tests",
                    "test",
                    "goldens",
                    "capture",
                    "mining",
                    "verify",
                    "validation",
                    "targetgen",
                    "perf",
                    "kernels",
                )
                for part in relative.parts
            ):
                raise ValueError("patch cannot change evaluators, evidence or learning machinery")
            if (
                relative.suffix not in (".py", ".cpp", ".cc", ".h", ".hpp", ".td")
                or type(source) is not str
                or not source.strip()
                or len(source.encode()) > 512 * 1024
                or not any(
                    relative == _relative(s.path) or relative.is_relative_to(_relative(s.path)) for s in surfaces
                )
            ):
                raise ValueError(f"source outside declared compiler surfaces: {path}")


@dataclass(frozen=True)
class CandidateWorkspace:
    root: Path
    source_roots: tuple[str, ...]
    compiler_sha256: str
    patch_sha256: str | None

    @property
    def environment(self):
        env = dict(os.environ)
        # Never inherit another editable checkout/overlay into a candidate build.
        env["PYTHONPATH"] = os.pathsep.join(str(self.root / value) for value in self.source_roots)
        env["MERLIN_REPO_ROOT"] = str(self.root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["MERLIN_RECONSTRUCTION_COMPILER_SHA256"] = self.compiler_sha256
        return env

    def run(self, argv, *, timeout=300):
        """Run a fresh compiler process; a cached parent import is not a candidate."""
        return subprocess.run(
            argv,
            cwd=self.root,
            env=self.environment,
            timeout=timeout,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )


def source_inventory(compiler_root, source_roots):
    """Pin the bytes copied, excluding interpreter caches and evaluator data."""
    root = Path(compiler_root).resolve()
    inventory = []
    for value in source_roots:
        path = root / str(_relative(value))
        if not path.is_dir() or path.is_symlink():
            raise ValueError("source roots must be real compiler source directories")
        for source in sorted(path.rglob("*")):
            if "__pycache__" in source.parts or source.suffix == ".pyc":
                continue
            if source.is_symlink():
                raise ValueError("candidate compiler source cannot contain external symlinks")
            if source.is_file():
                inventory.append((source.relative_to(root).as_posix(), hashlib.sha256(source.read_bytes()).hexdigest()))
    return tuple(inventory)


def stage_candidate(compiler_root, source_roots, destination, *, patch=None, surfaces=()):
    """Copy compiler roots, never a checkout with captures or evaluator artifacts."""
    root, destination = Path(compiler_root).resolve(), Path(destination)
    inventory = source_inventory(root, source_roots)
    destination.mkdir(parents=True, exist_ok=False)
    for value in source_roots:
        shutil.copytree(root / value, destination / value, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    copied = source_inventory(destination, source_roots)
    if copied != inventory:
        raise ValueError("compiler source changed while staging candidate")
    pins = dict(copied)
    if patch is not None:
        patch.validate(surfaces)
        for value, source in patch.files:
            if not any(_relative(value).is_relative_to(_relative(prefix)) for prefix in source_roots):
                raise ValueError("patch source must belong to a copied compiler root")
            target = destination / value
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
            pins[value] = hashlib.sha256(source.encode()).hexdigest()
    return CandidateWorkspace(
        destination, source_roots, digest(tuple(sorted(pins.items()))), None if patch is None else patch.sha256
    )


@dataclass(frozen=True)
class CompilerProposalAttempt:
    patch: CompilerPatch | None
    error: str | None = None
    sandboxed: bool = False
    usage: dict | None = None


def propose_compiler_patch(
    task,
    *,
    compiler_root,
    surfaces,
    workspace,
    feedback=(),
    model="opus",
    timeout=1800,
    runner=None,
    require_sandbox=True,
):
    """One actual sandbox-agent turn; tests can inject a runner explicitly.

    The sandbox helper is shared with the pass slot. A production failure to
    establish its isolation is a typed refusal, never an unsandboxed fallback.
    """
    from .pass_agent import _sandbox_condition, sandbox_argv

    ws, root = Path(workspace), Path(compiler_root).resolve()
    ws.mkdir(parents=True, exist_ok=False)
    current = ws / "current"
    for surface in surfaces:
        surface.__post_init__()
        selected = root / surface.path
        if not selected.exists():
            continue  # a declared new representation/adapter can be added
        paths = (selected,) if selected.is_file() else sorted(selected.rglob("*"))
        for source in paths:
            if source.is_symlink():
                raise ValueError("proposer source cannot follow external symlinks")
            if not source.is_file() or source.suffix not in (".py", ".cpp", ".cc", ".h", ".hpp", ".td"):
                continue
            target = current / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    card = dict(
        schema="merlin.reference_guided_compiler_proposal.v1",
        task=task,
        allowed_surfaces=[dict(path=s.path, kind=s.kind) for s in surfaces],
        feedback=list(feedback),
    )
    instructions = (
        "Implement the machine task using current compiler sources. Search library composition and scheduling "
        "dialect consumers before adding mechanisms. All choices must derive from current semantics, geometry, "
        "layouts and resource capabilities. Do not dispatch by reference/workload names or access evaluator "
        "inputs. Bounded search exhaustion is not proof of a missing pass. Write proposal.json with exactly "
        "files (a nonempty list of {path, source} complete replacement/addition sources) and rationale. "
        "Only listed compiler surfaces are editable. Do not modify current/. A claim or metadata-only "
        "annotation cannot satisfy the executable scheduling, numerical, storage, target and emission gates.\n"
        + json.dumps(card, sort_keys=True, indent=2)
    )
    (ws / "TASK.md").write_text(instructions, encoding="utf-8")
    try:
        sandbox = sandbox_argv(ws)
    except (ImportError, OSError, RuntimeError) as error:
        return CompilerProposalAttempt(None, f"sandbox_unavailable: {error}")
    if sandbox is None and require_sandbox:
        return CompilerProposalAttempt(None, "sandbox_unavailable: " + _sandbox_condition())
    if runner is None:
        try:
            proc = subprocess.run(
                list(sandbox or [])
                + ["claude", "-p", uuid.uuid4().hex + "\n" + instructions, "--model", model, "--output-format", "json"],
                cwd=ws,
                env=dict(os.environ),
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return CompilerProposalAttempt(None, f"proposer_error: {error}", sandbox is not None)
        (ws / "agent_transcript.json").write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode:
            return CompilerProposalAttempt(None, f"proposer_exit: {proc.returncode}", sandbox is not None)
        try:
            output = json.loads(proc.stdout)
        except ValueError:
            output = {}
    else:
        output = runner(prompt=instructions, workspace=ws, model=model, timeout=timeout, sandbox=sandbox)
    try:
        obj = json.loads((ws / "proposal.json").read_text(encoding="utf-8"))
        if (
            type(obj) is not dict
            or set(obj) != {"files", "rationale"}
            or type(obj["files"]) is not list
            or any(type(row) is not dict or set(row) != {"path", "source"} for row in obj["files"])
        ):
            raise ValueError("expected files and rationale")
        patch = CompilerPatch(tuple((row["path"], row["source"]) for row in obj["files"]), obj["rationale"])
        patch.validate(surfaces)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return CompilerProposalAttempt(None, f"invalid_proposal: {error}", sandbox is not None, output.get("usage"))
    return CompilerProposalAttempt(patch, sandboxed=sandbox is not None, usage=output.get("usage"))
