"""Reviewed public compiler source workspaces, separate from the private evaluator.

The source inventory is a copy boundary, not an API menu. Authors can add modules
inside compiler namespaces. No candidate code is imported by this owner.
"""

from __future__ import annotations

import ast
import contextvars
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath

from merlin.common.access import AUTHOR_WITHHELD_PREFIXES, module_matches
from merlin_experiments.access_policy import MODULE_ACCESS, require_current_policy

ENV = "MERLIN_FULL_COMPILER_SELECTION"
SCHEMA = "merlin.full_compiler_sources.v1"
DIRECTORY = "compiler_library"
MAX_BYTES = 128 << 20
SUPPORT_SUFFIXES = frozenset(
    (
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".td",
        ".inc",
        ".LICENSE",
        ".mlir",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".txt",
        ".def",
        ".tpp",
        ".cmake",
    )
)
# Compiler construction areas; dependencies are added from the same reviewed
# core source tree. No repo root, tests, kernels, corpora or artifacts are copied.
SOURCE_ROOTS = (
    "compile",
    "transforms",
    "sched",
    "semantic_compiler",
    "llvmlower",
    "frontends",
    "capture",
    "integrations",
    "common",
    "liveness",
    "xdsl_dialects",
)
MUTABLE_NAMESPACES = tuple("merlin." + p for p in SOURCE_ROOTS) + (
    "merlin.runtime",
    "merlin.targetgen",
    "merlin.verify",
)
PUBLIC_HELPERS = (
    "merlin.targetgen.contract.linalg_iface",
    "merlin.targetgen.isa_spec",
    "merlin.targetgen.isa_model",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_header_gen",
    "merlin.targetgen.oot_starterkit.cmdbuf",
    "merlin.perf.emitted_work",
    "merlin.perf.model_group_work",
)
# The shared author-withheld packages; the V22 author SDK stages none of them either.
DENIED_PREFIXES = AUTHOR_WITHHELD_PREFIXES
_UNSET = object()
_ACTIVE = contextvars.ContextVar("full_compiler_workspace", default=_UNSET)


def forbidden(module: str) -> bool:
    return any(module_matches(module, p) for p in DENIED_PREFIXES) or any(
        module_matches(module, p) for entry in MODULE_ACCESS for p in entry.modules
    )


def _relative(value):
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or any(p in (".git", "out", "tests", "__pycache__") for p in path.parts)
    ):
        raise ValueError(f"unsafe public compiler path: {value}")
    return path


def _module(relative):
    path = _relative(relative)
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not all(p.isidentifier() for p in parts):
        raise ValueError("invalid compiler module path")
    return ".".join(parts)


def _read(root, relative):
    path = root / _relative(relative)
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError(f"public compiler source is absent or indirect: {relative}")
    return path.read_bytes()


@dataclass(frozen=True)
class FullCompilerSelection:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    members: tuple
    mutable_namespaces: tuple[str, ...]
    diagnostics: tuple

    @property
    def python_root(self):
        return self.root

    def record(self):
        return {
            "schema": "merlin.full_compiler_selection.v1",
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "members": len(self.members),
            "mutable_namespaces": list(self.mutable_namespaces),
            "unavailable_dependencies": list(self.diagnostics),
        }

    def permits(self, module):
        return not forbidden(module) and (
            any(module_matches(module, p) for p in self.mutable_namespaces)
            or any(row[2] == module for row in self.members)
        )

    def verify(self):
        if (
            hashlib.sha256(_read(self.manifest_path.parent, self.manifest_path.name)).hexdigest()
            != self.manifest_sha256
        ):
            raise ValueError("full compiler source manifest changed")
        total = 0
        for path, digest, _ in self.members:
            payload = _read(self.root, path)
            total += len(payload)
            if total > MAX_BYTES or hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"full compiler baseline changed or exceeds bound: {path}")


def select(root, manifest_path, manifest_sha256):
    require_current_policy()
    root, manifest = Path(root).absolute(), Path(manifest_path).absolute()
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("full compiler root must be canonical")
    raw = _read(manifest.parent, manifest.name)
    if len(raw) > 4 << 20 or hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("full compiler manifest identity mismatch")
    data = json.loads(raw)
    if data.get("schema") != SCHEMA or not data.get("review_id"):
        raise ValueError("full compiler sources require explicit reviewed inventory")
    rows = []
    for row in data["members"]:
        path, digest, module = row["path"], row["sha256"], row["module"]
        if (
            _module(path) != module
            or (module is not None and forbidden(module))
            or (
                module is None
                and (
                    PurePosixPath(path).suffix not in SUPPORT_SUFFIXES
                    or forbidden(".".join(PurePosixPath(path).with_suffix("").parts))
                )
            )
        ):
            raise ValueError(f"withheld or non-code member in public compiler: {path}")
        rows.append((path, digest, module))
    if not rows or len(rows) > 8192 or len({r[0] for r in rows}) != len(rows):
        raise ValueError("full compiler inventory empty, duplicate or excessive")
    namespaces = tuple(data["mutable_namespaces"])
    allowed = set(MUTABLE_NAMESPACES)
    if not namespaces or set(namespaces) - allowed:
        raise ValueError("unreviewed mutable compiler namespace")
    result = FullCompilerSelection(
        root, manifest, manifest_sha256, tuple(rows), namespaces, tuple(data.get("unavailable_dependencies", ()))
    )
    result.verify()
    return result


def selected():
    value = _ACTIVE.get()
    if value is not _UNSET:
        return value
    raw = os.environ.get(ENV)
    if not raw:
        return None
    declaration = json.loads(raw)
    # Only trusted host orchestration supplies this environment. Never pass it
    # into the author/compiler sandbox as an admission authority.
    return select(declaration["root"], declaration["manifest_path"], declaration["manifest_sha256"])


@contextmanager
def scoped_full_compiler(selection):
    if selection is not None and type(selection) is not FullCompilerSelection:
        raise ValueError("full compiler scope requires an explicit original selection")
    token = _ACTIVE.set(selection)
    try:
        yield selection
    finally:
        _ACTIVE.reset(token)


def stage(selection, submission):
    """Complete compiler copy, preserving every existing continuation edit."""
    if type(selection) is not FullCompilerSelection:
        raise ValueError("full compiler staging needs reviewed selection")
    selection.verify()
    destination = Path(submission) / DIRECTORY
    if destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError("editable compiler destination is indirect")
    destination.mkdir(parents=True, exist_ok=True)
    copied, preserved = {}, {}
    for relative, digest, _ in selection.members:
        path = destination / relative
        if path.exists() or path.is_symlink():
            preserved[relative] = hashlib.sha256(_read(destination, relative)).hexdigest()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in path.parents):
            raise ValueError("editable compiler contains linked ancestors")
        payload = _read(selection.root, relative)
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o644)
        copied[relative] = digest
    preserved.update(
        {path: digest for path, digest in candidate_sources(selection, submission).items() if path not in copied}
    )
    expected = {path: digest for path, digest, _ in selection.members}
    differences = [
        dict(path=path, selected_baseline_sha256=expected[path], preserved_sha256=digest)
        for path, digest in sorted(preserved.items())
        if path in expected and digest != expected[path]
    ]
    versions = {
        "schema": "merlin.compiler_source_versions.v1",
        "scope": "initial staging only; later author edits have their own snapshot identities",
        "baseline_manifest_sha256": selection.manifest_sha256,
        "copied_members": len(copied),
        "preserved_matching_members": sum(expected.get(path) == digest for path, digest in preserved.items()),
        "preserved_differences": differences,
        "additional_authored_members": sorted(set(preserved) - set(expected)),
        "interpretation": "Differences may be authored edits or retained older sources. Inspect before explicit migration; no file was overwritten.",
        "qualification_issued": False,
    }
    return {
        "schema": "merlin.full_compiler_staging.v1",
        "selection": selection.record(),
        "destination": str(destination),
        "copied_missing_members": copied,
        "preserved_existing_members": preserved,
        "source_versions": versions,
        "backend_seed": "unchanged; no target implementation supplied",
    }


def prepare(repo, destination, *, review_id):
    """Operator preparation: generic compiler directories plus source dependencies.

    Missing/withheld dependencies are explicit, never silently granted. This
    copies implementation sources only, not module execution or corpus inputs.
    """
    require_current_policy()
    repo, destination = Path(repo).resolve(), Path(destination).absolute()
    if destination.exists() or not review_id:
        raise ValueError("public compiler preparation needs a new destination and review identity")
    source = repo / "src"
    pending = [
        p.relative_to(source).as_posix() for name in SOURCE_ROOTS for p in (source / "merlin" / name).rglob("*.py")
    ]
    pending += ["merlin/__init__.py", "merlin/compile_core.py", "merlin/compile_cli.py"]
    pending += [name.replace(".", "/") + ".py" for name in PUBLIC_HELPERS]
    members, denied, missing = {}, set(), set()
    while pending:
        relative = pending.pop()
        if relative in members:
            continue
        module = _module(relative)
        if forbidden(module):
            denied.add(module)
            continue
        if not (source / relative).is_file():
            missing.add(module)
            continue
        payload = _read(source, relative)
        members[relative] = (payload, module)
        package = module if relative.endswith("/__init__.py") else module.rpartition(".")[0]
        dependencies, direct = set(), set()
        for node in ast.walk(ast.parse(payload, filename=relative)):
            if isinstance(node, ast.Import):
                dependencies.update(a.name for a in node.names)
                direct.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = (
                    resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
                )
                dependencies.add(base)
                direct.add(base)
                dependencies.update(base + "." + a.name for a in node.names if a.name != "*")
        dependencies.update(".".join(module.split(".")[:i]) for i in range(1, len(module.split("."))))
        for dependency in dependencies:
            if not module_matches(dependency, "merlin"):
                continue
            choices = (dependency.replace(".", "/") + ".py", dependency.replace(".", "/") + "/__init__.py")
            found = next((c for c in choices if (source / c).is_file()), None)
            if found:
                pending.append(found)
            elif dependency in direct:
                missing.add(dependency)
    for path in (source / "merlin/runtime").glob("portable_*"):
        if path.suffix in (".c", ".h", ".LICENSE"):
            relative = path.relative_to(source).as_posix()
            members[relative] = (_read(source, relative), None)
    destination.mkdir(parents=True)
    root = destination / "python"
    root.mkdir()
    records = []
    for relative, (payload, module) in sorted(members.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o444)
        records.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "module": module})
    data = {
        "schema": SCHEMA,
        "review_id": review_id,
        "members": records,
        "source_roots": ["src/merlin/" + name for name in SOURCE_ROOTS],
        "mutable_namespaces": list(MUTABLE_NAMESPACES),
        "unavailable_dependencies": sorted(denied | missing),
        "scope": "generic compiler source; no corpora, kernels, grader, target backend or performance references",
    }
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    manifest.chmod(0o444)
    return select(root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())


def exact_staged_baseline(selection, submission):
    """A full generic baseline is not a target backend implementation seed."""
    root = Path(submission) / DIRECTORY
    expected = {p: digest for p, digest, _ in selection.members}
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            return False
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return actual == expected


def candidate_sources(selection, submission):
    """Bound a whole editable compiler tree; new implementation modules allowed."""
    root = Path(submission) / DIRECTORY
    if root.is_symlink() or not root.is_dir():
        raise ValueError("full editable compiler tree is missing or linked")
    total, files = 0, {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("editable compiler cannot contain links")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        module = _module(relative)
        if not path.is_file() or (module is not None and not selection.permits(module)):
            raise ValueError(f"editable compiler source outside public namespace: {relative}")
        if module is None:
            if path.suffix not in SUPPORT_SUFFIXES:
                raise ValueError(f"editable compiler contains non-source member: {relative}")
            # Data/config files do not create an escape from the public namespace
            # boundary (e.g. a private reference module renamed to .json).
            namespace = ".".join(PurePosixPath(relative).with_suffix("").parts)
            if not selection.permits(namespace) and relative not in {row[0] for row in selection.members}:
                raise ValueError(f"editable compiler support outside public namespace: {relative}")
            if forbidden(namespace):
                raise ValueError(f"editable compiler support outside public namespace: {relative}")
        payload = path.read_bytes()
        total += len(payload)
        if total > MAX_BYTES or len(files) >= 8192:
            raise ValueError("editable compiler source exceeds snapshot bound")
        files[relative] = hashlib.sha256(payload).hexdigest()
    return files


def apply_policy(policy, selection=None):
    """The writable author or immutable evaluated source is already bound by its owner."""
    from dataclasses import replace

    from merlin_experiments.phase2.compiler_libraries import _environment_prefix

    selection = selection if selection is not None else selected()
    if selection is None:
        return policy
    package = getattr(policy, "package", None)
    if package is None:
        package = Path(policy.workspace) / "submission"
    root = Path(package) / DIRECTORY
    if not root.is_dir() or root.is_symlink():
        raise ValueError("full compiler sandbox needs its staged source tree")
    # Do not mount the original checkout or private source inventory. The
    # existing workspace/candidate bind determines writable vs readonly access.
    return replace(policy, env_prefix="unset " + ENV + "; " + _environment_prefix(policy.env_prefix or "", root))
