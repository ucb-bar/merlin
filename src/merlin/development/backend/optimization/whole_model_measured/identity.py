"""What a whole-model measurement is a function of, and the small file primitives every owner shares.

A result belongs to the bytes that earned it, AND to the chain that turned those bytes into a
program, AND to the device that ran it.  This module names each of those identities:

* :func:`package_digest` -- the digest the rest of the loop already names a candidate by
  (:func:`merlin.common.tree_hash.hash_tree`), never a second one;
* :func:`program_digest` -- the bytes that can change the PROGRAM (documentation edits are not
  measurements);
* :func:`builder_identity` -- the builder callable AND the transitive closure of the modules it
  imports, read statically from the source files so the identity is a function of bytes on disk,
  never of which modules a process happens to have imported;
* :func:`store_root_for` -- the store a (builder identity, machine, build options) triple owns.

WHY THE CLOSURE IS STATIC.  A builder states its behaviour largely through the modules it calls into
(lowering passes, dialect helpers); a digest of its one named file misses every one of them.  The
first closure digest hashed every module of the builder's top-level package resident in
``sys.modules`` -- so two processes with different import histories (the launcher, a detached
worker) computed different identities for the same bytes.  Parsing the import statements of the
source files themselves, following only the builder's own top-level package, is the same closure
without that dependence.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from merlin.perf import package_identity as PI

BUILD_SCHEMA = "merlin_whole_model_build_record_v1"
IDENTITY_SCHEMA = "merlin.phase2.whole_model_measured.builder_identity.v2"


class IdentityError(RuntimeError):
    """An identity this measurement is attributed through cannot be established."""


def package_digest(package: Path) -> str:
    """The digest the rest of the loop names candidates by (:mod:`merlin.perf.package_identity`)."""
    try:
        return PI.package_digest(package)
    except PI.PackageIdentityError as exc:
        raise IdentityError(str(exc)) from exc


program_digest = PI.program_digest


def now() -> str:
    """A service timestamp: ``YYYYmmddTHHMMSSZ``, UTC."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def epoch_of(stamp: Any) -> float | None:
    """A service timestamp as epoch seconds, or None."""
    try:
        return datetime.strptime(str(stamp), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


@contextlib.contextmanager
def locked(directory: Path) -> Iterator[None]:
    """An exclusive advisory lock on ``directory`` (its ``.lock`` file)."""
    with (Path(directory) / ".lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def job_key(digest: str, replicate: int = 0) -> str:
    """A job's directory name: the digest, or the digest and which repeat of it this is."""
    return digest if replicate == 0 else f"{digest}.r{int(replicate)}"


def key_of(job: Mapping[str, Any]) -> str:
    return str(job.get("job_key") or job["package_sha256"])


# --------------------------------------------------------------- the builder's identity
def _installed_roots() -> list[Path]:
    """The interpreter's own library directories (standard library and installed distributions):
    modules under them are the environment's, not the build chain's source, and are not followed."""
    import sysconfig

    roots = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            roots.add(Path(value).resolve())
    return sorted(roots)


def _package_roots(top: str) -> list[Path]:
    """Every source directory top-level package ``top`` spans (an extended or namespace package spans
    several), from the imported package's own ``__path__``; empty for a non-package or an installed one."""
    installed = _installed_roots()
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    first = Path(next(iter(spec.submodule_search_locations))).resolve()
    if any(first == r or r in first.parents for r in installed):
        return []  # an installed distribution: never imported just to be excluded
    try:
        package = importlib.import_module(top)  # an extended package states every root only once imported
    except Exception:  # noqa: BLE001 -- an unimportable name contributes nothing to follow
        return []
    roots = []
    for entry in getattr(package, "__path__", None) or ():
        path = Path(entry).resolve()
        if path.is_dir() and not any(path == r or r in path.parents for r in installed):
            roots.append(path)
    return roots


def _module_file(roots: list[Path], top: str, dotted: str) -> tuple[Path, Path] | None:
    """``(root, source file)`` of ``dotted`` within the package roots, or None."""
    parts = dotted.split(".")
    if parts[0] != top:
        return None
    for root in roots:
        base = root.joinpath(*parts[1:])
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return root, candidate.resolve()
    return None


def _dotted_of(root: Path, top: str, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = [top, *relative.parts]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path, dotted: str) -> set[str]:
    """Every absolute module name ``path`` names in an import statement, at any depth (lazy imports
    inside functions included), with relative imports resolved."""
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return set()
    package = dotted if path.name == "__init__.py" else dotted.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package.split(".")
                if node.level > 1:
                    anchor = anchor[: -(node.level - 1)] if node.level - 1 < len(anchor) else []
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            if not base:
                continue
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return found


def module_closure(module_name: str) -> dict[str, str] | None:
    """``{package-relative path: sha256}`` of ``module_name`` and every SOURCE module it transitively
    imports (statically), across every top-level package that is not installed into the interpreter --
    the builder's own package AND the core it calls into -- each package ``__init__`` on the way included.

    Keys are relative to each package root's parent (``<top>/sub/mod.py``), never absolute, so the same
    bytes in a snapshot and in a checkout have the same identity.  ``None`` when the module's source
    cannot be located; the caller then refuses rather than guess a narrower identity."""
    roots_of: dict[str, list[Path]] = {}

    def locate(dotted: str) -> tuple[Path, Path] | None:
        top = dotted.partition(".")[0]
        if top not in roots_of:
            roots_of[top] = _package_roots(top)
        return _module_file(roots_of[top], top, dotted) if roots_of[top] else None

    located = locate(module_name)
    if located is None:
        return None
    seen: dict[Path, Path] = {}
    pending = [located]
    while pending:
        root, path = pending.pop()
        if path in seen:
            continue
        seen[path] = root
        top = next(t for t, rs in roots_of.items() if root in rs)
        dotted = _dotted_of(root, top, path)
        parts = dotted.split(".")
        for depth in range(1, len(parts)):
            init = locate(".".join(parts[:depth]))
            if init is not None and init[1] not in seen:
                pending.append(init)
        for name in _imports(path, dotted):
            target = locate(name)
            if target is not None and target[1] not in seen:
                pending.append(target)
    closure: dict[str, str] = {}
    for path, root in sorted(seen.items()):
        relative = path.relative_to(root.parent).as_posix()
        key, index = relative, 1
        while key in closure:
            index += 1
            key = f"{relative}#{index}"
        closure[key] = sha256_file(path)
    return closure


def closure_digest(closure: Mapping[str, str]) -> str:
    combined = hashlib.sha256()
    for relative in sorted(closure):
        combined.update(relative.encode("utf-8") + b"\0" + str(closure[relative]).encode("ascii") + b"\n")
    return combined.hexdigest()


def load_builder(spec: str, *, expected_sha256: str | None = None) -> Callable[..., Mapping[str, Any]]:
    """Resolve ``module:callable`` or ``/abs/file.py:callable``.

    A file-based builder must be pinned by ``expected_sha256``: a measurement is attributed to the
    build chain that produced its program, and an unpinned file can change between two jobs of one
    run without anything recording it.
    """
    target, sep, attribute = str(spec).rpartition(":")
    if not sep or not target or not attribute:
        raise IdentityError(f"builder spec {spec!r} is not module:callable or /path.py:callable")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_absolute() or not path.is_file():
            raise IdentityError(f"builder file {path} must be an absolute path to a file")
        observed = sha256_file(path)
        if not expected_sha256 or observed != expected_sha256:
            raise IdentityError(
                f"builder file {path} is {observed}, pinned {expected_sha256!r}; a file builder is admitted "
                "only at its pinned digest"
            )
        loader = importlib.util.spec_from_file_location(f"_whole_model_builder_{observed[:12]}", path)
        if loader is None or loader.loader is None:
            raise IdentityError(f"cannot load builder file {path}")
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise IdentityError(f"builder module {target!r} does not import: {exc}") from exc
    builder = getattr(module, attribute, None)
    if not callable(builder):
        raise IdentityError(f"builder {spec!r} does not name a callable")
    return builder


def builder_identity(spec: str, expected_sha256: str | None = None) -> dict[str, Any]:
    """The identity of the build chain a spec names.

    A MODULE spec is identified by its static transitive closure (:func:`module_closure`): the bytes
    the builder actually reads, not merely the one file its spec names.  A module whose closure
    cannot be read is REFUSED -- an identity silently narrowed to one file is how a behaviour-changing
    fix in a neighbouring module once went unrecorded.  A FILE spec is pinned (never re-hashed from
    disk): its digest is the expected one, admitted only at that exact value (see :func:`load_builder`).
    """
    target, _sep, attribute = str(spec).rpartition(":")
    if target.endswith(".py"):
        if not expected_sha256:
            raise IdentityError(f"file builder {spec!r} carries no pin")
        return {
            "schema": IDENTITY_SCHEMA,
            "spec": spec,
            "kind": "file",
            "callable": attribute,
            "sha256": expected_sha256,
        }
    closure = module_closure(target)
    if not closure:
        raise IdentityError(f"the builder module {target!r} has no readable source closure")
    return {
        "schema": IDENTITY_SCHEMA,
        "spec": spec,
        "kind": "module",
        "callable": attribute,
        "sha256": closure_digest(closure),
        "closure_files": len(closure),
    }


def store_root_for(
    base: Path,
    *,
    builder: Mapping[str, Any],
    machine: Mapping[str, Any],
    build_options: Mapping[str, Any] | None = None,
    builder_sha256: str | None = None,
) -> Path:
    """The store a (builder identity, machine, build options) triple owns under ``base``.

    A result is a function of the package bytes AND of the chain that turned them into a program AND
    of the device that ran it.  The chain is its IDENTITY -- the spec string and the digest of every
    module the spec's callable transitively imports (:func:`builder_identity`) -- so a
    behaviour-changing builder fix opens a new store rather than serving an old store's answers to a
    new builder's question.  ``builder_sha256`` may be passed when the caller already computed it.
    """
    spec = str(builder.get("spec") or "")
    if not spec:
        raise IdentityError("a store is keyed by a builder spec; none was given")
    digest = builder_sha256 or builder_identity(spec, builder.get("sha256")).get("sha256")
    identity = json.dumps(
        {
            "builder": {"spec": spec, "sha256": digest},
            "machine": dict(machine),
            "build_options": dict(build_options or {}),
        },
        sort_keys=True,
        default=str,
    )
    return Path(base) / hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def normalize_build_record(record: Mapping[str, Any], *, package_sha256: str) -> dict[str, Any]:
    """Require the fields a measurement is attributed through, and re-hash the ELF they name.

    Required: ``elf`` (an existing file whose bytes hash to ``elf_sha256``), ``parameter_header_sha256``
    (the ABI the program was compiled against), and ``expectations`` (per-group oracle digests or
    bounds, and the oracle classification).  ``groups`` (the per-group route census) is optional but is
    what makes the feedback actionable, so its absence is recorded.
    """
    from merlin.perf import whole_model_verdict as V

    if not isinstance(record, Mapping):
        raise IdentityError("the builder returned no record")
    elf = Path(str(record.get("elf") or ""))
    if not elf.is_absolute() or not elf.is_file():
        raise IdentityError(f"the build record names no ELF on disk ({elf})")
    observed = sha256_file(elf)
    declared = record.get("elf_sha256")
    if declared != observed:
        raise IdentityError(f"the build record says elf_sha256={declared!r} but the ELF is {observed}")
    header = record.get("parameter_header_sha256")
    if not isinstance(header, str) or len(header) != 64:
        raise IdentityError("the build record states no parameter_header_sha256; its ABI is UNKNOWN")
    expectations = record.get("expectations")
    V.Expectations.from_record(expectations if isinstance(expectations, Mapping) else {})
    groups = record.get("groups")
    protocol = record.get("protocol")
    return {
        "schema": BUILD_SCHEMA,
        "package_sha256": package_sha256,
        "elf": str(elf),
        "elf_sha256": observed,
        "parameter_header_sha256": header,
        "expectations": json.loads(json.dumps(dict(expectations or {}), default=str)),
        "groups": json.loads(json.dumps(list(groups), default=str)) if isinstance(groups, list) else None,
        "route_census": "present" if isinstance(groups, list) else "absent: feedback carries no routes",
        "protocol": dict(protocol) if isinstance(protocol, Mapping) else None,
        "builder_notes": json.loads(json.dumps(record.get("notes"), default=str)),
        "provenance": json.loads(json.dumps(record.get("provenance"), default=str)),
    }


__all__ = [
    "BUILD_SCHEMA",
    "IdentityError",
    "builder_identity",
    "closure_digest",
    "epoch_of",
    "job_key",
    "key_of",
    "load_builder",
    "locked",
    "module_closure",
    "normalize_build_record",
    "now",
    "package_digest",
    "program_digest",
    "read_json",
    "sha256_file",
    "store_root_for",
    "write_json_atomic",
]
