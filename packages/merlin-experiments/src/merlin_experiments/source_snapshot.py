"""Private, byte-verified source snapshots shared across experiment phases.

Source and corpus files are copies, not hard links. Runtime dependencies remain external and are
listed separately: their executable/hardware identities are verified by campaign preflight.
Ordinary source trees retain their existing link-dereferencing behavior; only selected external
providers enforce containment for every linked resource. This is not an atomic filesystem snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath

OMIT = {"__pycache__", "_qa_ws"}
SCHEMA = "merlin.performance-source-snapshot.v4"
ROOT_FILES = (".env", ".env.example", "pyproject.toml", "setup.cfg", "CMakeLists.txt")
PROVIDER_ROOT = "_selected_provider"
INPUT_ROOT = "_declared_inputs"


class SnapshotError(RuntimeError):
    pass


def selected_provider(target: str) -> dict:
    """Resolve source ownership once, in the trusted creating process."""
    from merlin.targetgen.target_registry import resolve

    info = resolve(target)
    if info.kind not in {"reference", "external"} or not info.base.is_dir():
        raise SnapshotError(f"selected target has no freezable support provider: {target}")
    return {"target": target, "resolved_target": info.name, "kind": info.kind, "source": str(info.base.resolve())}


def provider_environment(root: Path, receipt: dict) -> dict[str, str]:
    """Authoritative source selection, decoded without consulting the live registry."""
    _require_current(receipt)
    provider = receipt.get("selected_provider")
    return {
        "MERLIN_TARGET_PATH": str(root / provider["root"]) if provider and provider["kind"] == "external" else "",
        "MERLIN_TARGET_CONTRACT": "",
        "MERLIN_RTL_FACTS": "",
        "MERLIN_TARGETS_DIR": str(root / "merlin/targets"),
        "MERLIN_CONTRACT_DIR": str(root / "merlin/contract"),
        "MERLIN_SCHEMAS_DIR": str(root / "merlin/schemas"),
    }


def remap_input(root: Path, receipt: dict, path: Path, *, name: str | None = None) -> Path:
    """Remap source-owned inputs; external runtime evidence is not implicitly admitted."""
    path = path.absolute()
    if name is not None:
        declared = receipt.get("declared_inputs", {}).get(name)
        if declared is None or path != Path(declared["source"]):
            raise SnapshotError(f"input does not match sealed declaration: {name}")
        return root / declared["snapshot"]
    provider = receipt.get("selected_provider")
    if path.is_relative_to(root):
        relative = path.relative_to(root)
    elif provider and path.is_relative_to(Path(provider["source"])):
        relative = Path(provider["root"]) / path.relative_to(provider["source"])
    else:
        try:
            relative = path.relative_to(receipt["source_root"])
        except ValueError as exc:
            raise SnapshotError(f"input is outside frozen source ownership: {path}") from exc
    if relative.as_posix() not in receipt["files"]:
        raise SnapshotError(f"input is absent from frozen source inventory: {path}")
    return root / relative


def _relative(value: str) -> str:
    if not isinstance(value, str):
        raise SnapshotError("snapshot path must be a relative POSIX string")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or value == "."
        or "\\" in value
        or "\x00" in value
    ):
        raise SnapshotError(f"unsafe snapshot-relative path: {value!r}")
    return value


def _import_metadata(python_roots, legacy_roots, files, directories) -> dict:
    for roots in (python_roots, legacy_roots):
        if not isinstance(roots, list):
            raise SnapshotError("invalid explicit import roots")
        for relative in roots:
            if _relative(relative) not in directories:
                raise SnapshotError(f"import root is not a source directory: {relative}")
            if PurePosixPath(relative).parts[0] in {PROVIDER_ROOT, INPUT_ROOT}:
                raise SnapshotError("provider and input roots cannot be general Python import roots")
        if len(roots) != len(set(roots)):
            raise SnapshotError("duplicate explicit import roots")
    names = set()
    for relative in files:
        path = PurePosixPath(relative)
        if path.parent.as_posix() in legacy_roots and path.suffix == ".py" and path.stem != "__init__":
            if not path.stem.isidentifier():
                raise SnapshotError(f"invalid legacy Python module identity: {relative}")
            names.add(path.stem)
    return {"python_roots": python_roots, "legacy_roots": legacy_roots, "legacy_names": sorted(names)}


def _validate_roots(source: Path, roots: tuple[str, ...]) -> None:
    seen = []
    for relative in roots:
        path = PurePosixPath(_relative(relative))
        if path.parts[0] in {"out", ".venv", "third_party", PROVIDER_ROOT, INPUT_ROOT}:
            raise SnapshotError(f"runtime dependency cannot be a source root: {relative}")
        if any(path == old or path.is_relative_to(old) or old.is_relative_to(path) for old in seen):
            raise SnapshotError(f"overlapping snapshot source roots: {relative}")
        seen.append(path)
        candidate = source / relative
        if not candidate.is_dir() or not candidate.resolve().is_relative_to(source):
            raise SnapshotError(f"missing or escaping snapshot source: {candidate}")


def _copy_tree(source: Path, destination: Path, ignore, ancestors=frozenset()) -> None:
    """Dereference declared input links into independent bytes, refusing cycles/devices."""
    stat = source.stat()
    identity = (stat.st_dev, stat.st_ino)
    if identity in ancestors:
        raise SnapshotError(f"cyclic snapshot source directory: {source}")
    children = list(source.iterdir())
    omitted = ignore(str(source), [path.name for path in children])
    destination.mkdir()
    for child in children:
        if child.name in omitted:
            continue
        target = destination / child.name
        if child.is_dir():
            _copy_tree(child, target, ignore, ancestors | {identity})
        elif child.is_file():
            shutil.copy2(child, target)
        else:
            raise SnapshotError(f"nonregular snapshot source: {child}")
    shutil.copystat(source, destination)


def _tree_members(source: Path, ignore, ancestors=frozenset()) -> set[str]:
    stat = source.stat()
    identity = (stat.st_dev, stat.st_ino)
    if identity in ancestors:
        raise SnapshotError(f"cyclic snapshot source directory: {source}")
    children = list(source.iterdir())
    omitted = ignore(str(source), [path.name for path in children])
    members = {"."}
    for child in children:
        if child.name in omitted:
            continue
        if child.is_dir():
            members.update(f"{child.name}/{member}" for member in _tree_members(child, ignore, ancestors | {identity}))
        elif child.is_file():
            members.add(child.name)
        else:
            raise SnapshotError(f"nonregular snapshot source: {child}")
    return members


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seal(root: Path, stem: str, document: object) -> Path:
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = root / f"{stem}.{digest}.json"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return path


def load_seal(root: Path, stem: str) -> tuple[Path, dict]:
    files = list(root.glob(f"{stem}.*.json"))
    if len(files) != 1:
        raise SnapshotError(f"expected one {stem} seal in {root}")
    path = files[0]
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_mode & 0o222
        or path.name != f"{stem}.{sha_file(path)}.json"
    ):
        raise SnapshotError(f"invalid {stem} seal: {path}")
    return path, json.loads(path.read_text())


def create(
    source: Path,
    destination: Path,
    *,
    output_root: Path,
    source_roots: tuple[str, ...],
    python_roots: tuple[str, ...],
    legacy_roots: tuple[str, ...],
    internal_aliases: dict[str, str] | None = None,
    exclude_paths: tuple[str, ...] = (),
    directory_memberships: dict[str, tuple[str, ...] | None] | None = None,
    target_name: str | None = None,
    provider: dict | None = None,
    declared_inputs: dict[str, Path] | None = None,
) -> Path:
    """Publish a private snapshot. An interrupted destination cannot masquerade as a seal.

    ``directory_memberships`` binds caller-selected directory child names, or explicit
    absence with ``None``, before copying and again before sealing. An empty tuple requires
    an existing empty directory; absence rejects files, directories and dangling links.
    """
    source, destination = source.resolve(), destination.absolute()
    source_roots = tuple(source_roots)
    _validate_roots(source, source_roots)
    excludes = {PurePosixPath(_relative(relative)) for relative in exclude_paths}
    memberships = {}
    if directory_memberships is not None and not isinstance(directory_memberships, dict):
        raise SnapshotError("invalid source selection directory memberships")
    for relative, names in (directory_memberships or {}).items():
        _relative(relative)
        if names is None:
            memberships[relative] = None
            continue
        if not isinstance(names, tuple):
            raise SnapshotError("source selection membership must be a tuple of child names or None")
        for name in names:
            if len(PurePosixPath(_relative(name)).parts) != 1:
                raise SnapshotError(f"unsafe source selection child name: {name!r}")
        if len(names) != len(set(names)):
            raise SnapshotError("duplicate source selection child names")
        memberships[relative] = frozenset(names)

    def verify_memberships() -> None:
        for relative, names in memberships.items():
            selected = source / relative
            if names is None:
                if selected.exists() or selected.is_symlink():
                    raise SnapshotError(f"source selection membership changed: {relative}")
                continue
            try:
                actual = {path.name for path in selected.iterdir()}
            except OSError as exc:
                raise SnapshotError(f"source selection membership changed: {relative}") from exc
            if actual != names:
                raise SnapshotError(f"source selection membership changed: {relative}")

    verify_memberships()
    input_sources = {}
    for name, path in (declared_inputs or {}).items():
        if not isinstance(name, str) or not name.isidentifier():
            raise SnapshotError(f"invalid declared input name: {name!r}")
        path = Path(path).absolute()
        if path.is_symlink() or not path.is_file():
            raise SnapshotError(f"declared input is not an ordinary file: {path}")
        input_sources[name] = path
    provider_record = None
    if provider is not None:
        if set(provider) != {"target", "resolved_target", "kind", "source"} or provider["target"] != target_name:
            raise SnapshotError("provider selection does not match snapshot target")
        provider_source = Path(provider["source"]).resolve(strict=True)
        if not provider_source.is_dir() or provider["kind"] not in {"reference", "external"}:
            raise SnapshotError("invalid selected provider source")
        if destination.resolve().is_relative_to(provider_source) or provider_source.is_relative_to(
            destination.resolve()
        ):
            raise SnapshotError("snapshot destination overlaps selected provider")
        provider_record = {**provider, "source": str(provider_source)}
        if provider["kind"] == "reference":
            provider_record["root"] = provider_source.relative_to(source).as_posix()
            if not any(provider_source.is_relative_to(source / relative) for relative in source_roots):
                raise SnapshotError("reference provider is outside copied source roots")
        else:
            provider_record["root"] = PROVIDER_ROOT
    if destination.resolve().is_relative_to(source) and any(
        destination.resolve().is_relative_to((source / relative).resolve()) for relative in source_roots
    ):
        raise SnapshotError("snapshot destination overlaps a copied source tree")
    if target_name is not None and (
        not target_name or Path(target_name).name != target_name or target_name in (".", "..")
    ):
        raise SnapshotError("target_name must be one safe path component")
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"snapshot destination already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    files = {}

    def ignored(directory: str, names: list[str]) -> set[str]:
        omitted = set(names) & OMIT
        current = Path(directory).relative_to(source)
        omitted.update(name for name in names if PurePosixPath(current / name) in excludes)
        return omitted

    members = {relative: _tree_members(source / relative, ignored) for relative in source_roots}
    for relative in source_roots:
        src = source / relative
        if not src.is_dir():
            raise SnapshotError(f"missing snapshot source: {src}")
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        _copy_tree(src, destination / relative, ignored)
    external_provider = provider_record is not None and provider_record["kind"] == "external"
    if external_provider:

        def provider_ignore(directory, names):
            current = Path(directory)
            omitted = set(names) & {"__pycache__", ".git"}
            for name in names:
                if name in omitted:
                    continue
                path = current / name
                if not path.resolve(strict=True).is_relative_to(provider_source):
                    raise SnapshotError(f"selected provider resource escapes its root: {path}")
            return omitted

        provider_members = _tree_members(provider_source, provider_ignore)
        _copy_tree(provider_source, destination / PROVIDER_ROOT, provider_ignore)
    input_records = {}
    copied_inputs = {}
    for name, path in input_sources.items():
        canonical = path.resolve(strict=True)
        relative = None
        if provider_record and canonical.is_relative_to(provider_source):
            relative = Path(provider_record["root"]) / canonical.relative_to(provider_source)
        elif canonical.is_relative_to(source):
            relative = canonical.relative_to(source)
        if relative is None or not (destination / relative).is_file():
            relative = Path(INPUT_ROOT) / name / path.name
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied_inputs[relative.as_posix()] = path
        input_records[name] = {"source": str(path), "snapshot": relative.as_posix()}
    for relative in ROOT_FILES:
        src = source / relative
        if src.is_file():
            shutil.copy2(src, destination / relative)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            relative = path.relative_to(destination).as_posix()
            digest = sha_file(path)
            original = copied_inputs.get(relative) or (
                provider_source / path.relative_to(destination / PROVIDER_ROOT)
                if external_provider and path.is_relative_to(destination / PROVIDER_ROOT)
                else source / relative
            )
            if digest != sha_file(original):
                raise SnapshotError(f"source changed during snapshot: {relative}")
            files[relative] = digest
            path.chmod(path.stat().st_mode & ~0o222)
    if any(_tree_members(source / relative, ignored) != members[relative] for relative in source_roots):
        raise SnapshotError("source membership changed during snapshot")
    if external_provider:
        if _tree_members(provider_source, provider_ignore) != provider_members:
            raise SnapshotError("provider membership changed during snapshot")
        source_roots = (*source_roots, PROVIDER_ROOT)
    if copied_inputs:
        source_roots = (*source_roots, INPUT_ROOT)
    for name, path in input_sources.items():
        if path.is_symlink() or not path.is_file() or sha_file(path) != files[input_records[name]["snapshot"]]:
            raise SnapshotError(f"declared input changed during snapshot: {name}")
    aliases = dict(internal_aliases or {})
    for relative, target in aliases.items():
        _relative(relative)
        _relative(target)
        link = destination / relative
        if (
            link.exists()
            or link.is_symlink()
            or not (destination / target).is_dir()
            or (destination / target).is_symlink()
            or not (destination / target).resolve().is_relative_to(destination.resolve())
            or not link.parent.resolve().is_relative_to(destination.resolve())
        ):
            raise SnapshotError(f"invalid snapshot internal alias: {relative}")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(destination / target, link.parent), target_is_directory=True)
    directories = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_dir() and not path.is_symlink()
    )
    metadata = _import_metadata(list(python_roots), list(legacy_roots), files, directories)
    links = {}
    for relative, target in (
        ("out", output_root),
        (".venv", source / ".venv"),
        ("third_party", source / "third_party"),
    ):
        if target.exists():
            (destination / relative).symlink_to(target.resolve(), target_is_directory=True)
            links[relative] = str(target.resolve())
    document = {
        "schema": SCHEMA,
        "source_root": str(source),
        "source_roots": list(source_roots),
        "files": files,
        "external_links": links,
        "internal_aliases": aliases,
        "directories": directories,
        "selected_provider": provider_record,
        "declared_inputs": input_records,
        **metadata,
        "limits": "External tools and outputs are not immutable source; campaign pins verify engines.",
    }
    _validate_layout(document)
    verify_memberships()
    receipt = seal(destination, "snapshot", document)
    for path in destination.rglob("*"):
        if not path.is_symlink() and path.is_dir():
            path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(0o500)
    return receipt


def import_layout(root: Path, verified_receipt: dict) -> dict:
    """Decode explicit sealed ownership; historical execution needs its original verifier."""
    _require_current(verified_receipt)
    _validate_layout(verified_receipt)
    return {key: verified_receipt[key] for key in ("python_roots", "legacy_roots", "legacy_names")}


def _require_current(receipt: dict) -> None:
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise SnapshotError("source snapshot requires a newly frozen run with explicit import ownership")


def _validate_layout(receipt: dict) -> None:
    """Validate explicit inventories independently of the original live tree."""
    roots, directories = receipt.get("source_roots"), receipt.get("directories")
    files, links, aliases = receipt.get("files"), receipt.get("external_links"), receipt.get("internal_aliases")
    if (
        not isinstance(roots, list)
        or not isinstance(directories, list)
        or not isinstance(files, dict)
        or not isinstance(links, dict)
        or not isinstance(aliases, dict)
        or not isinstance(receipt.get("source_root"), str)
        or not Path(receipt["source_root"]).is_absolute()
    ):
        raise SnapshotError("malformed snapshot source inventory")
    for relative in [*roots, *directories, *files, *links, *aliases, *aliases.values()]:
        _relative(relative)
    if len(set(roots)) != len(roots) or directories != sorted(set(directories)):
        raise SnapshotError("duplicate snapshot source inventory")
    for index, relative in enumerate(roots):
        candidate = PurePosixPath(relative)
        if relative not in directories or candidate.parts[0] in {"out", ".venv", "third_party"}:
            raise SnapshotError(f"invalid snapshot source root: {relative}")
        if any(
            candidate.is_relative_to(PurePosixPath(other)) or PurePosixPath(other).is_relative_to(candidate)
            for other in roots[index + 1 :]
        ):
            raise SnapshotError("overlapping snapshot source roots")
    root_paths = [PurePosixPath(relative) for relative in roots]
    if any(
        relative not in ROOT_FILES and not any(PurePosixPath(relative).is_relative_to(owner) for owner in root_paths)
        for relative in files
    ):
        raise SnapshotError("snapshot files outside declared source roots")
    if any(
        not any(
            PurePosixPath(relative).is_relative_to(owner) or owner.is_relative_to(PurePosixPath(relative))
            for owner in root_paths
        )
        and not any(PurePosixPath(alias).is_relative_to(PurePosixPath(relative)) for alias in aliases)
        for relative in directories
    ):
        raise SnapshotError("snapshot directories outside declared source roots")
    if set(links) - {"out", ".venv", "third_party"} or any(
        not isinstance(target, str) or not Path(target).is_absolute() for target in links.values()
    ):
        raise SnapshotError("invalid external dependency links")
    for relative, target in aliases.items():
        _relative(relative)
        _relative(target)
        if (
            relative in files
            or relative in directories
            or relative in links
            or target not in directories
            or PurePosixPath(relative).parts[0] in {"out", ".venv", "third_party"}
            or any(
                PurePosixPath(relative).is_relative_to(PurePosixPath(other)) for other in aliases if other != relative
            )
        ):
            raise SnapshotError("invalid snapshot internal aliases")
    if any(
        not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
        for digest in files.values()
    ):
        raise SnapshotError("invalid source digest")
    expected = _import_metadata(receipt.get("python_roots"), receipt.get("legacy_roots"), files, directories)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise SnapshotError("snapshot import ownership changed")
    if receipt["schema"] == SCHEMA:
        inputs = receipt.get("declared_inputs")
        if not isinstance(inputs, dict):
            raise SnapshotError("missing declared input inventory")
        for name, row in inputs.items():
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or not isinstance(row, dict)
                or set(row) != {"source", "snapshot"}
                or not isinstance(row["source"], str)
                or not Path(row["source"]).is_absolute()
                or _relative(row["snapshot"]) not in files
            ):
                raise SnapshotError("invalid declared input mapping")
        if (INPUT_ROOT in roots) != any(
            PurePosixPath(row["snapshot"]).parts[0] == INPUT_ROOT for row in inputs.values()
        ):
            raise SnapshotError("declared input source inventory disagrees with mapping")
        mapped_inputs = {
            row["snapshot"] for row in inputs.values() if PurePosixPath(row["snapshot"]).parts[0] == INPUT_ROOT
        }
        if mapped_inputs != {relative for relative in files if PurePosixPath(relative).parts[0] == INPUT_ROOT}:
            raise SnapshotError("declared input files disagree with mapping")
        for name, row in inputs.items():
            relative = PurePosixPath(row["snapshot"])
            if relative.parts[0] == INPUT_ROOT and (
                len(relative.parts) != 3 or relative.parts[1] != name or relative.name != Path(row["source"]).name
            ):
                raise SnapshotError("invalid copied input mapping")
        provider = receipt.get("selected_provider")
        if provider is not None:
            if (
                not isinstance(provider, dict)
                or set(provider) != {"target", "resolved_target", "kind", "source", "root"}
                or any(not isinstance(provider[key], str) or not provider[key] for key in provider)
                or provider["kind"] not in {"reference", "external"}
                or not Path(provider["source"]).is_absolute()
                or _relative(provider["root"]) not in directories
                or (provider["kind"] == "external" and provider["root"] != PROVIDER_ROOT)
                or (
                    provider["kind"] == "reference"
                    and (
                        not any(PurePosixPath(provider["root"]).is_relative_to(owner) for owner in root_paths)
                        or Path(provider["source"]) != Path(receipt["source_root"]) / provider["root"]
                    )
                )
            ):
                raise SnapshotError("invalid sealed provider ownership")
            if any(
                PurePosixPath(relative).is_relative_to(PurePosixPath(provider["root"]))
                or PurePosixPath(provider["root"]).is_relative_to(PurePosixPath(relative))
                for relative in (*receipt["python_roots"], *receipt["legacy_roots"])
            ):
                raise SnapshotError("provider directories cannot be general Python import roots")
        if (PROVIDER_ROOT in roots) != bool(provider and provider["kind"] == "external"):
            raise SnapshotError("provider source inventory disagrees with selection")


def verify(root: Path) -> dict:
    if root.is_symlink():
        raise SnapshotError("snapshot root cannot be a symlink")
    path, receipt = load_seal(root, "snapshot")
    _require_current(receipt)
    _validate_layout(receipt)
    aliases = receipt["internal_aliases"]
    actual = set()
    actual_directories = set()
    actual_links = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        if base.stat().st_mode & 0o222:
            raise SnapshotError(f"snapshot directory became writable: {base}")
        if base != root:
            actual_directories.add(base.relative_to(root).as_posix())
        for name in list(dirs):
            candidate = base / name
            if candidate.is_symlink():
                dirs.remove(name)
                relative = candidate.relative_to(root).as_posix()
                actual_links.add(relative)
                if relative in aliases:
                    target = root / aliases[relative]
                    expected_link = os.path.relpath(target, candidate.parent)
                    if (
                        target.is_symlink()
                        or not target.is_dir()
                        or os.readlink(candidate) != expected_link
                        or candidate.resolve() != target.absolute()
                    ):
                        raise SnapshotError(f"snapshot internal alias changed: {relative}")
                elif receipt["external_links"].get(relative) != str(candidate.resolve(strict=True)):
                    raise SnapshotError(f"snapshot dependency link changed: {relative}")
        for name in names:
            candidate = base / name
            if candidate == path:
                continue
            relative = candidate.relative_to(root).as_posix()
            actual.add(relative)
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.stat().st_mode & 0o222
                or receipt["files"].get(relative) != sha_file(candidate)
            ):
                raise SnapshotError(f"snapshot source changed: {relative}")
    if actual != set(receipt["files"]):
        raise SnapshotError("snapshot source set changed")
    if actual_directories != set(receipt["directories"]):
        raise SnapshotError("snapshot directory set changed")
    if actual_links != set(receipt["external_links"]) | set(aliases):
        raise SnapshotError("snapshot link set changed")
    for relative, target in receipt["external_links"].items():
        link = root / relative
        if not link.is_symlink() or str(link.resolve(strict=True)) != target:
            raise SnapshotError(f"snapshot dependency link missing: {relative}")
    return receipt
