"""Public build closures, multi-tool authority, and the private-I/O barrier.

This module is the isolated v4 infrastructure tracer.  A public closure is a
symlink-free tree whose every file is named and hashed.  A separate authority
binds all build tools, the sysroot tree, and static libraries.  Private session
bytes can be loaded only with a sealed barrier returned after the closed tracer
recipe has rebuilt, partially linked, and re-verified its outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .paper_session_abi import (
    InputEndpoint,
    InputFrame,
    SessionDescriptor,
    assert_private_data_excluded,
    decode_request,
    decode_response,
    descriptor_from_dict,
    encode_request,
)
from .paper_session_tracer import (
    ENTRYPOINT,
    render_model_source,
    synthetic_prefill_decode_descriptor,
)

PUBLIC_SCHEMA = "merlin.paper.public-build-bundle/v1"
AUTHORITY_SCHEMA = "merlin.paper.multi-toolchain-authority/v1"
BARRIER_SCHEMA = "merlin.paper.verified-build-barrier/v1"
PRIVATE_SCHEMA = "merlin.paper.private-session-bundle/v1"
RESOURCE_ROLES_SCHEMA = "merlin.paper.public-resource-roles/v1"
TOOL_ROLES = frozenset({"c_compiler", "cxx_compiler", "linker", "cmake", "ninja"})

_RESOURCE_ROLES = frozenset({
    "c_source", "cxx_source", "header", "mlir", "producer_manifest",
    "session_descriptor", "static_library",
})
_TEXT_RESOURCE_ROLES = frozenset({
    "c_source", "cxx_source", "header", "mlir", "producer_manifest",
    "session_descriptor",
})

_TEXT_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".h", ".json", ".toml", ".txt", ".yaml", ".yml"})
_PRIVATE_PATH_TOKENS = frozenset({
    "correctness", "golden", "goldens", "private", "reference", "references",
    "session_goldens", "session_inputs", "session_quality",
})
_PRIVATE_CONTENT_MARKERS = (
    b"session_goldens", b"session_quality", b"correctness.npz", b"quality.npz",
    b"reference.npy", b"reference.npz", b"private/", b"private\\",
)
_BARRIER_SEAL = object()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str],
                where: str) -> None:
    actual = {str(key) for key in value}
    if actual != set(expected):
        raise ValueError(
            f"{where} fields differ from the closed schema: "
            f"missing={sorted(set(expected) - actual)} extra={sorted(actual - set(expected))}")


def _safe_relative(value: object, where: str) -> PurePosixPath:
    text = str(value or "")
    path = PurePosixPath(text)
    if (not text or path.is_absolute() or path.as_posix() != text
            or any(part in {"", ".", ".."} for part in path.parts)):
        raise ValueError(f"{where} must be a normalized relative path without traversal")
    return path


def _bound_path(root: Path, value: object, where: str) -> Path:
    relative = _safe_relative(value, where)
    path = root.joinpath(*relative.parts)
    if path.is_symlink():
        raise ValueError(f"{where} cannot be a symlink")
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{where} escapes its bundle")
    return resolved


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _load_json(path: Path, where: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{where} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where} is invalid JSON: {exc}") from exc
    return _mapping(value, where)


def _forbid_private_file(path: Path, relative: str) -> None:
    lowered = relative.lower()
    tokens = set()
    for part in PurePosixPath(lowered).parts:
        normalized = part.replace("-", "_").replace(".", "_")
        tokens.update(token for token in normalized.split("_") if token)
        tokens.add(normalized)
    if path.suffix.lower() in {".npy", ".npz"} or tokens & _PRIVATE_PATH_TOKENS:
        raise ValueError(f"public build closure contains a private stream/reference path: {relative}")
    if path.suffix.lower() in _TEXT_SUFFIXES:
        lowered_bytes = path.read_bytes().lower()
        if any(marker in lowered_bytes for marker in _PRIVATE_CONTENT_MARKERS):
            raise ValueError(
                f"public build closure contains a private stream/reference marker: {relative}")


def _tree_entries(root: Path, *, forbid_private: bool) -> tuple[dict[str, Any], ...]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"tree resource is absent: {root}")
    entries: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in [*names, *files]:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ValueError(f"tree resource contains a symlink: {candidate}")
        for name in sorted(files):
            path = directory_path / name
            if not path.is_file():
                raise ValueError(f"tree resource contains a non-regular file: {path}")
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative, "tree entry")
            if forbid_private:
                _forbid_private_file(path, relative)
            stat = path.stat()
            entries.append({
                "path": relative,
                "sha256": _sha_file(path),
                "size": stat.st_size,
                "executable": bool(stat.st_mode & 0o111),
            })
    return tuple(sorted(entries, key=lambda row: row["path"]))


def _tree_identity(root: Path, *, forbid_private: bool) -> dict[str, Any]:
    entries = _tree_entries(root, forbid_private=forbid_private)
    return {"sha256": _sha_bytes(_canonical(entries)), "files": list(entries)}


@dataclass(frozen=True)
class PublicBuildBundle:
    manifest_path: Path
    closure_root: Path
    tree_sha256: str
    manifest_sha256: str
    files: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ToolIdentity:
    role: str
    path: Path
    sha256: str
    version_sha256: str
    version_line: str


@dataclass(frozen=True)
class TargetABI:
    name: str
    target_triple: str
    march: str
    mabi: str
    features: tuple[str, ...]
    elf_class: int
    elf_machine: int
    elf_osabi: int
    elf_flags_mask: int
    elf_flags_value: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "target_triple": self.target_triple,
            "march": self.march, "mabi": self.mabi, "features": list(self.features),
            "elf_class": self.elf_class, "elf_machine": self.elf_machine,
            "elf_osabi": self.elf_osabi, "elf_flags_mask": self.elf_flags_mask,
            "elf_flags_value": self.elf_flags_value,
        }


@dataclass(frozen=True)
class TreeResourceIdentity:
    name: str
    path: Path
    sha256: str
    files: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FileResourceIdentity:
    name: str
    path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class MultiToolchainAuthority:
    path: Path
    tools: tuple[ToolIdentity, ...]
    sysroot: Path
    sysroot_sha256: str
    sysroot_files: tuple[dict[str, Any], ...]
    static_libraries: tuple[tuple[str, Path, str, int], ...]
    tree_resources: tuple[TreeResourceIdentity, ...]
    file_resources: tuple[FileResourceIdentity, ...]
    target_abi: TargetABI
    sha256: str

    def tool(self, role: str) -> ToolIdentity:
        matches = [tool for tool in self.tools if tool.role == role]
        if len(matches) != 1:
            raise ValueError(f"toolchain authority has no unique {role!r} tool")
        return matches[0]

    def tree_resource(self, name: str) -> TreeResourceIdentity:
        matches = [resource for resource in self.tree_resources if resource.name == name]
        if len(matches) != 1:
            raise ValueError(f"toolchain authority has no unique {name!r} tree resource")
        return matches[0]

    def file_resource(self, name: str) -> FileResourceIdentity:
        matches = [resource for resource in self.file_resources if resource.name == name]
        if len(matches) != 1:
            raise ValueError(f"toolchain authority has no unique {name!r} file resource")
        return matches[0]


class VerifiedBuildBarrier:
    """Unforgeable-by-construction token returned only by barrier verification."""

    __slots__ = ("public_manifest", "authority_path", "receipt_path", "receipt_sha256",
                 "runner", "composite_object", "descriptor", "_seal", "_verifier")

    def __init__(self, *, public_manifest: Path, authority_path: Path, receipt_path: Path,
                 receipt_sha256: str, runner: Path, composite_object: Path,
                 descriptor: SessionDescriptor, _seal: object, _verifier: object = None) -> None:
        if _seal is not _BARRIER_SEAL:
            raise TypeError("VerifiedBuildBarrier values come only from verify_build_barrier")
        self.public_manifest = public_manifest
        self.authority_path = authority_path
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256
        self.runner = runner
        self.composite_object = composite_object
        self.descriptor = descriptor
        self._seal = _seal
        self._verifier = _verifier


@dataclass(frozen=True)
class PrivateSessionBundle:
    root: Path
    request: bytes
    reference_response: bytes
    descriptor_sha256: str


def issue_verified_build_barrier(
        *, public_manifest: Path, authority_path: Path, receipt_path: Path,
        runner: Path, composite_object: Path, descriptor: SessionDescriptor,
        verifier: object) -> VerifiedBuildBarrier:
    """Issue a sealed token for a producer-specific closed replay verifier."""
    if not callable(verifier):
        raise TypeError("verified barrier requires a callable closed replay verifier")
    return VerifiedBuildBarrier(
        public_manifest=public_manifest, authority_path=authority_path,
        receipt_path=receipt_path, receipt_sha256=_sha_file(receipt_path),
        runner=runner, composite_object=composite_object, descriptor=descriptor,
        _seal=_BARRIER_SEAL, _verifier=verifier)


def write_public_resource_roles(
        closure_root: str | Path, resources: Mapping[str, str]) -> Path:
    """Declare the semantic role of every public build resource."""
    root = Path(closure_root).resolve()
    rows = []
    for raw_path, role in sorted(resources.items()):
        relative = _safe_relative(raw_path, "public resource path").as_posix()
        if relative == "resource_roles.json":
            raise ValueError("resource_roles.json cannot declare itself")
        if role not in _RESOURCE_ROLES:
            raise ValueError(f"unsupported public resource role {role!r}")
        rows.append({"path": relative, "role": role})
    path = root / "resource_roles.json"
    _write_json(path, {"schema": RESOURCE_ROLES_SCHEMA, "resources": rows})
    return path


def _verify_public_resource_roles(root: Path, files: Sequence[Mapping[str, Any]]) -> None:
    document = _load_json(root / "resource_roles.json", "public resource roles")
    _exact_keys(document, {"schema", "resources"}, "public resource roles")
    if document.get("schema") != RESOURCE_ROLES_SCHEMA:
        raise ValueError("public resource-role schema differs")
    raw_rows = document.get("resources")
    if not isinstance(raw_rows, list):
        raise ValueError("public resource roles must be a list")
    declared: dict[str, str] = {}
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, f"public resource roles[{index}]")
        _exact_keys(row, {"path", "role"}, f"public resource roles[{index}]")
        relative = _safe_relative(row.get("path"), f"public resource roles[{index}].path")
        role = str(row.get("role", ""))
        if relative.as_posix() in declared or role not in _RESOURCE_ROLES:
            raise ValueError("public resource roles contain a duplicate path or unknown role")
        declared[relative.as_posix()] = role
    if list(declared) != sorted(declared):
        raise ValueError("public resource roles are not in canonical path order")
    actual = {str(row["path"]) for row in files} - {"resource_roles.json"}
    if set(declared) != actual:
        raise ValueError(
            "public resource roles do not exactly cover the build closure: "
            f"missing={sorted(actual - set(declared))} "
            f"extra={sorted(set(declared) - actual)}")
    for relative, role in declared.items():
        path = _bound_path(root, relative, f"public {role} resource")
        if role in _TEXT_RESOURCE_ROLES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"public {role} resource is not UTF-8 text: {relative}") from exc
            if "\x00" in text:
                raise ValueError(f"public {role} resource contains binary bytes: {relative}")
        if role == "static_library" and not path.read_bytes().startswith(b"!<arch>\n"):
            raise ValueError(f"public static_library is not an ar archive: {relative}")
        if role == "session_descriptor":
            descriptor_from_dict(_load_json(path, f"public session descriptor {relative}"))
        elif role == "producer_manifest":
            assert_private_data_excluded(
                _load_json(path, f"public producer manifest {relative}"),
                f"public producer manifest {relative}")


def snapshot_public_build_bundle(closure_root: str | Path,
                                 manifest_path: str | Path) -> PublicBuildBundle:
    closure_root, manifest_path = Path(closure_root).resolve(), Path(manifest_path).resolve()
    if manifest_path.is_relative_to(closure_root):
        raise ValueError("public manifest must be outside the closure it describes")
    tree = _tree_identity(closure_root, forbid_private=False)
    _verify_public_resource_roles(closure_root, tree["files"])
    relative = os.path.relpath(closure_root, manifest_path.parent)
    relative_path = PurePosixPath(Path(relative).as_posix())
    _safe_relative(relative_path.as_posix(), "public closure root")
    document = {"schema": PUBLIC_SCHEMA, "closure_root": relative_path.as_posix(), "tree": tree}
    _write_json(manifest_path, document)
    return verify_public_build_bundle(manifest_path)


def verify_public_build_bundle(manifest_path: str | Path) -> PublicBuildBundle:
    manifest_path = Path(manifest_path).resolve()
    document = _load_json(manifest_path, "public build bundle")
    _exact_keys(document, {"schema", "closure_root", "tree"}, "public build bundle")
    if document.get("schema") != PUBLIC_SCHEMA:
        raise ValueError("public build bundle schema differs")
    closure_root = _bound_path(
        manifest_path.parent, document.get("closure_root"), "public closure root")
    expected_tree = _mapping(document.get("tree"), "public build tree")
    _exact_keys(expected_tree, {"sha256", "files"}, "public build tree")
    expected_files = expected_tree.get("files")
    if not isinstance(expected_files, list):
        raise ValueError("public build tree files must be a list")
    for index, raw in enumerate(expected_files):
        row = _mapping(raw, f"public build tree files[{index}]")
        _exact_keys(row, {"path", "sha256", "size", "executable"},
                    f"public build tree files[{index}]")
        _bound_path(closure_root, row.get("path"), f"public build tree files[{index}].path")
    actual = _tree_identity(closure_root, forbid_private=False)
    if actual != expected_tree:
        raise ValueError("public build closure tree differs from its canonical file manifest")
    _verify_public_resource_roles(closure_root, actual["files"])
    return PublicBuildBundle(
        manifest_path, closure_root, str(actual["sha256"]), _sha_file(manifest_path),
        tuple(actual["files"]),
    )


def _tool_identity(role: str, path: str | Path) -> ToolIdentity:
    raw = Path(path)
    # Preserve the invoked basename for multicall tools such as ld.lld -> lld;
    # resolving that symlink changes the selected driver flavor.  The content
    # hash still follows the symlink and binds the executable bytes.
    invoked = Path(os.path.abspath(raw))
    if not invoked.is_file() or not os.access(invoked, os.X_OK):
        raise ValueError(f"{role} tool is not an executable regular file: {invoked}")
    completed = subprocess.run(
        [str(invoked), "--version"], capture_output=True, timeout=15,
        cwd="/", stdin=subprocess.DEVNULL, close_fds=True,
        env={"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": ""})
    version = completed.stdout + completed.stderr
    if completed.returncode:
        raise ValueError(f"{role} tool does not support identity query --version")
    text = version.decode("utf-8", errors="replace").splitlines()
    return ToolIdentity(
        role, invoked, _sha_file(invoked), _sha_bytes(version), text[0] if text else "")


def write_multi_toolchain_authority(
        path: str | Path, *, tools: Mapping[str, str | Path], sysroot: str | Path,
        static_libraries: Mapping[str, str | Path], tree_resources: Mapping[str, str | Path],
        file_resources: Mapping[str, str | Path],
        target_abi: Mapping[str, object]) -> MultiToolchainAuthority:
    path = Path(path).resolve()
    if not TOOL_ROLES.issubset(tools):
        raise ValueError(
            f"toolchain roles must include {sorted(TOOL_ROLES)}, got {sorted(tools)}")
    identities = tuple(_tool_identity(role, tools[role]) for role in sorted(tools))
    sysroot_path = Path(sysroot).resolve()
    sysroot_tree = _tree_identity(sysroot_path, forbid_private=False)
    libraries = []
    for name, raw_path in sorted(static_libraries.items()):
        if not name.isascii() or not name.isidentifier():
            raise ValueError(f"static-library identity name is unsafe: {name!r}")
        library = Path(raw_path)
        if library.is_symlink():
            raise ValueError(f"static library cannot be a symlink: {library}")
        library = library.resolve()
        if not library.is_file():
            raise ValueError(f"static library is absent: {library}")
        libraries.append({
            # Producer-owned static libraries can live in the portable public closure.  Keeping
            # this path relative to the authority makes an otherwise identical retained graph
            # independently replayable after relocation.
            "name": name, "path": os.path.relpath(library, path.parent),
            "sha256": _sha_file(library),
            "size": library.stat().st_size,
        })
    resources = []
    for name, raw_path in sorted(tree_resources.items()):
        if not name.isascii() or not name.isidentifier():
            raise ValueError(f"tree-resource identity name is unsafe: {name!r}")
        resource_path = Path(raw_path)
        if resource_path.is_symlink():
            raise ValueError(f"tree resource root cannot be a symlink: {resource_path}")
        resource_path = resource_path.resolve()
        identity = _tree_identity(resource_path, forbid_private=False)
        resources.append({"name": name, "path": str(resource_path), **identity})
    file_identities = []
    for name, raw_path in sorted(file_resources.items()):
        if not name.isascii() or not name.isidentifier():
            raise ValueError(f"file-resource identity name is unsafe: {name!r}")
        resource_path = Path(raw_path)
        if resource_path.is_symlink():
            raise ValueError(f"file resource cannot be a symlink: {resource_path}")
        resource_path = resource_path.resolve()
        if not resource_path.is_file():
            raise ValueError(f"file resource is absent: {resource_path}")
        file_identities.append({
            "name": name, "path": str(resource_path), "sha256": _sha_file(resource_path),
            "size": resource_path.stat().st_size,
        })
    document = {
        "schema": AUTHORITY_SCHEMA,
        "tools": [
            {"role": row.role, "path": str(row.path), "sha256": row.sha256,
             "version_sha256": row.version_sha256, "version_line": row.version_line}
            for row in identities
        ],
        "sysroot": {"path": str(sysroot_path), **sysroot_tree},
        "static_libraries": libraries,
        "tree_resources": resources,
        "file_resources": file_identities,
        "target_abi": dict(target_abi),
    }
    _write_json(path, document)
    return load_multi_toolchain_authority(path)


def load_multi_toolchain_authority(path: str | Path) -> MultiToolchainAuthority:
    path = Path(path).resolve()
    document = _load_json(path, "multi-toolchain authority")
    _exact_keys(document, {"schema", "tools", "sysroot", "static_libraries",
                           "tree_resources", "file_resources", "target_abi"},
                "multi-toolchain authority")
    if document.get("schema") != AUTHORITY_SCHEMA:
        raise ValueError("multi-toolchain authority schema differs")
    raw_tools = document.get("tools")
    if not isinstance(raw_tools, list):
        raise ValueError("multi-toolchain authority tools must be a list")
    tools: list[ToolIdentity] = []
    for index, raw in enumerate(raw_tools):
        row = _mapping(raw, f"tools[{index}]")
        _exact_keys(row, {"role", "path", "sha256", "version_sha256", "version_line"},
                    f"tools[{index}]")
        role = str(row.get("role", ""))
        actual = _tool_identity(role, str(row.get("path", "")))
        if (actual.sha256 != row.get("sha256")
                or actual.version_sha256 != row.get("version_sha256")
                or actual.version_line != row.get("version_line")):
            raise ValueError(f"tool identity differs for role {role!r}")
        tools.append(actual)
    roles = {tool.role for tool in tools}
    if not TOOL_ROLES.issubset(roles) or len(roles) != len(tools):
        raise ValueError("multi-toolchain authority has missing required or duplicate tool roles")
    if [tool.role for tool in tools] != sorted(roles):
        raise ValueError("multi-toolchain authority tools are not in canonical role order")

    raw_sysroot = _mapping(document.get("sysroot"), "authority sysroot")
    _exact_keys(raw_sysroot, {"path", "sha256", "files"}, "authority sysroot")
    sysroot = Path(str(raw_sysroot.get("path", ""))).resolve()
    actual_sysroot = _tree_identity(sysroot, forbid_private=False)
    if actual_sysroot != {"sha256": raw_sysroot.get("sha256"), "files": raw_sysroot.get("files")}:
        raise ValueError("toolchain sysroot tree differs from its authority identity")

    raw_libraries = document.get("static_libraries")
    if not isinstance(raw_libraries, list):
        raise ValueError("authority static_libraries must be a list")
    libraries: list[tuple[str, Path, str, int]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_libraries):
        row = _mapping(raw, f"static_libraries[{index}]")
        _exact_keys(row, {"name", "path", "sha256", "size"}, f"static_libraries[{index}]")
        name = str(row.get("name", ""))
        library = Path(str(row.get("path", "")))
        if not library.is_absolute():
            library = path.parent / library
        if library.is_symlink():
            raise ValueError(f"static library cannot be a symlink: {library}")
        library = library.resolve()
        if (name in names or not library.is_file() or _sha_file(library) != row.get("sha256")
                or library.stat().st_size != row.get("size")):
            raise ValueError(f"static-library identity differs for {name!r}")
        names.add(name)
        libraries.append((name, library, str(row["sha256"]), int(row["size"])))
    if [row[0] for row in libraries] != sorted(row[0] for row in libraries):
        raise ValueError("static-library identities are not in canonical name order")
    raw_resources = document.get("tree_resources")
    if not isinstance(raw_resources, list):
        raise ValueError("authority tree_resources must be a list")
    resources: list[TreeResourceIdentity] = []
    resource_names: set[str] = set()
    for index, raw in enumerate(raw_resources):
        row = _mapping(raw, f"tree_resources[{index}]")
        _exact_keys(row, {"name", "path", "sha256", "files"}, f"tree_resources[{index}]")
        name = str(row.get("name", ""))
        resource_path = Path(str(row.get("path", "")))
        if resource_path.is_symlink():
            raise ValueError(f"tree resource root cannot be a symlink: {resource_path}")
        resource_path = resource_path.resolve()
        actual = _tree_identity(resource_path, forbid_private=False)
        if (name in resource_names or not name.isascii() or not name.isidentifier()
                or actual != {"sha256": row.get("sha256"), "files": row.get("files")}):
            raise ValueError(f"tree-resource identity differs for {name!r}")
        resource_names.add(name)
        resources.append(TreeResourceIdentity(
            name, resource_path, str(actual["sha256"]), tuple(actual["files"])))
    if [row.name for row in resources] != sorted(row.name for row in resources):
        raise ValueError("tree-resource identities are not in canonical name order")
    raw_file_resources = document.get("file_resources")
    if not isinstance(raw_file_resources, list):
        raise ValueError("authority file_resources must be a list")
    file_identities: list[FileResourceIdentity] = []
    file_names: set[str] = set()
    for index, raw in enumerate(raw_file_resources):
        row = _mapping(raw, f"file_resources[{index}]")
        _exact_keys(row, {"name", "path", "sha256", "size"}, f"file_resources[{index}]")
        name = str(row.get("name", ""))
        resource_path = Path(str(row.get("path", "")))
        if resource_path.is_symlink():
            raise ValueError(f"file resource cannot be a symlink: {resource_path}")
        resource_path = resource_path.resolve()
        if (name in file_names or not name.isascii() or not name.isidentifier()
                or not resource_path.is_file() or _sha_file(resource_path) != row.get("sha256")
                or resource_path.stat().st_size != row.get("size")):
            raise ValueError(f"file-resource identity differs for {name!r}")
        file_names.add(name)
        file_identities.append(FileResourceIdentity(
            name, resource_path, str(row["sha256"]), int(row["size"])))
    if [row.name for row in file_identities] != sorted(row.name for row in file_identities):
        raise ValueError("file-resource identities are not in canonical name order")
    raw_target = _mapping(document.get("target_abi"), "authority target ABI")
    _exact_keys(raw_target, {
        "name", "target_triple", "march", "mabi", "features", "elf_class",
        "elf_machine", "elf_osabi", "elf_flags_mask", "elf_flags_value",
    }, "authority target ABI")
    target_name = str(raw_target.get("name", ""))
    target_triple = str(raw_target.get("target_triple", ""))
    march, mabi = str(raw_target.get("march", "")), str(raw_target.get("mabi", ""))
    features = raw_target.get("features")
    elf_class, elf_machine = raw_target.get("elf_class"), raw_target.get("elf_machine")
    elf_osabi = raw_target.get("elf_osabi")
    flags_mask, flags_value = raw_target.get("elf_flags_mask"), raw_target.get("elf_flags_value")
    if (not target_name.isascii() or not target_name.isidentifier()
            or not target_triple.isascii() or not target_triple
            or not march.isascii() or not march or not mabi.isascii() or not mabi
            or not isinstance(features, list)
            or any(not isinstance(feature, str) or not feature.isascii() or not feature
                   for feature in features)
            or features != sorted(set(features))
            or elf_class not in {32, 64} or not isinstance(elf_machine, int)
            or isinstance(elf_machine, bool) or elf_machine < 1
            or not isinstance(elf_osabi, int) or isinstance(elf_osabi, bool)
            or not 0 <= elf_osabi <= 255
            or not isinstance(flags_mask, int) or isinstance(flags_mask, bool)
            or not isinstance(flags_value, int) or isinstance(flags_value, bool)
            or not 0 <= flags_mask <= 0xFFFFFFFF or not 0 <= flags_value <= 0xFFFFFFFF
            or flags_value & ~flags_mask):
        raise ValueError("authority target ABI is invalid")
    target = TargetABI(
        target_name, target_triple, march, mabi, tuple(features), elf_class,
        elf_machine, elf_osabi, flags_mask, flags_value)
    return MultiToolchainAuthority(
        path, tuple(tools), sysroot, str(actual_sysroot["sha256"]),
        tuple(actual_sysroot["files"]), tuple(libraries), tuple(resources), tuple(file_identities),
        target, _sha_file(path))


def materialize_synthetic_public_closure(root: str | Path) -> Path:
    """Write the multi-file public closure used by the synthetic tracer recipe."""
    root = Path(root).resolve()
    descriptor = synthetic_prefill_decode_descriptor()
    (root / "sources").mkdir(parents=True, exist_ok=True)
    (root / "descriptor").mkdir(parents=True, exist_ok=True)
    (root / "lib").mkdir(parents=True, exist_ok=True)
    (root / "sources" / "model_session.c").write_text(
        render_model_source(descriptor), encoding="utf-8")
    (root / "sources" / "runner.c").write_text(
        _freestanding_host_runner_source(), encoding="utf-8")
    (root / "sources" / "support.cc").write_text(_support_source(), encoding="utf-8")
    (root / "descriptor" / "session_descriptor.json").write_bytes(
        descriptor.canonical_bytes + b"\n")
    # A valid deterministic empty ar archive.  Production libraries will be
    # source-derived separately; this one exists to exercise library identity
    # and link-closure omission checks without another unpinned build action.
    (root / "lib" / "libpublic_anchor.a").write_bytes(b"!<arch>\n")
    write_public_resource_roles(root, {
        "descriptor/session_descriptor.json": "session_descriptor",
        "lib/libpublic_anchor.a": "static_library",
        "sources/model_session.c": "c_source",
        "sources/runner.c": "c_source",
        "sources/support.cc": "cxx_source",
    })
    return root


def _support_source() -> str:
    return 'extern "C" int merlin_public_support(void) { return 0; }\n'


def _freestanding_host_runner_source() -> str:
    """x86-64-only liveness driver with no ambient libc/startup dependency."""
    return f'''typedef __SIZE_TYPE__ size_t;
typedef unsigned char u8;
extern int {ENTRYPOINT}(const char *,const u8 *,size_t,u8 *,size_t,size_t *);
static u8 request[1048576]; static u8 response[1048576];
static long syscall3(long n,long a,long b,long c){{long r;__asm__ volatile(
  "syscall":"=a"(r):"a"(n),"D"(a),"S"(b),"d"(c):"rcx","r11","memory");return r;}}
static void run(void){{size_t used=0,out=0;long n,rc;
  while(used<sizeof(request)){{n=syscall3(0,0,(long)(request+used),sizeof(request)-used);if(n<=0)break;used+=(size_t)n;}}
  rc={ENTRYPOINT}(".",request,used,response,sizeof(response),&out);
  if(!rc&&syscall3(1,1,(long)response,out)!=(long)out)rc=93;
  syscall3(60,rc,0,0);for(;;){{}}
}}
__attribute__((noreturn)) void _start(void){{run();__builtin_unreachable();}}
'''


def _validate_synthetic_public_graph(public: PublicBuildBundle) -> None:
    expected = {
        "descriptor/session_descriptor.json", "lib/libpublic_anchor.a",
        "resource_roles.json", "sources/model_session.c", "sources/runner.c",
        "sources/support.cc",
    }
    if {str(row["path"]) for row in public.files} != expected:
        raise ValueError("synthetic tracer public resources differ from its exact path-role graph")
    descriptor = descriptor_from_dict(_load_json(
        public.closure_root / "descriptor/session_descriptor.json",
        "public session descriptor"))
    exact_text = {
        "sources/model_session.c": render_model_source(descriptor),
        "sources/runner.c": _freestanding_host_runner_source(),
        "sources/support.cc": _support_source(),
    }
    for relative, expected_source in exact_text.items():
        if (public.closure_root / relative).read_text(encoding="utf-8") != expected_source:
            raise ValueError(f"synthetic tracer {relative} differs from deterministic producer output")


def _controlled_build_env(authority: MultiToolchainAuthority) -> dict[str, str]:
    return {
        "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "SOURCE_DATE_EPOCH": "0",
        "PATH": os.pathsep.join(sorted({str(tool.path.parent) for tool in authority.tools})),
    }


def _run(argv: Sequence[str], where: str,
         authority: MultiToolchainAuthority | None = None) -> None:
    completed = subprocess.run(
        list(argv), capture_output=True, timeout=60,
        env=_controlled_build_env(authority) if authority is not None else None,
        cwd=str(authority.sysroot) if authority is not None else "/",
        stdin=subprocess.DEVNULL, close_fds=True)
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{where} failed ({completed.returncode}): {detail}")


def _elf_identity(path: Path) -> dict[str, Any]:
    """Inspect an ELF without relying on an unbound host ``nm``/``readelf``."""
    raw = path.read_bytes()
    if len(raw) < 64 or raw[:4] != b"\x7fELF" or raw[4] not in {1, 2} or raw[5] not in {1, 2}:
        raise ValueError(f"rebuilt output is not a supported ELF file: {path.name}")
    elf_class, byte_order = raw[4], raw[5]
    endian = "<" if byte_order == 1 else ">"
    elf_type, machine = struct.unpack_from(endian + "HH", raw, 16)
    osabi = raw[7]
    if elf_class == 2:
        elf_flags = struct.unpack_from(endian + "I", raw, 48)[0]
        section_offset = struct.unpack_from(endian + "Q", raw, 40)[0]
        section_size, section_count = struct.unpack_from(endian + "HH", raw, 58)
        section_format = endian + "IIQQQQIIQQ"
        symbol_format = endian + "IBBHQQ"
    else:
        elf_flags = struct.unpack_from(endian + "I", raw, 36)[0]
        section_offset = struct.unpack_from(endian + "I", raw, 32)[0]
        section_size, section_count = struct.unpack_from(endian + "HH", raw, 46)
        section_format = endian + "IIIIIIIIII"
        symbol_format = endian + "IIIBBH"
    expected_section_size = struct.calcsize(section_format)
    if (section_size != expected_section_size or section_count < 1
            or section_offset + section_size * section_count > len(raw)):
        raise ValueError(f"rebuilt ELF section table is malformed: {path.name}")
    sections = [
        struct.unpack_from(section_format, raw, section_offset + index * section_size)
        for index in range(section_count)
    ]
    definitions: list[str] = []
    for section in sections:
        section_type, offset, size, linked, entry_size = (
            section[1], section[4], section[5], section[6], section[9])
        if section_type not in {2, 11}:
            continue
        if (linked >= len(sections) or entry_size != struct.calcsize(symbol_format)
                or offset + size > len(raw) or size % entry_size):
            raise ValueError(f"rebuilt ELF symbol table is malformed: {path.name}")
        strings = sections[linked]
        string_offset, string_size = strings[4], strings[5]
        if string_offset + string_size > len(raw):
            raise ValueError(f"rebuilt ELF string table is malformed: {path.name}")
        string_data = raw[string_offset:string_offset + string_size]
        for at in range(offset, offset + size, entry_size):
            symbol = struct.unpack_from(symbol_format, raw, at)
            if elf_class == 2:
                name_at, info, section_index = symbol[0], symbol[1], symbol[3]
            else:
                name_at, info, section_index = symbol[0], symbol[3], symbol[5]
            if section_index == 0 or info >> 4 not in {1, 2} or name_at >= len(string_data):
                continue
            end = string_data.find(b"\0", name_at)
            if end < 0:
                raise ValueError(f"rebuilt ELF symbol name is malformed: {path.name}")
            definitions.append(string_data[name_at:end].decode("ascii", errors="strict"))
    return {
        "class": 64 if elf_class == 2 else 32,
        "type": elf_type,
        "machine": machine,
        "osabi": osabi,
        "flags": elf_flags,
        "global_definitions": tuple(sorted(definitions)),
    }


def _elf_matches_target(elf: Mapping[str, Any], target: TargetABI) -> bool:
    return (
        elf.get("class") == target.elf_class
        and elf.get("machine") == target.elf_machine
        and elf.get("osabi") == target.elf_osabi
        and int(elf.get("flags", -1)) & target.elf_flags_mask == target.elf_flags_value
    )


def _public_probe_request(descriptor: SessionDescriptor) -> bytes:
    return encode_request(descriptor, [
        InputFrame(InputEndpoint(program, input_index), step,
                   struct.pack(">Q", ordinal + 1))
        for ordinal, (program, input_index, step)
        in enumerate(descriptor.required_input_keys)
    ])


def _validate_public_session_runner(runner: Path, descriptor: SessionDescriptor,
                                    runtime_root: Path) -> bytes:
    completed = subprocess.run(
        [str(runner), str(runtime_root)], input=_public_probe_request(descriptor),
        capture_output=True, timeout=30,
        cwd=runtime_root, close_fds=True,
        env={"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": ""})
    if completed.returncode:
        raise ValueError(
            f"rebuilt runner rejected the public MRLNSES2 liveness request "
            f"({completed.returncode})")
    decoded = decode_response(completed.stdout, expected_descriptor=descriptor)
    if decoded.executed_calls != descriptor.calls or len(decoded.outputs) != descriptor.output.frames:
        raise ValueError("rebuilt runner did not produce the complete public session trajectory")
    return completed.stdout


def _tracer_recipe(public: PublicBuildBundle, authority: MultiToolchainAuthority,
                   output_root: Path) -> dict[str, list[str]]:
    source = public.closure_root / "sources" / "model_session.c"
    support = public.closure_root / "sources" / "support.cc"
    runner = public.closure_root / "sources" / "runner.c"
    library_paths = [str(row[1]) for row in authority.static_libraries]
    c_compiler = str(authority.tool("c_compiler").path)
    cxx_compiler = str(authority.tool("cxx_compiler").path)
    linker = str(authority.tool("linker").path)
    sysroot = str(authority.sysroot)
    expected_host = TargetABI(
        "host_tracer_x86_64", "x86_64-unknown-linux-gnu", "x86-64", "sysv",
        (), 64, 62, 0, 0xFFFFFFFF, 0)
    if authority.target_abi != expected_host:
        raise ValueError("synthetic host tracer requires explicit ELF64 x86-64 target authority")
    target_flags = [f"--target={authority.target_abi.target_triple}",
                    f"-march={authority.target_abi.march}"]
    if {resource.name for resource in authority.tree_resources} != {"compiler_resource_dir"}:
        raise ValueError("synthetic host tracer requires the exact compiler resource tree")
    compiler_resource = str(authority.tree_resource("compiler_resource_dir").path)
    compiler_flags = [*target_flags, f"-resource-dir={compiler_resource}"]
    expected_library = (public.closure_root / "lib" / "libpublic_anchor.a").resolve()
    if (len(authority.static_libraries) != 1
            or authority.static_libraries[0][0] != "public_anchor"
            or authority.static_libraries[0][1] != expected_library):
        raise ValueError("synthetic relink requires the exact bound public_anchor static library")
    return {
        "compile_model": [c_compiler, *compiler_flags, f"--sysroot={sysroot}",
                          "-ffreestanding", "-fno-builtin",
                          "-std=c11", "-O2", "-c", str(source),
                          "-o", str(output_root / "model_session.o")],
        "compile_support": [cxx_compiler, *compiler_flags, f"--sysroot={sysroot}",
                            "-ffreestanding", "-fno-builtin",
                            "-std=c++17", "-O2", "-c", str(support),
                            "-o", str(output_root / "support.o")],
        "compile_runner": [c_compiler, *compiler_flags, f"--sysroot={sysroot}",
                           "-ffreestanding", "-fno-builtin",
                           "-std=c11", "-O2", "-c", str(runner),
                           "-o", str(output_root / "runner.o")],
        "partial_link": [linker, "-m", "elf_x86_64", f"--sysroot={sysroot}", "-r",
                         str(output_root / "model_session.o"),
                         str(output_root / "support.o"), "-o", str(output_root / "composite.o")],
        "link_runner": [linker, "-m", "elf_x86_64", f"--sysroot={sysroot}",
                        "-nostdlib", "-static", "-e", "_start",
                        str(output_root / "runner.o"), str(output_root / "composite.o"),
                        *library_paths, "-o", str(output_root / "runner")],
    }


def _replay_synthetic_outputs(public: PublicBuildBundle,
                              authority: MultiToolchainAuthority,
                              descriptor: SessionDescriptor) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="merlin-paper-barrier-replay-") as raw_root:
        root = Path(raw_root).resolve()
        recipe = _tracer_recipe(public, authority, root)
        for name in ("compile_model", "compile_support", "compile_runner",
                     "partial_link", "link_runner"):
            _run(recipe[name], f"independent replay {name}", authority)
        replay = {
            "model_object": root / "model_session.o",
            "support_object": root / "support.o",
            "runner_object": root / "runner.o",
            "composite_object": root / "composite.o",
            "runner": root / "runner",
        }
        _verify_synthetic_output_graph(root, replay, receipt=None)
        composite_elf = _elf_identity(replay["composite_object"])
        runner_elf = _elf_identity(replay["runner"])
        if composite_elf["type"] != 1:
            raise ValueError("independent replay composite is not an ELF relocatable object")
        if runner_elf["type"] not in {2, 3}:
            raise ValueError("independent replay runner is not an ELF executable/shared image")
        if (not _elf_matches_target(composite_elf, authority.target_abi)
                or not _elf_matches_target(runner_elf, authority.target_abi)):
            raise ValueError("independent replay ELF does not match the bound target ABI")
        if composite_elf["machine"] != runner_elf["machine"]:
            raise ValueError("independent replay objects target different ELF machines")
        if composite_elf["global_definitions"].count(ENTRYPOINT) != 1:
            raise ValueError("independent replay does not export exactly one common session entrypoint")
        if composite_elf["global_definitions"].count("merlin_public_support") != 1:
            raise ValueError("independent replay did not consume the deterministic support object")
        _validate_public_session_runner(replay["runner"], descriptor, root)
        # Temporary files disappear at block exit, so return their independently
        # computed identities rather than paths.
        return {name: _sha_file(path) for name, path in replay.items()}


def _verify_synthetic_output_graph(root: Path, outputs: Mapping[str, Path], *,
                                   receipt: Path | None) -> None:
    root = root.resolve()
    expected = {path.resolve().relative_to(root).as_posix() for path in outputs.values()}
    if receipt is not None:
        expected.add(receipt.resolve().relative_to(root).as_posix())
    actual: set[str] = set()
    directories: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*names, *files]:
            if (base / name).is_symlink():
                raise ValueError("synthetic build output graph contains a symlink")
        directories.update((base / name).relative_to(root).as_posix() for name in names)
        actual.update((base / name).relative_to(root).as_posix() for name in files)
    if actual != expected or directories:
        raise ValueError("synthetic build output graph has omitted or extra paths")


def build_and_relink_synthetic(
        public_manifest: str | Path, authority_path: str | Path,
        output_root: str | Path) -> Path:
    """Run the closed multi-file tracer recipe and emit a barrier receipt."""
    public = verify_public_build_bundle(public_manifest)
    authority = load_multi_toolchain_authority(authority_path)
    _validate_synthetic_public_graph(public)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor_path = public.closure_root / "descriptor" / "session_descriptor.json"
    descriptor = descriptor_from_dict(_load_json(descriptor_path, "public session descriptor"))
    recipe = _tracer_recipe(public, authority, output_root)
    for name in ("compile_model", "compile_support", "compile_runner",
                 "partial_link", "link_runner"):
        _run(recipe[name], name, authority)
    outputs = {
        name: {"path": path.name, "sha256": _sha_file(path), "size": path.stat().st_size}
        for name, path in {
            "model_object": output_root / "model_session.o",
            "support_object": output_root / "support.o",
            "runner_object": output_root / "runner.o",
            "composite_object": output_root / "composite.o",
            "runner": output_root / "runner",
        }.items()
    }
    receipt = {
        "schema": BARRIER_SCHEMA,
        "public_manifest_sha256": public.manifest_sha256,
        "public_tree_sha256": public.tree_sha256,
        "toolchain_authority_sha256": authority.sha256,
        "descriptor_sha256": descriptor.sha256,
        "entrypoint": ENTRYPOINT,
        "target_abi": authority.target_abi.to_dict(),
        "recipe": recipe,
        "outputs": outputs,
    }
    receipt_path = output_root / "build_barrier_receipt.json"
    _write_json(receipt_path, receipt)
    _verify_synthetic_output_graph(
        output_root, {
            "model_object": output_root / "model_session.o",
            "support_object": output_root / "support.o",
            "runner_object": output_root / "runner.o",
            "composite_object": output_root / "composite.o",
            "runner": output_root / "runner",
        }, receipt=receipt_path)
    return receipt_path


def verify_build_barrier(public_manifest: str | Path, authority_path: str | Path,
                         receipt_path: str | Path) -> VerifiedBuildBarrier:
    """Verify public inputs, tools, relink recipe, and outputs before granting private I/O."""
    public = verify_public_build_bundle(public_manifest)
    authority = load_multi_toolchain_authority(authority_path)
    _validate_synthetic_public_graph(public)
    receipt_path = Path(receipt_path).resolve()
    receipt = _load_json(receipt_path, "build barrier receipt")
    _exact_keys(receipt, {
        "schema", "public_manifest_sha256", "public_tree_sha256",
        "toolchain_authority_sha256", "descriptor_sha256", "entrypoint", "target_abi",
        "recipe", "outputs",
    }, "build barrier receipt")
    if (receipt.get("schema") != BARRIER_SCHEMA or receipt.get("entrypoint") != ENTRYPOINT):
        raise ValueError("build barrier schema or entrypoint differs")
    if (receipt.get("public_manifest_sha256") != public.manifest_sha256
            or receipt.get("public_tree_sha256") != public.tree_sha256
            or receipt.get("toolchain_authority_sha256") != authority.sha256):
        raise ValueError("build barrier input identities differ")
    expected_target = authority.target_abi.to_dict()
    if receipt.get("target_abi") != expected_target:
        raise ValueError("build barrier target ABI differs from its authority")
    descriptor = descriptor_from_dict(_load_json(
        public.closure_root / "descriptor" / "session_descriptor.json",
        "public session descriptor"))
    if receipt.get("descriptor_sha256") != descriptor.sha256:
        raise ValueError("build barrier descriptor identity differs")
    expected_recipe = _tracer_recipe(public, authority, receipt_path.parent)
    if receipt.get("recipe") != expected_recipe:
        raise ValueError("build barrier recipe differs from the closed deterministic relink")
    outputs = _mapping(receipt.get("outputs"), "build barrier outputs")
    expected_output_names = {
        "model_object", "support_object", "runner_object", "composite_object", "runner"}
    _exact_keys(outputs, expected_output_names, "build barrier outputs")
    resolved: dict[str, Path] = {}
    for name in sorted(expected_output_names):
        row = _mapping(outputs[name], f"build barrier output {name}")
        _exact_keys(row, {"path", "sha256", "size"}, f"build barrier output {name}")
        path = _bound_path(receipt_path.parent, row.get("path"), f"build barrier output {name}")
        if (not path.is_file() or _sha_file(path) != row.get("sha256")
                or path.stat().st_size != row.get("size")):
            raise ValueError(f"build barrier output identity differs for {name}")
        resolved[name] = path
    _verify_synthetic_output_graph(receipt_path.parent, resolved, receipt=receipt_path)
    replay_hashes = _replay_synthetic_outputs(public, authority, descriptor)
    for name, path in resolved.items():
        if replay_hashes[name] != _sha_file(path):
            raise ValueError(
                f"build barrier output differs from independent clean replay for {name}")
    composite_elf = _elf_identity(resolved["composite_object"])
    runner_elf = _elf_identity(resolved["runner"])
    if (composite_elf["type"] != 1 or runner_elf["type"] not in {2, 3}
            or composite_elf["machine"] != runner_elf["machine"]
            or not _elf_matches_target(composite_elf, authority.target_abi)
            or not _elf_matches_target(runner_elf, authority.target_abi)
            or composite_elf["global_definitions"].count(ENTRYPOINT) != 1
            or composite_elf["global_definitions"].count("merlin_public_support") != 1):
        raise ValueError("build barrier ELF identity or common session export differs")
    _validate_public_session_runner(resolved["runner"], descriptor, receipt_path.parent)
    return issue_verified_build_barrier(
        public_manifest=public.manifest_path, authority_path=authority.path,
        receipt_path=receipt_path, runner=resolved["runner"],
        composite_object=resolved["composite_object"], descriptor=descriptor,
        verifier=verify_build_barrier)


def _reverify_barrier(barrier: VerifiedBuildBarrier) -> VerifiedBuildBarrier:
    if not isinstance(barrier, VerifiedBuildBarrier) or barrier._seal is not _BARRIER_SEAL:
        raise PermissionError("a verified build/relink barrier is required before private I/O")
    if _sha_file(barrier.receipt_path) != barrier.receipt_sha256:
        raise ValueError("build barrier receipt changed after verification")
    if not callable(barrier._verifier):
        raise PermissionError("verified build barrier has no closed replay verifier")
    return barrier._verifier(
        barrier.public_manifest, barrier.authority_path, barrier.receipt_path)


def materialize_private_session_bundle(
        root: str | Path, *, request: bytes, reference_response: bytes,
        descriptor: SessionDescriptor) -> Path:
    """Materialize synthetic private bytes separately from the public build closure."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "request.bin"
    response_path = root / "reference_response.bin"
    request_path.write_bytes(request)
    response_path.write_bytes(reference_response)
    manifest = {
        "schema": PRIVATE_SCHEMA,
        "descriptor_sha256": descriptor.sha256,
        "request": {"path": request_path.name, "sha256": _sha_file(request_path)},
        "reference_response": {
            "path": response_path.name, "sha256": _sha_file(response_path)},
    }
    manifest_path = root / "private_session.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def load_private_session_bundle(
        root: str | Path, *, barrier: VerifiedBuildBarrier | None) -> PrivateSessionBundle:
    """Open private bytes only after re-verifying the explicit build/relink barrier."""
    if barrier is None:
        raise PermissionError("a verified build/relink barrier is required before private I/O")
    verified = _reverify_barrier(barrier)
    root = Path(root).resolve()
    manifest = _load_json(root / "private_session.json", "private session manifest")
    _exact_keys(manifest, {
        "schema", "descriptor_sha256", "request", "reference_response"},
        "private session manifest")
    if manifest.get("schema") != PRIVATE_SCHEMA:
        raise ValueError("private session bundle schema differs")
    if manifest.get("descriptor_sha256") != verified.descriptor.sha256:
        raise ValueError("private session descriptor differs from the rebuilt public session")

    def payload(field: str) -> bytes:
        row = _mapping(manifest.get(field), f"private session {field}")
        _exact_keys(row, {"path", "sha256"}, f"private session {field}")
        path = _bound_path(root, row.get("path"), f"private session {field}")
        if not path.is_file():
            raise ValueError(f"private session {field} is absent")
        value = path.read_bytes()
        if _sha_bytes(value) != row.get("sha256"):
            raise ValueError(f"private session {field} hash differs")
        return value

    request = payload("request")
    reference_response = payload("reference_response")
    decode_request(request, expected_descriptor=verified.descriptor)
    decode_response(reference_response, expected_descriptor=verified.descriptor)
    return PrivateSessionBundle(
        root, request, reference_response, str(manifest["descriptor_sha256"]))


def run_after_barrier(barrier: VerifiedBuildBarrier, request: bytes) -> bytes:
    """Execute the rebuilt runner only while the barrier remains valid."""
    verified = _reverify_barrier(barrier)
    completed = subprocess.run(
        [str(verified.runner), str(verified.runner.parent)], input=request,
        capture_output=True, timeout=30, cwd=verified.runner.parent, close_fds=True,
        env={"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": ""})
    if completed.returncode:
        raise RuntimeError(f"rebuilt session runner failed with status {completed.returncode}")
    decode_response(completed.stdout, expected_descriptor=verified.descriptor)
    return completed.stdout
