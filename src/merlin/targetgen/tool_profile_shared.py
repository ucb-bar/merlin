"""Closed source-release preparation for ordinary shared-pass authoring.

This is host preparation, never candidate-selected dependency discovery. It
combines the existing reviewed SDK export with preceding common facilities.
Only the exporter's three fixed namespace initializers replace source bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.export import export_shared_scheduling, namespace_replacements, public_modules
from merlin.compile.scheduling.library_sources import library_modules

_NAMESPACE_MODULES_V6 = (
    "merlin",
    "merlin.common",
    "merlin.compile",
    "merlin.compile.scheduling",
    "merlin.llvmlower",
    "merlin.sched",
    "merlin.sched.codegen",
    "merlin.sched.ir",
    "merlin.sched.isa",
    "merlin.semantic_compiler",
    "merlin.targetgen",
    "merlin.transforms",
    "merlin.verify",
)


_NAMESPACE_MODULES_V7 = (*_NAMESPACE_MODULES_V6,)


def _namespace_modules(version):
    library_modules(version)
    return _NAMESPACE_MODULES_V6 if version == 6 else _NAMESPACE_MODULES_V7


def shared_inventory(namespaces, public, *, version=6):
    """Stable exact membership; package initializers never become public APIs."""
    namespaces = tuple(dict.fromkeys((*namespaces, *(name + ".__init__" for name in _namespace_modules(version)))))
    public = tuple(dict.fromkeys((*public, *public_modules(version))))
    implementation = tuple(
        name for name in library_modules(version) if name not in _namespace_modules(version) and name not in public
    )
    return namespaces, public, implementation


def _manifest_version(document, version):
    if type(document) is not dict:
        return False
    if version == 6:
        return document.get("schema") == "merlin.shared_scheduling_library.v1" and "inventory_version" not in document
    return (
        document.get("schema") == "merlin.shared_scheduling_library.v2"
        and type(document.get("inventory_version")) is int
        and document["inventory_version"] == 7
    )


def _digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _read(root, relative):
    path = root / relative
    if not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("shared compiler release requires canonical unlinked source files")
    return path.read_bytes()


def reviewed_shared_manifest(root, *, version=6):
    """Reopen the fixed export; this supplies no runtime or numerical authority."""
    modules, namespaces = library_modules(version), namespace_replacements(version)
    payload = _read(root, "shared-library.json")
    document = json.loads(payload)
    if not _manifest_version(document, version):
        raise ValueError("shared compiler release requires the reviewed SDK export")
    if document.get("library_sha256") != _digest({k: v for k, v in document.items() if k != "library_sha256"}):
        raise ValueError("shared compiler export identity changed")
    if document.get("public_modules") != list(public_modules(version)):
        raise ValueError("shared compiler public API inventory changed")
    members = document.get("members", [])
    if not isinstance(members, list) or [row.get("module") for row in members] != [*modules, None, None]:
        raise ValueError("shared compiler source inventory changed")
    expected_paths = [
        name.replace(".", "/") + ("/__init__.py" if name in _namespace_modules(version) else ".py") for name in modules
    ] + ["LICENSE", "NOTICE"]
    for row, relative in zip(members, expected_paths, strict=True):
        if row.get("path") != relative or hashlib.sha256(_read(root, relative)).hexdigest() != row.get("sha256"):
            raise ValueError("shared compiler release bytes differ from its export")
    replacements = document.get("replaced_namespace_source_sha256")
    if type(replacements) is not dict or set(replacements) != set(namespaces):
        raise ValueError("shared compiler namespace replacement inventory changed")
    for module, text in namespaces.items():
        relative = module.replace(".", "/") + "/__init__.py"
        if _read(root, relative) != text.encode():
            raise ValueError("shared compiler namespace replacement changed")
        digest = replacements[module]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("shared compiler original namespace identity is invalid")
    return document


def validate_profile_release(profile):
    """Validate embedded export identity and its exact relation to profile bytes."""
    from .tool_profile import _SHARED_SUBSTRATES

    version = _SHARED_SUBSTRATES[profile["common_substrate"]]
    modules, namespaces = library_modules(version), namespace_replacements(version)
    document = profile.get("shared_scheduling_library")
    if not isinstance(document, dict) or document.get("library_sha256") != _digest(
        {k: v for k, v in document.items() if k != "library_sha256"}
    ):
        raise ValueError("shared compiler profile requires its unchanged export identity")
    if not _manifest_version(document, version) or document.get("public_modules") != list(public_modules(version)):
        raise ValueError("shared compiler profile API changed")
    rows = {row["module"].removesuffix(".__init__"): row for row in profile["common_sources"]}
    members = document.get("members", [])
    if not isinstance(members, list) or [row.get("module") for row in members] != [*modules, None, None]:
        raise ValueError("shared compiler profile export membership changed")
    replacements = document.get("replaced_namespace_source_sha256")
    if type(replacements) is not dict or set(replacements) != set(namespaces):
        raise ValueError("shared compiler profile namespace inventory changed")
    for member in members:
        module = member["module"]
        if module is None:
            continue
        row = rows[module]
        relative = module.replace(".", "/") + ("/__init__.py" if module in _namespace_modules(version) else ".py")
        if member.get("path") != relative or member.get("sha256") != row["sha256"]:
            raise ValueError("shared compiler profile differs from exported bytes")
        if module in namespaces and row["sha256"] != hashlib.sha256(namespaces[module].encode()).hexdigest():
            raise ValueError("shared compiler profile namespace replacement changed")
    for module, digest in replacements.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("shared compiler profile original namespace identity is invalid")
    resources = profile.get("shared_scheduling_resources")
    if type(resources) is not list or len(resources) != 3:
        raise ValueError("shared compiler profile requires the exact manifest and license resources")
    root = None
    for row, name in zip(resources, ("shared-library.json", "LICENSE", "NOTICE"), strict=True):
        if type(row) is not dict or set(row) != {"path", "sha256"} or not isinstance(row["path"], str):
            raise ValueError("shared compiler profile resource is malformed")
        path = Path(row["path"])
        digest = row["sha256"]
        if (
            not path.is_absolute()
            or path.name != name
            or ".." in path.parts
            or (root is not None and path.parent != root)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("shared compiler profile resource identity changed")
        root = path.parent
        if name != "shared-library.json" and digest != next(m["sha256"] for m in members if m["path"] == name):
            raise ValueError("shared compiler profile license differs from its export")


def host_source_digest(profile, row):
    """Only the three reviewed namespaces have different host/public bytes."""
    module = row["module"].removesuffix(".__init__")
    return (
        profile.get("shared_scheduling_library", {})
        .get("replaced_namespace_source_sha256", {})
        .get(module, row["sha256"])
    )


def _tool_sources(root, tools):
    """Existing explicit tool grants only; publication still belongs to each arm."""
    from .tool_registry import spec

    if type(tools) is not tuple or any(type(name) is not str for name in tools) or len(set(tools)) != len(tools):
        raise ValueError("shared release requires an explicit unique tool tuple")
    selected = set()
    for name in tools:
        for logical in spec(name).bundle_paths:
            prefix = "merlin/python/"
            if not logical.startswith(prefix):
                raise ValueError("shared release tool source is outside its existing Python grant")
            relative = logical.removeprefix(prefix).rstrip("/")
            path = root / relative
            if path.is_symlink() or not path.exists():
                raise ValueError("shared release selected tool source is absent or linked")
            for source in path.rglob("*") if path.is_dir() else (path,):
                if "__pycache__" in source.parts:
                    continue
                if source.is_symlink() or (not source.is_dir() and not source.is_file()):
                    raise ValueError("shared release tool grant contains linked or special sources")
                if source.is_file():
                    selected.add(source.relative_to(root).as_posix())
                if len(selected) > 4096:
                    raise ValueError("shared release tool sources exceed the fixed file bound")
    return sorted(selected)


def prepare_shared_source_release(
    directory: Path, *, python_source_root: Path, license_directory: Path, tools: tuple[str, ...] = (), version=6
) -> dict:
    """Prepare one source root before Phase 1; never alter a frozen run's grants."""
    from .tool_profile import COMMON_SUBSTRATE_V6, COMMON_SUBSTRATE_V7, substrate_data, substrate_inventory

    library_modules(version)
    substrate = COMMON_SUBSTRATE_V6 if version == 6 else COMMON_SUBSTRATE_V7
    replacements = namespace_replacements(version)
    root, directory = Path(python_source_root), Path(directory)
    actual = Path(__file__).resolve().parents[2]
    if not root.is_absolute() or root.resolve() != root or root != actual:
        raise ValueError("shared release must use the current explicitly selected installed source root")
    document = export_shared_scheduling(directory, license_directory=license_directory, version=version)
    namespaces, public, automatic = substrate_inventory(substrate)
    _, _, implementation = shared_inventory(namespaces, public, version=version)
    relatives = [name.replace(".", "/") + ".py" for name in (*namespaces, *public, *implementation, *automatic)]
    relatives += ["merlin/" + relative for relative in substrate_data(substrate)]
    relatives += _tool_sources(root, tools)
    if sum((root / relative).stat().st_size for relative in set(relatives)) > 32 << 20:
        raise ValueError("shared source release exceeds the fixed 32 MiB preparation bound")
    before = {relative: _read(root, relative) for relative in dict.fromkeys(relatives)}
    for module, digest in document["replaced_namespace_source_sha256"].items():
        relative = module.replace(".", "/") + "/__init__.py"
        if hashlib.sha256(before[relative]).hexdigest() != digest:
            raise ValueError("shared namespace source changed during preparation")
    for relative, payload in before.items():
        module = relative.removesuffix("/__init__.py").replace("/", ".")
        if module in replacements:
            continue
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError("shared export differs from its selected source release")
        destination.write_bytes(payload)
    if any(_read(root, relative) != payload for relative, payload in before.items()):
        raise ValueError("shared source release changed during preparation")
    reviewed_shared_manifest(directory, version=version)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-source-root", required=True, type=Path)
    parser.add_argument("--license-directory", required=True, type=Path)
    parser.add_argument("--library-version", type=int, choices=(6, 7), default=6)
    parser.add_argument("--tool", action="append", default=[], help="existing tool registry name to stage, not grant")
    args = parser.parse_args(argv)
    result = prepare_shared_source_release(
        args.output_dir,
        python_source_root=args.python_source_root,
        license_directory=args.license_directory,
        tools=tuple(args.tool),
        version=args.library_version,
    )
    print(json.dumps({"library_sha256": result["library_sha256"], "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
