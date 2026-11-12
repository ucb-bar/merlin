"""Closed source-release preparation for ordinary shared-pass authoring.

This is host preparation, never candidate-selected dependency discovery. It
combines the existing reviewed SDK export with preceding common facilities.
Only the exporter's declared namespace initializers replace source bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.compiler.schedule.export import (
    SCHEMA,
    SOURCE_BINDING_MODULES,
    SOURCE_BINDING_PROTOCOL,
    export_shared_scheduling,
    namespace_replacements,
    public_modules,
)
from merlin.compiler.schedule.library_sources import CURRENT_SDK_VERSION, library_modules, require_current_sdk

# Package namespaces of the current release. The exporter's reviewed package
# replacements also determine source paths: a namespace is staged as its
# ``__init__.py``, never as a same-named implementation module.
_NAMESPACE_MODULES = (
    "merlin",
    "merlin.common",
    "merlin.compiler",
    "merlin.compiler.schedule",
    "merlin.compiler.lower.model",
    "merlin.compiler.kernel_ir",
    "merlin.compiler.semantic",
    "merlin.development.targetgen",
    "merlin.compiler.transforms",
    "merlin.verification",
    "merlin.compiler.dialects",
    "merlin.compiler.frontend",
    "merlin.integrations",
    "merlin.runtime",
    "merlin.compiler.dialects.lowering",
    "merlin.capture",
    "merlin.development.targetgen.oot_starterkit",
    "merlin.analysis.perf",
    "merlin.analysis",
    "merlin.compiler.ir",
    "merlin.compiler.passes",
    "merlin.compiler.selection",
    "merlin.compiler.quantization",
    "merlin.compiler.lower",
    "merlin.compiler.lower.scalar",
)


# The release excludes protected SDK sources. Arm tool grants name whole files, and some of those lie
# inside packages withheld from every author (``AUTHOR_WITHHELD_PREFIXES``): the operator CCA spine and
# the compatibility shims of analyses that moved out of it. Directory grants also carry modules the
# access policy declares grader/oracle surfaces (``MODULE_ACCESS``; the RTL bridge's bit-exact cosim
# under ``targetgen/rtl/``). The release stages none of them; the CCA calls stay reachable through
# their host broker. The generic analyses that staged tool sources import (eagerly or lazily) are
# staged instead, as exact files below ``merlin/``. Machine observation helpers remain outside the
# shared editable SDK; selected infrastructure tools retain their independently declared grants.
_SDK_SUPPORT_SOURCES = (
    "analysis/__init__.py",
    "compiler/ir/contraction_shapes.py",
    "analysis/disassembly.py",
    "analysis/intrinsics_header.py",
    "analysis/rocc_commands.py",
)


def sdk_support_sources(version=CURRENT_SDK_VERSION) -> tuple[str, ...]:
    """Exact extra source files (relative to ``merlin/``) the release stages for its tool grants."""
    require_current_sdk(version)
    return _SDK_SUPPORT_SOURCES


def staged_module(relative: str) -> str:
    """The import identity of one staged file (a data file belongs to its directory's package)."""
    parts = relative.split("/")
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts.pop()
    else:
        parts.pop()
    return ".".join(parts)


def _selected_inventory(namespaces, public, modules, exported):
    namespaces = tuple(
        dict.fromkeys((*namespaces, *(name + ".__init__" for name in _NAMESPACE_MODULES if name in modules)))
    )
    public = tuple(dict.fromkeys((*public, *exported)))
    implementation = tuple(name for name in modules if name not in _NAMESPACE_MODULES and name not in public)
    return namespaces, public, implementation


def shared_inventory(namespaces, public, *, version=CURRENT_SDK_VERSION):
    """Stable exact membership; package initializers never become public APIs."""
    require_current_sdk(version)
    namespaces = tuple(dict.fromkeys((*namespaces, *(name + ".__init__" for name in _NAMESPACE_MODULES))))
    public = tuple(dict.fromkeys((*public, *public_modules())))
    implementation = tuple(name for name in library_modules() if name not in _NAMESPACE_MODULES and name not in public)
    return namespaces, public, implementation


def _current_manifest(document) -> bool:
    """Whether a release manifest is the current SDK export; retired releases are refused by name."""
    if type(document) is not dict:
        return False
    if "source_binding_protocol" in document and document["source_binding_protocol"] != SOURCE_BINDING_PROTOCOL:
        return False
    inventory = document.get("inventory_version")
    if type(inventory) is int and inventory != CURRENT_SDK_VERSION:
        require_current_sdk(inventory, what="shared compiler release inventory")
    return document.get("schema") == SCHEMA and type(inventory) is int and inventory == CURRENT_SDK_VERSION


def _release_inventory(document):
    if not _current_manifest(document):
        raise ValueError("shared compiler profile API changed")
    modules, public = library_modules(), public_modules()
    if "source_binding_protocol" in document:
        modules, public = (*modules, *SOURCE_BINDING_MODULES), (*public, *SOURCE_BINDING_MODULES)
    members = document.get("members", [])
    if (
        isinstance(members, list)
        and all(type(row) is dict for row in members)
        and [row.get("module") for row in members] == [*modules, None, None]
    ):
        if document.get("public_modules") != list(public):
            raise ValueError("shared compiler public API inventory changed")
        return modules, public
    raise ValueError("shared compiler source inventory changed")


def profile_inventory(profile, namespaces, public):
    """Reconstruct only the exact selected release's frozen import roster."""
    from merlin.development.targetgen.tool_profile import require_current_substrate

    require_current_substrate(profile.get("common_substrate"))
    modules, release_public = _release_inventory(profile.get("shared_scheduling_library"))
    return _selected_inventory(namespaces, public, modules, release_public)


def _digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _read(root, relative):
    path = root / relative
    if not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("shared compiler release requires canonical unlinked source files")
    return path.read_bytes()


def reviewed_shared_manifest(root, *, version=CURRENT_SDK_VERSION):
    """Reopen the fixed export; this supplies no runtime or numerical authority."""
    require_current_sdk(version, what="shared compiler release")
    namespaces = namespace_replacements()
    payload = _read(root, "shared-library.json")
    document = json.loads(payload)
    if not _current_manifest(document):
        raise ValueError("shared compiler release requires the reviewed SDK export")
    if document.get("library_sha256") != _digest({k: v for k, v in document.items() if k != "library_sha256"}):
        raise ValueError("shared compiler export identity changed")
    modules, _ = _release_inventory(document)
    members = document.get("members", [])
    expected_paths = [
        name.replace(".", "/") + ("/__init__.py" if name in _NAMESPACE_MODULES else ".py") for name in modules
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
    namespaces = namespace_replacements()
    document = profile.get("shared_scheduling_library")
    if not isinstance(document, dict) or document.get("library_sha256") != _digest(
        {k: v for k, v in document.items() if k != "library_sha256"}
    ):
        raise ValueError("shared compiler profile requires its unchanged export identity")
    _release_inventory(document)
    rows = {row["module"].removesuffix(".__init__"): row for row in profile["common_sources"]}
    members = document.get("members", [])
    replacements = document.get("replaced_namespace_source_sha256")
    if type(replacements) is not dict or set(replacements) != set(namespaces):
        raise ValueError("shared compiler profile namespace inventory changed")
    for member in members:
        module = member["module"]
        if module is None:
            continue
        row = rows[module]
        relative = module.replace(".", "/") + ("/__init__.py" if module in _NAMESPACE_MODULES else ".py")
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
    """Existing explicit tool grants only; publication still belongs to each arm.

    Every file of an author-inaccessible module (withheld package or access-policy grader/oracle
    module) is dropped rather than staged.
    """
    from merlin.common.access import author_inaccessible
    from merlin.development.targetgen.tool_registry import spec

    if type(tools) is not tuple or any(type(name) is not str for name in tools) or len(set(tools)) != len(tools):
        raise ValueError("shared release requires an explicit unique tool tuple")
    selected = {}
    for name in tools:
        for logical in spec(name).bundle_paths:
            prefix = "merlin/python/"
            if not logical.startswith(prefix):
                raise ValueError("shared release tool source is outside its existing Python grant")
            relative = logical.removeprefix(prefix).rstrip("/")
            sources = {relative: relative}
            if name == "lean_proof_authoring":
                # The registry still names the verify compatibility paths; also stage each
                # path's owner below ``merlin/verification/``.
                if relative == "merlin/verify/lean/Merlin/Foundation.lean":
                    # The historical resource slot is an exact generated copy of its one owner.
                    sources[relative] = "merlin/verification/lean/Merlin/Foundation.lean"
                elif relative.endswith(".py") and relative.startswith("merlin/verify/"):
                    owner = "merlin/verification/" + relative.removeprefix("merlin/verify/")
                    sources[owner] = owner
            for destination, selected_source in sources.items():
                path = root / selected_source
                if path.is_symlink() or not path.exists():
                    raise ValueError("shared release selected tool source is absent or linked")
                for source in path.rglob("*") if path.is_dir() else (path,):
                    if "__pycache__" in source.parts:
                        continue
                    if source.is_symlink() or (not source.is_dir() and not source.is_file()):
                        raise ValueError("shared release tool grant contains linked or special sources")
                    original = source.relative_to(root).as_posix()
                    staged = (Path(destination) / source.relative_to(path)).as_posix() if path.is_dir() else destination
                    if source.is_file() and not any(author_inaccessible(staged_module(p)) for p in (original, staged)):
                        selected[staged] = original
                    if len(selected) > 4096:
                        raise ValueError("shared release tool sources exceed the fixed file bound")
    return dict(sorted(selected.items()))


def prepare_shared_source_release(
    directory: Path,
    *,
    python_source_root: Path,
    license_directory: Path | None = None,
    tools: tuple[str, ...] = (),
    version=CURRENT_SDK_VERSION,
    source_binding: bool = False,
) -> dict:
    """Prepare one source root before Phase 1; never alter a frozen run's grants."""
    from merlin.common.access import author_inaccessible
    from merlin.development.targetgen.tool_profile import (
        CURRENT_COMMON_SUBSTRATE,
        substrate_data,
        substrate_inventory,
    )

    require_current_sdk(version, what="shared source release")
    replacements = namespace_replacements()
    root, directory = Path(python_source_root), Path(directory)
    actual = Path(__file__).resolve().parents[3]
    if not root.is_absolute() or root.resolve() != root or root != actual:
        raise ValueError("shared release must use the current explicitly selected installed source root")
    document = export_shared_scheduling(directory, license_directory=license_directory, source_binding=source_binding)
    namespaces, public, automatic = substrate_inventory(CURRENT_COMMON_SUBSTRATE)
    modules, release_public = _release_inventory(document)
    namespaces, public, implementation = _selected_inventory(namespaces, public, modules, release_public)
    relatives = [name.replace(".", "/") + ".py" for name in (*namespaces, *public, *implementation, *automatic)]
    relatives += ["merlin/" + relative for relative in substrate_data(CURRENT_COMMON_SUBSTRATE)]
    tool_sources = _tool_sources(root, tools)
    relatives += ["merlin/" + relative for relative in _SDK_SUPPORT_SOURCES]
    sources = {relative: relative for relative in relatives}
    sources.update(tool_sources)
    leaked = sorted(
        {relative for relative in (*sources, *sources.values()) if author_inaccessible(staged_module(relative))}
    )
    if leaked:
        raise ValueError("author SDK release would stage withheld sources: " + ", ".join(leaked[:4]))
    if sum((root / relative).stat().st_size for relative in sources.values()) > 32 << 20:
        raise ValueError("shared source release exceeds the fixed 32 MiB preparation bound")
    before = {relative: _read(root, relative) for relative in dict.fromkeys(sources.values())}
    for module, digest in document["replaced_namespace_source_sha256"].items():
        relative = module.replace(".", "/") + "/__init__.py"
        if hashlib.sha256(before[relative]).hexdigest() != digest:
            raise ValueError("shared namespace source changed during preparation")
    for relative, selected_source in sources.items():
        payload = before[selected_source]
        module = relative.removesuffix("/__init__.py").replace("/", ".")
        if module in replacements:
            continue
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f"shared export differs from its selected source release: {relative}")
        destination.write_bytes(payload)
    if any(_read(root, relative) != payload for relative, payload in before.items()):
        raise ValueError("shared source release changed during preparation")
    reviewed_shared_manifest(directory)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-source-root", required=True, type=Path)
    parser.add_argument("--license-directory", required=True, type=Path)
    parser.add_argument("--library-version", type=int, choices=(CURRENT_SDK_VERSION,), default=CURRENT_SDK_VERSION)
    parser.add_argument("--tool", action="append", default=[], help="existing tool registry name to stage, not grant")
    parser.add_argument("--source-binding", action="store_true", help="explicit typed-source/LLVM lexer APIs")
    args = parser.parse_args(argv)
    result = prepare_shared_source_release(
        args.output_dir,
        python_source_root=args.python_source_root,
        license_directory=args.license_directory,
        tools=tuple(args.tool),
        version=args.library_version,
        source_binding=args.source_binding,
    )
    print(json.dumps({"library_sha256": result["library_sha256"], "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
