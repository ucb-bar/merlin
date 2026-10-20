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

from merlin.compile.scheduling.export import (
    _NAMESPACES,
    PUBLIC_MODULES,
    SOURCE_BINDING_MODULES,
    SOURCE_BINDING_PROTOCOL,
    export_shared_scheduling,
)
from merlin.compile.scheduling.library_sources import LIBRARY_MODULES

_NAMESPACE_MODULES = (
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

# Exact pre-configuration V6 release, independent of future live API additions.
# Archived profiles retain these grants and their original ordering.
_V6_PRE_CONFIGURATION_MODULES = (
    "merlin", "merlin.common", "merlin.common.dense_float_ir", "merlin.compile",
    "merlin.compile.scheduling", "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.compact_dense", "merlin.compile.scheduling.compact_verification",
    "merlin.compile.scheduling.configuration", "merlin.compile.scheduling.copy_elimination",
    "merlin.compile.scheduling.dense_contracts", "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.device_spills", "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_contracts", "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources", "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_contracts", "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_contracts", "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.row_softmax_verification", "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts", "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.spill_contracts", "merlin.compile.scheduling.storage_contracts",
    "merlin.compile.scheduling.staged_dense", "merlin.compile.scheduling.staged_dense_verification",
    "merlin.compile.scheduling.temporary_storage", "merlin.llvmlower", "merlin.llvmlower.cpu_softmax",
    "merlin.sched", "merlin.sched.codegen", "merlin.sched.ir", "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain", "merlin.sched.ir.kernel", "merlin.sched.ir.parse",
    "merlin.sched.isa", "merlin.sched.isa.types", "merlin.semantic_compiler",
    "merlin.semantic_compiler.allocate", "merlin.semantic_compiler.allocation_types",
    "merlin.semantic_compiler.egg_bridge", "merlin.semantic_compiler.extract",
    "merlin.semantic_compiler.model", "merlin.semantic_compiler.rules", "merlin.targetgen",
    "merlin.targetgen.isa_dialect_gen", "merlin.targetgen.isa_spec", "merlin.transforms",
    "merlin.transforms.certificates", "merlin.transforms.dense_ir", "merlin.transforms.dense_planning",
    "merlin.transforms.normalization", "merlin.transforms.pipeline_state", "merlin.transforms.pure_scalar",
    "merlin.transforms.scalar_witness", "merlin.transforms.unrolling", "merlin.verify",
    "merlin.verify.epilogue_applicability",
)
_V6_PRE_CONFIGURATION_PUBLIC = (
    "merlin.compile.scheduling.block_schedule", "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.dense_source", "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.device_spills", "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_semantics", "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.pointwise_source", "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_softmax", "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts", "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.temporary_storage", "merlin.llvmlower.cpu_softmax",
    "merlin.compile.scheduling.staged_dense", "merlin.sched.ir.expr", "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel", "merlin.sched.ir.parse", "merlin.sched.isa.types",
    "merlin.semantic_compiler.allocation_types", "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec", "merlin.transforms.dense_ir", "merlin.transforms.dense_planning",
    "merlin.transforms.pipeline_state",
)


def shared_inventory(namespaces, public, *, library_modules=LIBRARY_MODULES, public_modules=PUBLIC_MODULES):
    """Stable exact membership; package initializers never become public APIs."""
    namespaces = tuple(dict.fromkeys((*namespaces, *(name + ".__init__" for name in _NAMESPACE_MODULES
                                                   if name in library_modules))))
    public = tuple(dict.fromkeys((*public, *public_modules)))
    implementation = tuple(name for name in library_modules if name not in _NAMESPACE_MODULES and name not in public)
    return namespaces, public, implementation


def _release_inventory(document):
    """Recognize exact versioned rosters; never upgrade an archived import grant."""
    members = document.get("members", []) if isinstance(document, dict) else []
    if isinstance(members, list):
        actual = [row.get("module") for row in members if isinstance(row, dict)]
        if document.get("schema") == "merlin.shared_scheduling_library.v2":
            modules = (*LIBRARY_MODULES, *SOURCE_BINDING_MODULES)
            public = (*PUBLIC_MODULES, *SOURCE_BINDING_MODULES)
            if (
                document.get("source_binding_protocol") == SOURCE_BINDING_PROTOCOL
                and actual == [*modules, None, None]
                and document.get("public_modules") == list(public)
            ):
                return modules, public
            raise ValueError("shared compiler source-binding inventory changed")
        if document.get("schema") != "merlin.shared_scheduling_library.v1":
            raise ValueError("shared compiler release API changed")
        for complete, public in (
            (LIBRARY_MODULES, PUBLIC_MODULES),
            (_V6_PRE_CONFIGURATION_MODULES, _V6_PRE_CONFIGURATION_PUBLIC),
        ):
            original = tuple(module for module in complete if module not in
                             ("merlin.common", "merlin.common.dense_float_ir"))
            for modules in (complete, original):
                if len(actual) == len(members) and actual == [*modules, None, None]:
                    if document.get("public_modules") != list(public):
                        raise ValueError("shared compiler public API inventory changed")
                    return modules, public
    raise ValueError("shared compiler source inventory changed")


def _release_modules(document):
    return _release_inventory(document)[0]


def profile_inventory(profile, namespaces, public):
    """Derive unchanged grants from this frozen release, not the current checkout."""
    modules, release_public = _release_inventory(profile.get("shared_scheduling_library"))
    return shared_inventory(
        namespaces, public, library_modules=modules, public_modules=release_public
    )


def _digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _read(root, relative):
    path = root / relative
    if not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("shared compiler release requires canonical unlinked source files")
    return path.read_bytes()


def reviewed_shared_manifest(root):
    """Reopen the fixed export; this supplies no runtime or numerical authority."""
    payload = _read(root, "shared-library.json")
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("schema") not in {
        "merlin.shared_scheduling_library.v1", "merlin.shared_scheduling_library.v2"
    }:
        raise ValueError("shared compiler release requires the reviewed SDK export")
    if document.get("library_sha256") != _digest({k: v for k, v in document.items() if k != "library_sha256"}):
        raise ValueError("shared compiler export identity changed")
    members = document.get("members", [])
    library_modules = _release_modules(document)
    expected_paths = [
        name.replace(".", "/") + ("/__init__.py" if name in _NAMESPACE_MODULES else ".py") for name in library_modules
    ] + ["LICENSE", "NOTICE"]
    for row, relative in zip(members, expected_paths, strict=True):
        if row.get("path") != relative or hashlib.sha256(_read(root, relative)).hexdigest() != row.get("sha256"):
            raise ValueError("shared compiler release bytes differ from its export")
    replacements = document.get("replaced_namespace_source_sha256")
    if type(replacements) is not dict or set(replacements) != set(_NAMESPACES):
        raise ValueError("shared compiler namespace replacement inventory changed")
    for module, text in _NAMESPACES.items():
        relative = module.replace(".", "/") + "/__init__.py"
        if _read(root, relative) != text.encode():
            raise ValueError("shared compiler namespace replacement changed")
        digest = replacements[module]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("shared compiler original namespace identity is invalid")
    return document


def validate_profile_release(profile):
    """Validate embedded export identity and its exact relation to profile bytes."""
    document = profile.get("shared_scheduling_library")
    if not isinstance(document, dict) or document.get("library_sha256") != _digest(
        {k: v for k, v in document.items() if k != "library_sha256"}
    ):
        raise ValueError("shared compiler profile requires its unchanged export identity")
    if document.get("schema") not in {"merlin.shared_scheduling_library.v1", "merlin.shared_scheduling_library.v2"}:
        raise ValueError("shared compiler profile API changed")
    rows = {row["module"].removesuffix(".__init__"): row for row in profile["common_sources"]}
    members = document.get("members", [])
    _release_modules(document)
    replacements = document.get("replaced_namespace_source_sha256")
    if type(replacements) is not dict or set(replacements) != set(_NAMESPACES):
        raise ValueError("shared compiler profile namespace inventory changed")
    for member in members:
        module = member["module"]
        if module is None:
            continue
        row = rows[module]
        relative = module.replace(".", "/") + ("/__init__.py" if module in _NAMESPACE_MODULES else ".py")
        if member.get("path") != relative or member.get("sha256") != row["sha256"]:
            raise ValueError("shared compiler profile differs from exported bytes")
        if module in _NAMESPACES and row["sha256"] != hashlib.sha256(_NAMESPACES[module].encode()).hexdigest():
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
    directory: Path, *, python_source_root: Path, license_directory: Path, tools: tuple[str, ...] = (),
    source_binding: bool = False,
) -> dict:
    """Prepare one source root before Phase 1; never alter a frozen run's grants."""
    from .tool_profile import COMMON_SUBSTRATE_V6, substrate_data, substrate_inventory

    root, directory = Path(python_source_root), Path(directory)
    actual = Path(__file__).resolve().parents[2]
    if not root.is_absolute() or root.resolve() != root or root != actual:
        raise ValueError("shared release must use the current explicitly selected installed source root")
    document = export_shared_scheduling(directory, license_directory=license_directory, source_binding=source_binding)
    namespaces, public, automatic = substrate_inventory(COMMON_SUBSTRATE_V6)
    modules, release_public = _release_inventory(document)
    namespaces, public, implementation = shared_inventory(
        namespaces, public, library_modules=modules, public_modules=release_public
    )
    relatives = [name.replace(".", "/") + ".py" for name in (*namespaces, *public, *implementation, *automatic)]
    relatives += ["merlin/" + relative for relative in substrate_data(COMMON_SUBSTRATE_V6)]
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
        if module in _NAMESPACES:
            continue
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError("shared export differs from its selected source release")
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
    parser.add_argument("--tool", action="append", default=[], help="existing tool registry name to stage, not grant")
    parser.add_argument("--source-binding", action="store_true", help="explicit V2 typed-source/LLVM lexer release")
    args = parser.parse_args(argv)
    result = prepare_shared_source_release(
        args.output_dir,
        python_source_root=args.python_source_root,
        license_directory=args.license_directory,
        tools=tuple(args.tool),
        source_binding=args.source_binding,
    )
    print(json.dumps({"library_sha256": result["library_sha256"], "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
