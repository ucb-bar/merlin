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
    SOURCE_BINDING_MODULES,
    SOURCE_BINDING_PROTOCOL,
    export_shared_scheduling,
    namespace_replacements,
    public_modules,
)
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
_NAMESPACE_MODULES_V8 = (*_NAMESPACE_MODULES_V6,)


def _namespace_modules(version):
    library_modules(version)
    return {6: _NAMESPACE_MODULES_V6, 7: _NAMESPACE_MODULES_V7, 8: _NAMESPACE_MODULES_V8}[version]


def _selected_inventory(namespaces, public, modules, exported, *, version):
    namespaces = tuple(
        dict.fromkeys((*namespaces, *(name + ".__init__" for name in _namespace_modules(version) if name in modules)))
    )
    public = tuple(dict.fromkeys((*public, *exported)))
    implementation = tuple(name for name in modules if name not in _namespace_modules(version) and name not in public)
    return namespaces, public, implementation


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
    schema = document.get("schema")
    binding = "source_binding_protocol" in document
    if binding and document["source_binding_protocol"] != SOURCE_BINDING_PROTOCOL:
        return False
    if version == 6:
        # Incoming archived V2 source-binding exports preceded explicit V8.
        return "inventory_version" not in document and schema == (
            "merlin.shared_scheduling_library.v2" if binding else "merlin.shared_scheduling_library.v1"
        )
    return (
        version in (7, 8)
        and schema == {7: "merlin.shared_scheduling_library.v2", 8: "merlin.shared_scheduling_library.v3"}[version]
        and type(document.get("inventory_version")) is int
        and document["inventory_version"] == version
        and (version == 8 or not binding)
    )


# Closed archived V6 inventories are replay-only. New exports always use the
# fixed V6 or explicitly selected V7 inventory; archival recognition grants no
# source, runtime, numerical or execution authority.
_V6_STAGED_MODULES = (
    "merlin",
    "merlin.common",
    "merlin.common.dense_float_ir",
    "merlin.compile",
    "merlin.compile.scheduling",
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.compact_verification",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.configuration_ir",
    "merlin.compile.scheduling.copy_elimination",
    "merlin.compile.scheduling.dense_contracts",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_contracts",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_contracts",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_contracts",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.row_softmax_verification",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.spill_contracts",
    "merlin.compile.scheduling.storage_contracts",
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.staged_dense_verification",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.llvmlower",
    "merlin.llvmlower.cpu_softmax",
    "merlin.sched",
    "merlin.sched.codegen",
    "merlin.sched.ir",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa",
    "merlin.sched.isa.types",
    "merlin.semantic_compiler",
    "merlin.semantic_compiler.allocate",
    "merlin.semantic_compiler.allocation_types",
    "merlin.semantic_compiler.egg_bridge",
    "merlin.semantic_compiler.extract",
    "merlin.semantic_compiler.model",
    "merlin.semantic_compiler.rules",
    "merlin.targetgen",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms",
    "merlin.transforms.certificates",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.normalization",
    "merlin.transforms.pipeline_state",
    "merlin.transforms.pure_scalar",
    "merlin.transforms.scalar_witness",
    "merlin.transforms.unrolling",
    "merlin.verify",
    "merlin.verify.epilogue_applicability",
)

_V6_STAGED_PUBLIC = (
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.configuration_ir",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.llvmlower.cpu_softmax",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa.types",
    "merlin.semantic_compiler.allocation_types",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.pipeline_state",
)

_V6_PRE_CONFIGURATION_MODULES = (
    "merlin",
    "merlin.common",
    "merlin.common.dense_float_ir",
    "merlin.compile",
    "merlin.compile.scheduling",
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.compact_verification",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.copy_elimination",
    "merlin.compile.scheduling.dense_contracts",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_contracts",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_contracts",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_contracts",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.row_softmax_verification",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.spill_contracts",
    "merlin.compile.scheduling.storage_contracts",
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.staged_dense_verification",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.llvmlower",
    "merlin.llvmlower.cpu_softmax",
    "merlin.sched",
    "merlin.sched.codegen",
    "merlin.sched.ir",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa",
    "merlin.sched.isa.types",
    "merlin.semantic_compiler",
    "merlin.semantic_compiler.allocate",
    "merlin.semantic_compiler.allocation_types",
    "merlin.semantic_compiler.egg_bridge",
    "merlin.semantic_compiler.extract",
    "merlin.semantic_compiler.model",
    "merlin.semantic_compiler.rules",
    "merlin.targetgen",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms",
    "merlin.transforms.certificates",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.normalization",
    "merlin.transforms.pipeline_state",
    "merlin.transforms.pure_scalar",
    "merlin.transforms.scalar_witness",
    "merlin.transforms.unrolling",
    "merlin.verify",
    "merlin.verify.epilogue_applicability",
)

_V6_PRE_CONFIGURATION_PUBLIC = (
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.llvmlower.cpu_softmax",
    "merlin.compile.scheduling.staged_dense",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa.types",
    "merlin.semantic_compiler.allocation_types",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.pipeline_state",
)

# Exact subsequent release: configuration source APIs preceded the parallel
# scheduling additions. Keep this historical roster independent of live exports.
_V6_CONFIGURATION_MODULES = tuple(
    name for module in _V6_PRE_CONFIGURATION_MODULES
    for name in ((module, "merlin.compile.scheduling.configuration_ir")
                 if module == "merlin.compile.scheduling.configuration" else (module,))
)
_V6_CONFIGURATION_PUBLIC = (
    "merlin.compile.scheduling.block_schedule", "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.dense_source", "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.configuration_ir", "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion", "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources", "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_limits", "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.schedule_pipeline", "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.selected_contracts", "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.temporary_storage", "merlin.llvmlower.cpu_softmax",
    "merlin.sched.ir.expr", "merlin.sched.ir.integer_domain", "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse", "merlin.sched.isa.types", "merlin.semantic_compiler.allocation_types",
    "merlin.targetgen.isa_dialect_gen", "merlin.targetgen.isa_spec", "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning", "merlin.transforms.pipeline_state",
)


# Exact already-published incoming manifests; replay recognition grants no upgrade.
_INCOMING_V6_MODULES = (
    "merlin",
    "merlin.common",
    "merlin.common.dense_float_ir",
    "merlin.compile",
    "merlin.compile.scheduling",
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.zero_initializer_groups",
    "merlin.compile.scheduling.shared_input_verification",
    "merlin.compile.scheduling.shared_input_schedule",
    "merlin.compile.scheduling.shared_input_groups",
    "merlin.compile.scheduling.panel_transfers",
    "merlin.compile.scheduling.copy_pipeline",
    "merlin.compile.scheduling.configuration_loops",
    "merlin.compile.scheduling.command_compaction",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.compact_verification",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.configuration_ir",
    "merlin.compile.scheduling.copy_elimination",
    "merlin.compile.scheduling.dense_contracts",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_contracts",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.llvm_program",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_contracts",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_contracts",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.row_softmax_verification",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.spill_contracts",
    "merlin.compile.scheduling.storage_contracts",
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.staged_dense_verification",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.compile.scheduling.window_transfer",
    "merlin.llvmlower",
    "merlin.llvmlower.cpu_softmax",
    "merlin.sched",
    "merlin.sched.codegen",
    "merlin.sched.ir",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa",
    "merlin.sched.isa.types",
    "merlin.semantic_compiler",
    "merlin.semantic_compiler.allocate",
    "merlin.semantic_compiler.allocation_types",
    "merlin.semantic_compiler.egg_bridge",
    "merlin.semantic_compiler.extract",
    "merlin.semantic_compiler.model",
    "merlin.semantic_compiler.rules",
    "merlin.targetgen",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms",
    "merlin.transforms.certificates",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.normalization",
    "merlin.transforms.pipeline_state",
    "merlin.transforms.pure_scalar",
    "merlin.transforms.scalar_witness",
    "merlin.transforms.unrolling",
    "merlin.verify",
    "merlin.verify.epilogue_applicability",
)
_INCOMING_V6_PUBLIC = (
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.shared_input_verification",
    "merlin.compile.scheduling.shared_input_schedule",
    "merlin.compile.scheduling.shared_input_groups",
    "merlin.compile.scheduling.panel_transfers",
    "merlin.compile.scheduling.copy_pipeline",
    "merlin.compile.scheduling.configuration_loops",
    "merlin.compile.scheduling.command_compaction",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.configuration_ir",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.llvm_program",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.compile.scheduling.window_transfer",
    "merlin.llvmlower.cpu_softmax",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa.types",
    "merlin.semantic_compiler.allocation_types",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.pipeline_state",
)


def _release_inventory(document, *, version=6):
    if not _manifest_version(document, version):
        raise ValueError("shared compiler profile API changed")
    binding = "source_binding_protocol" in document
    candidates = [(library_modules(version), public_modules(version))]
    incoming = [
        (
            tuple(name for name in _INCOMING_V6_MODULES if name not in omitted),
            tuple(name for name in _INCOMING_V6_PUBLIC if name not in omitted),
        )
        for omitted in (
            (),
            ("merlin.compile.scheduling.window_transfer",),
            ("merlin.compile.scheduling.window_transfer", "merlin.compile.scheduling.llvm_program"),
        )
    ]
    if version == 7:
        # Exact first V7 roster before the explicitly separated staged extension.
        candidates.append((library_modules(6), public_modules(6)))
    if version == 6:
        historical = [*incoming, (_V6_CONFIGURATION_MODULES, _V6_CONFIGURATION_PUBLIC)]
        candidates = historical if binding else [*candidates, *historical]
        if not binding:
            for modules, public in (
                (_V6_STAGED_MODULES, _V6_STAGED_PUBLIC),
                (_V6_PRE_CONFIGURATION_MODULES, _V6_PRE_CONFIGURATION_PUBLIC),
                *incoming,
            ):
                candidates.extend(
                    (
                        (modules, public),
                        (
                            tuple(
                                name
                                for name in modules
                                if name not in ("merlin.common", "merlin.common.dense_float_ir")
                            ),
                            public,
                        ),
                    )
                )
    if binding:
        candidates = [
            ((*modules, *SOURCE_BINDING_MODULES), (*public, *SOURCE_BINDING_MODULES)) for modules, public in candidates
        ]
    members = document.get("members", [])
    if isinstance(members, list) and all(type(row) is dict for row in members):
        actual = [row.get("module") for row in members]
        for modules, public in candidates:
            if actual == [*modules, None, None]:
                if document.get("public_modules") != list(public):
                    raise ValueError("shared compiler public API inventory changed")
                return modules, public
    raise ValueError("shared compiler source inventory changed")


def profile_inventory(profile, namespaces, public):
    """Reconstruct only the exact selected release's frozen import roster."""
    from .tool_profile import _SHARED_SUBSTRATES

    version = _SHARED_SUBSTRATES[profile["common_substrate"]]
    modules, release_public = _release_inventory(profile.get("shared_scheduling_library"), version=version)
    return _selected_inventory(namespaces, public, modules, release_public, version=version)


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
    modules, _ = _release_inventory(document, version=version)
    members = document.get("members", [])
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
    namespaces = namespace_replacements(version)
    document = profile.get("shared_scheduling_library")
    if not isinstance(document, dict) or document.get("library_sha256") != _digest(
        {k: v for k, v in document.items() if k != "library_sha256"}
    ):
        raise ValueError("shared compiler profile requires its unchanged export identity")
    _release_inventory(document, version=version)
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
    directory: Path,
    *,
    python_source_root: Path,
    license_directory: Path,
    tools: tuple[str, ...] = (),
    version=6,
    source_binding: bool = False,
) -> dict:
    """Prepare one source root before Phase 1; never alter a frozen run's grants."""
    from .tool_profile import _SHARED_SUBSTRATES, substrate_data, substrate_inventory

    library_modules(version)
    substrate = next((name for name, number in _SHARED_SUBSTRATES.items() if number == version), None)
    if substrate is None:
        raise ValueError("selected shared inventory has no explicit common-substrate integration")
    replacements = namespace_replacements(version)
    root, directory = Path(python_source_root), Path(directory)
    actual = Path(__file__).resolve().parents[2]
    if not root.is_absolute() or root.resolve() != root or root != actual:
        raise ValueError("shared release must use the current explicitly selected installed source root")
    document = export_shared_scheduling(
        directory, license_directory=license_directory, version=version, source_binding=source_binding
    )
    namespaces, public, automatic = substrate_inventory(substrate)
    modules, release_public = _release_inventory(document, version=version)
    namespaces, public, implementation = _selected_inventory(
        namespaces, public, modules, release_public, version=version
    )
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
    parser.add_argument("--library-version", type=int, choices=(6, 7, 8), default=6)
    parser.add_argument("--tool", action="append", default=[], help="existing tool registry name to stage, not grant")
    parser.add_argument("--source-binding", action="store_true", help="explicit version-8 typed-source/LLVM lexer APIs")
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
