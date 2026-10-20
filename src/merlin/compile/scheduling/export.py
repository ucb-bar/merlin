"""Export canonical shared compiler sources for explicit host review."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version as package_version
from pathlib import Path

from .library_sources import library_modules, shared_library_sources

# These package initializers expose only shared leaf implementations. They
# omit installed host orchestration and backend discovery, without changing
# any pass, checker, instruction declaration or numerical implementation.
_NAMESPACES = {
    "merlin.compile.scheduling": '"""Shared scheduling namespace; public APIs are explicit leaf modules."""\n',
    "merlin.transforms": '"""Shared transform namespace; host proof orchestration is not exported."""\n',
    "merlin.sched.isa": (
        '"""Shared instruction declarations; backend discovery is not exported."""\n\n'
        "from .types import OPERAND_KINDS, InstrDef, InstructionSet, IsaError, Operand\n"
    ),
}

PUBLIC_MODULES_V6 = (
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

PUBLIC_MODULES_V7 = (*PUBLIC_MODULES_V6, "merlin.compile.scheduling.staged_dense")
PUBLIC_MODULES = PUBLIC_MODULES_V6
_NAMESPACES_V6 = dict(_NAMESPACES)
_NAMESPACES_V7 = dict(_NAMESPACES_V6)


def public_modules(version=6):
    library_modules(version)
    return PUBLIC_MODULES_V6 if version == 6 else PUBLIC_MODULES_V7


def namespace_replacements(version=6):
    library_modules(version)
    return dict(_NAMESPACES_V6 if version == 6 else _NAMESPACES_V7)


def export_shared_scheduling(directory: Path, *, license_directory: Path | None = None, version=6) -> dict:
    """Write one declared source set, without granting candidate import access.

    The host must review the members and separately select its existing
    CompilerLibraryContract. No target, provider, grader, reference output or
    calibration record is discovered or included. Namespace glue is the only
    source replacement; every algorithm/checker file is copied byte for byte.
    """
    modules, namespaces = library_modules(version), namespace_replacements(version)
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("shared compiler export requires an unlinked output directory")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("shared compiler export requires an empty output directory")
    sources = shared_library_sources(version=version)
    exporter = Path(__file__).read_bytes()
    source_root = Path(__file__).resolve().parents[3]
    if license_directory is None:
        if source_root.name != "src" or not (source_root.parent / "pyproject.toml").is_file():
            raise ValueError("installed shared export requires an explicit license_directory / --license-directory")
        license_directory = source_root.parent
    resources = {name: (Path(license_directory) / name).read_bytes() for name in ("LICENSE", "NOTICE")}
    members, namespace_changes = [], {}
    for module in modules:
        stem = source_root.joinpath(*module.split("."))
        source = stem.with_suffix(".py")
        if not source.is_file():
            source = stem / "__init__.py"
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sources[module]:
            raise ValueError("shared compiler sources changed during export; regenerate in a fresh directory")
        if module in namespaces:
            namespace_changes[module] = hashlib.sha256(payload).hexdigest()
            payload = namespaces[module].encode()
        relative = source.relative_to(source_root)
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        members.append({"module": module, "path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest()})
    for name, payload in resources.items():
        (directory / name).write_bytes(payload)
        members.append({"module": None, "path": name, "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "schema": "merlin.shared_scheduling_library.v1" if version == 6 else "merlin.shared_scheduling_library.v2",
        "public_modules": list(public_modules(version)),
        "members": members,
        "replaced_namespace_source_sha256": namespace_changes,
        "external_package_versions": {name: package_version(name) for name in ("xdsl", "numpy", "PyYAML", "z3-solver")},
        "dependency_qualification": "versions_only; external implementations and optional native resources unqualified",
        "admission": "requires_separate_host_review_and_explicit_compiler_library_contract",
        "review_resources": ["shared-library.json"],
    }
    if version == 7:
        manifest["inventory_version"] = 7
    if sources != shared_library_sources(version=version) or exporter != Path(__file__).read_bytes():
        raise ValueError("shared compiler sources changed during export; regenerate in a fresh directory")
    if any((Path(license_directory) / name).read_bytes() != payload for name, payload in resources.items()):
        raise ValueError("shared compiler resources changed during export; regenerate in a fresh directory")
    manifest["library_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (directory / "shared-library.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_dense_planner(directory: Path, *, include_block_schedule: bool = False) -> dict:
    """Write exact public sources and return identities for the complete release."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("dense_planning", "dense_ir"):
        source = (Path(__file__).parents[2] / "transforms" / (name + ".py")).read_bytes()
        destination = directory / ("common_" + name + ".py")
        destination.write_bytes(source)
        files[destination.name] = hashlib.sha256(source).hexdigest()
    if include_block_schedule:
        source = Path(__file__).with_name("block_schedule.py").read_bytes()
        destination = directory / "common_block_schedule.py"
        destination.write_bytes(source)
        files[destination.name] = hashlib.sha256(source).hexdigest()
    return {
        "schema": "merlin.dense_planner.v2" if include_block_schedule else "merlin.dense_planner.v1",
        "files_sha256": files,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-block-schedule", action="store_true")
    parser.add_argument("--library-version", type=int, choices=(6, 7), default=6)
    parser.add_argument(
        "--shared-library", action="store_true", help="export all shared scheduling families for host review"
    )
    parser.add_argument(
        "--license-directory", type=Path, help="selected LICENSE and NOTICE directory for installed sources"
    )
    args = parser.parse_args(argv)
    if args.shared_library and args.include_block_schedule:
        parser.error("--shared-library and --include-block-schedule select different export formats")
    if args.library_version != 6 and not args.shared_library:
        parser.error("--library-version requires --shared-library")
    if args.license_directory is not None and not args.shared_library:
        parser.error("--license-directory requires --shared-library")
    if args.shared_library:
        identity = export_shared_scheduling(
            args.output_dir, license_directory=args.license_directory, version=args.library_version
        )
    else:
        identity = export_dense_planner(args.output_dir, include_block_schedule=args.include_block_schedule)
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
