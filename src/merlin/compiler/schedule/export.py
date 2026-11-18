"""Export canonical shared compiler sources for explicit host review."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from importlib.metadata import distribution
from importlib.metadata import version as package_version
from pathlib import Path

from merlin.common.paths import module_source_path, python_source_dir
from merlin.compiler.schedule.library_sources import (
    CURRENT_SDK_VERSION,
    library_modules,
    require_current_sdk,
    shared_library_sources,
)

# ---- export ------------------------------------------------------------------------------------

SCHEMA = "merlin.shared_scheduling_library.v21"

PUBLIC_MODULES = (
    "merlin.compiler.records",
    "merlin.compiler.backend.protocol",
    "merlin.compiler.backend.interface",
    "merlin.compiler.ir.interface_contract",
    "merlin.compiler.ir.readout",
    "merlin.compiler.lower.quant.requantization",
    "merlin.compiler.lower.interface_materialization",
    "merlin.compiler.dialects.interface",
    "merlin.compiler.schedule.blocks",
    "merlin.compiler.schedule.shared_operands.shared_inputs",
    "merlin.compiler.schedule.panels.transfers",
    "merlin.compiler.schedule.storage.buffer_pipelines",
    "merlin.compiler.schedule.loops.configuration_loops",
    "merlin.compiler.schedule.loops.command_compaction",
    "merlin.compiler.schedule.dense.compact",
    "merlin.compiler.schedule.dense.loaded_sums",
    "merlin.compiler.schedule.spatial.max_stripes",
    "merlin.compiler.schedule.dense.source",
    "merlin.compiler.schedule.loops.configuration",
    "merlin.compiler.schedule.loops.configuration_ir",
    "merlin.compiler.schedule.storage.spills",
    "merlin.compiler.schedule.elementwise.fusion",
    "merlin.compiler.schedule.library_sources",
    "merlin.compiler.schedule.loops.counted_loops",
    "merlin.compiler.schedule.elementwise.pointwise_source",
    "merlin.compiler.schedule.pipeline.program_limits",
    "merlin.compiler.schedule.elementwise.row_softmax_checks",
    "merlin.compiler.schedule.elementwise.row_softmax",
    "merlin.compiler.schedule.pipeline.stages",
    "merlin.compiler.schedule.dense.staged",
    "merlin.compiler.schedule.pipeline.selected_contracts",
    "merlin.compiler.schedule.storage.temporaries",
    "merlin.compiler.schedule.convolution.windows",
    "merlin.compiler.lower.options.nonlinear",
    "merlin.compiler.kernel_ir.ir",
    "merlin.compiler.kernel_ir.parse",
    "merlin.compiler.semantic.allocation_types",
    "merlin.target.isa.dialect_generation",
    "merlin.target.isa.spec",
    "merlin.compiler.transforms.dense.ir",
    "merlin.compiler.transforms.dense.planning",
    "merlin.compiler.transforms.pipeline_state",
    "merlin.compiler.schedule.spatial.producer",
    "merlin.compiler.schedule.spatial.readouts",
    "merlin.compiler.lower.host.storage",
    "merlin.compiler.lower.host.writers",
    "merlin.compiler.lower.host.borrowed_effects",
    "merlin.compiler.schedule.dense.plans",
    "merlin.compiler.dialects.schedule",
    "merlin.compiler.schedule.panels.pipeline",
    "merlin.compiler.schedule.panels.construction",
    "merlin.compiler.schedule.panels.retained",
    "merlin.compiler.schedule.loops.scf_commands",
    "merlin.compiler.schedule.loops.configuration_contexts",
    "merlin.compiler.schedule.panels.readouts",
    "merlin.compiler.schedule.panels.load_coalescing",
    "merlin.compiler.schedule.panels.tile_loads",
    "merlin.compiler.schedule.panels.segmented_pipeline",
    "merlin.compiler.schedule.panels.array_retention",
    "merlin.compiler.schedule.api_catalog",
    "merlin.compiler.schedule.spatial.rolling_window",
    "merlin.compiler.lower.host.builder",
    "merlin.compiler.schedule.shared_operands.projections",
    "merlin.compiler.transforms.canonicalization",
    "merlin.compiler.lower.source.region",
    "merlin.compiler.schedule.panels.ready_readout",
    "merlin.compiler.schedule.panels.ready_schedule",
    "merlin.compiler.schedule.loops.completion_events",
    "merlin.compiler.kernel_ir.loops",
    "merlin.compiler.frontend.preprocess",
    "merlin.compiler.partition.outline",
    "merlin.compiler.ir.logical_abi",
    "merlin.compiler.schedule.convolution.exploration",
    "merlin.compiler.schedule.convolution.choices",
    "merlin.compiler.schedule.shared_operands.shared_rhs",
    "merlin.compiler.transforms.exploration.branch_policy",
    "merlin.compiler.transforms.exploration.pass_orders",
    "merlin.compiler.transforms.exploration.cycle_feedback",
    "merlin.compiler.transforms.exploration.shape_assessment",
    "merlin.compiler.transforms.exploration.shape_proposals",
    "merlin.compiler.schedule.panels.epochs",
    "merlin.compiler.schedule.convolution.dense_views",
    "merlin.compiler.schedule.dense.batched",
    "merlin.compiler.schedule.convolution.output_groups",
    "merlin.compiler.selection.knobs",
    "merlin.compiler.transforms.dense.row_batching",
    "merlin.compiler.lower.host.propagation",
    "merlin.compiler.schedule.convolution.input_residency",
    "merlin.analysis.programs.storage_encoding",
    "merlin.compiler.lower.host.layout",
    "merlin.compiler.schedule.panels.loaded_sums",
    "merlin.compiler.lower.device.outline",
    "merlin.compiler.ir.llvm_context",
    "merlin.compiler.lower.quant.late_quant_rne",
    "merlin.compiler.lower.source.interval",
    "merlin.compiler.lower.source.interval_llvm",
    "merlin.compiler.lower.source.finite",
    "merlin.compiler.schedule.spatial.rolling_lookahead",
    "merlin.compiler.lower.source.repeated_pointwise",
    "merlin.compiler.lower.source.row_slice",
    "merlin.compiler.lower.options.weight_prepack_runtime",
    "merlin.compiler.transforms.dense.shared_rhs_views",
    "merlin.compiler.frontend.repairs",
    "merlin.compiler.ir.provenance",
    "merlin.compiler.quantization.linalg",
    "merlin.compiler.schedule.convolution.input_rows",
    "merlin.compiler.schedule.shared_operands.shared_lhs",
    "merlin.compiler.schedule.pipeline.contracts",
    "merlin.compiler.ir.static_cfg",
    "merlin.compiler.transforms.radix_product",
    "merlin.compiler.lower.quant.guarded_mean",
    "merlin.compiler.lower.quant.enclosed_readout",
    "merlin.compiler.lower.quant.affine_joint",
    "merlin.compiler.lower.quant.affine_rectifier",
    "merlin.compiler.lower.quant.affine_domain",
    "merlin.compiler.lower.source.integer_domains",
    "merlin.compiler.lower.source.shared_permutation",
    "merlin.compiler.lower.source.segmented_inputs",
    "merlin.compiler.frontend.quantized",
    "merlin.compiler.partition.callback_regions",
    "merlin.compiler.build.host_regions",
    "merlin.compiler.build.llvm_bundle",
    "merlin.compiler.lower.quant.affine_pair",
)

# These package initializers expose only shared leaf implementations. They
# omit installed host orchestration and backend discovery, without changing
# any pass, checker, instruction declaration or numerical implementation.
_PREPROCESSING_NAMESPACE = '"""Selected shared preprocessing leaves; no driver or backend discovery."""\n'
NAMESPACES = {
    "merlin.compiler.backend": '"""Public compiler declaration and protocol records; no provider discovery or source admission."""\n',
    "merlin.compiler.transforms": '"""Shared transforms and selected preprocessing leaves; host proof orchestration is not exported."""\n',
    "merlin.compiler.dialects": '"""Shared xDSL helpers; dialect modules are explicitly selected."""\n',
    "merlin.runtime": '"""Selected host arithmetic helpers; execution and oracle modules are not exported."""\n',
    "merlin.compiler.dialects.lowering": '"""Selected graph lowering leaves; other pipeline owners are not exported."""\n',
    "merlin.development.targetgen.oot_starterkit": '"""Selected generic LLVM parsing leaf; no package invocation or grading."""\n',
    "merlin.analysis.perf": '"""Selected storage-layout leaf; performance analyses and measurement are not exported."""\n',
    "merlin": '"""Selected shared compiler SDK; public APIs are explicit leaf modules."""\n',
    "merlin.compiler.frontend": '"""Selected shared preprocessing leaves; no driver or backend discovery."""\n',
    "merlin.compiler.quantization": '"""Selected shared preprocessing leaves; no driver or backend discovery."""\n',
    "merlin.compiler.lower": '"""Selected shared preprocessing leaves; no driver or backend discovery."""\n',
    "merlin.compiler": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.ir": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.shared_operands": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.panels": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.storage": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.loops": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.dense": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.spatial": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.elementwise": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.pipeline": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.convolution": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.lower.options": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.kernel_ir": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.semantic": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.semantic.search": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.development": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.target": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.target.isa": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.transforms.dense": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.verification": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.verification.references": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.lower.quant": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.lower.host": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.lower.source": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.common": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.lower.model": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.partition": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.transforms.exploration": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.selection": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.analysis": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.analysis.programs": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.lower.device": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.development.targetgen": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.capture": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.lower.c": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.schedule.nests": '"""Selected shared compiler leaves; public APIs belong to explicit modules."""\n',
    "merlin.compiler.build": '"""Selected callback compilation leaves; no operator build or evaluator."""\n',
}

SOURCE_BINDING_PROTOCOL = "merlin.shared_source_binding.v1"
SOURCE_BINDING_MODULES = ("merlin.compiler.schedule.source_binding",)


def public_modules(version=CURRENT_SDK_VERSION):
    """The current release's public API modules."""
    require_current_sdk(version, what="shared compiler public API")
    return PUBLIC_MODULES


def namespace_replacements(version=CURRENT_SDK_VERSION):
    """The current release's reviewed namespace initializer replacements."""
    require_current_sdk(version, what="shared compiler namespace")
    return dict(NAMESPACES)


def _source_identities(source_root, *, source_binding):
    sources = shared_library_sources()
    if source_binding:
        for module in SOURCE_BINDING_MODULES:
            path = source_root.joinpath(*module.split(".")).with_suffix(".py")
            sources[module] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def _license_root(source_root):
    """Resolve the actual checkout or installed distribution's license resources."""
    project = source_root.parent
    if source_root.name == "src" and (project / "pyproject.toml").is_file():
        return project
    selected = distribution("merlin")
    directories = {}
    for member in selected.files or ():
        if member.name in {"LICENSE", "NOTICE"} and "licenses" in member.parts:
            path = Path(selected.locate_file(member))
            if path.is_file() and not any(item.is_symlink() for item in (path, *path.parents)):
                directories.setdefault(path.parent, set()).add(path.name)
    matches = [path for path, names in directories.items() if names == {"LICENSE", "NOTICE"}]
    if len(matches) != 1:
        raise ValueError("installed shared export requires unambiguous LICENSE and NOTICE resources")
    return matches[0]


def export_shared_scheduling(
    directory: Path,
    *,
    license_directory: Path | None = None,
    version=CURRENT_SDK_VERSION,
    source_binding: bool = False,
) -> dict:
    """Write one declared source set, without granting candidate import access.

    The host must review the members and separately select its existing
    CompilerLibraryContract. No target, provider, grader, reference output or
    calibration record is discovered or included. Namespace glue is the only
    source replacement; every algorithm/checker file is copied byte for byte.

    Only the current SDK release is exported; retired releases are never
    regenerated from current sources. Import closure and candidate grants still
    need independent host review.
    """
    require_current_sdk(version, what="shared compiler export")
    modules, namespaces = library_modules(), namespace_replacements()
    if type(source_binding) is not bool:
        raise ValueError("shared source-binding export requires an explicit Boolean")
    if source_binding:
        modules = (*modules, *SOURCE_BINDING_MODULES)
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("shared compiler export requires an unlinked output directory")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("shared compiler export requires an empty output directory")
    exporter = Path(__file__).read_bytes()
    source_root = python_source_dir().resolve()
    sources = _source_identities(source_root, source_binding=source_binding)
    if license_directory is None:
        license_directory = _license_root(source_root)
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
        "schema": SCHEMA,
        "public_modules": list((*PUBLIC_MODULES, *SOURCE_BINDING_MODULES) if source_binding else PUBLIC_MODULES),
        "members": members,
        "replaced_namespace_source_sha256": namespace_changes,
        "external_package_versions": {name: package_version(name) for name in ("xdsl", "numpy", "PyYAML", "z3-solver")},
        "dependency_qualification": "versions_only; external implementations and optional native resources unqualified",
        "interface_compiler_api": {
            "schema": "merlin.interface_sdk.v1",
            "source_kind": "merlin.interface.v1",
            "parser": "merlin.compiler.ir.interface_contract.parse_interface",
            "declarations": "merlin.compiler.backend.interface",
            "records": "merlin.compiler.records",
            "host_source_preparation": "not_exported",
        },
        "admission": "requires_separate_host_review_and_explicit_compiler_library_contract",
        "review_resources": ["shared-library.json"],
    }
    manifest["inventory_version"] = CURRENT_SDK_VERSION
    if source_binding:
        manifest["source_binding_protocol"] = SOURCE_BINDING_PROTOCOL
    from merlin.compiler.schedule.api_catalog import scheduling_capabilities

    manifest["scheduling_capabilities"] = scheduling_capabilities()
    if (
        sources != _source_identities(source_root, source_binding=source_binding)
        or exporter != Path(__file__).read_bytes()
    ):
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
        source = module_source_path(
            {
                "dense_planning": "merlin.compiler.transforms.dense.planning",
                "dense_ir": "merlin.compiler.transforms.dense.ir",
            }[name]
        ).read_bytes()
        destination = directory / ("common_" + name + ".py")
        destination.write_bytes(source)
        files[destination.name] = hashlib.sha256(source).hexdigest()
    if include_block_schedule:
        source = block_schedule_source()
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
    parser.add_argument(
        "--library-version",
        type=int,
        choices=(CURRENT_SDK_VERSION,),
        default=CURRENT_SDK_VERSION,
    )
    parser.add_argument(
        "--shared-library", action="store_true", help="export all shared scheduling families for host review"
    )
    parser.add_argument(
        "--license-directory", type=Path, help="selected LICENSE and NOTICE directory for installed sources"
    )
    parser.add_argument("--source-binding", action="store_true", help="explicit typed-source/LLVM lexer APIs")
    args = parser.parse_args(argv)
    if args.shared_library and args.include_block_schedule:
        parser.error("--shared-library and --include-block-schedule select different export formats")
    if args.license_directory is not None and not args.shared_library:
        parser.error("--license-directory requires --shared-library")
    if args.source_binding and not args.shared_library:
        parser.error("--source-binding requires --shared-library")
    if args.shared_library:
        identity = export_shared_scheduling(
            args.output_dir,
            license_directory=args.license_directory,
            version=args.library_version,
            source_binding=args.source_binding,
        )
    else:
        identity = export_dense_planner(args.output_dir, include_block_schedule=args.include_block_schedule)
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()


# ---- standalone --------------------------------------------------------------------------------
# Assemble a stdlib-only scheduling artifact from the actual editable owners.
#
# Generated packages may vendor this derived module without depending on Merlin.
# The artifact contains the original definitions, with only internal imports and
# owner docstrings removed. Source hashes identify every contributing owner;
# the generated file is not a second editable implementation.

OWNERS = (
    "merlin.compiler.schedule.nests.model",
    "merlin.compiler.schedule.nests.construction",
    "merlin.compiler.schedule.nests.execution",
)


def block_schedule_source() -> bytes:
    """Return one closed, inspectable module with no Merlin runtime imports."""
    pieces = [
        '"""Generated shared block schedule; edit the recorded source owners."""\n',
        "from __future__ import annotations\n",
        "from dataclasses import dataclass, field, replace\n",
        "from typing import Any\n",
    ]
    for owner in OWNERS:
        source = module_source_path(owner).read_bytes()
        text = source.decode("utf-8")
        lines = text.splitlines(keepends=True)
        pieces.append(f"\n# Owner: {owner}; sha256={hashlib.sha256(source).hexdigest()}\n")
        for node in ast.parse(text).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            first = min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", ()))])
            pieces.append("\n" + "".join(lines[first - 1 : node.end_lineno]) + "\n")
    return "".join(pieces).encode("utf-8")
