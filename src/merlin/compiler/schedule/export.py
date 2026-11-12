"""Export canonical shared compiler sources for explicit host review."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import distribution
from importlib.metadata import version as package_version
from pathlib import Path

from merlin.compiler.schedule.library_sources import (
    CURRENT_SDK_VERSION,
    library_modules,
    require_current_sdk,
    shared_library_sources,
)

SCHEMA = "merlin.shared_scheduling_library.v21"

PUBLIC_MODULES = (
    "merlin.compiler.schedule.blocks",
    "merlin.compiler.schedule.shared_input_verification",
    "merlin.compiler.schedule.shared_input_schedule",
    "merlin.compiler.schedule.shared_input_groups",
    "merlin.compiler.schedule.panel_transfers",
    "merlin.compiler.schedule.copy_pipeline",
    "merlin.compiler.schedule.configuration_loops",
    "merlin.compiler.schedule.command_compaction",
    "merlin.compiler.schedule.compact_dense",
    "merlin.compiler.schedule.compact_loaded_sum",
    "merlin.compiler.schedule.spatial_gather",
    "merlin.compiler.schedule.dense_source",
    "merlin.compiler.schedule.configuration",
    "merlin.compiler.schedule.configuration_ir",
    "merlin.compiler.schedule.device_spills",
    "merlin.compiler.schedule.fusion",
    "merlin.compiler.schedule.fusion_semantics",
    "merlin.compiler.schedule.library_sources",
    "merlin.compiler.schedule.llvm_program",
    "merlin.compiler.schedule.pointwise_source",
    "merlin.compiler.schedule.program_limits",
    "merlin.compiler.schedule.row_softmax",
    "merlin.compiler.schedule.schedule_pipeline",
    "merlin.compiler.schedule.staged_dense",
    "merlin.compiler.schedule.selected_contracts",
    "merlin.compiler.schedule.spill_conformance",
    "merlin.compiler.schedule.temporary_storage",
    "merlin.compiler.schedule.window_transfer",
    "merlin.llvmlower.cpu_softmax",
    "merlin.compiler.kernel_ir.ir",
    "merlin.compiler.kernel_ir.parse",
    "merlin.semantic_compiler.allocation_types",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.pipeline_state",
    "merlin.compiler.schedule.periodic_dense",
    "merlin.compiler.schedule.spatial_producer",
    "merlin.compiler.schedule.spatial_readout",
    "merlin.llvmlower.spatial_scratch_abi",
    "merlin.llvmlower.fresh_tensor_writer",
    "merlin.llvmlower.private_workspace_pool",
    "merlin.llvmlower.borrowed_effects",
    "merlin.compiler.schedule.schedule_ir",
    "merlin.xdsl_dialects.schedule",
    "merlin.compiler.schedule.loop_schedule",
    "merlin.compiler.schedule.dense_panel_pipeline",
    "merlin.compiler.schedule.dense_panel_contracts",
    "merlin.compiler.schedule.dense_panel_ir",
    "merlin.compiler.schedule.dense_retained",
    "merlin.compiler.schedule.scf_commands",
    "merlin.compiler.schedule.configuration_contexts",
    "merlin.compiler.schedule.configuration_loop_cleanup",
    "merlin.compiler.schedule.dense_panel_readout",
    "merlin.compiler.schedule.tile_load_coalescing",
    "merlin.compiler.schedule.panel_tile_loads",
    "merlin.compiler.schedule.dense_segmented_pipeline",
    "merlin.compiler.schedule.dense_array_retention",
    "merlin.compiler.schedule.api_catalog",
    "merlin.compiler.schedule.rolling_window",
    "merlin.compiler.schedule.rolling_spatial",
    "merlin.xdsl_dialects.rolling_window",
    "merlin.compiler.schedule.spatial_readout_cache",
    "merlin.compiler.schedule.dense_readout_overlap",
    "merlin.llvmlower.host_builder",
    "merlin.compiler.schedule.projection_views",
    "merlin.transforms.tensor_views",
    "merlin.llvmlower.source_region",
    "merlin.llvmlower.source_region_host",
    "merlin.compiler.passes.composition",
    "merlin.compiler.schedule.dense_ready_readout",
    "merlin.compiler.schedule.scf_carried_events",
    "merlin.compiler.kernel_ir.loops",
    "merlin.compiler.frontend.preprocess",
    "merlin.xdsl_dialects.lowering.outline",
    "merlin.compiler.schedule.completion_streams",
    "merlin.compiler.schedule.configuration_motion",
    "merlin.compiler.schedule.spatial_max_stripes",
    "merlin.compiler.ir.logical_abi",
    "merlin.compiler.schedule.convolution_exploration",
    "merlin.compiler.schedule.convolution_choices",
    "merlin.compiler.schedule.dense_rhs_residency",
    "merlin.transforms.pipeline_policy",
    "merlin.transforms.pass_exploration",
    "merlin.transforms.evaluated_exploration",
    "merlin.transforms.shape_policy",
    "merlin.compiler.schedule.dense_panel_epochs",
    "merlin.compiler.schedule.pointwise_convolution",
    "merlin.compiler.schedule.batched_dense",
    "merlin.compiler.schedule.convolution_output_groups",
    "merlin.compiler.schedule.loaded_sum_epochs",
    "merlin.compiler.selection.knobs",
    "merlin.transforms.tuning_selection",
    "merlin.transforms.dense_reduction_layout",
    "merlin.llvmlower.layout_propagation",
    "merlin.compiler.schedule.convolution_input_reuse",
    "merlin.perf.factored_storage",
    "merlin.llvmlower.constant_panel_layout",
    "merlin.compiler.schedule.dense_loaded_sum_composition",
    "merlin.llvmlower.quantized_sum_source",
    "merlin.llvmlower.generated_private_workspace",
    "merlin.targetgen.oot_starterkit.llvm_context",
    "merlin.llvmlower.late_quant_rne",
    "merlin.llvmlower.source_expression_interval",
    "merlin.llvmlower.source_expression_interval_llvm",
    "merlin.llvmlower.bounded_rne_word_cells",
    "merlin.llvmlower.scaled_integer_finite",
    "merlin.llvmlower.scaled_integer_finite_llvm",
    "merlin.compiler.schedule.projection_dense",
    "merlin.compiler.schedule.rolling_lookahead",
    "merlin.compiler.schedule.convolution_input_first_use",
    "merlin.compiler.schedule.operand_slots",
    "merlin.compiler.schedule.dense_shared_rhs",
    "merlin.compiler.schedule.horizontal_convolution",
    "merlin.llvmlower.repeated_pointwise",
    "merlin.llvmlower.contraction_row_slice",
    "merlin.llvmlower.weight_prepack_runtime",
    "merlin.transforms.dense_row_batching",
    "merlin.transforms.shared_rhs_views",
    "merlin.compiler.frontend.repairs",
    "merlin.compiler.ir.provenance",
    "merlin.compiler.quantization.linalg",
    "merlin.compiler.transforms.matmul",
    "merlin.compiler.transforms.dead",
    "merlin.compiler.lower.scalar.casts",
    "merlin.compiler.schedule.affine_gather_runs",
    "merlin.compiler.schedule.compact_dense_group",
)

# These package initializers expose only shared leaf implementations. They
# omit installed host orchestration and backend discovery, without changing
# any pass, checker, instruction declaration or numerical implementation.
_PREPROCESSING_NAMESPACE = '"""Selected shared preprocessing leaves; no driver or backend discovery."""\n'
NAMESPACES = {
    "merlin.compiler.schedule": '"""Shared scheduling namespace; public APIs are explicit leaf modules."""\n',
    "merlin.transforms": '"""Shared transform namespace; host proof orchestration is not exported."""\n',
    "merlin.xdsl_dialects": '"""Shared xDSL helpers; dialect modules are explicitly selected."""\n',
    "merlin.runtime": '"""Selected host arithmetic helpers; execution and oracle modules are not exported."""\n',
    "merlin.xdsl_dialects.lowering": '"""Selected graph lowering leaves; other pipeline owners are not exported."""\n',
    "merlin.targetgen.oot_starterkit": '"""Selected generic LLVM parsing leaf; no package invocation or grading."""\n',
    "merlin.perf": '"""Selected storage-layout leaf; performance analyses and measurement are not exported."""\n',
    "merlin": '"""Selected shared compiler SDK; public APIs are explicit leaf modules."""\n',
    "merlin.compiler.frontend": _PREPROCESSING_NAMESPACE,
    "merlin.compiler.quantization": _PREPROCESSING_NAMESPACE,
    "merlin.compiler.transforms": _PREPROCESSING_NAMESPACE,
    "merlin.compiler.lower": _PREPROCESSING_NAMESPACE,
    "merlin.compiler.lower.scalar": _PREPROCESSING_NAMESPACE,
}

SOURCE_BINDING_PROTOCOL = "merlin.shared_source_binding.v1"
SOURCE_BINDING_MODULES = ("merlin.compiler.schedule.direct_source", "merlin.compiler.schedule.source_tokens")


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
    source_root = Path(__file__).resolve().parents[3]
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
        source = (Path(__file__).parents[2] / "transforms" / (name + ".py")).read_bytes()
        destination = directory / ("common_" + name + ".py")
        destination.write_bytes(source)
        files[destination.name] = hashlib.sha256(source).hexdigest()
    if include_block_schedule:
        from merlin.compiler.schedule.standalone import block_schedule_source

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
