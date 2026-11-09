"""Build and admit a saved host/device image; no simulator launch or numerical comparison."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.runtime import modelinputs as MI
from merlin.compiler.backend import host
from merlin.compiler.backend.host import HostPackage
from merlin.runtime.backends import spike_model
from merlin.runtime.boards import CONSOLE_HTIF, FLOW_BAREMETAL, Board, load_boards

from .modeloptions import BaremetalModelError, _sha


@dataclass(frozen=True)
class SelectBuildResult:
    selected: Board
    pkg: HostPackage
    isas: list[str]
    marches: list[str]
    device_tree: str | None


def select_build(
    *,
    board,
    catalog_path,
    device,
    device_path,
    dts_path,
    host_bounded_rne,
    host_elementwise_fusion,
    host_roundeven,
    package_path,
    receipt,
    require_isa_dts,
    storage_policy,
    target,
):
    """Admit board, host ISA and selected device source ownership for the build."""
    boards = load_boards(catalog_path)
    selected = boards.get(board)
    if selected is None or selected.target != target:
        raise BaremetalModelError("selected board is absent or names a different target")
    if selected.flow != FLOW_BAREMETAL or selected.console != CONSOLE_HTIF:
        raise BaremetalModelError("selected board is not a bare-metal HTIF board")
    if selected.harts != 1 or selected.code_reserve is None or selected.host_dts_sha256 is None:
        raise BaremetalModelError("complete saved-model executor needs one hart, code reserve and pinned host DTS")
    if selected.dram_base != spike_model.DRAM_BASE:
        raise BaremetalModelError("selected board DRAM differs from the current bare-metal runner base")
    pkg = host.load_host_package(package_path)
    if pkg.backend not in {"scalar", "rvv"}:
        raise BaremetalModelError(f"unsupported whole-model host package backend {pkg.backend!r}")
    if host_elementwise_fusion and pkg.backend != "scalar":
        raise BaremetalModelError("selected host elementwise fusion requires the scalar host lane")
    isas = require_isa_dts(pkg.cflags, dts_path, expected_sha256=selected.host_dts_sha256)
    if len(isas) != selected.harts or len(set(isas)) != 1:
        raise BaremetalModelError("DTS CPU ISA roster does not match selected board")
    marches = [flag.removeprefix("-march=") for flag in pkg.cflags if flag.startswith("-march=")]
    if len(marches) != 1:
        raise BaremetalModelError("host package must declare exactly one -march")
    if host_bounded_rne or host_roundeven:
        from merlin.compiler.backend.isa import require_host_isa

        if pkg.backend != "scalar":
            raise BaremetalModelError("host RNE choices require the scalar host lane")
        require_host_isa(["-march=rv64gc"], marches[0])
    if device is not None:
        device_tree = MI.strict_tree_sha256(device_path)
        receipt["inputs"]["device"] = {
            "name": device.device,
            "package": str(device.package_dir),
            "package_tree": device_tree,
        }
        if getattr(device, "pre_group_layout", None) is not None:
            receipt["inputs"]["device"]["pre_group_layout"] = device.pre_group_layout.record()
        if storage_policy is not None:
            receipt["inputs"]["device"]["intermediate_storage"] = storage_policy.record()
        if getattr(device, "descriptor_results", False):
            receipt["inputs"]["device"]["descriptor_results"] = True
    else:
        device_tree = None
    return SelectBuildResult(selected=selected, pkg=pkg, isas=isas, marches=marches, device_tree=device_tree)


@dataclass(frozen=True)
class BuildSavedResult:
    built: dict[str, Any]
    readonly_record: dict | None
    elf: Path
    elf_sha: str
    sidecar: Path
    profile_table: Path
    profile_table_sha: str | None
    graph_artifacts: dict | None
    early_layout_artifacts: dict | None
    storage_artifacts: dict | None
    preparation: Path | None
    preparation_sha: str | None
    preparation_inputs: tuple[tuple[Path, str], ...] | None
    provider_record: dict | None
    recheck_host_graph_artifacts: Callable | None
    receipt_borrowed_lifetime: Callable | None
    recheck_storage: Callable | None
    revalidate_host_provider: Callable | None
    require_observation_selection: Callable | None


def build_saved(
    *,
    arena_mb,
    blob_weights,
    capture_path,
    dense_input_views,
    device,
    early_layout_policy,
    full,
    graph_optimizations,
    group_profile,
    host_bounded_rne,
    host_elementwise_fusion,
    host_llvm_bitcode,
    host_math_policy,
    host_math_symbols,
    host_roundeven,
    isas,
    math_archive_symbols,
    max_readonly_input_bytes,
    op_profile,
    output_path,
    pkg,
    prepack_weights,
    readback,
    readonly_input_indices,
    receipt,
    recomputation_check,
    respect_captured_quantization_scope,
    selected,
    share_exact_tensor_preparations,
    shared_source_lowerings,
    source_choices,
    source_optimizations,
    source_recomputation,
    storage_policy,
    weights_source,
):
    """Compile and bind actual ELF, preparation and profile products without executing them."""
    graph_artifacts = None
    early_layout_artifacts = None
    storage_artifacts = None
    preparation = None
    preparation_sha = None
    preparation_inputs = None
    provider_record = None
    recheck_host_graph_artifacts = None
    receipt_borrowed_lifetime = None
    recheck_storage = None
    revalidate_host_provider = None
    require_observation_selection = None
    build_options = {"math_archive_symbols": math_archive_symbols} if math_archive_symbols is not None else {}
    if recomputation_check is not None:
        recomputation_check()
        build_options["source_recomputation"] = source_recomputation
    if graph_optimizations is not None:
        build_options["graph_optimizations"] = graph_optimizations
    if shared_source_lowerings:
        receipt["inputs"]["shared_source_lowerings"] = [type(value).__name__ for value in shared_source_lowerings]
        build_options["shared_source_lowerings"] = shared_source_lowerings
    if weights_source is not None:
        receipt["inputs"]["weights_source"] = str(Path(weights_source).resolve(strict=True))
        build_options["weights_source"] = weights_source
    if host_math_symbols is not None:
        build_options["host_math_symbols"] = host_math_symbols
    if host_math_policy != "native":
        build_options["host_math_policy"] = host_math_policy
    if respect_captured_quantization_scope:
        receipt["inputs"]["respect_captured_quantization_scope"] = True
    if host_llvm_bitcode:
        receipt["inputs"]["host_llvm_bitcode"] = True
    if op_profile:
        receipt["inputs"]["op_profile"] = True
        build_options["op_profile"] = True
    if respect_captured_quantization_scope:
        from merlin.compiler.quantization.scope import FEATURE as captured_quantization_scope

        build_options["features"] = frozenset({captured_quantization_scope})
    if prepack_weights:
        from merlin.llvmlower.weight_prepack import ensure_registered

        receipt["inputs"]["prepack_weights"] = True
        build_options["features"] = frozenset((*build_options.get("features", ()), ensure_registered()))
    if dense_input_views:
        receipt["inputs"]["dense_input_views"] = True
        build_options["dense_input_views"] = True
    if share_exact_tensor_preparations:
        from merlin.llvmlower.tensor_provenance_cse import ensure_registered

        receipt["inputs"]["share_exact_tensor_preparations"] = True
        build_options["features"] = frozenset((*build_options.get("features", ()), ensure_registered()))
    if host_elementwise_fusion:
        from merlin.llvmlower.impr_features import FUSE_AFTER_GENERALIZE_NAME

        receipt["inputs"]["host_elementwise_fusion"] = True
        build_options["features"] = frozenset((*build_options.get("features", ()), FUSE_AFTER_GENERALIZE_NAME))
    if host_bounded_rne:
        from merlin.llvmlower.late_quant_rne import merlin_host_llvm_transform
        from merlin.llvmlower.toolchain import clang

        receipt["inputs"]["host_bounded_rne"] = {"host_isa": "rv64gc", "combine_clamp": True}
        build_options["host_llvm_transform"] = merlin_host_llvm_transform(
            Path(clang()).resolve().parent, host_isa="rv64gc", combine_clamp=True
        )
    if host_roundeven:
        receipt["inputs"]["host_roundeven"] = {"host_isa": "rv64gc", "abi": "lp64d", "symbol": "roundevenf"}
        build_options["host_roundeven"] = True
    from merlin.llvmlower.blob_weights import requested as blob_weights_requested

    build_started = time.monotonic()
    with blob_weights_requested(blob_weights):
        built = spike_model.build(
            capture_path,
            output_path / "build",
            arena_mb=arena_mb,
            dram_base=selected.dram_base,
            dram_bytes=selected.dram_bytes,
            code_reserve=selected.code_reserve,
            int8_compute=bool(pkg.is_int8),
            backend=pkg.backend,
            rvv_schedule=pkg.schedule_text if pkg.backend == "rvv" else None,
            cflags_override=list(pkg.cflags),
            vlen=selected.vlen if pkg.backend == "rvv" else None,
            console=selected.console,
            device=device,
            full_readback=full,
            **(
                {
                    "readonly_input_indices": readonly_input_indices,
                    "max_readonly_input_bytes": max_readonly_input_bytes,
                }
                if readonly_input_indices
                else {}
            ),
            group_profile=bool(group_profile),
            **({"host_llvm_bitcode": True} if host_llvm_bitcode else {}),
            **build_options,
            **source_choices.options(),
        )
    receipt["compilation"] = {
        "wall_seconds": time.monotonic() - build_started,
        "scope": "ordinary model build and selected compiler checks; excludes capture and target execution",
        "peak_memory_bytes": None,
    }
    if not isinstance(built.get("build_hash"), str) or not built["build_hash"]:
        raise BaremetalModelError("bare-metal build returned no citable build hash")
    readonly_record = None
    if readonly_input_indices:
        from merlin.runtime.input_observation import require_observation_selection

        readonly_record = json.loads(json.dumps(built.get("readonly_input_observation")))
        require_observation_selection(
            readonly_record, indices=readonly_input_indices, max_bytes=max_readonly_input_bytes
        )
    if not isinstance(built.get("index_lowering"), dict) or built["index_lowering"].get("schema") not in {
        "merlin.selected-index-lowering.v1",
        "merlin.selected-index-lowering.v2",
    }:
        raise BaremetalModelError("bare-metal build returned no selected index-lowering record")
    if built["index_lowering"]["schema"] == "merlin.selected-index-lowering.v2":
        from merlin.llvmlower.target_data_layout import selected_index_bits

        observation = {key: value for key, value in built["index_lowering"].items() if key != "effective_pipeline"}
        if selected_index_bits(observation) != built["index_lowering"].get("index_bits"):
            raise BaremetalModelError("bare-metal build returned an invalid implicit-pointer observation")
    elf = Path(built["elf"])
    source_choices.accept(built.get("source_optimizations"), build_dir=output_path / "build", executable=elf)
    if recomputation_check is not None:
        recomputation_check()
        record = source_recomputation.record()
        if (
            built.get("source_recomputation") != record
            or record["executable"]["path"] != str(elf.absolute())
            or record["executable"]["sha256"] != _sha(elf)
        ):
            raise BaremetalModelError("ordinary recomputation returned a foreign build record or final ELF")
        recomputation_check()
    if not elf.is_file() or elf.is_symlink() or not elf.resolve().is_relative_to(output_path):
        raise BaremetalModelError("bare-metal build returned no safe ELF in its output")
    elf_sha = _sha(elf)
    arch = spike_model.arch_extensions(elf)
    MI.require_elf_isa_supported(arch, isas[0], require_scalar=False)
    receipt["output"] = {
        "elf": str(elf),
        "elf_sha256": elf_sha,
        "elf_arch_extensions": arch,
        "build_hash": built.get("build_hash"),
        "index_lowering": built["index_lowering"],
        "matrix_routing": built.get("matrix_routing"),
        "host_artifacts": built.get("host_artifacts"),
        **({"host_bitcode_transport": built["host_bitcode_transport"]} if host_llvm_bitcode else {}),
        "device_artifacts": built.get("device_artifacts"),
        **({"source_optimizations": source_choices.record()} if source_optimizations is not None else {}),
        **({"source_recomputation": source_recomputation.record()} if recomputation_check is not None else {}),
    }
    if host_bounded_rne:
        if not isinstance(built.get("host_llvm_transform"), dict):
            raise BaremetalModelError("selected host bounded RNE returned no pre-object transform record")
        receipt["output"]["host_llvm_transform"] = built["host_llvm_transform"]
    if getattr(device, "descriptor_results", False):
        if built.get("descriptor_results", {}).get("selected") is not True:
            raise BaremetalModelError("selected descriptor bridge returned no actual selection receipt")
        receipt["output"]["descriptor_results"] = built["descriptor_results"]
    if graph_optimizations is not None:
        from merlin.llvmlower.graph_optimizations import recheck_host_graph_artifacts

        graph_artifacts = recheck_host_graph_artifacts(output_path / "build" / "lower", graph_optimizations)
        receipt["output"]["host_graph_optimization"] = graph_artifacts
    if early_layout_policy is not None:
        from merlin.llvmlower.graph_optimizations import recheck_host_graph_artifacts

        early_layout_artifacts = recheck_host_graph_artifacts(
            output_path / "build" / "pre_group_layout", early_layout_policy
        )
        receipt["output"]["pre_group_layout"] = early_layout_artifacts
    if storage_policy is not None:
        from merlin.llvmlower.borrowed_arena import receipt_borrowed_lifetime
        from merlin.llvmlower.host_storage_policy import recheck as recheck_storage

        storage_artifacts = recheck_storage(
            output_path / "build" / "intermediate_storage",
            storage_policy,
            borrowed_lifetime=receipt_borrowed_lifetime(built),
        )
        receipt["output"]["intermediate_storage"] = storage_artifacts
    if shared_source_lowerings:
        from merlin.llvmlower.prepared_model_transform import RECEIPT as PREPARATION_RECEIPT
        from merlin.runtime.host_provider import revalidate_host_provider

        preparation = output_path / "build" / "prepared_model_transform" / PREPARATION_RECEIPT
        if preparation.is_symlink() or not preparation.is_file():
            raise BaremetalModelError("shared source build returned no safe preparation receipt")
        preparation_sha = _sha(preparation)
        prepared_record = json.loads(preparation.read_text())
        if prepared_record.get("schema") != "merlin.prepared_model_transform.v1":
            raise BaremetalModelError("shared source build returned an unsupported preparation receipt")
        preparation_inputs = tuple(
            (Path(prepared_record[role + "_path"]), prepared_record[role + "_sha256"])
            for role in ("source", "input_snapshot", "selected")
        )
        receipt["output"]["shared_source_preparation"] = {"path": str(preparation), "sha256": preparation_sha}
        provider_record = json.loads(json.dumps(built.get("host_provider")))
        revalidate_host_provider(provider_record)
        if provider_record is None:
            if prepared_record["selected_sha256"] != prepared_record["input_snapshot_sha256"]:
                raise BaremetalModelError("shared source replacement lost its completed provider record")
        else:
            expected_sources = [
                {
                    "path": str(Path(prepared_record["selected_path"]).resolve()),
                    "sha256": prepared_record["selected_sha256"],
                },
                *(
                    {
                        "path": str(Path(built["host_artifacts"][role]["path"]).resolve()),
                        "sha256": built["host_artifacts"][role]["sha256"],
                    }
                    for role in ("llvm_ir", "object")
                ),
            ]
            if (
                provider_record["executable"] != {"path": str(elf.resolve()), "sha256": elf_sha}
                or provider_record["source_model"] != expected_sources
            ):
                raise BaremetalModelError("shared source provider belongs to a different ELF or prepared model")
        for key in ("host_provider", "linked_elf_admission"):
            if key in built:
                receipt["output"][key] = json.loads(json.dumps(built[key]))
        for value in shared_source_lowerings:
            value.selected_contracts.revalidate()
    # Bind the producer's record, rather than minting a new command history.
    # Its consumer independently verifies completed commands and input bytes.
    from merlin.llvmlower.compilation_recipe import FILENAME as RECIPE_FILENAME

    recipe = elf.parent / RECIPE_FILENAME
    if recipe.exists():
        if recipe.is_symlink() or not recipe.is_file():
            raise BaremetalModelError("bare-metal compilation recipe is absent or indirect")
        receipt["output"]["compilation_recipe"] = {"path": str(recipe), "sha256": _sha(recipe)}
    from merlin.llvmlower.device_offload import SIDECAR_NAME

    sidecar = output_path / "build" / SIDECAR_NAME
    if sidecar.is_file() and not sidecar.is_symlink():
        receipt["output"]["device_sidecar"] = {"path": str(sidecar), "sha256": _sha(sidecar)}
    if device is not None and "device_sidecar" not in receipt["output"]:
        raise BaremetalModelError("device was selected but the build emitted no device dispatch sidecar")
    if device is not None:
        receipt["output"]["device_dispatch_evidence"] = "static_sidecar_only; execution not established"
    receipt["output"]["readback"] = readback
    profile_table = output_path / "build" / "op_profile_table.json"
    profile_table_sha = None
    if op_profile:
        if not profile_table.is_file() or profile_table.is_symlink():
            raise BaremetalModelError("operation profiling selected but no safe operation table was emitted")
        profile_table_sha = _sha(profile_table)
        receipt["output"]["op_profile_table"] = {"path": str(profile_table), "sha256": profile_table_sha}
    return BuildSavedResult(
        built=built,
        readonly_record=readonly_record,
        elf=elf,
        elf_sha=elf_sha,
        sidecar=sidecar,
        profile_table=profile_table,
        profile_table_sha=profile_table_sha,
        graph_artifacts=graph_artifacts,
        early_layout_artifacts=early_layout_artifacts,
        storage_artifacts=storage_artifacts,
        preparation=preparation,
        preparation_sha=preparation_sha,
        preparation_inputs=preparation_inputs,
        provider_record=provider_record,
        recheck_host_graph_artifacts=recheck_host_graph_artifacts,
        receipt_borrowed_lifetime=receipt_borrowed_lifetime,
        recheck_storage=recheck_storage,
        revalidate_host_provider=revalidate_host_provider,
        require_observation_selection=require_observation_selection,
    )
