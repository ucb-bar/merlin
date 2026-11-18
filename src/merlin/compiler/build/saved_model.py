"""Validate saved-model options and explicit input/output ownership before building."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.paths import out_dir
from merlin.compiler.backend import host
from merlin.compiler.backend.host import HostPackage
from merlin.runtime.backends.baremetal import model_build as _model_elf_metadata
from merlin.runtime.backends.baremetal import model_build as _model_memory_layout
from merlin.runtime.backends.baremetal import model_build as spike_build
from merlin.runtime.boards.catalog import CONSOLE_HTIF, FLOW_BAREMETAL, Board, load_boards
from merlin.runtime.execution import model_inputs as MI

# ---- modeloptions ------------------------------------------------------------------------------


class BaremetalModelError(ValueError):
    """Selected bytes cannot support the requested whole-model claim."""


def _sha(path: Path) -> str:
    return MI.file_sha256(path)


def _output_root(path: str | Path) -> Path:
    lexical = Path(path)
    if lexical.is_symlink():
        raise BaremetalModelError(f"output must not be a symlink: {lexical}")
    output = lexical.resolve()
    root = out_dir().resolve()
    if output == root or not output.is_relative_to(root) or output.exists():
        raise BaremetalModelError(f"output must be a fresh directory below {root}: {output}")
    return output


@dataclass(frozen=True)
class PrepareSavedResult:
    recomputation_check: Callable | None
    full: bool
    source_choices: object
    host_bounded_rne: bool
    early_layout_policy: object
    storage_policy: object
    capture_path: Path
    package_path: Path
    catalog_path: Path
    dts_path: Path
    output_path: Path
    device_path: Path | None


def prepare_saved(
    *,
    arena_mb,
    blob_weights,
    board,
    board_catalog,
    capture,
    dense_input_views,
    device,
    dts,
    graph_optimizations,
    host_bounded_rne,
    host_elementwise_fusion,
    host_llvm_bitcode,
    host_math_policy,
    host_roundeven,
    max_readonly_input_bytes,
    op_profile,
    output,
    package,
    prepack_weights,
    readback,
    readonly_input_indices,
    respect_captured_quantization_scope,
    rtl_facts,
    run,
    share_exact_tensor_preparations,
    shared_source_lowerings,
    source_optimizations,
    source_recomputation,
    target,
    timeout_s,
    tolerance,
):
    """Validate selected options and bind ordinary source/output paths before build IO."""
    recomputation_check = None
    if source_recomputation is not None:
        from merlin.runtime.backends.baremetal.source_recomputation import freeze_source_recomputation

        recomputation_check = freeze_source_recomputation(source_recomputation)
    from merlin.runtime.backends.baremetal.host_provider import SourceOptimizationBuild
    from merlin.runtime.host_math import host_math_recipe

    try:
        host_math_recipe(host_math_policy)
    except ValueError as error:
        raise BaremetalModelError(str(error)) from error
    if run not in {"none", "spike", "gsim", "verilator"}:
        raise BaremetalModelError(f"unsupported bare-metal run {run!r}")
    if readback not in {"prefix", "full"}:
        raise BaremetalModelError(f"unsupported readback {readback!r}; choose prefix or full")
    if tolerance is not None and (
        not isinstance(tolerance, dict)
        or set(tolerance) != {"atol", "rtol"}
        or any(isinstance(v, bool) or not isinstance(v, int | float) or v < 0 for v in tolerance.values())
    ):
        raise BaremetalModelError("tolerance declares exactly non-negative atol and rtol")
    full = readback == "full"
    source_choices = SourceOptimizationBuild(source_optimizations)
    from merlin.runtime.harness.output_observation import validate_selection

    validate_selection(readonly_input_indices, max_readonly_input_bytes)
    if readonly_input_indices and not full:
        raise BaremetalModelError("readonly input observation requires full readback")
    if type(respect_captured_quantization_scope) is not bool:
        raise BaremetalModelError("captured quantization scope must be an explicit Boolean")
    if type(host_llvm_bitcode) is not bool:
        raise BaremetalModelError("host LLVM bitcode transport must be an explicit Boolean")
    if type(op_profile) is not bool:
        raise BaremetalModelError("operation profiling must be an explicit Boolean")
    if type(prepack_weights) is not bool:
        raise BaremetalModelError("weight prepacking must be an explicit Boolean optimization choice")
    if type(dense_input_views) is not bool:
        raise BaremetalModelError("dense input views must be an explicit Boolean choice")
    if type(share_exact_tensor_preparations) is not bool:
        raise BaremetalModelError("tensor preparation sharing must be an explicit Boolean choice")
    if type(host_elementwise_fusion) is not bool:
        raise BaremetalModelError("host elementwise fusion must be an explicit Boolean optimization choice")
    if type(host_bounded_rne) is not bool:
        raise BaremetalModelError("host bounded RNE must be an explicit Boolean optimization choice")
    if type(host_roundeven) is not bool:
        raise BaremetalModelError("host roundeven must be an explicit Boolean optimization choice")
    from merlin.compiler.lower.quant.host_bounded_rne import selected as bounded_rne_selected

    host_bounded_rne = bounded_rne_selected(host_bounded_rne)
    if type(shared_source_lowerings) is not tuple:
        raise BaremetalModelError("shared source lowerings require an immutable tuple of typed selections")
    if graph_optimizations is not None:
        from merlin.compiler.lower.host.graph import HostGraphPolicy

        if type(graph_optimizations) is HostGraphPolicy:
            graph_optimizations.revalidate()
            if graph_optimizations.target != target:
                raise BaremetalModelError("host graph software target differs from the selected model target")
    early_layout_policy = getattr(device, "pre_group_layout", None)
    if early_layout_policy is not None:
        from merlin.compiler.lower.host.graph import HostGraphPolicy

        if type(early_layout_policy) is not HostGraphPolicy or early_layout_policy.target != target:
            raise BaremetalModelError("pre-group layout requires original software for this target")
        early_layout_policy.revalidate()
    storage_policy = getattr(device, "intermediate_storage", None)
    if storage_policy is not None:
        from merlin.compiler.lower.host.graph import HostGraphPolicy

        if type(storage_policy) is not HostGraphPolicy or storage_policy.target != target:
            raise BaremetalModelError("intermediate storage requires original software for this target")
        storage_policy.revalidate()
    if shared_source_lowerings:
        from merlin.compiler.lower.source.dense import SourceDenseLowering
        from merlin.compiler.lower.source.families import SourcePointwiseLowering, SourceRowLowering

        allowed = (SourceDenseLowering, SourcePointwiseLowering, SourceRowLowering)
        if any(type(value) not in allowed for value in shared_source_lowerings):
            raise BaremetalModelError("shared source lowerings require explicit dense, pointwise or row selections")
        if len({type(value) for value in shared_source_lowerings}) != len(shared_source_lowerings):
            raise BaremetalModelError("duplicate shared source family selections are ambiguous")
        for value in shared_source_lowerings:
            value.selected_contracts.revalidate()
    if type(blob_weights) is not bool:
        raise BaremetalModelError("blob weights must be an explicit Boolean")
    if type(arena_mb) is not int or arena_mb < 1 or type(timeout_s) is not int or timeout_s < 1:
        raise BaremetalModelError("arena_mb and timeout_s must be positive integers")
    paths = [Path(capture), Path(package), Path(board_catalog), Path(dts)]
    if rtl_facts is not None:
        paths.append(Path(rtl_facts))
    if any(path.is_symlink() for path in paths):
        raise BaremetalModelError("explicit inputs must not be symlinks")
    capture_path, package_path, catalog_path, dts_path = (path.resolve(strict=True) for path in paths[:4])
    output_path = _output_root(output)
    if recomputation_check is not None:
        recomputation_check()
        source_recomputation.capture(
            capture_path,
            {
                "target": target,
                "board": board,
                "run": run,
                "package": str(package_path),
                "board_catalog": str(catalog_path),
                "dts": str(dts_path),
                "rtl_facts": str(rtl_facts) if rtl_facts else None,
                "readback": readback,
            },
        )
        recomputation_check()
    if output_path.is_relative_to(capture_path) or output_path.is_relative_to(package_path):
        raise BaremetalModelError("output must not be nested inside a saved capture or package")
    device_path = None
    if device is not None:
        from merlin.compiler.lower.device.device_build import DeviceRouting

        if not isinstance(device, DeviceRouting):
            raise BaremetalModelError("device must be an explicit DeviceRouting")
        device_path = Path(device.package_dir)
        if device_path.is_symlink():
            raise BaremetalModelError("selected device package must not be a symlink")
        device_path = device_path.resolve(strict=True)
        if output_path.is_relative_to(device_path):
            raise BaremetalModelError("output must not be nested inside the selected device package")
    return PrepareSavedResult(
        recomputation_check=recomputation_check,
        full=full,
        source_choices=source_choices,
        host_bounded_rne=host_bounded_rne,
        early_layout_policy=early_layout_policy,
        storage_policy=storage_policy,
        capture_path=capture_path,
        package_path=package_path,
        catalog_path=catalog_path,
        dts_path=dts_path,
        output_path=output_path,
        device_path=device_path,
    )


# ---- saved -------------------------------------------------------------------------------------
# Build and admit a saved host/device image; no simulator launch or numerical comparison.


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
    if selected.dram_base != _model_memory_layout.DRAM_BASE:
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
        from merlin.compiler.backend.host import require_host_isa

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
        from merlin.compiler.lower.options.quantization_scope import FEATURE as captured_quantization_scope

        build_options["features"] = frozenset({captured_quantization_scope})
    if prepack_weights:
        from merlin.compiler.lower.options.weight_prepack import ensure_registered

        receipt["inputs"]["prepack_weights"] = True
        build_options["features"] = frozenset((*build_options.get("features", ()), ensure_registered()))
    if dense_input_views:
        receipt["inputs"]["dense_input_views"] = True
        build_options["dense_input_views"] = True
    if share_exact_tensor_preparations:
        from merlin.compiler.lower.options.tensor_provenance_cse import ensure_registered

        receipt["inputs"]["share_exact_tensor_preparations"] = True
        build_options["features"] = frozenset((*build_options.get("features", ()), ensure_registered()))
    if host_elementwise_fusion:
        from merlin.compiler.optimizations.registry import FUSE_AFTER_GENERALIZE_NAME

        receipt["inputs"]["host_elementwise_fusion"] = True
        build_options["features"] = frozenset((*build_options.get("features", ()), FUSE_AFTER_GENERALIZE_NAME))
    if host_bounded_rne:
        from merlin.compiler.lower.model.toolchain import clang
        from merlin.compiler.lower.quant.late_quant_rne import merlin_host_llvm_transform

        receipt["inputs"]["host_bounded_rne"] = {"host_isa": "rv64gc", "combine_clamp": True}
        build_options["host_llvm_transform"] = merlin_host_llvm_transform(
            Path(clang()).resolve().parent, host_isa="rv64gc", combine_clamp=True
        )
    if host_roundeven:
        receipt["inputs"]["host_roundeven"] = {"host_isa": "rv64gc", "abi": "lp64d", "symbol": "roundevenf"}
        build_options["host_roundeven"] = True
    from merlin.compiler.lower.c.weights import requested as blob_weights_requested

    build_started = time.monotonic()
    with blob_weights_requested(blob_weights):
        built = spike_build.build(
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
        from merlin.runtime.harness.output_observation import require_observation_selection

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
        from merlin.compiler.lower.model.target_data_layout import selected_index_bits

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
    arch = _model_elf_metadata.arch_extensions(elf)
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
        from merlin.compiler.lower.host.graph import recheck_host_graph_artifacts

        graph_artifacts = recheck_host_graph_artifacts(output_path / "build" / "lower", graph_optimizations)
        receipt["output"]["host_graph_optimization"] = graph_artifacts
    if early_layout_policy is not None:
        from merlin.compiler.lower.host.graph import recheck_host_graph_artifacts

        early_layout_artifacts = recheck_host_graph_artifacts(
            output_path / "build" / "pre_group_layout", early_layout_policy
        )
        receipt["output"]["pre_group_layout"] = early_layout_artifacts
    if storage_policy is not None:
        from merlin.compiler.lower.host.borrowed_arena import receipt_borrowed_lifetime
        from merlin.compiler.lower.host.host_storage_policy import recheck as recheck_storage

        storage_artifacts = recheck_storage(
            output_path / "build" / "intermediate_storage",
            storage_policy,
            borrowed_lifetime=receipt_borrowed_lifetime(built),
        )
        receipt["output"]["intermediate_storage"] = storage_artifacts
    if shared_source_lowerings:
        from merlin.compiler.lower.model.prepared_model_transform import RECEIPT as PREPARATION_RECEIPT
        from merlin.runtime.backends.baremetal.host_provider import revalidate_host_provider

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
    from merlin.compiler.lower.model.compilation_recipe import FILENAME as RECIPE_FILENAME

    recipe = elf.parent / RECIPE_FILENAME
    if recipe.exists():
        if recipe.is_symlink() or not recipe.is_file():
            raise BaremetalModelError("bare-metal compilation recipe is absent or indirect")
        receipt["output"]["compilation_recipe"] = {"path": str(recipe), "sha256": _sha(recipe)}
    from merlin.compiler.lower.device.offload import SIDECAR_NAME

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
