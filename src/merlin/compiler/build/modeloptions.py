"""Validate saved-model options and explicit input/output ownership before building."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from merlin.common.paths import out_dir
from merlin.runtime import modelinputs as MI


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
        from merlin.runtime.source_recomputation_build import freeze_source_recomputation

        recomputation_check = freeze_source_recomputation(source_recomputation)
    from merlin.runtime.host_math import host_math_recipe
    from merlin.runtime.source_optimization_build import SourceOptimizationBuild

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
    from merlin.runtime.input_observation import validate_selection

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
    from merlin.llvmlower.host_bounded_rne import selected as bounded_rne_selected

    host_bounded_rne = bounded_rne_selected(host_bounded_rne)
    if type(shared_source_lowerings) is not tuple:
        raise BaremetalModelError("shared source lowerings require an immutable tuple of typed selections")
    if graph_optimizations is not None:
        from merlin.llvmlower.host_graph_policy import HostGraphPolicy

        if type(graph_optimizations) is HostGraphPolicy:
            graph_optimizations.revalidate()
            if graph_optimizations.target != target:
                raise BaremetalModelError("host graph software target differs from the selected model target")
    early_layout_policy = getattr(device, "pre_group_layout", None)
    if early_layout_policy is not None:
        from merlin.llvmlower.host_graph_policy import HostGraphPolicy

        if type(early_layout_policy) is not HostGraphPolicy or early_layout_policy.target != target:
            raise BaremetalModelError("pre-group layout requires original software for this target")
        early_layout_policy.revalidate()
    storage_policy = getattr(device, "intermediate_storage", None)
    if storage_policy is not None:
        from merlin.llvmlower.host_graph_policy import HostGraphPolicy

        if type(storage_policy) is not HostGraphPolicy or storage_policy.target != target:
            raise BaremetalModelError("intermediate storage requires original software for this target")
        storage_policy.revalidate()
    if shared_source_lowerings:
        from merlin.llvmlower.source_dense import SourceDenseLowering
        from merlin.llvmlower.source_pointwise import SourcePointwiseLowering
        from merlin.llvmlower.source_rows import SourceRowLowering

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
        from merlin.llvmlower.device_build import DeviceRouting

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
