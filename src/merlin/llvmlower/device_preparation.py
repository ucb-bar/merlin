"""Preserve original device preparation and a source-bound placement selection.

The original route still owns eligibility, package execution and qualification.
This module only carries a fixed selection through the existing preparation
callbacks; it supplies no numerical, ABI, completion or timing authority.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _DeviceOffloadInvocation:
    selection: object
    record: dict | None
    sidecar: tuple[Path, str] | None = None
    sources: tuple[tuple[Path, str], ...] = ()
    llvm_transform: tuple | None = None
    target_schedule: tuple | None = None
    connected_spatial: tuple | None = None
    spatial_host: object | None = None
    quantized_sum: tuple | None = None
    sibling_source: tuple | None = None
    sibling_host: object | None = None
    pre_group_layout: tuple | None = None
    intermediate_storage: tuple | None = None


def _layout_policy(device, field="pre_group_layout"):
    selected = getattr(device, field, None)
    if selected is None:
        return None
    from .host_graph_policy import HostGraphPolicy

    if type(selected) is not HostGraphPolicy or selected.target != getattr(device, "device", None):
        raise ValueError("pre-group layout needs original software for the selected device")
    selected.revalidate()
    return selected, json.dumps(selected.record(), sort_keys=True)


def _require_layout_policy(device, frozen, field="pre_group_layout"):
    actual = _layout_policy(device, field)
    original = getattr(frozen, field) if type(frozen) is _DeviceOffloadInvocation else None
    if ((actual is None) != (original is None)
            or actual is not None and (actual[0] is not original[0] or actual[1] != original[1])):
        raise ValueError(f"{field} selection changed after original admission")


def device_route_requested(device):
    return device is not None and any(
        getattr(device, name, None) is not None
        for name in (
            "select", "exact_selection", "catalog_manifest", "catalog_builder", "offload_selection",
            "connected_spatial", "sibling_source", "quantized_sum"
        )
    )


def require_device_offload_artifacts(invocation):
    """Reopen realized file identities without invoking any caller callbacks."""
    if invocation is None:
        return
    pins = (*invocation.sources, *((invocation.sidecar,) if invocation.sidecar is not None else ()))
    for path, digest in pins:
        if _artifact_digest(path) != digest:
            raise ValueError("device offload original realized source or sidecar changed")


def _artifact_file(path):
    if path.resolve() != path or path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError("device offload artifact needs its original direct regular file")
    return path


def _artifact_digest(path):
    from merlin.common.digest import sha256_file

    return sha256_file(_artifact_file(path))


def _artifact_payload(path):
    from merlin.common.strict_json import MAX_RECORD_BYTES

    path = _artifact_file(path)
    maximum = MAX_RECORD_BYTES
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("device offload artifact exceeds its compilation byte budget")
    return payload


def retain_device_offload_source(frozen, path, *, compilation=None):
    """Pin an actual trusted preparation product before any further callbacks.

    Only the normal compiler preparation stages may issue the next source.
    An opaque callback's result cannot establish source-call correspondence.
    """
    if frozen is None:
        return
    require_device_offload_artifacts(frozen)
    path = Path(path).absolute()
    pin = (path, _artifact_digest(path))
    if pin not in frozen.sources:
        frozen.sources = (*frozen.sources, pin)
    if compilation is not None:
        for index, (source, digest) in enumerate(frozen.sources):
            # Checked weight preparation can already own this exact realized
            # model dependency. Retain its original role/pin and the explicit
            # source-chain pin above, rather than minting a duplicate recipe
            # identity. A path match never permits changed bytes.
            preparation = getattr(compilation, "record", {}).get("preparation", {})
            existing = [pin for pin in preparation.values() if pin.get("path") == str(source)]
            if existing:
                if (len(existing) != 1 or existing[0].get("sha256") != digest
                        or existing[0].get("bytes") != source.stat().st_size):
                    raise ValueError("device source differs from its already retained preparation identity")
                continue
            compilation.bind_preparation(f"device_offload_source_{index}", source)
        if frozen.sidecar is not None:
            compilation.bind_preparation("device_offload_sidecar", frozen.sidecar[0])


def require_supported_device_preparation(frozen, transform):
    if (
        frozen is not None and (
            frozen.selection is not None or frozen.connected_spatial is not None or frozen.sibling_source is not None or frozen.quantized_sum is not None
        )
        and transform is not None
    ):
        raise ValueError("source-bound device selection does not support a later opaque prepared-model rewrite")


def finish_device_offload_selection(
    device, frozen, *, admission, result, elf, linked_sha256, target_configuration, artifacts
):
    """Run live checks before the final callback-free realized-source check."""
    from .device_build import (
        freeze_device_elf_custody,
        revalidate_device_elf_custody,
        revalidate_device_target_artifacts,
    )

    original = (
        None
        if frozen is None
        else (
            frozen.selection, frozen.sources, frozen.sidecar, frozen.llvm_transform, frozen.target_schedule,
            frozen.connected_spatial, frozen.spatial_host, frozen.sibling_source, frozen.sibling_host, frozen.quantized_sum,
        )
    )
    custody = (
        None
        if frozen is None
        else freeze_device_elf_custody(device, admission, elf=elf, linked_sha256=linked_sha256, result=result)
    )
    revalidate_device_target_artifacts(device, target_configuration, artifacts)
    schedule = _require_device_target_schedule(device, frozen)
    if schedule is not None:
        from .device_target_passes import schedule_artifacts

        schedule_artifacts(device, schedule, artifacts)
    require_device_offload_selection(device, frozen)
    if custody is not None:
        revalidate_device_elf_custody(device, admission, custody)
    if frozen is not None:
        selected, sources, sidecar, transform, target_schedule, spatial, spatial_host, sibling, sibling_host, quantized = original
        if (
            frozen.selection is not selected
            or getattr(device, "offload_selection", None) is not selected
            or (frozen.sources, frozen.sidecar) != (sources, sidecar)
            or frozen.llvm_transform is not transform
            or frozen.target_schedule is not target_schedule
            or frozen.connected_spatial is not spatial
            or frozen.spatial_host is not spatial_host
            or frozen.sibling_source is not sibling
            or frozen.sibling_host is not sibling_host
            or frozen.quantized_sum is not quantized
        ):
            raise ValueError("device offload selection or realized artifact pins changed during final admission")
        if selected is not None:
            selected.revalidate_context(routing=device)  # No selector/cost callback follows final admission.
    require_device_offload_artifacts(frozen)
    _require_device_llvm_transform(device, frozen)
    _require_device_target_schedule(device, frozen)
    # No provider/package callback follows this final original-owner check.
    from .connected_spatial_route import require_host, require_route

    spatial = require_route(device, frozen.connected_spatial if frozen is not None else None)
    if spatial is not None:
        require_host(frozen.spatial_host, spatial)
    from .sibling_source_route import require_host as require_sibling_host
    from .sibling_source_route import require_route as require_sibling

    sibling = require_sibling(device, frozen.sibling_source if frozen is not None else None)
    if sibling is not None:
        require_sibling_host(frozen.sibling_host, sibling)
    require_device_offload_artifacts(frozen)


def _require_device_llvm_transform(device, frozen):
    chain = getattr(device, "llvm_transform_chain", None)
    original = frozen.llvm_transform if type(frozen) is _DeviceOffloadInvocation else None
    if chain is None and original is None:
        return None
    from .device_object_optimization import recheck_transform_chain

    recheck_transform_chain(chain, original)
    return chain


def _freeze_target_schedule(selected, target):
    if selected is None:
        return None
    from .device_target_passes import freeze_schedule

    return freeze_schedule(selected, target)


def _require_device_target_schedule(device, frozen):
    selected = getattr(device, "target_schedule", None)
    original = frozen.target_schedule if type(frozen) is _DeviceOffloadInvocation else None
    if selected is None and original is None:
        return None
    if getattr(device, "target_configuration", None) is not None:
        raise ValueError("target schedule and configuration selections are mutually exclusive")
    from .device_target_passes import require_schedule

    return require_schedule(selected, getattr(device, "device", None), original)


def freeze_device_offload_selection(device):
    from .connected_spatial_route import freeze_route

    spatial = freeze_route(device)
    from .quantized_sum_route import freeze_route as freeze_quantized_sum
    quantized = freeze_quantized_sum(device)
    layout = _layout_policy(device)
    storage = _layout_policy(device, "intermediate_storage")
    from .sibling_source_route import freeze_route as freeze_sibling

    sibling = freeze_sibling(device)
    schedule = _freeze_target_schedule(getattr(device, "target_schedule", None), getattr(device, "device", None))
    if schedule is not None and getattr(device, "target_configuration", None) is not None:
        raise ValueError("target schedule and configuration selections are mutually exclusive")
    chain = getattr(device, "llvm_transform_chain", None)
    transform = None
    if chain is not None:
        from .device_object_optimization import freeze_transform_chain
        from .host_transform_chain import HostLLVMArtifact

        original, stages, sources = freeze_transform_chain(chain)
        transform = original, stages, (*sources, HostLLVMArtifact.capture(Path(__file__)))
    selected = getattr(device, "offload_selection", None)
    if selected is None:
        result = (
            None
            if all(value is None for value in (transform, schedule, spatial, sibling, quantized, layout, storage))
            else _DeviceOffloadInvocation(
                None, None, llvm_transform=transform, target_schedule=schedule, connected_spatial=spatial,
                sibling_source=sibling, quantized_sum=quantized, pre_group_layout=layout, intermediate_storage=storage
            )
        )
        _require_device_target_schedule(device, result)
        return result
    from .device_offload_selection import DeviceOffloadSelection

    if type(selected) is not DeviceOffloadSelection:
        raise ValueError("device offload requires its original typed source selection")
    if getattr(device, "post_offload_transform", None) is not None:
        raise ValueError("source-bound device selection does not support a later opaque host ABI rewrite")
    record = selected.record(routing=device)
    if getattr(device, "offload_selection", None) is not selected:
        raise ValueError("device offload selection changed during original verification")
    result = _DeviceOffloadInvocation(selected, record, llvm_transform=transform, target_schedule=schedule,
                                     pre_group_layout=layout, intermediate_storage=storage)
    _require_device_llvm_transform(device, result)
    _require_device_target_schedule(device, result)
    return result


def require_device_offload_selection(device, frozen):
    _require_layout_policy(device, frozen)
    _require_layout_policy(device, frozen, "intermediate_storage")
    from .quantized_sum_route import require_route as require_quantized_sum
    require_quantized_sum(device, frozen.quantized_sum if type(frozen) is _DeviceOffloadInvocation else None)
    from .connected_spatial_route import require_host, require_route
    from .sibling_source_route import require_host as require_sibling_host
    from .sibling_source_route import require_route as require_sibling

    sibling = require_sibling(device, frozen.sibling_source if type(frozen) is _DeviceOffloadInvocation else None)
    if sibling is not None and frozen.sibling_host is not None:
        require_sibling_host(frozen.sibling_host, sibling)
    spatial = require_route(device, frozen.connected_spatial if type(frozen) is _DeviceOffloadInvocation else None)
    if spatial is not None and frozen.spatial_host is not None:
        require_host(frozen.spatial_host, spatial)
    _require_device_target_schedule(device, frozen)
    _require_device_llvm_transform(device, frozen)
    selected = getattr(device, "offload_selection", None)
    if frozen is None:
        if selected is not None:
            raise ValueError("device offload selection appeared after preparation selection")
        return None
    if type(frozen) is not _DeviceOffloadInvocation:
        raise ValueError("device offload requires its original invocation")
    original, record = frozen.selection, frozen.record
    if selected is not original:
        raise ValueError("device offload original source selection changed")
    pins = (frozen.sources, frozen.sidecar)
    transform = frozen.llvm_transform
    schedule = frozen.target_schedule
    require_device_offload_artifacts(frozen)
    if selected is None:
        return None
    # record performs the full live transaction, including rechecks after
    # original callbacks. Reading its result adds no new callback boundary.
    if selected.record(routing=device) != record or getattr(device, "offload_selection", None) is not original:
        raise ValueError("device offload source selection changed during original verification")
    if (
        (frozen.sources, frozen.sidecar) != pins
        or frozen.llvm_transform is not transform
        or frozen.target_schedule is not schedule
    ):
        raise ValueError("device offload realized artifact pins changed during verification")
    require_device_offload_artifacts(frozen)
    _require_device_llvm_transform(device, frozen)
    _require_device_target_schedule(device, frozen)
    return selected


def prepare_device_route(device, prepared, work, *, early_complete_groups=False, selection=None):
    """Run the existing target preparation order without changing inert routes."""
    require_device_offload_selection(device, selection)
    if not device_route_requested(device):
        return prepared
    from .device_build import apply_post_offload_transform
    from .device_offload import BY_CONTRACTION, SIDECAR_NAME, rewrite_prepared_file

    if getattr(device, "prepared_transform", None) is not None:
        transformed = Path(device.prepared_transform(Path(prepared), Path(work) / "device_prepared"))
        require_device_offload_selection(device, selection)
        if not transformed.is_file():
            raise ValueError("device prepared transform returned no model file")
        prepared = transformed
    exact = getattr(device, "exact_selection", None)
    if exact is not None:
        exact.check_package(device.package_dir)
        exact.check_backend_contract()
    grain = str(getattr(device, "granularity", None) or BY_CONTRACTION)
    catalog_manifest = getattr(device, "catalog_manifest", None)
    catalog_object = getattr(device, "catalog_object", None)
    if getattr(device, "catalog_builder", None) is not None:
        catalog_manifest, catalog_object = device.catalog_builder(Path(prepared), Path(work) / "device_catalog")
    selected = require_device_offload_selection(device, selection)
    options = {} if selected is None else dict(offload_selection=selected, routing=device)
    spatial = getattr(device, "connected_spatial", None)
    if spatial is not None:
        snapshot = Path(work) / "connected_spatial" / "original-input.mlir"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(Path(prepared).read_bytes())
        retain_device_offload_source(selection, snapshot)
    sibling = getattr(device, "sibling_source", None)
    if sibling is not None and selection.sibling_host is None:
        raise ValueError("sibling route lost its original early host preparation")
    moved = (
        selection.sibling_host.rewrite if sibling is not None else None
        if early_complete_groups or sibling is not None
        else rewrite_prepared_file(
            prepared,
            work,
            device.device,
            select=device.select,
            exact_selection=exact,
            catalog_manifest=catalog_manifest,
            catalog_object=catalog_object,
            granularity=grain,
            capture=getattr(device, "capture", None),
            model=str(getattr(device, "model", "") or ""),
            **options,
        )
    )
    if spatial is not None:
        from .connected_spatial_route import prepare_host, restore_group_rewrite

        if early_complete_groups:
            from .device_offload import load_sidecar

            moved = restore_group_rewrite(load_sidecar(work))

        prepared, selection.spatial_host = prepare_host(
            device, spatial, prepared, moved, Path(work) / "connected_spatial"
        )
        moved = selection.spatial_host.rewrite
        moved.write_sidecar(work)
        retain_device_offload_source(selection, prepared)
    quantized = getattr(device, "quantized_sum", None)
    if quantized is not None:
        from .quantized_sum_route import prepare_host as prepare_quantized_sum
        if early_complete_groups:
            from .device_offload import load_sidecar
            from .connected_spatial_route import restore_group_rewrite
            moved = restore_group_rewrite(load_sidecar(work))
        prepared, moved = prepare_quantized_sum(device, quantized, prepared, moved, Path(work)/"quantized_sum")
        moved.write_sidecar(work)
        retain_device_offload_source(selection, prepared)
    if sibling is not None:
        moved.write_sidecar(work)
    if selected is not None or spatial is not None or sibling is not None or quantized is not None:
        from merlin.common.strict_json import loads

        path = (Path(work) / SIDECAR_NAME).absolute()
        payload = _artifact_payload(path)
        if moved is None or loads(payload.decode()) != moved.to_sidecar():
            raise ValueError("device offload lost its original realized sidecar")
        selection.sidecar = (path, hashlib.sha256(payload).hexdigest())
        retain_device_offload_source(selection, prepared)
    require_device_offload_selection(device, selection)
    prepared = apply_post_offload_transform(
        device, Path(prepared), Path(work) / SIDECAR_NAME, Path(work) / "device_host_abi"
    )
    require_device_offload_selection(device, selection)
    if moved is not None:
        print(
            f"[device] routed {moved.moved} {grain}(s) to {device.device} across "
            f"{len(moved.signatures)} signature(s)"
            + (f"; declined: {[why for _s, why in moved.skipped]}" if moved.skipped else "")
        )
    return prepared



def prepare_sibling_device_host(device, source, work, selection):
    """Prepare original sibling SSA before ordinary host normalization/offload."""
    selected = getattr(device, "sibling_source", None)
    if selected is None:
        return source
    from .sibling_source_route import prepare_host

    require_device_offload_selection(device, selection)
    if selection.sibling_host is not None:
        raise ValueError("sibling original host was prepared more than once")
    path, host = prepare_host(device, selected, source, Path(work) / "sibling_source")
    selection.sibling_host = host
    host.rewrite.write_sidecar(work)
    retain_device_offload_source(selection, Path(source).absolute())
    retain_device_offload_source(selection, path.absolute())
    for name in ("core.mlir", "interface.mlir", "receipt.json"):
        retain_device_offload_source(selection, (path.parent / name).absolute())
    require_device_offload_selection(device, selection)
    return path


def build_prepared_device_objects(
    device,
    work,
    *,
    offload_selection,
    device_admission_selection,
    target_configuration_selection,
    device_ir_budget,
    shim_cflags,
    descriptor_results=False,
):
    """Build the actual routed sidecar once at the caller's selected stage.

    This is the ordinary model build block, not a placement or ABI selection.
    ``shim_cflags`` retains the original host compiler's lazy flag query; no
    tool query runs when the source routed no device symbols. Original final
    linked-image admission remains with the caller.
    """
    from merlin.common.digest import sha256_file

    from .device_build import require_device_elf_admission, require_device_target_configuration
    from .device_offload import build_arguments as _device_build_arguments
    from .device_offload import load_sidecar as _load_device_sidecar

    if type(descriptor_results) is not bool:
        raise ValueError("descriptor results require an original boolean")
    _dev_build = None
    require_device_offload_selection(device, offload_selection)
    selected_chain = _require_device_llvm_transform(device, offload_selection)
    selected_schedule = _require_device_target_schedule(device, offload_selection)
    _dev_side = _load_device_sidecar(work)
    if getattr(device, "connected_spatial", None) is not None and (
        not _dev_side.get("signatures") or _dev_side.get("catalog_manifest") is not None
        or offload_selection.spatial_host is None
    ):
        raise ValueError("connected spatial route requires its actual original host and package-emitted objects")
    if selected_chain is not None and (
        not _dev_side.get("signatures") or _dev_side.get("catalog_manifest") is not None
    ):
        raise ValueError("selected device LLVM stages require routed package-emitted LLVM objects")
    if selected_schedule is not None and (
        not _dev_side.get("signatures") or _dev_side.get("catalog_manifest") is not None
    ):
        raise ValueError("selected target schedule requires routed package-emitted IR objects")
    _dev_args = _device_build_arguments(
        _dev_side, expected_granularity=device.granularity if device is not None else None
    )
    _dev_sigs = _dev_args["signatures"]
    device_artifacts = None
    _catalog_requested = device is not None and (
        getattr(device, "catalog_manifest", None) is not None or getattr(device, "catalog_builder", None) is not None
    )
    if _catalog_requested and not _dev_sigs:
        raise RuntimeError("requested external device catalog routed no model contractions")
    if _catalog_requested and (not _dev_side.get("catalog_manifest") or not _dev_side.get("catalog_object")):
        raise RuntimeError("device offload sidecar lost the external catalog artifacts")
    if not _catalog_requested and _dev_side.get("catalog_manifest"):
        raise RuntimeError("device offload sidecar names a catalog that this build did not request")
    if _dev_sigs:
        if device is None:
            raise RuntimeError(
                f"{len(_dev_sigs)} device signature(s) were offloaded but no `device=` routing is "
                "available to build them against; the image would not link"
            )
        require_device_elf_admission(device, device_admission_selection)
        require_device_offload_selection(device, offload_selection)
        target_configuration = require_device_target_configuration(device, target_configuration_selection)
        if _dev_side.get("device") != device.device:
            raise RuntimeError("device offload sidecar does not match selected device")
        exact = getattr(device, "exact_selection", None)
        if exact is not None:
            routed_ids = {row.get("operation_id") for row in _dev_side.get("routed") or ()}
            if (
                routed_ids != set(exact.by_operation_id)
                or _dev_side.get("package_sha256") != exact.package_sha256
                or _dev_side.get("transport") != exact.transport
                or _dev_side.get("abi_sha256") != exact.abi_sha256
                or _dev_side.get("certification_sha256") != list(exact.certification_sha256)
            ):
                raise RuntimeError("device offload sidecar lost exact operation or package identity")
        _dev_dts = _dev_args["dtypes"]
        if _dev_side.get("catalog_manifest") is not None:
            if target_configuration is not None:
                raise ValueError("target configuration cleanup cannot observe compiled catalog IR")
            from .device_catalog import build_catalog_objects

            _dev_build = build_catalog_objects(
                device.device,
                _dev_sigs,
                _dev_dts,
                manifest_path=_dev_side["catalog_manifest"],
                object_path=_dev_side["catalog_object"],
                source_sha256=_dev_side.get("model_sha256") or "",
                routed=_dev_side.get("routed") or (),
                workdir=work / "device",
                codegen_target="riscv",
                # Device shims are C and need the same selected bare-metal
                # headers as the trusted host helper runtime.
                cflags=shim_cflags(),
            )
        else:
            from .device_build import build_device_objects

            _dev_build = build_device_objects(
                device.device,
                _dev_sigs,
                _dev_dts,
                package_dir=device.package_dir,
                workdir=work / "device",
                operand_dtype=device.operand_dtype,
                accum_dtype=device.accum_dtype,
                codegen_target="riscv",
                numeric_policy=device.numeric_policy,
                entries=_dev_args["entries"],
                # Every routed signature is mandatory below. Once one fails, later objects
                # cannot make this whole-model image link or satisfy its selected route.
                stop_on_first_failure=True,
                # the SAME ISA the rest of the image is built for -- see device_build._flags
                # Device shims are C and need the same selected bare-metal
                # headers as the trusted host helper runtime.
                cflags=shim_cflags(),
                expected_interfaces=_dev_side.get("expected_interfaces") or None,
                package_sha256=_dev_side.get("package_sha256"),
                call_buffers=_dev_args.get("call_buffers"),  # every pointer of a group's logical interface
                device_ir_budget=device_ir_budget,
                **({"descriptor_results": True} if descriptor_results else {}),
                **(
                    {"llvm_transform_chain": _require_device_llvm_transform(device, offload_selection)}
                    if selected_chain is not None
                    else {}
                ),
                **({"target_configuration": target_configuration} if target_configuration is not None else {}),
                **(
                    {"target_schedule": _require_device_target_schedule(device, offload_selection)}
                    if selected_schedule is not None
                    else {}
                ),
                **(
                    {"sibling_source": (device.sibling_source, offload_selection.sibling_host)}
                    if getattr(device, "sibling_source", None) is not None else {}
                ),
                **(
                    {"connected_spatial": (device.connected_spatial, offload_selection.spatial_host)}
                    if getattr(device, "connected_spatial", None) is not None else {}
                ),
            )
        require_device_target_configuration(device, target_configuration_selection)
        _require_device_target_schedule(device, offload_selection)
        if selected_schedule is not None:
            from .device_target_passes import schedule_artifacts

            paths = schedule_artifacts(
                device,
                selected_schedule,
                {"kernels": {symbol: {"custody": _dev_build.custody.get(symbol, {})} for symbol in _dev_sigs}},
            )
            for path in paths:
                retain_device_offload_source(offload_selection, path)
        if exact is not None:
            exact.check_package(device.package_dir)
            exact.check_backend_contract()
        _require_device_llvm_transform(device, offload_selection)
        if not _dev_build.ok or set(_dev_build.kernels) != set(_dev_sigs):
            raise RuntimeError(
                f"device offload did not build every routed kernel for {device.device!r}: {_dev_build.skipped}"
            )
        if selected_chain is not None:
            from .device_object_optimization import recheck_transform_chain

            for symbol in _dev_sigs:
                emission = _dev_build.custody.get(symbol, {}).get("llvm_transform")
                if type(emission) is not dict:
                    raise ValueError("selected device LLVM stages were not consumed by the object builder")
                recheck_transform_chain(selected_chain, offload_selection.llvm_transform, emission)
        if _dev_side.get("catalog_manifest") is None:
            from merlin.llvmlower.device_object_optimization import artifacts as optimized_artifacts

            # Producer-owned symbol mapping, never reconstructed from signature order.
            device_artifacts = {
                "target": device.device,
                "kernels": {
                    symbol: {
                        "kernel_symbol": kernel,
                        "custody": _dev_build.custody[symbol],
                        "build_metrics": _dev_build.build_metrics.get(symbol),
                        "artifacts": {
                            role: {
                                "path": str(path.absolute()),
                                "sha256": sha256_file(path),
                                "size_bytes": path.stat().st_size,
                            }
                            for role, path in (
                                ("llvm_mlir", work / "device" / f"{symbol}.device.mlir"),
                                ("llvm_ir", work / "device" / f"{symbol}.ll"),
                                *optimized_artifacts(_dev_build.custody[symbol]).items(),
                                ("object", work / "device" / f"{symbol}.o"),
                            )
                        },
                    }
                    for symbol, kernel in _dev_build.kernels.items()
                },
            }
        print(
            f"[device] linked {len(_dev_build.kernels)} kernel(s) + shim for {device.device}"
            f" ({_dev_side.get('granularity') or 'contraction'} granularity;"
            f" built_from={sorted(set(_dev_build.built_from.values()))};"
            f" {_dev_build.object_dedup.get('unique_artifacts')} distinct compiled artifact(s))"
            + (f"; declined: {[w for _s, w in _dev_build.skipped]}" if _dev_build.skipped else "")
        )

    return _dev_side, _dev_args, _dev_build, device_artifacts


def apply_post_offload_transform(routing, prepared: Path, sidecar: Path, workdir: Path) -> Path:
    """Apply an explicit host ABI rewrite while preserving the routing sidecar.

    The callback owns any bridge implementation. It receives the exact routed IR,
    immutable offload sidecar, and private output directory; no target ABI is
    inferred here. Absent callbacks preserve existing behavior and bytes.
    """
    callback = getattr(routing, "post_offload_transform", None)
    if callback is None:
        return Path(prepared)
    import hashlib
    import json

    prepared, sidecar, workdir = Path(prepared), Path(sidecar), Path(workdir)
    original_sidecar = sidecar.read_bytes()
    original_source = prepared.read_bytes()
    workdir.mkdir(parents=True, exist_ok=True)
    selected = Path(callback(prepared, sidecar, workdir))
    if sidecar.read_bytes() != original_sidecar:
        raise ValueError("post-offload transform changed source routing identity")
    if not selected.is_file():
        raise ValueError("post-offload transform returned no model file")
    # Later preparation can overwrite the input path. Retain the actual
    # pre-callback source for checked call/ABI observations, not just its hash.
    snapshot = workdir / "post_offload_transform.source.mlir"
    if snapshot.is_symlink() or selected.resolve() == snapshot.resolve() or prepared.resolve() == snapshot.resolve():
        raise ValueError("post-offload transform used the reserved original-source snapshot path")
    snapshot.write_bytes(original_source)

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    (workdir / "post_offload_transform.json").write_text(
        json.dumps(
            {
                "source_path": str(prepared.resolve()),
                "source_sha256": digest(original_source),
                "source_snapshot": str(snapshot.resolve()),
                "routing_sidecar_sha256": digest(original_sidecar),
                "selected_path": str(selected.resolve()),
                "selected_sha256": digest(selected.read_bytes()),
            },
            indent=2,
        )
        + "\n"
    )
    return selected
