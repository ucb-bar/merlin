"""Preserve original device preparation and a source-bound placement selection.

The original route still owns eligibility, package execution and qualification.
This module only carries a fixed selection through the existing preparation
callbacks; it supplies no numerical, ABI, completion or timing authority.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _DeviceOffloadInvocation:
    selection: object
    record: dict
    sidecar: tuple[Path, str] | None = None
    sources: tuple[tuple[Path, str], ...] = ()


def device_route_requested(device):
    return device is not None and any(
        getattr(device, name, None) is not None
        for name in ("select", "exact_selection", "catalog_manifest", "catalog_builder", "offload_selection")
    )


def require_device_offload_artifacts(invocation):
    """Reopen realized file identities without invoking any caller callbacks."""
    if invocation is None:
        return
    pins = (*invocation.sources, *((invocation.sidecar,) if invocation.sidecar is not None else ()))
    for path, digest in pins:
        if hashlib.sha256(_artifact_payload(path)).hexdigest() != digest:
            raise ValueError("device offload original realized source or sidecar changed")


def _artifact_payload(path):
    if path.resolve() != path or path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError("device offload artifact needs its original direct regular file")
    maximum = 16 << 20
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
    pin = (path, hashlib.sha256(_artifact_payload(path)).hexdigest())
    if pin not in frozen.sources:
        frozen.sources = (*frozen.sources, pin)
    if compilation is not None:
        for index, (source, _) in enumerate(frozen.sources):
            compilation.bind_preparation(f"device_offload_source_{index}", source)
        if frozen.sidecar is not None:
            compilation.bind_preparation("device_offload_sidecar", frozen.sidecar[0])


def require_supported_device_preparation(frozen, transform):
    if frozen is not None and transform is not None:
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

    original = None if frozen is None else (frozen.selection, frozen.sources, frozen.sidecar)
    custody = (
        None
        if frozen is None
        else freeze_device_elf_custody(device, admission, elf=elf, linked_sha256=linked_sha256, result=result)
    )
    revalidate_device_target_artifacts(device, target_configuration, artifacts)
    require_device_offload_selection(device, frozen)
    if custody is not None:
        revalidate_device_elf_custody(device, admission, custody)
    if frozen is not None:
        selected, sources, sidecar = original
        if (
            frozen.selection is not selected
            or getattr(device, "offload_selection", None) is not selected
            or (frozen.sources, frozen.sidecar) != (sources, sidecar)
        ):
            raise ValueError("device offload selection or realized artifact pins changed during final admission")
        selected.revalidate_context(routing=device)  # No selector/cost callback follows final admission.
    require_device_offload_artifacts(frozen)


def freeze_device_offload_selection(device):
    selected = getattr(device, "offload_selection", None)
    if selected is None:
        return None
    from .device_offload_selection import DeviceOffloadSelection

    if type(selected) is not DeviceOffloadSelection:
        raise ValueError("device offload requires its original typed source selection")
    if getattr(device, "post_offload_transform", None) is not None:
        raise ValueError("source-bound device selection does not support a later opaque host ABI rewrite")
    record = selected.record(routing=device)
    if getattr(device, "offload_selection", None) is not selected:
        raise ValueError("device offload selection changed during original verification")
    return _DeviceOffloadInvocation(selected, record)


def require_device_offload_selection(device, frozen):
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
    require_device_offload_artifacts(frozen)
    # record performs the full live transaction, including rechecks after
    # original callbacks. Reading its result adds no new callback boundary.
    if selected.record(routing=device) != record or getattr(device, "offload_selection", None) is not original:
        raise ValueError("device offload source selection changed during original verification")
    if (frozen.sources, frozen.sidecar) != pins:
        raise ValueError("device offload realized artifact pins changed during verification")
    require_device_offload_artifacts(frozen)
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
    moved = (
        None
        if early_complete_groups
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
    if selected is not None:
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
