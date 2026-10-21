"""Keep ordinary complete-group manifests aligned with checked weight preparation."""

import json
from dataclasses import replace
from pathlib import Path

from merlin.common.digest import sha256_file

from .device_build import DeviceRouting
from .weight_prepack import PrepackRefused, revalidate_build_preparation


def rebind(device, *, source: Path, prepared: Path, work: Path, compilation=None):
    """Rebind only the ordinary all-group selector, without moving source certificates.

    Exact selections, catalogs, provider transforms and other source-bound owners
    need their own checked composition. They refuse instead of borrowing the
    prepared bundle's manifest or silently retaining the original weights.
    """
    source, prepared = Path(source).resolve(), Path(prepared).resolve()
    if source == prepared or device is None:
        return device
    from merlin.compile.complete_group_routing import select_complete_group

    if (
        type(device) is not DeviceRouting
        or device.select is not select_complete_group
        or not device.require_complete_groups
        or device.granularity != "group"
        or any(
            getattr(device, key) is not None
            for key in (
                "exact_selection",
                "catalog_manifest",
                "catalog_object",
                "catalog_builder",
                "prepared_transform",
                "post_offload_transform",
                "offload_selection",
                "descriptor_effects",
                "llvm_transform_chain",
                "target_schedule",
                "target_configuration",
            )
        )
        or device.descriptor_results
    ):
        raise PrepackRefused("weight preparation has no checked composition with this source-bound device route")
    capture = None if device.capture is None else Path(device.capture).resolve()
    if capture not in {source, source / "model.mlir"}:
        raise PrepackRefused("complete-group capture differs from the original weight-preparation source")
    receipt = Path(work) / "bundle_preparation.json"
    revalidate_build_preparation(receipt)
    record = json.loads(receipt.read_bytes())
    if record["source_bundle"] != str(source) or record["prepared_bundle"] != str(prepared):
        raise PrepackRefused("device manifest preparation differs from the checked bundle selection")
    selected = prepared / "model.mlir" if capture.name == "model.mlir" else prepared
    bound = Path(work) / "device_capture_preparation.json"
    bound.write_text(
        json.dumps(
            {
                "schema": "merlin.prepacked_device_capture.v1",
                "source_capture": str(capture),
                "prepared_capture": str(selected),
                "preparation_sha256": sha256_file(receipt),
                "scope": "ordinary complete-group selector and checked weight manifest; no source-certificate transfer",
            },
            indent=2,
        )
        + "\n"
    )
    if compilation is not None:
        compilation.bind_preparation("device_capture_preparation", bound)
    return replace(device, capture=selected)
