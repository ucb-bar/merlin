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
                "connected_spatial",
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
    sibling = getattr(device, "sibling_source", None)
    if sibling is not None:
        import hashlib

        from merlin.common import mlir_query as mq
        from merlin.xdsl_dialects._common import text

        from .sibling_source_route import SiblingSourceRoute

        if type(sibling) is not SiblingSourceRoute:
            raise PrepackRefused("weight preparation needs the original typed sibling selection")
        sibling.record()
        original = mq.parse((source / "model.mlir").read_text())
        if hashlib.sha256(text(original, generic=True).encode()).hexdigest() != sibling.source_ir_sha256:
            raise PrepackRefused("weight preparation source differs from the original sibling selection")
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


def original_sibling_source(device, selected, *, actual_source, work):
    """Reopen an exact checked layout derivation without moving source authority.

    The sibling selection continues to bind the original SSA. Its prepared host
    and runtime consume the same independently verified layout/manifest rewrite.
    This function neither accepts candidate preparation receipts nor changes
    effects, numerical policy or source eligibility.
    """
    import hashlib

    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects._common import text

    receipt = Path(work) / "bundle_preparation.json"
    capture_receipt = Path(work) / "device_capture_preparation.json"
    revalidate_build_preparation(receipt)
    record = json.loads(receipt.read_bytes())
    binding = json.loads(capture_receipt.read_bytes())
    source = Path(record['source_bundle'])
    prepared = Path(record['prepared_bundle'])
    actual_source = Path(actual_source).resolve()
    capture = Path(device.capture).resolve()
    if (actual_source != prepared / 'model.mlir'
            or capture not in (prepared, actual_source)
            or binding.get('preparation_sha256') != sha256_file(receipt)
            or Path(binding.get('prepared_capture', '')).resolve() != capture
            or Path(binding.get('source_capture', '')).resolve() not in (source, source / 'model.mlir')):
        raise PrepackRefused('sibling source does not consume its original checked weight preparation')
    original = mq.parse((source / 'model.mlir').read_text())
    if hashlib.sha256(text(original, generic=True).encode()).hexdigest() != selected.source_ir_sha256:
        raise PrepackRefused('prepared sibling source lost its original source selection')
    return original, {
        'path': str(receipt.resolve()), 'sha256': sha256_file(receipt),
        'capture_binding': {'path': str(capture_receipt.resolve()), 'sha256': sha256_file(capture_receipt)},
        'scope': 'checked bit-preserving immutable weight layout; original source and effects retained',
    }
