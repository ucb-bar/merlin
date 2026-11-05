"""Original-owner component routing; unmatched groups use ordinary lowering."""

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class RoutedSiblingComponents:
    source: object
    module: object
    components: tuple
    rewrite: object
    receipt: dict
    integrity_sha256: str


def _snapshot(host, *, source_sha256=None, host_sha256=None):
    from .sibling_source_route import _sha, _text
    from .sibling_source_route import _snapshot as component_snapshot

    source_sha256 = _sha(_text(host.source)) if source_sha256 is None else source_sha256
    host_sha256 = _sha(_text(host.module)) if host_sha256 is None else host_sha256
    if any(c.source is not host.source or c.module is not host.module for c in host.components):
        raise ValueError("sibling components must retain their common source and host")
    return {
        "source": source_sha256,
        "host": host_sha256,
        "components": [component_snapshot(c, source_sha256=source_sha256, host_sha256=host_sha256)
                       for c in host.components],
        "rewrite": host.rewrite.to_sidecar(),
        "receipt": host.receipt,
    }


def require_components(host, selected):
    from .sibling_source_route import _require_host_integrity, _sha, _text

    source_sha256, host_sha256 = _sha(_text(host.source)), _sha(_text(host.module))
    if (
        not selected.partial
        or not host.components
        or source_sha256 != selected.source_ir_sha256
        or _sha(json.dumps(_snapshot(host, source_sha256=source_sha256, host_sha256=host_sha256),
                           sort_keys=True, allow_nan=False)) != host.integrity_sha256
    ):
        raise ValueError("original sibling component source, host or custody changed")
    host.source.verify()
    host.module.verify()
    checked_preparations = set()
    for component in host.components:
        _require_host_integrity(component, selected, source_sha256=source_sha256,
                                host_sha256=host_sha256, verify_graphs=False,
                                checked_preparations=checked_preparations)


def prepare_components(device, selected, source, directory):
    from merlin.frontends.linalg_mlir import parse_mlir_file

    from .sibling_source_route import _sha, _text

    selected.record()
    actual = parse_mlir_file(source)
    original, preparation = actual, None
    if _sha(_text(actual)) != selected.source_ir_sha256:
        from .prepacked_device_capture import original_sibling_source

        original, preparation = original_sibling_source(device, selected, actual_source=source, work=directory.parent)
    host = build_components(device, selected, actual, original=original, preparation=preparation)
    return write_components(host, directory)


def build_components(device, selected, actual, *, original, preparation=None):
    """Compose on checked source while retaining its original-owner identity.

    The caller must establish any preceding source transformation separately;
    this builder supplies no equivalence or ancestry permission for it.
    """
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    from .device_offload import BY_GROUP, DeviceRewrite, Routed, rewrite_groups_to_device
    from .device_shim import kernel_abi_for, logical_buffers
    from .exact_offload import _package_sha256
    from .sibling_components import prepare_sibling_components
    from .sibling_source_route import RoutedSiblingHost, _command_buffer, _sha, _text
    from .sibling_source_route import _snapshot as single_snapshot

    selected.record()
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    prepared = prepare_sibling_components(actual, effects=selected.effects,
        output_path=OF.declared_path(device.device), values=OF.Values(device.capture))
    if not prepared.components:
        raise ValueError("source has no independently checked shared-input component")
    abi = kernel_abi_for(device.device)
    if abi is None or abi.version != 2:
        raise ValueError("sibling components require the selected ordinary logical ABI")
    # Unknown/unsupported clusters and every unmatched original group remain
    # actual source here. The unchanged normal group owner performs their work.
    module = prepared.module
    ordinary = rewrite_groups_to_device(
        module,
        device.device,
        select=device.select,
        capture=device.capture,
        model=device.model,
        require_complete_groups=getattr(device, "require_complete_groups", False),
    )
    # Freeze unmatched interfaces through the same original group constructor
    # ordinary device_build uses; a partial exact-interface roster is invalid.
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.targetgen import corpus_spec as CS

    binding = _mesh_tile_binding(
        device.device, device.operand_dtype, device.accum_dtype, numeric_policy=device.numeric_policy
    )
    interfaces = dict(ordinary.expected_interfaces)
    for symbol, entry in ordinary.entries.items():
        _, interface = CS.build(entry, binding)
        interfaces[symbol] = {"mlir": interface, "sha256": _sha(interface)}
    ordinary = replace(ordinary, expected_interfaces=interfaces)
    components = []
    offset = max((r.group for r in ordinary.routed if r.group is not None), default=-1) + 1
    original_sha256, module_sha256 = _sha(_text(original)), _sha(_text(module))
    actual_sha256 = original_sha256 if actual is original else _sha(_text(actual))
    for index, item in enumerate(prepared.components):
        cb, group = _command_buffer(item.core, device.device, readouts=item.receipt.get("readouts"))
        interface = emit_interface_mlir(cb)
        buffers, reason = logical_buffers(interface)
        if buffers is None:
            raise ValueError("component lost its complete logical ABI: " + reason)
        from .sibling_readout_commands import require_writer_interface

        require_writer_interface(module, item.writer, buffers)
        symbol = item.writer.symbol + "_device"
        entry = {
            "schema": "original_sibling_source_entry_v1",
            "name": symbol,
            "source_ir_sha256": selected.source_ir_sha256,
            "core_ir_sha256": _sha(_text(item.core)),
        }
        rewrite = DeviceRewrite(
            device.device,
            routed=(
                Routed(symbol, (group.m, sum(group.widths)), (group.k,), ("i8", "i8", "i32"), group=offset + index),
            ),
            signatures={symbol: (group.m, sum(group.widths), group.k)},
            granularity=BY_GROUP,
            entries={symbol: entry},
            programs={symbol: cb},
            model_sha256=selected.source_ir_sha256,
            package_sha256=_package_sha256(Path(device.package_dir)),
            arg_access={symbol: tuple("write" if b.role.startswith("out_") else "read" for b in buffers)},
            expected_interfaces={symbol: {"mlir": interface, "sha256": _sha(interface)}},
            call_buffers={symbol: [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers]},
        )
        host = RoutedSiblingHost(
            original,
            module,
            item.core,
            item.writer,
            symbol,
            interface,
            rewrite,
            {
                **item.receipt,
                "source_ir_sha256": selected.source_ir_sha256,
                "prepared_source_ir_sha256": actual_sha256,
                "weight_preparation": preparation,
                "source_group": group.record(),
            },
            "",
        )
        host = replace(host, integrity_sha256=_sha(json.dumps(
            single_snapshot(host, source_sha256=original_sha256, host_sha256=module_sha256),
            sort_keys=True, allow_nan=False)))
        components.append(host)
    maps = ("signatures", "entries", "programs", "arg_access", "expected_interfaces", "call_buffers")
    merged = {name: dict(getattr(ordinary, name)) for name in maps}
    for component in components:
        for name in maps:
            incoming = getattr(component.rewrite, name)
            if set(incoming).intersection(merged[name]):
                raise ValueError("source component collided with an ordinary routed symbol")
            merged[name].update(incoming)
    rewrite = replace(
        ordinary,
        **merged,
        model_sha256=selected.source_ir_sha256,
        package_sha256=_package_sha256(Path(device.package_dir)),
        routed=(*ordinary.routed, *(r for c in components for r in c.rewrite.routed)),
    )
    host = RoutedSiblingComponents(original, module, tuple(components), rewrite, prepared.receipt, "")
    host = replace(host, integrity_sha256=_sha(json.dumps(
        _snapshot(host, source_sha256=original_sha256, host_sha256=module_sha256),
        sort_keys=True, allow_nan=False)))
    require_components(host, selected)
    return host


def write_components(host, directory):
    from xdsl.dialects.builtin import ModuleOp

    from .sibling_source_route import _text

    components, rewrite = host.components, host.rewrite
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "host.mlir"
    path.write_text(_text(host.module) + "\n")
    (directory / "core.mlir").write_text(
        _text(ModuleOp([op.clone() for c in components for op in c.core.body.block.ops])) + "\n"
    )
    (directory / "interface.mlir").write_text(components[0].interface)
    (directory / "receipt.json").write_text(
        json.dumps({**host.receipt, "interfaces": rewrite.expected_interfaces}, indent=2) + "\n"
    )
    return path, host
