"""Original-owner sibling extraction through the ordinary logical device ABI.

Only the original caller selects this source/effect binding. Host readouts and
observers remain original SSA; the backend receives the closed raw integer
products. Source preparation establishes no target execution qualification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .borrowed_effects import BorrowedBufferEffects


def _sha(payload):
    return hashlib.sha256(payload.encode()).hexdigest()


def _text(module):
    from merlin.xdsl_dialects._common import text

    return text(module, generic=True)


@dataclass(frozen=True)
class SiblingSourceRoute:
    source_ir_sha256: str
    effects: BorrowedBufferEffects

    def record(self):
        if (
            type(self.source_ir_sha256) is not str
            or len(self.source_ir_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_ir_sha256)
            or type(self.effects) is not BorrowedBufferEffects
        ):
            raise ValueError("sibling route needs original source identity and typed borrowed effects")
        self.effects.__post_init__()
        return asdict(self)


@dataclass(frozen=True)
class RoutedSiblingHost:
    source: object
    module: object
    core: object
    writer: object
    device_symbol: str
    interface: str
    rewrite: object
    receipt: dict
    integrity_sha256: str


def freeze_route(device):
    selected = getattr(device, "sibling_source", None)
    if selected is None:
        return None
    if type(selected) is not SiblingSourceRoute:
        raise ValueError("sibling route requires its original typed selection")
    if getattr(device, "granularity", None) != "group" or any(
        getattr(device, key, None) is not None for key in (
            "connected_spatial", "exact_selection", "catalog_manifest", "catalog_object", "catalog_builder",
            "prepared_transform", "post_offload_transform", "offload_selection",
        )
    ):
        raise ValueError("sibling ownership requires a group route without opaque source or ABI rewrites")
    return selected, selected.record()


def require_route(device, frozen):
    selected = getattr(device, "sibling_source", None)
    if frozen is None:
        if selected is not None:
            raise ValueError("sibling source selection appeared after preparation")
        return None
    original, record = frozen
    if selected is not original or freeze_route(device)[1] != record:
        raise ValueError("sibling original selection changed")
    return selected


def _snapshot(host):
    return {
        "source": _sha(_text(host.source)), "host": _sha(_text(host.module)), "core": _sha(_text(host.core)),
        "writer": asdict(host.writer), "device_symbol": host.device_symbol, "interface": _sha(host.interface),
        "rewrite": host.rewrite.to_sidecar(), "receipt": host.receipt,
    }


def require_host(host, selected):
    if type(host) is not RoutedSiblingHost or (
        _sha(_text(host.source)) != selected.source_ir_sha256
        or host.writer.borrowed_effects != selected.effects
        or _sha(json.dumps(_snapshot(host), sort_keys=True, allow_nan=False)) != host.integrity_sha256
    ):
        raise ValueError("original sibling source, host, interface or effects changed")
    host.source.verify()
    host.module.verify()
    host.core.verify()
    preparation = host.receipt.get('weight_preparation')
    if preparation is not None:
        from merlin.common.digest import sha256_file

        from .weight_prepack import revalidate_build_preparation

        for pin in (preparation, preparation['capture_binding']):
            if sha256_file(pin['path']) != pin['sha256']:
                raise ValueError('sibling original weight preparation changed')
        revalidate_build_preparation(preparation['path'])


def _command_buffer(core, target):
    """Restate every independently checked original product, in source order."""
    from xdsl.dialects import func

    from merlin.compile.scheduling.projection_provenance import shared_projection_provenance
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups

    groups = find_shared_input_groups(core)
    if len(groups) != 1:
        raise ValueError("sibling owner requires one complete source-checked shared-input group")
    group = groups[0]
    group.verify_current(core)
    provenance = shared_projection_provenance(core, group)
    fn = next(op for op in core.body.block.ops if isinstance(op, func.FuncOp))
    operations = tuple(fn.body.block.ops)
    contractions = tuple(operations[i] for i in group.contraction_ordinals)
    if (
        set(group.source_op_indices) != set(range(len(operations) - 1))
        or not isinstance(operations[-1], func.ReturnOp)
        or tuple(operations[-1].arguments) != tuple(op.results[0] for op in contractions)
    ):
        raise ValueError("sibling core has extra operations or incomplete/reordered results")
    args = tuple(fn.body.block.args)
    lhs = contractions[0].operands[0]
    weights = tuple(op.operands[1] for op in contractions)
    if args != (lhs, *weights) or len(set(weights)) != len(weights):
        raise ValueError("sibling core must retain its exact shared input and distinct member weights")
    names = {arg: "input_0" if index == 0 else "weight_" + str(index - 1) for index, arg in enumerate(args)}
    tensors = {
        names[arg]: {"shape": list(arg.type.get_shape()), "dtype": str(arg.type.element_type),
                     "role": "input" if index == 0 else "weight"}
        for index, arg in enumerate(args)
    }
    commands = []
    for index, weight in enumerate(weights):
        resident, accumulator, output = (prefix + str(index) for prefix in ("resident_", "acc_", "out_"))
        commands.extend((
            {"opcode": "RES_PACK", "operands": {"src": names[weight], "dst": resident},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": names[lhs], "rhs": resident, "dst": accumulator}},
            {"opcode": "COMMIT", "operands": {"src": accumulator, "dst": output},
             "attributes": {"epilogue": [], "output_dtype": "i32"}},
            {"opcode": "EVICT", "operands": {"handle": resident}},
        ))
        if provenance:
            commands[-3]["attributes"] = provenance[index]
    return {"abi_version": "0.1", "target": target, "tensors": tensors, "commands": commands}, group


def prepare_host(device, selected, source, directory):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    from .device_offload import BY_GROUP, DeviceRewrite, Routed
    from .device_shim import kernel_abi_for, logical_buffers
    from .source_region_host import prepare_sibling_source_host

    selected.record()
    actual = parse_mlir_file(source)
    original, weight_preparation = actual, None
    if _sha(_text(actual)) != selected.source_ir_sha256:
        from .prepacked_device_capture import original_sibling_source

        if not (directory.parent / 'bundle_preparation.json').is_file():
            raise ValueError('sibling selection differs from original source SSA; no checked layout preparation')
        original, weight_preparation = original_sibling_source(
            device, selected, actual_source=source, work=directory.parent,
        )
    prepared = prepare_sibling_source_host(actual, effects=selected.effects)
    cb, group = _command_buffer(prepared.core, device.device)
    abi = kernel_abi_for(device.device)
    if abi is None or abi.version != 2:
        raise ValueError("sibling owner requires the selected ordinary logical kernel ABI")
    if getattr(device, "require_complete_groups", False):
        from .device_offload import rewrite_groups_to_device

        census = rewrite_groups_to_device(
            actual.clone(), device.device, select=device.select, capture=device.capture, model=device.model,
            require_complete_groups=True,
        )
        if census.moved != len(group.widths):
            raise ValueError("sibling source does not cover every independently required accelerator group")
    interface = emit_interface_mlir(cb)
    buffers, reason = logical_buffers(interface)
    if buffers is None:
        raise ValueError("original sibling interface has no complete logical ABI: " + reason)
    from .exact_offload import _package_sha256

    symbol = prepared.writer.symbol + "_device"
    entry = {"schema": "original_sibling_source_entry_v1", "name": symbol,
             "source_ir_sha256": selected.source_ir_sha256, "core_ir_sha256": _sha(_text(prepared.core))}
    rewrite = DeviceRewrite(
        device.device,
        routed=(Routed(symbol, (group.m, sum(group.widths)), (group.k,), ("i8", "i8", "i32"), group=0),),
        signatures={symbol: (group.m, sum(group.widths), group.k)}, granularity=BY_GROUP,
        entries={symbol: entry}, programs={symbol: cb}, model_sha256=selected.source_ir_sha256,
        package_sha256=_package_sha256(Path(device.package_dir)),
        arg_access={symbol: ("read",) * (len(group.widths) + 1) + ("write",) * len(group.widths)},
        expected_interfaces={symbol: {"mlir": interface, "sha256": _sha(interface)}},
        call_buffers={symbol: [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers]},
    )
    host = RoutedSiblingHost(
        original, prepared.module, prepared.core, prepared.writer, symbol, interface, rewrite,
        {**prepared.receipt, "source_ir_sha256": selected.source_ir_sha256,
         "prepared_source_ir_sha256": _sha(_text(actual)),
         "weight_preparation": weight_preparation,
         "interface_sha256": _sha(interface), "source_group": group.record(),
         "original_target_and_linked_image_admission_required": True}, "",
    )
    host = replace(host, integrity_sha256=_sha(json.dumps(_snapshot(host), sort_keys=True, allow_nan=False)))
    require_host(host, selected)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "host.mlir"
    path.write_text(_text(host.module) + "\n")
    (directory / "core.mlir").write_text(_text(host.core) + "\n")
    (directory / "interface.mlir").write_text(interface)
    (directory / "receipt.json").write_text(json.dumps(host.receipt, indent=2, sort_keys=True) + "\n")
    return path, host


def borrowed_bridge_source(symbol, borrowed_symbol, call, result_arguments):
    """Emit stock descriptor transport, reproducible from the source-bound ABI."""
    import re

    if any(type(value) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None
           for value in (symbol, borrowed_symbol)):
        raise ValueError("sibling descriptor bridge needs closed C symbols")
    outputs = [index for index, row in enumerate(call) if row["role"].startswith("out_")]
    if len(outputs) < 2 or outputs != list(result_arguments):
        raise ValueError("sibling descriptor bridge lost original ordered complete outputs")
    descriptors = ", ".join(f"const merlin_lmemref_{len(row['shape'])}d *d{i}" for i, row in enumerate(call))
    return (
        f"void _mlir_ciface_{borrowed_symbol}({descriptors}) {{\n"
        f"  {symbol}_results result;\n"
        f"  _mlir_ciface_{symbol}(&result, " + ", ".join(f"d{i}" for i in range(len(call))) + ");\n}\n"
    )


def with_borrowed_bridge(unit, selected, host):
    """Discard returned identity descriptors without changing any buffer or call."""
    require_host(host, selected)
    if host.device_symbol not in unit.symbols:
        raise ValueError("sibling descriptor bridge lost its ordinary logical shim")
    bridge = borrowed_bridge_source(
        host.device_symbol, host.writer.borrowed_symbol,
        host.rewrite.call_buffers[host.device_symbol], host.writer.result_arguments,
    )
    return replace(unit, text=unit.text + "\n" + bridge)


def require_emission(artifact, selected, host, *, entry_symbol):
    """Read the real LLVM entry in either supported textual spelling."""
    from xdsl.dialects import llvm
    from xdsl.parser import Parser

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    require_host(host, selected)
    module = Parser(make_llvm_context(), artifact).parse_module()
    module.verify()
    entries = [op for op in module.body.block.ops
               if isinstance(op, llvm.FuncOp) and op.sym_name.data == entry_symbol]
    if len(entries) != 1:
        raise ValueError("sibling artifact has no unique contract-named LLVM kernel entry")
    entry = entries[0]
    if (
        len(entry.function_type.inputs) != len(host.rewrite.call_buffers[host.device_symbol])
        or any(not isinstance(t, llvm.LLVMPointerType) for t in entry.function_type.inputs)
        or not isinstance(entry.function_type.output, llvm.LLVMVoidType)
        or not entry.body.blocks
    ):
        raise ValueError("sibling artifact kernel entry differs from its original complete pointer ABI")
