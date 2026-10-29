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
    partial: bool = False

    def record(self):
        if (
            type(self.source_ir_sha256) is not str
            or len(self.source_ir_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_ir_sha256)
            or type(self.partial) is not bool
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
    from .head_source_route import HeadBatchSourceRoute

    if type(selected) not in (SiblingSourceRoute, HeadBatchSourceRoute):
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


def _snapshot(host, *, source_sha256=None, host_sha256=None):
    # A component owner can serialize its shared graph once per validation.
    # These are invocation-local values, never cached across source mutations.
    return {
        "source": _sha(_text(host.source)) if source_sha256 is None else source_sha256,
        "host": _sha(_text(host.module)) if host_sha256 is None else host_sha256,
        "core": _sha(_text(host.core)),
        "writer": asdict(host.writer), "device_symbol": host.device_symbol, "interface": _sha(host.interface),
        "rewrite": host.rewrite.to_sidecar(), "receipt": host.receipt,
    }


def require_host(host, selected):
    from .head_source_route import HeadBatchSourceRoute, RoutedHeads, component_selection
    from .head_source_route import require_host as require_heads

    if type(selected) is HeadBatchSourceRoute:
        if type(host) is RoutedHeads:
            return require_heads(host, selected)
        return require_host(host, component_selection(selected))
    from .sibling_component_route import RoutedSiblingComponents, require_components

    if type(host) is RoutedSiblingComponents:
        return require_components(host, selected)
    if type(host) is not RoutedSiblingHost:
        raise ValueError("original sibling source, host, interface or effects changed")
    _require_host_integrity(host, selected, source_sha256=_sha(_text(host.source)),
                            host_sha256=_sha(_text(host.module)))


def _require_host_integrity(host, selected, *, source_sha256, host_sha256, verify_graphs=True,
                            checked_preparations=None):
    """Check one component against freshly read shared source/host identities."""
    if type(host) is not RoutedSiblingHost or (
        source_sha256 != selected.source_ir_sha256
        or host.writer.borrowed_effects != selected.effects
        or _sha(json.dumps(_snapshot(host, source_sha256=source_sha256, host_sha256=host_sha256),
                           sort_keys=True, allow_nan=False)) != host.integrity_sha256
    ):
        raise ValueError("original sibling source, host, interface or effects changed")
    if verify_graphs:
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
        # Disjoint components often share one exact model/weight preparation.
        # Their receipt pins are still read above; the expensive source/weight
        # proof need run only once inside this callback-free validation batch.
        # A new require_components invocation starts with a fresh set.
        key = json.dumps(preparation, sort_keys=True, allow_nan=False)
        if checked_preparations is None or key not in checked_preparations:
            revalidate_build_preparation(preparation['path'])
            if checked_preparations is not None:
                checked_preparations.add(key)


def _command_buffer(core, target, *, readouts=None):
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
    # A literal zero can be shared by several original splats. The group
    # recognizer omits that shared producer from its per-seed interval, while
    # the exact zero-seeded reader already checked every root's initializer.
    # Account for only those actual literal dependencies, never arbitrary ops.
    zero_literals = {operations.index(op.operands[-1].owner.operands[0].owner) for op in contractions}
    if (
        set(group.source_op_indices) | zero_literals != set(range(len(operations) - 1))
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
    from .sibling_readout_commands import source_readout_commands

    member_readouts = source_readout_commands(group, readouts, tensors)
    commands = []
    for index, weight in enumerate(weights):
        resident, accumulator, output = (prefix + str(index) for prefix in ("resident_", "acc_", "out_"))
        commands.extend((
            {"opcode": "RES_PACK", "operands": {"src": names[weight], "dst": resident},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": names[lhs], "rhs": resident, "dst": accumulator}},
            {"opcode": "COMMIT", "operands": {"src": accumulator, "dst": output},
             "attributes": member_readouts[index]},
            {"opcode": "EVICT", "operands": {"handle": resident}},
        ))
        if provenance:
            commands[-3]["attributes"] = provenance[index]
    return {"abi_version": "0.1", "target": target, "tensors": tensors, "commands": commands}, group


def prepare_host(device, selected, source, directory):
    from .head_source_route import HeadBatchSourceRoute
    from .head_source_route import prepare_host as prepare_heads

    if type(selected) is HeadBatchSourceRoute:
        return prepare_heads(device, selected, source, directory)
    if selected.partial:
        from .sibling_component_route import prepare_components

        return prepare_components(device, selected, source, directory)
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
    if any(type(value) is not str or not value.isascii() or not value.isidentifier()
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
    return replace(unit, text=unit.text + "\n" + _borrowed_bridge(unit, host))


def _borrowed_bridge(unit, host):
    if host.device_symbol not in unit.symbols:
        raise ValueError("sibling descriptor bridge lost its ordinary logical shim")
    return borrowed_bridge_source(
        host.device_symbol, host.writer.borrowed_symbol,
        host.rewrite.call_buffers[host.device_symbol], host.writer.result_arguments,
    )


def with_borrowed_bridges(unit, selected, host, *, symbols=None):
    """Assemble original sibling bridges under one fresh complete-owner check.

    The invocation is read-only and contains no external callbacks. It validates
    the complete routed owner, then derives every bridge from its original
    component roster. ``symbols`` may select an immutable subset of successfully
    built original components; it cannot add or reorder that roster. No proof
    or cache escapes this function, and the single-bridge API remains fresh on
    every call. This does not authorize reuse across package emission callbacks.
    """
    from .device_shim import ShimUnit

    if (type(unit) is not ShimUnit or type(unit.text) is not str
        or type(unit.symbols) is not tuple
        or any(type(symbol) is not str for symbol in unit.symbols)
        or symbols is not None and (type(symbols) is not tuple
            or any(type(symbol) is not str for symbol in symbols)
            or len(set(symbols)) != len(symbols))):
        raise ValueError("sibling bridge batch needs a logical shim and distinct original symbols")
    require_host(host, selected)
    components = component_hosts(host)
    original_symbols = tuple(component.device_symbol for component in components)
    if (len(set(original_symbols)) != len(original_symbols)
        or symbols is not None and not set(symbols) <= set(original_symbols)):
        raise ValueError("sibling bridge batch changed its original component roster")
    wanted = set(original_symbols if symbols is None else symbols)
    bridges = tuple(_borrowed_bridge(unit, component) for component in components
                    if component.device_symbol in wanted)
    return replace(unit, text=unit.text + "".join("\n" + bridge for bridge in bridges))


def require_emission(artifact, selected, host, *, entry_symbol):
    """Read the real LLVM entry in either supported textual spelling."""
    require_host(host, selected)
    require_logical_emission(artifact, entry_symbol=entry_symbol,
                             arity=len(host.rewrite.call_buffers[host.device_symbol]))


def require_logical_emission(artifact, *, entry_symbol, arity):
    """Read either actual LLVM spelling without a textual entry-name guess."""
    from xdsl.dialects import llvm
    from xdsl.parser import Parser

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    module = Parser(make_llvm_context(), artifact).parse_module()
    module.verify()
    entries = [op for op in module.body.block.ops
               if isinstance(op, llvm.FuncOp) and op.sym_name.data == entry_symbol]
    if len(entries) != 1:
        raise ValueError("sibling artifact has no unique contract-named LLVM kernel entry")
    entry = entries[0]
    if (
        len(entry.function_type.inputs) != arity
        or any(not isinstance(t, llvm.LLVMPointerType) for t in entry.function_type.inputs)
        or not isinstance(entry.function_type.output, llvm.LLVMVoidType)
        or not entry.body.blocks
    ):
        raise ValueError("sibling artifact kernel entry differs from its original complete pointer ABI")


def component_hosts(host):
    """The exact independently checked logical writers of either route."""
    from .head_source_route import RoutedHeads
    from .sibling_component_route import RoutedSiblingComponents

    if type(host) is RoutedHeads:
        # The ordinary rank-three descriptor shim already owns the head loop.
        # It needs no sibling-specific borrowed multioutput bridge.
        return () if host.component_host is None else host.component_host.components

    return host.components if type(host) is RoutedSiblingComponents else (host,)
