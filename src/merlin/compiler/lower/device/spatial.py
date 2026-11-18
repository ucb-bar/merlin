"""Explicit closed spatial grouping before ordinary per-interface extraction.

Match current tensor SSA and the ordinary rewrite's source-bound statements.
Neither dimensions nor symbol spelling select a model. Both original numerical
programs travel unchanged; this pass supplies no device/effect qualification.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from merlin.compiler.lower.host.storage import prepare_spatial_scratch_abi, spatial_logical_buffers
from merlin.compiler.lower.host.writers import FreshTensorWriterContract

# ---- connected_spatial -------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectedSpatialSelection:
    module: object
    rewrite: object
    receipt: dict


@dataclass(frozen=True)
class ConnectedSpatialRegion:
    """An exact current-SSA slice, with its original values for input binding."""

    module: object
    rewrite: object
    inputs: tuple
    outputs: tuple
    receipt: dict


def extract_connected_spatial_region(module, rewrite, source_pair, *, symbol="forward"):
    """Extract an already matched adjacent call pair, never reconstruct its SSA.

    The complete source matcher owns closure and the two numerical statements.
    This only clones the actual calls and their fresh destinations, parameterizes
    their free SSA values and returns every externally observed member result.
    Those original input/output values remain available to a source-owned runner;
    no input bytes, golden values, borrowing effects or target grants are inferred.
    The caller's complete model and every unrelated observer remain unchanged.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ModuleOp, UnitAttr
    from xdsl.ir import Block, Region

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.ir.boundary import free_values
    from merlin.compiler.lower.host.storage import _symbol

    _symbol(symbol)
    if type(source_pair) is not tuple or len(source_pair) != 2 or any(type(s) is not str for s in source_pair):
        raise ValueError("region extraction requires an explicit current source callee pair")
    before = text(module, generic=True)
    grouped = group_connected_spatial(module, rewrite)
    selected = [row for row in grouped.receipt["groups"] if row["components"] == list(source_pair)]
    if len(selected) != 1:
        raise ValueError("region extraction requires a unique closed current-source spatial pair")
    calls = {
        op.callee.root_reference.data: op
        for op in module.walk()
        if isinstance(op, func.CallOp) and op.callee.root_reference.data in source_pair
    }
    producer, pool = (calls[name] for name in source_pair)
    block = producer.parent
    start = producer.arguments[-1].owner
    if not isinstance(start, tensor.EmptyOp) or start.parent is not block:
        raise ValueError("spatial region has no current fresh producer destination")
    current = tuple(block.ops)
    first, last = current.index(start), current.index(pool)
    operations = current[first : last + 1]
    if not operations or any(op not in (producer, pool) and not isinstance(op, tensor.EmptyOp) for op in operations):
        raise ValueError("spatial extraction cannot cross an intervening operation or effect")
    params = tuple(free_values(operations))
    members = {op for top in operations for op in top.walk()}
    outputs = tuple(
        value for op in operations for value in op.results if any(use.operation not in members for use in value.uses)
    )
    if outputs != tuple(pool.results):
        raise ValueError("spatial region has an unaccounted original output or no observed final result")
    body = Block(arg_types=[value.type for value in params])
    value_map = dict(zip(params, body.args, strict=True))
    for op in operations:
        cloned = op.clone(value_mapper=value_map)
        body.add_op(cloned)
        value_map.update(zip(op.results, cloned.results, strict=True))
    body.add_op(func.ReturnOp(*(value_map[value] for value in outputs)))
    owner = func.FuncOp(symbol, ([v.type for v in params], [v.type for v in outputs]), Region(body))
    owner.attributes["llvm.emit_c_interface"] = UnitAttr()
    declarations = [
        op.clone() for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data in source_pair
    ]
    if symbol in source_pair:
        raise ValueError("region entry symbol collides with its original source declaration")
    region = ModuleOp([owner, *declarations])
    region.verify()
    maps = {
        name: {s: deepcopy(getattr(rewrite, name)[s]) for s in source_pair if s in getattr(rewrite, name)}
        for name in ("signatures", "programs", "entries", "arg_access", "call_buffers", "expected_interfaces")
    }
    routed = tuple(r for r in rewrite.routed if r.symbol in source_pair)
    prepack = deepcopy(rewrite.prepack)
    if prepack is not None:
        indices = {r.group for r in routed}
        prepack["groups"] = [row for row in prepack["groups"] if row["group"] in indices]
        prepack["skipped"] = [row for row in prepack["skipped"] if row["group"] in indices]
        prepack["folded_bias_elements"] = sum(row.get("bias", {}).get("elements", 0) for row in prepack["groups"])
        prepack["device_weights"] = sum("array" in (row.get("weight") or {}) for row in prepack["groups"])
    selected_rewrite = replace(
        rewrite,
        **maps,
        routed=routed,
        prepack=prepack,
        skipped=(),
        blob_weights=None,
    )
    # Exercise the same normal matcher on the actual extracted SSA. It must
    # retain the complete statements; only the public region ABI is new.
    rebound = group_connected_spatial(region, selected_rewrite)
    if len(rebound.receipt["groups"]) != 1:
        raise ValueError("extracted source lost its ordinary connected spatial admission")
    if text(module, generic=True) != before:
        raise ValueError("source extraction mutated the original full graph")
    source_owner = producer.parent_op()
    while source_owner is not None and not isinstance(source_owner, func.FuncOp):
        source_owner = source_owner.parent_op()
    if source_owner is None:
        raise ValueError("spatial source region has no original function owner")
    positions = {op: index for index, op in enumerate(current)}
    return ConnectedSpatialRegion(
        region,
        selected_rewrite,
        params,
        outputs,
        {
            "schema": "connected_spatial_source_region_v1",
            "source_host_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "source_owner": source_owner.sym_name.data,
            "source_pair": list(source_pair),
            "source_operation_ordinals": [positions[op] for op in operations],
            "region_sha256": hashlib.sha256(text(region, generic=True).encode()).hexdigest(),
            "input_types": [str(value.type) for value in params],
            "output_types": [str(value.type) for value in outputs],
            "input_binding": "original free SSA values, in first-use order",
            "output_binding": "all externally observed original member results, in source order",
            "original_full_graph_preserved": True,
            "original_source_selection": selected[0],
            "region_selection": rebound.receipt,
            "target_execution": False,
            "numerical_evaluation": False,
        },
    )


def public_spatial_source_graph(entry, binding):
    """Build typed host SSA from two original explicitly supplied public programs.

    This is a source graph, not captured-model or execution evidence. Existing
    source builders validate its programs and complete edge before declarations
    are minted. Their original entries travel in the ordinary group sidecar;
    subsequent source grouping, target and numerical gates remain mandatory.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import IntegerType, ModuleOp, TensorType, UnitAttr
    from xdsl.ir import Block, Region

    from merlin.compiler.lower.device.device_shim import logical_buffers
    from merlin.compiler.lower.device.offload import BY_GROUP, DeviceRewrite, Routed
    from merlin.compiler.lower.host.storage import _symbol
    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1

    if entry.get("op") != "spatial_producer_pool":
        raise ValueError("two original public spatial source programs required")
    capsule, _ = _merlin_model_builders_1.build(entry, binding)  # Includes complete composed-edge validation.
    components = deepcopy(entry["components"])
    symbols, interfaces, buffers = [], [], []
    for component in components:
        symbol = _symbol(component["name"])
        _, interface = _merlin_model_builders_1.build(component, binding)
        logical, reason = logical_buffers(interface)
        if logical is None or sum(b.kind == "output" for b in logical) != 1 or logical[-1].kind != "output":
            raise ValueError("public spatial component has unsupported external observers: " + reason)
        symbols.append(symbol)
        interfaces.append(interface)
        buffers.append(logical)
    if len(set(symbols)) != 2 or symbols[0] == "forward" or symbols[1] == "forward":
        raise ValueError("public component symbols must be distinct private declarations")
    first, second = buffers
    if len(second) != 2 or (first[-1].shape, first[-1].dtype) != (second[0].shape, second[0].dtype):
        raise ValueError("public spatial source edge changes its complete shape or format")

    def tensor_type(buffer):
        if buffer.dtype not in ("i8", "i32"):
            raise ValueError("public spatial source requires its exact signed integer formats")
        return TensorType(IntegerType(int(buffer.dtype[1:])), buffer.shape)

    block = Block(arg_types=[tensor_type(b) for b in first[:-1]])
    declarations = []
    edge = None
    for index, (symbol, logical) in enumerate(zip(symbols, buffers, strict=True)):
        destination = tensor.EmptyOp([], tensor_type(logical[-1]))
        arguments = list(block.args) if index == 0 else [edge]
        call = func.CallOp(symbol, [*arguments, destination.tensor], [tensor_type(logical[-1])])
        block.add_ops([destination, call])
        edge = call.res[0]
        declaration = func.FuncOp.external(symbol, [tensor_type(b) for b in logical], [tensor_type(logical[-1])])
        declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
        declarations.append(declaration)
    block.add_op(func.ReturnOp(edge))
    forward = func.FuncOp("forward", ([tensor_type(b) for b in first[:-1]], [tensor_type(second[-1])]), Region(block))
    forward.attributes["llvm.emit_c_interface"] = UnitAttr()
    module = ModuleOp([forward, *declarations])
    module.verify()
    rewrite = DeviceRewrite(
        binding.target,
        routed=tuple(Routed(symbol, (), (), ("i8", "i8", "i8")) for symbol in symbols),
        signatures={symbol: () for symbol in symbols},
        granularity=BY_GROUP,
        entries=dict(zip(symbols, components, strict=True)),
        programs={s: {"schema": "group_program_v1", "entry": e} for s, e in zip(symbols, components, strict=True)},
        arg_access={s: (*("read" for _ in bs[:-1]), "write") for s, bs in zip(symbols, buffers, strict=True)},
        call_buffers={
            s: [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in bs]
            for s, bs in zip(symbols, buffers, strict=True)
        },
    )
    return (
        module,
        rewrite,
        {
            "schema": "public_spatial_source_graph_v1",
            "origin": "two explicitly supplied public component programs; not a captured model",
            "source_entry_sha256": hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(),
            "component_interface_sha256": [hashlib.sha256(i.encode()).hexdigest() for i in interfaces],
            "source_pair": symbols,
            "observed_outputs": [capsule["operation"]["attributes"]["out"]],
            "target_execution": False,
        },
    )


def spatial_pair_interface(entry, binder, *, source_reference):
    """Compose the two original ordinary requests without changing their stages."""
    from merlin.compiler.lower.device.whole_program import group_interface
    from merlin.target.contract.interfaces.grammar import emit_interface_mlir, parse_interface_mlir

    if entry.get("schema") != "connected_spatial_entry_v1":
        raise ValueError("explicit connected spatial source entry required")
    components = entry.get("components")
    if not isinstance(components, list) or len(components) != 2:
        raise ValueError("connected spatial source must have exactly two original components")
    originals = [
        parse_interface_mlir(group_interface(e, binder, source_reference=source_reference)) for e in components
    ]
    first, second = deepcopy(originals)
    if (first["target"], first["abi_version"]) != (second["target"], second["abi_version"]):
        raise ValueError("connected source targets and ABIs differ")
    convolution = [c for c in first["commands"] if c["opcode"] == "CONV2D"]
    if len(convolution) != 1 or [c["opcode"] for c in second["commands"]] != ["MAXPOOL2D"]:
        raise ValueError("connected source requires one convolution and a standalone spatial maximum")
    if [c["opcode"] for c in first["commands"]] != ["RES_PACK", "CONV2D", "EVICT"]:
        raise ValueError("connected convolution contains additional operations")
    internal = convolution[0]["operands"]["dst"]
    source = second["commands"][0]["operands"]["src"]
    if first["tensors"][internal] != {**second["tensors"][source], "role": "output"}:
        raise ValueError("connected source edge changes tensor shape or format")
    # Rename every value, including resident handles, before identifying the edge.
    for prefix, cb in (("producer_", first), ("pool_", second)):
        cb["tensors"] = {prefix + n: s for n, s in cb["tensors"].items()}
        for c in cb["commands"]:
            c["operands"] = {k: prefix + v for k, v in c["operands"].items()}
            if "bias" in c.get("attributes", {}):
                c["attributes"]["bias"] = prefix + c["attributes"]["bias"]
    second["tensors"].pop("pool_" + source)
    second["commands"][0]["operands"]["src"] = "producer_" + internal
    joined = {
        "abi_version": first["abi_version"],
        "target": first["target"],
        "tensors": {**first["tensors"], **second["tensors"]},
        "commands": [*first["commands"], *second["commands"]],
    }
    return emit_interface_mlir(joined)


def group_connected_spatial(module, rewrite):
    """Return a detached opt-in selection; preserve every other observer/call.

    The narrow first route requires unique callee instances, adjacent calls
    apart from fresh tensor.empty destinations, and private source declarations.
    Other graphs remain separate requests. Source numerical entries are copied,
    never replaced with a multiplied readout or an inferred equivalence claim.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, TensorType, UnitAttr
    from xdsl.rewriter import Rewriter

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.lower.device.offload import DeviceRewrite, Routed

    if type(rewrite) is not DeviceRewrite:
        raise ValueError("ordinary DeviceRewrite required for connected source selection")
    module.verify()
    before = text(module, generic=True)
    selected = module.clone()
    calls = [op for op in selected.walk() if isinstance(op, func.CallOp)]
    declarations = {f.sym_name.data: f for f in selected.body.block.ops if isinstance(f, func.FuncOp)}
    by_symbol = {}
    for call in calls:
        by_symbol.setdefault(call.callee.root_reference.data, []).append(call)
    maps = {
        name: deepcopy(getattr(rewrite, name))
        for name in ("signatures", "programs", "entries", "arg_access", "call_buffers", "expected_interfaces")
    }
    removed, added, records = set(), [], []
    for producer in calls:
        ps = producer.callee.root_reference.data
        pe = rewrite.entries.get(ps, {})
        if pe.get("op") != "conv2d" or ps in removed or len(producer.res) != 1:
            continue
        uses = list(producer.res[0].uses)
        if len(uses) != 1 or uses[0].index != 0 or not isinstance(uses[0].operation, func.CallOp):
            continue
        pool = uses[0].operation
        qs = pool.callee.root_reference.data
        qe = rewrite.entries.get(qs, {})
        if qe.get("op") != "maxpool2d" or qs in removed or producer.parent is not pool.parent:
            continue
        if len(by_symbol[ps]) != 1 or len(by_symbol[qs]) != 1 or len(pool.res) != 1:
            continue
        if ps not in rewrite.programs or qs not in rewrite.programs:
            continue
        if any(
            rewrite.programs[s].get("entry")
            != {k: v for k, v in e.items() if k in rewrite.programs[s].get("entry", {})}
            for s, e in ((ps, pe), (qs, qe))
        ):
            continue
        access = rewrite.arg_access
        if access.get(ps) not in (("read", "read", "write"), ("read", "read", "read", "write")) or access.get(qs) != (
            "read",
            "write",
        ):
            continue
        if len(producer.arguments) != len(access[ps]) or len(pool.arguments) != 2:
            continue
        if any(not isinstance(v.type, TensorType) for v in (*producer.arguments, *pool.arguments, *pool.res)):
            continue
        if any(
            s not in declarations
            or declarations[s].body.blocks
            or getattr(declarations[s].sym_visibility, "data", None) != "private"
            for s in (ps, qs)
        ):
            continue
        if any(
            not isinstance(c.arguments[-1].owner, tensor.EmptyOp) or len(list(c.arguments[-1].uses)) != 1
            for c in (producer, pool)
        ):
            continue
        between, cursor = [], producer.next_op
        while cursor is not None and cursor is not pool:
            between.append(cursor)
            cursor = cursor.next_op
        if cursor is None or any(not isinstance(op, tensor.EmptyOp) for op in between):
            continue
        pbuf, qbuf = rewrite.call_buffers.get(ps, []), rewrite.call_buffers.get(qs, [])
        if len(pbuf) != len(producer.arguments) or len(qbuf) != 2:
            continue

        def matches(buffer, value):
            return buffer.get("shape") == list(value.type.get_shape()) and buffer.get("dtype") == str(
                value.type.element_type
            )

        if any(not matches(b, v) for b, v in zip([*pbuf, *qbuf], [*producer.arguments, *pool.arguments])):
            continue
        original_types = [str(v.type) for v in (*producer.arguments, *pool.arguments, *pool.res)]
        entry = {
            "schema": "connected_spatial_entry_v1",
            "kind": "model_slice",
            "source_role": pe.get("source_role", "model_derived"),
            "source_reference": pe.get("source_reference", "explicit closed source host graph"),
            "op": "spatial_producer_pool",
            "operand_dtype": pe.get("operand_dtype"),
            "components": [deepcopy(pe), deepcopy(qe)],
        }
        identity = hashlib.sha256(
            json.dumps({"entry": entry, "types": original_types}, sort_keys=True).encode()
        ).hexdigest()
        symbol = "merlin_connected_spatial_" + identity[:16]
        if symbol in declarations:
            raise ValueError("connected spatial symbol already exists")
        operands = [*producer.arguments[:-1], pool.arguments[-1]]
        declaration = func.FuncOp.external(symbol, [v.type for v in operands], [pool.res[0].type])
        declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
        declaration.properties["arg_attrs"] = ArrayAttr(
            [DictionaryAttr({"bufferization.access": StringAttr(a)}) for a in (*access[ps][:-1], "write")]
        )
        selected.body.block.add_op(declaration)
        replacement = func.CallOp(symbol, operands, [pool.res[0].type])
        Rewriter.replace_op(pool, replacement)
        destination = producer.arguments[-1].owner
        Rewriter.erase_op(producer)
        Rewriter.erase_op(destination)
        for name in maps:
            maps[name].pop(ps, None)
            maps[name].pop(qs, None)
        entry["name"] = symbol
        maps["signatures"][symbol] = ()
        maps["entries"][symbol] = entry
        maps["programs"][symbol] = {
            "schema": "connected_spatial_program_v1",
            "components": [deepcopy(rewrite.programs[ps]), deepcopy(rewrite.programs[qs])],
        }
        maps["arg_access"][symbol] = (*access[ps][:-1], "write")
        maps["call_buffers"][symbol] = [*deepcopy(pbuf[:-1]), deepcopy(qbuf[-1])]
        removed.update((ps, qs))
        originals = [r for r in rewrite.routed if r.symbol in (ps, qs)]
        if len(originals) != 2:
            raise ValueError("connected source route inventory differs from current unique calls")
        added.append(
            Routed(
                symbol,
                (),
                (),
                ("i8", "i8", "i8"),
                source_node_ids=tuple(n for r in originals for n in r.source_node_ids),
            )
        )
        records.append(
            {
                "symbol": symbol,
                "components": [ps, qs],
                "source_binding_sha256": identity,
                "producer_result_uses": 1,
                "external_intermediate_observers": [],
                "logical_argument_types": [str(v.type) for v in operands],
                "logical_result_types": [str(v.type) for v in replacement.res],
                "quantization": "both original ordered rounded readouts retained",
            }
        )
    updated = replace(
        rewrite, **maps, routed=tuple(r for r in rewrite.routed if r.symbol not in removed) + tuple(added)
    )
    selected.verify()
    if text(module, generic=True) != before:
        raise ValueError("connected selection mutated original source")
    return ConnectedSpatialSelection(
        selected,
        updated,
        {
            "schema": "connected_spatial_selection_v1",
            "source_host_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "selected_host_sha256": hashlib.sha256(text(selected, generic=True).encode()).hexdigest(),
            "groups": records,
            "target_execution": False,
            "numerical_evaluation": False,
        },
    )


# ---- connected_spatial_host --------------------------------------------------------------------
# Source-bound host ownership and closed private device selection for spatial pairs.
#
# The original public host owns scratch. The separately selected device helper
# has a private pointer ABI and cannot bring allocator/runtime imports across the
# closed device object boundary. Neither preparation establishes target effects.


@dataclass(frozen=True)
class ConnectedSpatialHost:
    scratch: object
    rewrite: object
    source_pair: tuple[str, str]
    source_entry: dict
    source_entry_sha256: str
    receipt: dict


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _check_source(plan, entry):
    """Check every source geometry, observer and ordered coefficient boundary."""
    from merlin.compiler.lower.quant.requantization import f32
    from merlin.compiler.schedule.blocks import SpatialMaxReadout
    from merlin.compiler.schedule.convolution.windows import Window2D
    from merlin.compiler.schedule.spatial.producer import PackedSpatialProducerPlan, SpatialProducerPlan
    from merlin.compiler.schedule.spatial.readouts import prove_spatial_readout

    if type(plan) not in (SpatialProducerPlan, PackedSpatialProducerPlan):
        raise ValueError("explicit checked spatial producer plan required")
    plan.__post_init__()
    if type(plan) is PackedSpatialProducerPlan:
        plan = plan.source
    if entry.get("schema") != "connected_spatial_entry_v1":
        raise ValueError("original connected spatial source entry required")
    producer, pool = entry["components"]
    p, s = plan.producer, plan.spatial
    pe, qe = producer.get("epilogue"), pool.get("epilogue")
    if (
        producer.get("op") != "conv2d"
        or pool.get("op") != "maxpool2d"
        or pe not in (["bias_add", "acc_scale"], ["bias_add", "acc_scale", "relu"])
        or qe != ["acc_scale", "maxpool"]
        or not p.bias
        or producer.get("output_dtype") != "i8"
        or pool.get("output_dtype") != "i8"
        or plan.readout_mode not in ("staged_readouts", "single_readout")
    ):
        raise ValueError("host spatial route requires the original rounded signed-i8 stages")
    window = Window2D(
        (producer.get("batch", 1), producer["Himg"], producer["Wimg"], producer["ci"]),
        (producer["kh"], producer["kw"]),
        tuple(producer["stride"]),
        tuple(producer.get("dilation", [1, 1])),
        tuple(producer["padding"]),
    )
    spatial = SpatialMaxReadout(
        tuple(pool["pool_in_dims"]),
        tuple(pool["pool_size"]),
        tuple(pool["pool_stride"]),
        tuple(pool["pool_padding"]),
        pool["pool_pad_value"],
    )
    if (
        window != p.window
        or spatial != s
        or producer["N"] != p.output_channels
        or pool["N"] != p.output_channels
        or pool["M"] != window.input_shape[0] * s.input_hw[0] * s.input_hw[1]
    ):
        raise ValueError("private spatial plan differs from the complete original source geometry")
    if ("acc_scale" in producer) == ("acc_scale_by_column" in producer):
        raise ValueError("source producer requires exactly one coefficient representation")
    scales = producer.get("acc_scale_by_column", [producer.get("acc_scale")] * p.output_channels)
    if len(scales) != p.output_channels:
        raise ValueError("source producer coefficients do not cover every output channel")
    readout_proofs = []
    for index, chain in enumerate(plan.readouts):
        group = tuple(f32(v) for v in scales[index * p.column_tile : (index + 1) * p.column_tile])
        if (
            not group
            or any(v != group[0] for v in group)
            or chain.producer is None
            or chain.producer.scales != (group[0],)
            or chain.producer.relu != ("relu" in pe)
            or chain.pool.scales != (f32(pool["acc_scale"]),)
            or chain.pool.relu
        ):
            raise ValueError("private spatial plan changes an original ordered Q1/Q2 boundary")
        if plan.readout_mode == "staged_readouts":
            if chain.replacement is not None:
                raise ValueError("staged spatial route cannot replace an original Q1/Q2 boundary")
        else:
            # The original Q1/Q2 coefficients above remain source-bound. Only
            # the existing complete signed-i32 transition proof may justify
            # consuming one replacement readout; no scalar samples suffice.
            readout_proofs.append(prove_spatial_readout(chain))
    return readout_proofs


def prepare_connected_spatial_host(
    module,
    rewrite,
    source_pair,
    plan,
    contract,
    workspace,
    *,
    kernel_symbol,
    pointer_alignments,
    stack_budget_bytes=None,
):
    """Group current source SSA and hoist scratch into its original public owner.

    Caller-supplied writer, noescape and completion effects remain independent
    premises. The ordinary route still owns source admission, device image,
    numerical checks and execution. This returns no qualification or permission.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

    if type(source_pair) is not tuple or len(source_pair) != 2 or any(type(s) is not str for s in source_pair):
        raise ValueError("explicit original producer/pool SSA callee pair required")
    selected = group_connected_spatial(module, rewrite)
    groups = [r for r in selected.receipt["groups"] if r["components"] == list(source_pair)]
    if len(groups) != 1 or type(contract) is not FreshTensorWriterContract or contract.symbol != groups[0]["symbol"]:
        raise ValueError("current source has no unique closed pair matching the supplied writer contract")
    # A narrow request cannot accidentally regroup unrelated source pairs.
    if len(selected.receipt["groups"]) != 1:
        raise ValueError("each private spatial preparation explicitly selects its complete grouped source set")
    entry = selected.rewrite.entries[contract.symbol]
    readout_proofs = _check_source(plan, entry)
    # Reattach only ordinary source-derived declaration effects that text export
    # drops. Generic printing retains them for normal upstream bufferization.
    for declaration in selected.module.body.block.ops:
        if not isinstance(declaration, func.FuncOp) or declaration.body.blocks:
            continue
        modes = selected.rewrite.arg_access.get(declaration.sym_name.data)
        if modes is None:
            continue
        if len(modes) != len(declaration.function_type.inputs) or any(m not in ("read", "write") for m in modes):
            raise ValueError("original source declaration access inventory differs")
        declaration.properties["arg_attrs"] = ArrayAttr(
            [DictionaryAttr({"bufferization.access": StringAttr(mode)}) for mode in modes]
        )
    artifact = prepare_spatial_scratch_abi(
        selected.module,
        plan,
        contract,
        workspace,
        kernel_symbol=kernel_symbol,
        pointer_alignments=pointer_alignments,
        source_binding_sha256=groups[0]["source_binding_sha256"],
        stack_budget_bytes=stack_budget_bytes,
    )
    return ConnectedSpatialHost(
        artifact,
        selected.rewrite,
        source_pair,
        entry,
        _digest(entry),
        {
            "schema": "connected_spatial_host_owner_v1",
            "source": selected.receipt,
            "source_entry_sha256": _digest(entry),
            "source_readout_proofs": readout_proofs,
            "host_owner": artifact.owner_symbol,
            "logical_buffers": spatial_logical_buffers(plan),
            "private_helper_pointer_count": len(pointer_alignments) + 1,
            "scratch": artifact.receipt,
            "target_execution": False,
        },
    )


def select_closed_spatial_device_helper(module, host, plan, *, helper_symbol: str, entry_symbol: str):
    """Select a literal closed private helper while preserving every operation.

    The discarded public transport wrapper is separately present in the prepared
    original host owner. No external call, global, capture or nested region can
    enter this device image; its normal object-symbol admission stays strict.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.lower.host.builder import PTR, FnBuilder, append_void_region
    from merlin.compiler.lower.host.storage import _symbol

    if type(host) is not ConnectedSpatialHost or type(module) is not ModuleOp:
        raise ValueError("current source-bound host and typed device module required")
    if (
        host.source_entry_sha256 != _digest(host.source_entry)
        or plan.identity != host.scratch.plan_sha256
        or hashlib.sha256(text(host.scratch.module, generic=True).encode()).hexdigest()
        != host.scratch.selected_source_sha256
    ):
        raise ValueError("source-bound spatial host or plan changed before device helper selection")
    _check_source(plan, host.source_entry)
    _symbol(helper_symbol)
    _symbol(entry_symbol)
    functions = [f for f in module.body.block.ops if isinstance(f, llvm.FuncOp) and f.sym_name.data == helper_symbol]
    count = len(spatial_logical_buffers(plan)) + 1
    if len(functions) != 1 or functions[0].function_type != llvm.LLVMFunctionType([PTR] * count, llvm.LLVMVoidType()):
        raise ValueError("private spatial device helper differs from original logical buffers plus bounded scratch")
    before = text(module, generic=True)
    builder = FnBuilder([PTR] * count)
    receipt = append_void_region(builder, functions[0], builder.entry.args)
    selected = ModuleOp(
        [
            llvm.FuncOp(
                entry_symbol, functions[0].function_type, linkage=llvm.LinkageAttr("external"), body=builder.finish()
            )
        ]
    )
    selected.verify()
    if text(module, generic=True) != before:
        raise ValueError("private device selection changed its original artifact")
    return selected, {
        "schema": "closed_spatial_device_helper_v1",
        "source_symbol": helper_symbol,
        "entry_symbol": entry_symbol,
        "private_pointers": count,
        "source_entry_sha256": host.source_entry_sha256,
        "plan_sha256": plan.identity,
        "region": receipt,
        "allocator_imports": [],
        "target_execution": False,
    }


# ---- connected_spatial_route -------------------------------------------------------------------
# Explicit ordinary host ownership for one current-source spatial pair.
#
# The original caller selects the source pair, exact staged plan and separately
# qualified borrowing/workspace premises. No emitted command buffer can select
# this route. Numerical, target and final-image admission remain ordinary gates.

if TYPE_CHECKING:
    from merlin.compiler.lower.host.writers import FreshTensorWriterContract, PrivateWorkspaceContract
    from merlin.compiler.schedule.spatial.producer import PackedSpatialProducerPlan, SpatialProducerPlan


@dataclass(frozen=True)
class ConnectedSpatialRoute:
    source_pair: tuple[str, str]
    plan: SpatialProducerPlan | PackedSpatialProducerPlan
    writer: FreshTensorWriterContract
    workspace: PrivateWorkspaceContract
    pointer_alignments: tuple[int, ...]
    helper_symbol: str
    stack_budget_bytes: int | None = None

    def record(self):
        from merlin.compiler.lower.host.storage import _symbol
        from merlin.compiler.lower.host.writers import FreshTensorWriterContract, PrivateWorkspaceContract
        from merlin.compiler.schedule.spatial.producer import PackedSpatialProducerPlan, SpatialProducerPlan

        if (
            type(self.source_pair) is not tuple
            or len(self.source_pair) != 2
            or any(type(s) is not str for s in self.source_pair)
            or self.source_pair[0] == self.source_pair[1]
            or type(self.plan) not in (SpatialProducerPlan, PackedSpatialProducerPlan)
            or type(self.writer) is not FreshTensorWriterContract
            or type(self.workspace) is not PrivateWorkspaceContract
            or type(self.pointer_alignments) is not tuple
        ):
            raise ValueError("connected spatial route requires original typed source, plan and ownership premises")
        self.plan.__post_init__()
        if self.stack_budget_bytes is not None and (
            type(self.stack_budget_bytes) is not int or not 0 < self.stack_budget_bytes < 1 << 63
        ):
            raise ValueError("private stack storage requires an explicit positive bounded byte budget")
        _symbol(self.helper_symbol)
        # Return a value snapshot, never live nested dictionaries from a caller.
        return json.loads(json.dumps(asdict(self), sort_keys=True, allow_nan=False))


def freeze_route(device):
    selected = getattr(device, "connected_spatial", None)
    if selected is None:
        return None
    if type(selected) is not ConnectedSpatialRoute:
        raise ValueError("connected spatial route requires its original typed selection")
    if getattr(device, "granularity", None) != "group" or any(
        getattr(device, name, None) is not None
        for name in (
            "exact_selection",
            "catalog_manifest",
            "catalog_object",
            "catalog_builder",
            "offload_selection",
            "post_offload_transform",
        )
    ):
        raise ValueError("connected spatial ownership requires a stated group route without opaque ABI rewrites")
    return selected, ConnectedSpatialRoute.record(selected)


def require_route(device, frozen):
    selected = getattr(device, "connected_spatial", None)
    if frozen is None:
        if selected is not None:
            raise ValueError("connected spatial selection appeared after original preparation")
        return None
    original, record = frozen
    if selected is not original or freeze_route(device)[1] != record:
        raise ValueError("connected spatial original selection changed")
    return selected


def require_host(host, selected):
    from merlin.compiler.dialects.xdsl_context import text

    if type(host) is not ConnectedSpatialHost or (
        host.source_pair != selected.source_pair
        or host.source_entry_sha256 != _digest(host.source_entry)
        or selected.plan.identity != host.scratch.plan_sha256
        or hashlib.sha256(text(host.scratch.module, generic=True).encode()).hexdigest()
        != host.scratch.selected_source_sha256
        or host.scratch.pointer_alignments != selected.pointer_alignments
        or host.scratch.receipt["pool"].get("stack_budget_bytes") != selected.stack_budget_bytes
    ):
        raise ValueError("connected spatial original host preparation changed")
    _check_source(selected.plan, host.source_entry)


def restore_group_rewrite(sidecar):
    """Restore ordinary early group preparation data; it grants no selection."""
    from merlin.compiler.lower.device.offload import DeviceRewrite, Routed, build_arguments

    args = build_arguments(sidecar, expected_granularity="group")
    fields = {k: v for k, v in sidecar.items() if k in DeviceRewrite.__dataclass_fields__}
    sequence_fields = ("parallel", "reduction", "dtypes", "tensor_types", "source_node_ids")
    fields["routed"] = tuple(
        Routed(**{k: tuple(v) if k in sequence_fields else v for k, v in row.items()}) for row in sidecar["routed"]
    )
    fields["signatures"] = args["signatures"]
    fields["arg_access"] = {k: tuple(v) for k, v in sidecar["arg_access"].items()}
    # Blob storage remains with the normal early preparation owner. Rewriting
    # this JSON inventory must not rewrite its separately retained blobs.
    return DeviceRewrite(**fields)


def prepare_host(device, selected, prepared, rewrite, directory):
    """Prepare the actual rewritten host before its ordinary sidecar is frozen."""
    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file
    from merlin.compiler.lower.device.build import kernel_symbol
    from merlin.compiler.lower.device.shim import kernel_abi_for

    module = parse_mlir_file(prepared)
    grouped = group_connected_spatial(module, rewrite)
    matches = [r for r in grouped.receipt["groups"] if r["components"] == list(selected.source_pair)]
    if len(matches) != 1 or matches[0]["symbol"] != selected.writer.symbol:
        raise ValueError("connected route lost its original current-source pair")
    symbol = matches[0]["symbol"]
    abi = kernel_abi_for(device.device)
    if abi is None or abi.version != 2:
        raise ValueError("connected spatial route requires the selected ordinary logical kernel ABI")
    index = sorted(grouped.rewrite.signatures).index(symbol)
    host = prepare_connected_spatial_host(
        module,
        rewrite,
        selected.source_pair,
        selected.plan,
        selected.writer,
        selected.workspace,
        kernel_symbol=kernel_symbol(abi.symbol, index),
        pointer_alignments=selected.pointer_alignments,
        stack_budget_bytes=selected.stack_budget_bytes,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "host.mlir"
    path.write_text(text(host.scratch.module, generic=True) + "\n")
    (directory / "receipt.json").write_text(json.dumps(host.receipt, sort_keys=True, indent=2) + "\n")
    return path, host


def select_emission(artifact, selected, host, *, entry_symbol):
    """Keep the checked producer artifact intact; derive one literal closed helper."""
    from xdsl.parser import Parser

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.ir.llvm_context import make_llvm_context

    require_host(host, selected)
    module = Parser(make_llvm_context(), artifact).parse_module()
    result, receipt = select_closed_spatial_device_helper(
        module,
        host,
        selected.plan,
        helper_symbol=selected.helper_symbol,
        entry_symbol=entry_symbol,
    )
    receipt = {
        **receipt,
        "checked_producer_sha256": hashlib.sha256(artifact.encode()).hexdigest(),
        "selected_artifact_sha256": hashlib.sha256(text(result, generic=True).encode()).hexdigest(),
    }
    return text(result, generic=True), receipt


def with_private_bridge(unit, selected, host, *, symbol, kernel):
    """Append only the original host's descriptor transport to ordinary shims."""
    from merlin.compiler.lower.host.storage import _bridge

    require_host(host, selected)
    if symbol != selected.writer.symbol or symbol in unit.symbols:
        raise ValueError("private spatial bridge has ambiguous original source membership")
    bridge = _bridge(selected.plan, selected.writer, selected.workspace, kernel, selected.pointer_alignments)
    if bridge != host.scratch.bridge_source:
        raise ValueError("private spatial bridge differs from the original host-bound kernel")
    return replace(
        unit, text=unit.text + "\n" + bridge, symbols=(*unit.symbols, symbol), kernels=(*unit.kernels, kernel)
    )
