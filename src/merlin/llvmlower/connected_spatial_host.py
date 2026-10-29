"""Source-bound host ownership and closed private device selection for spatial pairs.

The original public host owns scratch. The separately selected device helper
has a private pointer ABI and cannot bring allocator/runtime imports across the
closed device object boundary. Neither preparation establishes target effects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .connected_spatial import group_connected_spatial
from .fresh_tensor_writer import FreshTensorWriterContract
from .spatial_scratch_abi import prepare_spatial_scratch_abi, spatial_logical_buffers


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
    from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
    from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan
    from merlin.compile.scheduling.spatial_readout import prove_spatial_readout
    from merlin.compile.scheduling.window_transfer import Window2D
    from merlin.llvmlower.requantization import f32

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

    from merlin.xdsl_dialects._common import text

    from .host_builder import PTR, FnBuilder, append_void_region
    from .spatial_scratch_abi import _symbol

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
