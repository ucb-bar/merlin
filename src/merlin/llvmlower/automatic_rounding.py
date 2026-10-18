"""Compact exact rounding for bounded integer observations in serial host IR.

The selected native LLVM toolchain implements its standard roundeven intrinsic.
This source rewrite makes no ISA, rounding-mode, latency or device-support claim.
Other opaque operations remain barriers. Destination selection runs beforehand.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from xdsl.dialects import arith, builtin, func
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Operation
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import ParseError

from merlin.common.mlir_query import op_name

from . import quant_round_check as check


def _container(generic):
    fn = generic.parent_op()
    if (not isinstance(fn, func.FuncOp)
            or set(generic.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}
            or any(not k.startswith("prov.") for k in generic.attributes)
            or set(fn.properties) - {"sym_name", "function_type", "sym_visibility"}
            or any(k != "llvm.emit_c_interface" and not k.startswith("prov.") for k in fn.attributes)):
        return False
    parent = fn.parent_op()
    return (isinstance(parent, builtin.ModuleOp) and not parent.properties
            and all(k in {"llvm.data_layout", "llvm.target_triple"} or k.startswith("prov.")
                    for k in parent.attributes))


def _candidate(op, block):
    if not isinstance(op, arith.AddiOp) or not isinstance(op.result.type, builtin.IntegerType):
        return None
    truncated = op.operands[0].owner
    if not isinstance(truncated, arith.FPToSIOp) or truncated.parent is not block:
        return None
    value = truncated.operands[0]
    if not check.bounded_round_input(value, op.result.type):
        return None
    external = {}
    for child in block.ops:
        for operand in child.operands:
            if operand not in block.args and getattr(operand.owner, "parent", None) is not block:
                external.setdefault(operand, check._term("external", operand.type, parameter=len(external)))
    original = check._outputs(block, external, legalize=False, values=(op.result,))[0]
    source = check._outputs(block, external, legalize=False, values=(value,))[0]
    if original != check.residual_round_expression(source, value.type, op.result.type):
        return None
    # Retaining a live intermediate would duplicate work. Constants and the
    # bounded operand may have other readers; all other old round nodes close.
    nodes, pending = set(), [op]
    while pending:
        current = pending.pop()
        if current in nodes:
            continue
        nodes.add(current)
        for operand in current.operands:
            owner = operand.owner
            if operand is not value and isinstance(owner, Operation) and owner.parent is block:
                if not isinstance(owner, arith.ConstantOp):
                    pending.append(owner)
    if any(use.operation not in nodes for node in nodes - {op} for result in node.results for use in result.uses):
        return None
    return value


def rewrite_rounding(module, *, max_added_operations=4096, max_scalar_operations=512):
    from .automatic_host import _tensor_body

    count = 0
    for generic in tuple(module.walk()):
        if not isinstance(generic, GenericOp) or not _tensor_body(generic) or not _container(generic):
            continue
        block = generic.body.block
        if len(tuple(block.ops)) > max_scalar_operations:
            continue
        for op in tuple(block.ops):
            if count >= max_added_operations:
                return count
            if (value := _candidate(op, block)) is None:
                continue
            rounded = builtin.UnregisteredOp.with_name("llvm.intr.roundeven").create(
                operands=[value], result_types=[value.type])
            converted = arith.FPToSIOp(rounded.results[0], op.result.type)
            converted.attributes.update(op.attributes)
            Rewriter.replace_op(op, [rounded, converted])
            count += 1
    return count


def prepare_automatic_rounding(source, *, serial, index_bits=64, data_layout=None,
                               max_added_operations=4096, max_scalar_operations=512):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms import certificates, pipeline_state
    from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession
    from merlin.xdsl_dialects._common import text

    from . import automatic_host, optional_passes, quant_round

    if type(serial) is not bool or type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise ValueError("rounding selection requires an explicit lowering family and index width")
    if data_layout is not None and (not isinstance(data_layout, str) or not data_layout):
        raise ValueError("rounding selection requires its original selected data layout")
    if any(type(v) is not int or v < 0 for v in (max_added_operations, max_scalar_operations)):
        raise ValueError("rounding selection requires nonnegative code and scalar-IR budgets")
    digest = hashlib.sha256(source.encode()).hexdigest()
    owners = [Path(__file__), *(Path(module.__file__) for module in (
        check, automatic_host, quant_round, optional_passes, pipeline_state, certificates)),
        Path(__file__).with_name("lower.py")]
    def library():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in owners}
    pinned = library()
    context = PipelineContext(
        hardware=None if data_layout is None else "llvm-datalayout:" + hashlib.sha256(data_layout.encode()).hexdigest(),
        numerical_policy="typed_source:" + digest,
        resources=f"index_bits={index_bits};added_ops={max_added_operations};scalar_ops={max_scalar_operations}",
        execution="serial_native_llvm;value_semantics;standard_roundeven_intrinsic",
        library=hashlib.sha256(json.dumps(pinned, sort_keys=True).encode()).hexdigest(),
    )
    report = {
        "schema": "merlin.automatic_host_rounding.v1", "rewrites": 0,
        "source_file_sha256": digest, "output_file_sha256": digest,
        "context": asdict(context), "library_sources_sha256": pinned, "checks": [],
        "machine_conformance": "not_established",
        "cost_policy": "compact a closed residual-round DAG to a standard exact intrinsic; no cycle estimate",
    }
    if not serial or not optional_passes.switched(
        "automatic-host-rounding", os.environ.get("MERLIN_AUTOMATIC_HOST_ROUNDING", "1") != "0",
    ) or not max_added_operations or not max_scalar_operations:
        report["diagnostic"] = "disabled, insufficient budget or unsupported lowering family; original source retained"
        return source, report
    try:
        original = parse_mlir_text(source)
    except ParseError:
        report["diagnostic"] = "source is outside the typed rounding parser; original source retained"
        return source, report
    current = original.clone()
    session = PipelineSession(current, context=context, ir_level="host_tensor", max_repairs=0)
    session.run(PassSpec("rounding_source", 0, mutates_ir=False, ir_level="host_tensor", produces=("source",)),
                lambda: {"source": not any(op.successors or any(len(r.blocks) > 1 for r in op.regions)
                                           for op in current.walk())})
    if not session.analysis("source"):
        report["diagnostic"] = "unsupported source control flow; original source retained"
        return source, report
    session.run(PassSpec("rounding_representation", 1, ir_level="host_tensor", requires=("source",),
                         invalidates=("*",), produces=("emission",)),
                lambda: {"emission": rewrite_rounding(current, max_added_operations=max_added_operations,
                                                     max_scalar_operations=max_scalar_operations)})
    session.run(PassSpec("rounding_conformance", 2, mutates_ir=False, ir_level="host_tensor",
                         requires=("emission",)),
                lambda: (check.verify_quantization_rewrite(original, current, bounded_intrinsics=True), {})[1])
    count = session.analysis("emission")
    observed = sum(op_name(op) == "llvm.intr.roundeven" for op in current.walk()) - sum(
        op_name(op) == "llvm.intr.roundeven" for op in original.walk())
    if count != observed:
        raise ValueError("rounding accounting differs from actual emitted intrinsic operations")
    if library() != pinned:
        raise ValueError("rounding library changed during source selection; replan")
    output = text(current, generic=True) if count else source
    report.update({"rewrites": count, "output_file_sha256": hashlib.sha256(output.encode()).hexdigest(),
                   "checks": [asdict(record) for record in session.records]})
    if not count:
        report["diagnostic"] = "no closed bounded residual-round DAG; original source retained"
    return output, report
