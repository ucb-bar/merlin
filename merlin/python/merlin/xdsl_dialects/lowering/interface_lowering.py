"""``schedule`` decisions -> ``interface`` ops (merlin-materialize-interface stage).

Rebuilds the function body in the interface dialect: for each weight selected for
residency, emit one ``interface.resident_pack``; rewrite each matmul against it as
``interface.matmul`` -> ``interface.commit`` (no epilogue stages for the plain
repeated-RHS workload; raw i32 accumulations are committed as output_dtype i32); evict
after the last use. Contract/schedule decoration is consumed (dropped). The cross-op
analyses (use-after-evict, place legality, discharged checks) run on the result.
"""
from __future__ import annotations

from .._common import HAS_XDSL
from .input_workload import find_matmuls, matmul_lhs_rhs


class LoweringError(RuntimeError):
    pass


def lower_to_interface(module):
    """Build a fresh interface-level module from the scheduled module."""
    if not HAS_XDSL:
        return module
    from xdsl.ir import Block, Region
    from xdsl.dialects.builtin import (ArrayAttr, FunctionType, ModuleOp, StringAttr,
                                       TensorType)
    from xdsl.dialects.func import FuncOp, ReturnOp

    from .. import interface as i
    from .. import schedule as s
    from . import analyses

    # Which payload values were selected for residency (+ requested visibility)?
    selected = {}
    for op in module.walk():
        if (isinstance(op, s.SelectInterfaceOp)
                and op.interface.data == "resident_packed_tensor"):
            vis = op.visibility.data if op.visibility is not None else None
            selected[op.value] = vis
    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        raise LoweringError("no func.func in module")
    fn = fns[0]
    src_block = fn.body.blocks[0]
    matmuls = find_matmuls(module)
    if not matmuls:
        raise LoweringError("no matmul payload to materialize")

    arg_types = [a.type for a in src_block.args]
    blk = Block(arg_types=arg_types)
    value_map = dict(zip(src_block.args, blk.args))

    ops = []
    packs = {}  # old weight SSA value -> ResidentPackOp
    for old_w, vis in selected.items():
        if old_w not in value_map:
            raise LoweringError("selected weight is not a function argument")
        w = value_map[old_w]
        props = {"layout": i.LayoutAttr(i.Layout.PACKED_RHS),
                 "lifetime": i.LifetimeAttr(i.Lifetime.REGION)}
        if vis is not None:
            props["visibility"] = i.VisibilityAttr(vis)
        pack = i.ResidentPackOp(
            operands=[w],
            result_types=[i.ResidentTensorType(w.type, StringAttr("packed_rhs"))],
            properties=props)
        packs[old_w] = pack
        ops.append(pack)

    outs = []
    out_types = []
    for mm in matmuls:
        old_lhs, old_rhs = matmul_lhs_rhs(mm)
        lhs = value_map[old_lhs]
        acc_type = i.AccumulatorType(mm.results[0].type)
        if old_rhs in packs:
            rhs = packs[old_rhs].res
        else:
            rhs = value_map[old_rhs]
        imm = i.MatmulOp(operands=[lhs, rhs], result_types=[acc_type])
        out_t = mm.results[0].type
        if not isinstance(out_t, TensorType):
            raise LoweringError("matmul result is not a tensor")
        commit = i.CommitOp(operands=[imm.acc], result_types=[out_t], properties={
            "epilogue": ArrayAttr([]),
            "output_dtype": StringAttr("i32")})
        ops += [imm, commit]
        outs.append(commit.out)
        out_types.append(out_t)

    for pack in packs.values():
        ops.append(i.ResidentEvictOp(operands=[pack.res]))
    ops.append(ReturnOp(*outs))
    blk.add_ops(ops)
    new_fn = FuncOp(fn.sym_name.data, FunctionType.from_lists(arg_types, out_types),
                    Region([blk]))
    out = ModuleOp([new_fn])

    problems = (analyses.check_no_use_after_evict(out)
                + analyses.check_place_legality(module)
                + analyses.check_contract_discharged(module))
    if problems:
        raise LoweringError("; ".join(problems))
    return out
