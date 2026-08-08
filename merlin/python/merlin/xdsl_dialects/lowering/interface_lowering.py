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

    from .. import contract as c
    from .. import interface as i
    from .. import schedule as s
    from . import analyses

    def _out_dtype(t):
        """The commit output dtype token, DERIVED from the accumulator element type — not
        assumed i32 (a f32 matmul commits f32; an i8→i32 matmul commits i32)."""
        from xdsl.dialects.builtin import IntegerType
        et = t.element_type
        return f"i{et.width.data}" if isinstance(et, IntegerType) else str(et)

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

    # Rebuild the body in PROGRAM ORDER, threading a value map that grows as each op is
    # materialized — so a matmul CHAIN (layer N's committed output feeding layer N+1) and any
    # other intermediate resolve, not just block-arg operands. Each matmul -> interface.matmul
    # -> commit; the committed tensor is recorded so downstream consumers find it.
    matmul_set = set(matmuls)
    inline_packs = {}  # base weight value -> inline ResidentPackOp (single-use weights)
    ret_op = None
    for op in list(src_block.ops):
        if isinstance(op, ReturnOp):
            ret_op = op
            continue
        if op in matmul_set:
            old_lhs, old_rhs = matmul_lhs_rhs(op)
            if old_lhs not in value_map:
                raise LoweringError("matmul lhs is not a materialized value")
            lhs = value_map[old_lhs]
            acc_type = i.AccumulatorType(op.results[0].type)
            if old_rhs in packs:
                rhs = packs[old_rhs].res
            elif old_rhs in value_map:
                # Every matmul weight is placed resident on the mesh — the reuse>=2 gate only
                # decides whether it is HOISTED/kept across dispatches (the `packs` set), not
                # whether it is packed at all. A single-use weight (common in a whole model) is
                # still packed here, once, so the target matmul always sees a resident RHS.
                base = value_map[old_rhs]
                pk = inline_packs.get(base)
                if pk is None:
                    pk = i.ResidentPackOp(
                        operands=[base],
                        result_types=[i.ResidentTensorType(base.type, StringAttr("packed_rhs"))],
                        properties={"layout": i.LayoutAttr(i.Layout.PACKED_RHS),
                                    "lifetime": i.LifetimeAttr(i.Lifetime.REGION)})
                    ops.append(pk)
                    inline_packs[base] = pk
                rhs = pk.res
            else:
                raise LoweringError("matmul rhs is not a materialized value")
            imm = i.MatmulOp(operands=[lhs, rhs], result_types=[acc_type])
            out_t = op.results[0].type
            if not isinstance(out_t, TensorType):
                raise LoweringError("matmul result is not a tensor")
            commit = i.CommitOp(operands=[imm.acc], result_types=[out_t], properties={
                "epilogue": ArrayAttr([]),
                "output_dtype": StringAttr(_out_dtype(out_t))})
            ops += [imm, commit]
            value_map[op.results[0]] = commit.out
            continue
        # Contract/schedule decoration and matmul-init scaffolding (the zero-point constant,
        # the empty accumulator init) are consumed. Any other payload op is one this stage does
        # not yet lower to an interface form — fail closed so the boundary is visible, never
        # silently dropped (that would miscompile a whole model down to its matmuls alone).
        if (op.dialect_name() in (c.DIALECT_NAME, s.DIALECT_NAME)
                or op.name in ("arith.constant", "tensor.empty")):
            continue
        raise LoweringError(f"interface lowering does not yet handle payload op '{op.name}'")

    outs = []
    out_types = []
    if ret_op is not None:
        for operand in ret_op.operands:
            if operand not in value_map:
                raise LoweringError("return operand is not a materialized value")
            outs.append(value_map[operand])
            out_types.append(value_map[operand].type)

    for pack in list(packs.values()) + list(inline_packs.values()):
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
