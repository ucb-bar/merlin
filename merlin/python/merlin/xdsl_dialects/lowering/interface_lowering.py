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


# linalg elementwise op name -> interface.vector_map combine token (two real tensor operands).
_ELEMENTWISE_COMBINE = {"linalg.add": "add", "linalg.mul": "mul"}

_VIEW_OPS = ("tensor.expand_shape", "tensor.collapse_shape", "tensor.cast", "linalg.copy",
             "linalg.transpose")


def _resolved_name(op) -> str | None:
    """Op name, resolving model2MLIR's ``quant_ext`` ops (parsed as ``builtin.unregistered``).
    Total: a block argument's owner is a Block (no ``name``) -> None."""
    name = getattr(op, "name", None)
    if name == "builtin.unregistered":
        on = getattr(op, "op_name", None)
        return on.data if on is not None else name
    return name


def _view_input(owner):
    """The single tensor input of a view/layout op (operands[0], or inputs[0] for transpose)."""
    if _resolved_name(owner) == "linalg.transpose":
        return owner.inputs[0]
    return owner.operands[0]


def _map_through_views(value, value_map):
    """Follow view/layout ops from ``value`` to a value present in ``value_map`` (a materialized
    block arg / earlier result); return the mapped new value, or None."""
    cur = value
    for _ in range(16):
        if cur in value_map:
            return value_map[cur]
        owner = getattr(cur, "owner", None)
        if _resolved_name(owner) in _VIEW_OPS:
            cur = _view_input(owner)
            continue
        return None
    return None


def _dequant_source(rhs, value_map):
    """If a matmul RHS is fed by ``quant_ext.dequantize_per_channel`` (the int8 weight-only idiom),
    resolve it to (weight_value, scale_value, axis) in the NEW block's values — else None. The
    weight and scale are traced through any view/layout ops back to materialized function args."""
    cur = rhs
    for _ in range(8):
        owner = getattr(cur, "owner", None)
        nm = _resolved_name(owner)
        if nm in _VIEW_OPS:
            cur = _view_input(owner)
            continue
        if nm is not None and nm.startswith("quant_ext.dequantize"):
            w = _map_through_views(owner.operands[0], value_map)
            scale = (_map_through_views(owner.operands[1], value_map)
                     if len(owner.operands) > 1 else None)
            if w is None or scale is None:
                return None
            axis = 1
            axis_attr = (getattr(owner, "properties", {}) or {}).get("axis")
            if axis_attr is not None:
                axis = int(axis_attr.value.data)
            return w, scale, axis
        return None
    return None


def _is_zero_fill(value) -> bool:
    """True when ``value`` is a ``linalg.fill`` of a zero constant — i.e. the second operand of a
    ``linalg.max`` that makes it a relu (max(x, 0)). Derived structurally from the IR, no regex."""
    owner = getattr(value, "owner", None)
    # A block argument's owner is a Block (no ``name``) — getattr keeps the check total.
    if getattr(owner, "name", None) != "linalg.fill":
        return False
    scalar = owner.inputs[0]
    c = getattr(scalar, "owner", None)
    if getattr(c, "name", None) != "arith.constant":
        return False
    attr = getattr(c, "value", None)
    inner = getattr(attr, "value", None)
    data = getattr(inner, "data", None)
    return data is not None and float(data) == 0.0


def _lower_vector_map(op, value_map, i):
    """Lower a linalg elementwise/activation op to a single interface.vector_map.

    ``linalg.add``/``linalg.mul`` -> a two-operand combine; ``linalg.max`` against a zero fill ->
    an identity copy with a relu activation (the standard relu idiom max(x, 0)). Any other shape
    fails closed — the engine's vector path models exactly this vocabulary."""
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    name = op.name
    if name in _ELEMENTWISE_COMBINE:
        lhs, rhs = value_map[op.inputs[0]], value_map[op.inputs[1]]
        props = {"combine": StringAttr(_ELEMENTWISE_COMBINE[name])}
    elif name == "linalg.max" and _is_zero_fill(op.inputs[1]):
        a = value_map[op.inputs[0]]
        lhs = rhs = a                       # identity copy of lhs; rhs is unused by the engine
        props = {"combine": StringAttr("identity"),
                 "activation": ArrayAttr([StringAttr("relu")])}
    else:
        raise LoweringError(
            f"interface lowering does not model vector op '{name}' "
            "(only elementwise add/mul and relu = max(x, 0) map to the engine's vector path)")
    return i.VectorMapOp(operands=[lhs, rhs], result_types=[op.results[0].type], properties=props)


def lower_to_interface(module):
    """Build a fresh interface-level module from the scheduled module."""
    if not HAS_XDSL:
        return module
    from xdsl.ir import Block, Region
    from xdsl.dialects.builtin import (ArrayAttr, FunctionType, IntegerAttr, ModuleOp,
                                       StringAttr, TensorType)
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
            operands=[w, None],
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
            dq = None if (old_rhs in packs or old_rhs in value_map) \
                else _dequant_source(old_rhs, value_map)
            if old_rhs in packs:
                rhs = packs[old_rhs].res
            elif dq is not None:
                # int8 weight-only: pack the i8 weight and dequantize it per channel at pack time
                # (a float resident weight the target matmul consumes normally). The dequantize op
                # and its scale are consumed here — the target sees one resident_pack.
                w_val, scale_val, axis = dq
                pk = inline_packs.get(w_val)
                if pk is None:
                    pk = i.ResidentPackOp(
                        operands=[w_val, scale_val],
                        result_types=[i.ResidentTensorType(old_rhs.type, StringAttr("packed_rhs"))],
                        properties={"layout": i.LayoutAttr(i.Layout.PACKED_RHS),
                                    "lifetime": i.LifetimeAttr(i.Lifetime.REGION),
                                    "dequant_axis": IntegerAttr(axis, 64)})
                    ops.append(pk)
                    inline_packs[w_val] = pk
                rhs = pk.res
            elif old_rhs in value_map:
                # Every matmul weight is placed resident on the mesh — the reuse>=2 gate only
                # decides whether it is HOISTED/kept across dispatches (the `packs` set), not
                # whether it is packed at all. A single-use weight (common in a whole model) is
                # still packed here, once, so the target matmul always sees a resident RHS.
                base = value_map[old_rhs]
                pk = inline_packs.get(base)
                if pk is None:
                    pk = i.ResidentPackOp(
                        operands=[base, None],
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
        # Non-matmul payload: elementwise combine / activation ops run on the target's vector lanes
        # (interface.vector_map). These are the residual adds, gating multiplies, and activations of
        # a whole model — threaded through the same value_map so they chain with the matmuls.
        if op.name in _ELEMENTWISE_COMBINE or op.name == "linalg.max":
            vmap = _lower_vector_map(op, value_map, i)
            ops.append(vmap)
            value_map[op.results[0]] = vmap.out
            continue
        # Contract/schedule decoration and op-init scaffolding (the zero-point constant, the empty
        # init tensors, the linalg.fill that materializes a relu's zero) are consumed. Any other
        # payload op is one this stage does not yet lower to an interface form — fail closed so the
        # boundary is visible, never silently dropped (that would miscompile a whole model).
        if (op.dialect_name() in (c.DIALECT_NAME, s.DIALECT_NAME)
                or op.name in ("arith.constant", "tensor.empty", "linalg.fill", "tensor.splat")
                or _resolved_name(op).startswith("quant_ext.dequantize")):
            # A dequantize feeding a matmul RHS is consumed by that matmul's dequant-pack above; a
            # dead one is harmless to drop. Its scale/weight args flow through as pack operands.
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
