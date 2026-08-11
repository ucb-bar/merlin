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


def support_ops(block, payload: set) -> set:
    """Ops in ``block`` whose results feed ONLY the payload — safely consumed by materialization.

    For the matmul payload these are the accumulator inits (``tensor.empty``) and the quantized
    matmul's zero-point constants: the rebuilt body re-creates their effect, so dropping the
    original op loses nothing. Computed as a fixpoint rather than a hardcoded op-name list, so it
    stays correct for any frontend's spelling of "value that exists only to feed the contraction".

    An op with NO results can never qualify: it is kept out because a result-less op is how a side
    effect is spelled (a store, a copy), and silently dropping one changes what the program does.
    """
    support: set = set()
    changed = True
    while changed:
        changed = False
        for op in block.ops:
            if op in support or op in payload or not op.results:
                continue
            if all(use.operation in payload or use.operation in support
                   for res in op.results for use in res.uses):
                support.add(op)
                changed = True
    return support


def unaccounted_ops(block, payload) -> list:
    """Ops in ``block`` that interface materialization would neither rebuild nor safely drop.

    This is the single definition of "what the staged pipeline can carry", used both by the guard
    below and by the router in :mod:`merlin.compile_core`. They were separate op-name lists once,
    and drifted immediately: the router called a zeroing ``linalg.fill`` generic computation while
    materialization correctly treated it as support, so a perfectly stageable matmul was routed to
    the LLVM path. One computation cannot disagree with itself.
    """
    from .. import contract as c
    from .. import schedule as s

    decoration = {c.DIALECT_NAME, s.DIALECT_NAME}
    payload = set(payload)
    support = support_ops(block, payload)
    terminator = block.last_op
    return [op for op in block.ops
            if op not in support and op not in payload and op is not terminator
            and op.dialect_name() not in decoration]


def _check_payload_complete(fn, src_block, payload: list) -> None:
    """Fail closed if materialization would silently drop part of the computation.

    :func:`lower_to_interface` REBUILDS the function body from scratch as pack / matmul / commit /
    evict + return. Anything it does not recognize simply never gets emitted — and because every
    stage after it verifies the *rebuilt* module, a dropped masked store, epilogue or grid loop
    produces a module that verifies at all six stages, emits a command buffer, and computes
    something other than what was asked for. Nothing downstream can notice.

    That is a false-PASS generator, and it matters most for exactly the payloads a kernel frontend
    submits, so the drop is turned into an error naming what would have been lost.
    """
    dropped = [op.name for op in unaccounted_ops(src_block, payload)]
    if dropped:
        raise LoweringError(
            "interface materialization would silently drop " f"{len(dropped)} op(s) of the payload: "
            + ", ".join(sorted(set(dropped)))
            + ". The staged pipeline rebuilds the function body as resident_pack/matmul/commit/evict, "
              "so it can only carry matmul-family computation; anything else (masked store, "
              "elementwise epilogue, grid loop) has to be expressed as an interface op before it can "
              "descend. Failing here instead of compiling a different program.")

    # An epilogue would be caught above, but a payload whose RESULTS are not the matmul results is a
    # second way to lose computation (the rebuilt return carries commits of the matmuls, in order).
    terminator = src_block.last_op
    if terminator is not None:
        returned = list(terminator.operands)
        expected = [mm.results[0] for mm in payload]
        if returned != expected:
            raise LoweringError(
                f"function @{fn.sym_name.data} returns {len(returned)} value(s) that are not exactly "
                f"the {len(expected)} matmul result(s), in order — interface materialization returns "
                "the commits of the matmuls it found, so the difference would be dropped.")


def lower_to_interface(module):
    """Build a fresh interface-level module from the scheduled module."""
    if not HAS_XDSL:
        return module
    from xdsl.dialects.builtin import ArrayAttr, FunctionType, ModuleOp, StringAttr, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

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
    if len(fns) > 1:
        # Only fns[0] is materialized, so the rest would vanish without a word.
        raise LoweringError(
            f"{len(fns)} func.func in module ({', '.join(f.sym_name.data for f in fns)}) — interface "
            "materialization rebuilds a single function; submit one kernel per module")
    fn = fns[0]
    if len(fn.body.blocks) != 1:
        raise LoweringError(
            f"@{fn.sym_name.data} has {len(fn.body.blocks)} blocks — only the entry block is "
            "materialized, so control flow must be resolved before this stage")
    src_block = fn.body.blocks[0]
    matmuls = find_matmuls(module)
    if not matmuls:
        raise LoweringError("no matmul payload to materialize")
    _check_payload_complete(fn, src_block, matmuls)

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
