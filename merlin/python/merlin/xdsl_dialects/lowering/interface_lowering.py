"""``schedule`` decisions -> ``interface`` ops (merlin-materialize-interface stage).

Rebuilds the function body in the interface dialect: for each weight selected for
residency, emit one ``interface.resident_pack``; rewrite each matmul against it as
``interface.matmul`` -> ``interface.commit`` (raw i32 accumulations are committed as
output_dtype i32); evict after the last use. A bias-add consumer of a contraction is
absorbed into that contraction's commit as a ``bias_add`` epilogue stage naming the bias
tensor; every other epilogue shape stays unaccounted and is refused. Contract/schedule
decoration is consumed (dropped). A contraction whose accumulator init is not provably zero is
REFUSED, not rebuilt: the commit accumulates from zero, so the init would be dropped silently. The
cross-op analyses (use-after-evict, place legality, discharged checks) run on the result.
"""
from __future__ import annotations

from .._common import HAS_XDSL
from .input_workload import elementwise_operands, find_elementwise, find_matmuls, matmul_lhs_rhs


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


def _is_zero_scalar(value) -> bool:
    """True when ``value`` is an ``arith.constant`` whose scalar value is zero.

    Total and fail-closed: a block argument, a computed value, a constant whose attribute this
    cannot read, or a value whose data is not a number all answer False (NOT provably zero) rather
    than raising or guessing.
    """
    c = getattr(value, "owner", None)
    # A block argument's owner is a Block (no ``name``) — getattr keeps the check total.
    if getattr(c, "name", None) != "arith.constant":
        return False
    attr = getattr(c, "value", None)
    inner = getattr(attr, "value", None)
    data = getattr(inner, "data", None)
    if data is None:
        return False
    try:
        return float(data) == 0.0
    except (TypeError, ValueError):
        return False


def _is_zero_dense_constant(owner) -> bool:
    """True when ``owner`` is an ``arith.constant`` of an all-zero dense tensor attribute."""
    if getattr(owner, "name", None) != "arith.constant":
        return False
    get_values = getattr(getattr(owner, "value", None), "get_values", None)
    if get_values is None:
        return False
    try:
        values = list(get_values())
        return bool(values) and all(float(v) == 0.0 for v in values)
    except Exception:                        # unreadable attribute -> not provably zero
        return False


def _is_zero_fill(value) -> bool:
    """True when ``value`` is a ``linalg.fill`` of a zero constant — i.e. the second operand of a
    ``linalg.max`` that makes it a relu (max(x, 0)). Derived structurally from the IR, no regex.

    The fill's own ``outs`` is irrelevant: a fill writes its scalar into EVERY element, so whatever
    it was destined for is overwritten.
    """
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) != "linalg.fill":
        return False
    return _is_zero_scalar(owner.inputs[0])


def init_contributes_nothing(value) -> bool:
    """Can this accumulator init be dropped without changing the program?

    The interface rebuild emits ``interface.matmul`` -> ``interface.commit``, which accumulates from
    a ZERO accumulator and never reads the source op's ``outs``. So an init that holds a value is
    part of the computation (``C + A@B``), and dropping it compiles a different program. This is the
    proof that it holds nothing, and it is a PROOF, not a pattern — every answer below is read out
    of the IR structurally, and everything undecidable answers False:

    * ``tensor.empty`` — an uninitialized tensor has no defined contents, so there is no value that
      could be lost (this is the idiom every frontend here emits for "no accumulation");
    * ``linalg.fill`` of a zero constant — the standard explicit zeroing;
    * ``tensor.splat`` of a zero constant, and an all-zero dense ``arith.constant``.

    A function argument, a computed value (a ``tensor.pad``, an earlier layer's output), a non-zero
    constant, and a fill whose scalar cannot be resolved are all NOT provably zero and answer False.
    """
    owner = getattr(value, "owner", None)
    name = _resolved_name(owner)             # None for a block argument
    if name == "tensor.empty":
        return True
    if name == "linalg.fill":
        return _is_zero_fill(value)
    if name == "tensor.splat":
        operands = list(getattr(owner, "operands", ()) or ())
        return bool(operands) and _is_zero_scalar(operands[0])
    if name == "arith.constant":
        return _is_zero_dense_constant(owner)
    return False


def accumulated_inits(op) -> list:
    """The ``outs`` init values of ``op`` that ``op``'s OWN BODY reads.

    The bug this exists for: ``unaccounted_ops`` enumerates OPS, and an init that arrives as a block
    argument is not an op, so a ``linalg.matmul`` accumulating onto a non-zero function argument
    passed every completeness check and lowered to the un-biased program. Whether an init is part of
    the computation is not a property of the op's NAME (this repo derives, it does not hardcode): it
    is whether the op's region reads the corresponding ``outs`` block argument. A contraction's body
    is ``yield add(mul(a, b), out)`` — it reads it; ``linalg.add``/``mul``/``max``/``fill`` bodies do
    not read theirs, and their init really is just a destination.

    Fails closed: an op with no inspectable body, or a body whose argument count cannot be lined up
    with its ``outs``, reports EVERY init as accumulated.
    """
    outs = list(getattr(op, "outputs", ()) or ())
    if not outs:
        return []
    regions = list(getattr(op, "regions", ()) or ())
    blocks = list(regions[0].blocks) if regions else []
    if not blocks:
        return outs
    args = list(blocks[0].args)
    if len(args) < len(outs):
        return outs
    out_args = args[len(args) - len(outs):]
    return [v for v, a in zip(outs, out_args) if len(list(a.uses)) > 0]


def nonzero_accumulator_inits(payload) -> list:
    """``[(op, init value)]`` — payload ops that accumulate onto an init this cannot prove is zero.

    The companion of :func:`unaccounted_ops` for VALUES: that one answers "which ops would the
    rebuild drop", this one answers "which incoming values would it drop". Both are consulted by
    :func:`_check_payload_complete`; a router that asks the first should ask this one too, or it
    will route to the staged path a payload the staged path must refuse.
    """
    return [(op, v) for op in payload for v in accumulated_inits(op)
            if not init_contributes_nothing(v)]


def _value_origin(value) -> str:
    """Where a value came from, for a refusal message: the defining op, or the block argument."""
    owner = getattr(value, "owner", None)
    name = _resolved_name(owner)
    if name is None:
        idx = getattr(value, "index", None)
        return "block argument #%d" % idx if idx is not None else "a block argument"
    return "the result of '%s'" % name


def _lower_vector_map(op, value_map, i):
    """Lower a linalg elementwise/activation op to a single interface.vector_map.

    ``linalg.add``/``linalg.mul`` -> a two-operand combine; ``linalg.max`` against a zero fill ->
    an identity copy with a relu activation (the standard relu idiom max(x, 0)). Any other shape
    fails closed — the engine's vector path models exactly this vocabulary."""
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    def materialized(operand):
        """The rebuilt value for ``operand`` — a named refusal, never a bare KeyError.

        A vector operand can be missing for the same reason an init can be wrong: ``support_ops``
        absorbed the op that produced it (a non-zero ``linalg.fill``, a dense constant) on the
        assumption the rebuild re-creates its effect, and for a vector lane it does not. The lookup
        used to be a raw dict subscript, which surfaced as a ``KeyError`` naming an SSA value — a
        crash a caller cannot catch by type and a reader cannot act on.
        """
        if operand not in value_map:
            raise LoweringError(
                f"vector op '{op.name}' reads an operand that materialization does not produce "
                f"({_value_origin(operand)}, type {operand.type}) — the interface vector lane "
                "consumes materialized tensors, so an operand folded away as contraction "
                "scaffolding (a constant, a non-zero fill) has no value to read. Refusing rather "
                "than emitting a vector_map over a tensor the engine never materializes.")
        return value_map[operand]

    name = op.name
    if name in _ELEMENTWISE_COMBINE:
        lhs, rhs = materialized(op.inputs[0]), materialized(op.inputs[1])
        props = {"combine": StringAttr(_ELEMENTWISE_COMBINE[name])}
    elif name == "linalg.max" and _is_zero_fill(op.inputs[1]):
        a = materialized(op.inputs[0])
        lhs = rhs = a                       # identity copy of lhs; rhs is unused by the engine
        props = {"combine": StringAttr("identity"),
                 "activation": ArrayAttr([StringAttr("relu")])}
    else:
        raise LoweringError(
            f"interface lowering does not model vector op '{name}' "
            "(only elementwise add/mul and relu = max(x, 0) map to the engine's vector path)")
    return i.VectorMapOp(operands=[lhs, rhs], result_types=[op.results[0].type], properties=props)


def _is_vector_lane_op(op) -> bool:
    """Does the interface rebuild materialize this op as an ``interface.vector_map``?

    Consulted by BOTH the payload-completeness check and the rebuild loop below, for the reason
    :func:`unaccounted_ops` states about its own pair: two op-name lists that must agree will drift.
    They already did here — the loop grew an elementwise branch while the check still counted only
    matmuls, so every mixed payload looked like a dropped computation.
    """
    return op.name in _ELEMENTWISE_COMBINE or op.name == "linalg.max"


def _iterators_all_parallel(op) -> bool:
    """Every iterator of a ``linalg.generic`` is ``parallel`` (no reduction dimension)."""
    from xdsl.dialects.linalg.attrs import IteratorType

    its = list(op.iterator_types)
    return bool(its) and all(getattr(a, "data", None) is IteratorType.PARALLEL for a in its)


def _is_trailing_broadcast(amap, rank: int) -> bool:
    """``(d0..d{rank-1}) -> (d{rank-1})``: the map that indexes a bias by the output's LAST axis.

    Deliberately narrow. The engine's bias stage is ``Tensor.add_bias``, which adds a length-n
    vector to every row of an (m, n) tensor — a PER-COLUMN bias. A row bias
    (``(d0, d1) -> (d0)``) is a different computation that no engine here implements, so it must
    not match: matching it would emit a `bias_add` the runtime would execute against the wrong
    axis (or reject), which is exactly the fail-open this stage exists to prevent.
    """
    from xdsl.ir.affine import AffineDimExpr

    if amap.num_dims != rank or amap.num_symbols:
        return False
    res = amap.results
    return (len(res) == 1 and isinstance(res[0], AffineDimExpr)
            and res[0].position == rank - 1)


def _adds_its_two_inputs(op) -> bool:
    """The generic's body is exactly ``linalg.yield(add(%in0, %in1))``.

    Structural, per this repo's no-regex rule: the body's op list is walked and its operands are
    compared to the block arguments by IDENTITY. Either operand order is accepted because addition
    commutes; the ``outs`` block argument must be UNREAD (an init-reading body is an accumulation,
    not a bias add). Anything else — a second op, a different arithmetic op, a yield of something
    other than the sum — does not match, and therefore stays unaccounted and refused.
    """
    blocks = list(op.body.blocks)
    if len(blocks) != 1:
        return False
    body = blocks[0]
    body_ops = list(body.ops)
    if len(body_ops) != 2 or len(body.args) != 3:
        return False
    add, yld = body_ops
    if add.name not in ("arith.addf", "arith.addi") or yld.name != "linalg.yield":
        return False
    if len(add.operands) != 2 or len(add.results) != 1:
        return False
    if set(add.operands) != {body.args[0], body.args[1]}:
        return False
    if any(use for use in body.args[2].uses):
        return False
    return list(yld.operands) == [add.results[0]]


def find_bias_epilogues(block, matmuls) -> dict:
    """``{matmul op -> (generic op, bias value)}`` — the bias-add consumers a commit can absorb.

    A ``linalg.generic`` qualifies only when ALL of the following hold, each read out of the IR
    structurally (never matched as text):

    * all-parallel iterators, exactly two inputs, one output, one result;
    * input 0 is the result of one of ``matmuls``, mapped by the IDENTITY map, and that result has
      no other consumer (fusing rewrites the contraction's only readout — a second consumer would
      otherwise silently receive the biased tensor in place of the raw accumulation);
    * input 1 is mapped by the trailing-axis broadcast (see :func:`_is_trailing_broadcast`) and is
      a rank-1 tensor whose extent and element type match the output's last axis;
    * the output map is the identity and the result type equals the contraction's;
    * the body is exactly an ``arith.addf``/``arith.addi`` of the two inputs (see
      :func:`_adds_its_two_inputs`).

    Everything else — a row bias, a multiply, a two-op body, a masked store, a generic with a
    reduction — does NOT match, stays out of the payload, and is therefore refused by
    :func:`_check_payload_complete` rather than quietly dropped.
    """
    from xdsl.dialects.builtin import TensorType
    from xdsl.ir.affine import AffineMap

    mm_set = set(matmuls)
    found: dict = {}
    for op in block.ops:
        if op.name != "linalg.generic":
            continue
        if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
            continue
        if not _iterators_all_parallel(op):
            continue
        maps = [m.data for m in op.indexing_maps]
        if len(maps) != 3:
            continue
        rank = len(list(op.iterator_types))
        ident = AffineMap.identity(rank)
        if maps[0] != ident or maps[2] != ident:
            continue
        if not _is_trailing_broadcast(maps[1], rank):
            continue
        if not _adds_its_two_inputs(op):
            continue
        acc, bias = op.inputs
        owner = getattr(acc, "owner", None)
        if owner not in mm_set or owner in found:
            continue
        if len(list(acc.uses)) != 1:
            continue
        acc_t, out_t = acc.type, op.results[0].type
        if not isinstance(acc_t, TensorType) or acc_t != out_t:
            continue
        bias_t = bias.type
        if not isinstance(bias_t, TensorType):
            continue
        bias_shape = list(bias_t.get_shape())
        out_shape = list(out_t.get_shape())
        if len(bias_shape) != 1 or bias_shape[0] != out_shape[-1]:
            continue
        if bias_t.element_type != out_t.element_type:
            continue
        found[owner] = (op, bias)
    return found


def runtime_tensor_names(block_args, pack_sources) -> dict:
    """Command-buffer tensor name for each tensor value of a rebuilt interface block.

    ``interface.commit`` references its bias BY NAME because the runtime engine has no SSA — its
    environment is keyed by the names the command buffer's resource table declares. Those names are
    minted by ``runtime_lowering.lower_to_runtime``'s naming pre-pass, two stages further down: a
    pack source is ``W``/``W1``.., every other tensor block argument is ``A0``.. in block order.
    This stage has to produce the SAME name, so the rule is stated here in the one form the two
    stages can be compared in, and ``merlin/tests/ir/test_interface_bias_epilogue.py`` pins them
    against each other by lowering all the way to the emitted buffer and reading the bias operand
    back out of it — a drift between the two turns that test red instead of producing a commit that
    names a tensor the engine has never heard of.
    """
    from xdsl.dialects.builtin import TensorType

    names: dict = {}
    n_w = 0
    for v in pack_sources:
        names[v] = "W" if n_w == 0 else "W%d" % n_w
        n_w += 1
    n_a = 0
    for arg in block_args:
        if arg in names:
            continue
        if isinstance(arg.type, TensorType):
            names[arg] = "A%d" % n_a
            n_a += 1
    return names


def payload_ops(block, matmuls) -> list:
    """Everything in ``block`` the interface rebuild materializes as computation.

    One definition for the rebuild loop, the completeness guard, and (once it adopts this) the
    router: contractions, the vector-lane elementwise/activation ops, and the bias-add generics a
    commit absorbs as an epilogue stage. A bias generic belongs here precisely BECAUSE it is
    rebuilt — as a ``bias_add`` stage on its contraction's commit rather than as an op of its own.
    """
    fused = find_bias_epilogues(block, matmuls)
    return (list(matmuls)
            + [op for op in block.ops if _is_vector_lane_op(op)]
            + [g for g, _ in fused.values()])


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

    # An epilogue would be caught above, but a payload whose RESULTS are not returned is a second way
    # to lose computation (the rebuilt return carries the materialized payload results, in order).
    #
    # "Which results" is the TERMINAL ones: payload ops whose results no other payload op consumes.
    # Comparing against every payload result instead is wrong the moment the payload composes — a
    # two-layer chain returns only the last layer, and a matmul feeding an elementwise combine returns
    # only the combine. For a set of independent matmuls the terminal set IS all of them, so this is
    # the same check that shipped, stated in the form that survives composition.
    terminator = src_block.last_op
    if terminator is not None:
        returned = list(terminator.operands)
        payload_set = set(payload)
        expected = [res for op in payload
                    if not any(use.operation in payload_set
                               for r in op.results for use in r.uses)
                    for res in op.results]
        if returned != expected:
            raise LoweringError(
                f"function @{fn.sym_name.data} returns {len(returned)} value(s) that are not exactly "
                f"the {len(expected)} terminal payload result(s), in order — interface materialization "
                "returns the rebuilt results of the payload it found, so the difference would be "
                "dropped.")

    _check_inits_accounted(fn, payload)


def _check_inits_accounted(fn, payload: list) -> None:
    """Fail closed if the rebuild would drop an accumulator INIT.

    The third way to lose computation, and the one the two checks above cannot see: they enumerate
    OPS, and ``outs(%c)`` where ``%c`` is a function argument is a VALUE, not an op. Such a matmul
    has an empty ``unaccounted_ops`` and a terminator that returns exactly the payload result, so it
    lowered cleanly and computed ``A@B`` where the program said ``C + A@B`` — measured as a 3.55
    absolute error against a program the compiler reported no problem with.

    The rebuild has no way to carry it: ``interface.commit``'s epilogue vocabulary is a per-column
    ``bias_add`` (a length-n vector broadcast over rows), not a full-tensor accumulate, so there is
    no correct lowering of a general init to emit. Re-associating it as a post-commit add is not the
    same program either — it moves ``C`` out of the reduction, which is not bit-exact in float, and
    this is the RTL-certified contraction path. So it is refused, naming what would have been lost.
    """
    bad = nonzero_accumulator_inits(payload)
    if not bad:
        return
    op, init = bad[0]
    raise LoweringError(
        f"@{fn.sym_name.data}: '{op.name}' accumulates onto an init that is not provably zero "
        f"({_value_origin(init)}, type {init.type}) — and {len(bad)} such init(s) in all. Interface "
        "materialization rebuilds a contraction as interface.matmul -> interface.commit from a ZERO "
        "accumulator and never reads the source `outs`, so that init would be DROPPED and the "
        "emitted program would compute A@B instead of init + A@B. Only an uninitialized "
        "tensor.empty, a linalg.fill of a zero constant, a zero tensor.splat or an all-zero dense "
        "constant can be proven to contribute nothing; `interface.commit`'s epilogue is a "
        "per-column bias_add, which cannot carry a general init. Refusing instead of compiling a "
        "different program — hoist the init out of the contraction (a bias-add epilogue, or an "
        "explicit add of the contraction's result) or compile this function through the LLVM path.")


def _lower_elementwise_to_interface(fn, src_block, elementwise: list):
    """Materialize an elementwise payload as ``interface.elementwise`` ops.

    Kept separate from the matmul path rather than folded into it. The matmul path is RTL-certified
    on real hardware and its output is byte-compared against a second frontend; a refactor that
    merged the two would put that at risk to save a few lines. The completeness guard is the same
    one, so neither path can silently drop computation.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from .. import interface as i

    payload = [op for op, _ in elementwise]
    dropped = [op.name for op in unaccounted_ops(src_block, payload)]
    if dropped:
        raise LoweringError(
            f"interface materialization would silently drop {len(dropped)} op(s) of the elementwise "
            f"payload: {', '.join(sorted(set(dropped)))}. Failing here instead of compiling a "
            "different program.")
    # Same value-level guard as the matmul path: an op that READS its `outs` is accumulating onto
    # that init, and this rebuild does not carry one either.
    _check_inits_accounted(fn, payload)

    arg_types = [a.type for a in src_block.args]
    blk = Block(arg_types=arg_types)
    value_map = dict(zip(src_block.args, blk.args))

    ops = []
    for op, combine in elementwise:
        lhs, rhs = elementwise_operands(op)
        for operand in (lhs, rhs):
            if operand not in value_map:
                raise LoweringError(
                    "elementwise operand is not a function argument — the interface layer maps "
                    "operands through the function's own arguments, so a computed operand cannot "
                    "be materialized (chain the kernel or re-raise it to an argument)")
        new = i.ElementwiseOp(
            operands=[value_map[lhs], value_map[rhs]],
            result_types=[op.results[0].type],
            properties={"combine": StringAttr(combine)})
        value_map[op.results[0]] = new.out
        ops.append(new)

    terminator = src_block.last_op
    returned = list(terminator.operands) if terminator is not None else []
    expected = [op.results[0] for op, _ in elementwise]
    if returned != expected:
        raise LoweringError(
            f"function @{fn.sym_name.data} returns {len(returned)} value(s) that are not exactly the "
            f"{len(expected)} elementwise result(s), in order — the difference would be dropped.")

    out_types = [v.type for v in expected]
    ops.append(ReturnOp(*[value_map[v] for v in expected]))
    blk.add_ops(ops)
    new_fn = FuncOp(fn.sym_name.data, FunctionType.from_lists(arg_types, out_types), Region([blk]))
    return ModuleOp([new_fn])


def lower_to_interface(module):
    """Build a fresh interface-level module from the scheduled module."""
    if not HAS_XDSL:
        return module
    from xdsl.dialects.builtin import (ArrayAttr, FunctionType, IntegerAttr, ModuleOp,
                                       StringAttr, TensorType)
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

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
    elementwise = find_elementwise(module)
    if not matmuls and not elementwise:
        raise LoweringError("no matmul or elementwise payload to materialize")
    if elementwise and not matmuls:
        # Elementwise-only: its own materializer, deliberately not folded into the matmul path.
        return _lower_elementwise_to_interface(fn, src_block, elementwise)
    # A MIXED payload (a residual add or gating multiply between matmul layers — what a whole model
    # actually looks like) descends through the rebuild loop below, whose vector-lane branch emits
    # each combine as an interface.vector_map threaded through the same value map as the matmuls.
    # That branch is still here; it was the pre-loop refusal in front of it that made every mixed
    # workload unreachable, and with it gemmini's whole-model-on-mesh baseline and the two- and
    # three-layer chain tests.
    # A bias-add consumer of a contraction is REBUILT (as a `bias_add` stage on that contraction's
    # commit), so it is payload, not a drop. The bias tensor is named, not passed by SSA, so refuse
    # here — before anything is emitted — any bias whose name cannot be derived.
    bias_epilogues = find_bias_epilogues(src_block, matmuls)
    src_args = set(src_block.args)
    for _gen, bias_val in bias_epilogues.values():
        if bias_val not in src_args:
            raise LoweringError(
                "a bias-add epilogue's bias tensor is not a function argument, so it has no "
                "command-buffer name — `interface.commit` references its bias BY NAME (the engine "
                "has no SSA) and only the function's own tensor arguments are declared in the "
                "buffer's resource table. A computed or constant bias would have to be named "
                "something the engine never materializes, so it is refused rather than invented: "
                "raise the bias to a function argument.")
    fused_generics = {gen for gen, _ in bias_epilogues.values()}
    payload = payload_ops(src_block, matmuls)
    _check_payload_complete(fn, src_block, payload)

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
    pending_bias = []  # (CommitOp, old bias SSA value) — named once the pack set is final
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
            fused = bias_epilogues.get(op)
            stages = [StringAttr("bias_add")] if fused is not None else []
            commit = i.CommitOp(operands=[imm.acc], result_types=[out_t], properties={
                "epilogue": ArrayAttr(stages),
                "output_dtype": StringAttr(_out_dtype(out_t))})
            ops += [imm, commit]
            value_map[op.results[0]] = commit.out
            if fused is not None:
                # The generic's result IS the committed tensor now; its own materialization is the
                # stage on this commit. The `bias` NAME is filled in after the loop, once every
                # resident_pack is emitted — the naming rule keys off which arguments are pack
                # sources, and the last inline pack is not known until then.
                value_map[fused[0].results[0]] = commit.out
                pending_bias.append((commit, fused[1]))
            continue
        if op in fused_generics:
            # Rebuilt as the `bias_add` stage of its contraction's commit above, and its result is
            # already mapped to that commit — emitting it again would apply the bias twice.
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

    # Name each fused bias with the command-buffer name its argument will carry (see
    # :func:`runtime_tensor_names`). Done here because the pack SOURCES — which decide the naming —
    # are exactly the resident_pack ops now sitting in ``ops``, in the order the later stages walk.
    if pending_bias:
        pack_sources = [o.src for o in ops if isinstance(o, i.ResidentPackOp)]
        names = runtime_tensor_names(blk.args, pack_sources)
        for commit, old_bias in pending_bias:
            new_bias = value_map[old_bias]
            name = names.get(new_bias)
            if name is None:
                raise LoweringError(
                    "a bias-add epilogue's bias tensor has no command-buffer name — refusing "
                    "rather than committing a `bias_add` stage the engine cannot resolve")
            commit.properties["bias"] = StringAttr(name)

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
