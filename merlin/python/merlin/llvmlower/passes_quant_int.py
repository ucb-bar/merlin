"""Integer (W8A8) int8 compute datapath.

Today the int8 capture is WEIGHT-ONLY: ``quant_ext.dequantize_per_channel(i8, scale, zp)``
produces an f32 weight that feeds an f32 contraction — the matmul/attention runs in float.
This module rewrites every contraction (``linalg.matmul`` and the batched ``linalg.generic``
matmuls/attention from ``collapse_overrank_matmul``) into a real integer contraction:

  * each f32 activation operand is dynamically quantized to i8 (symmetric, per output-row:
    ``s = max|x|/127`` reduced over the operand's contraction dim, zero-point 0);
  * a ``dequantize_per_channel`` weight operand is used directly as i8 (its per-channel scale
    carried forward), the dequant dropped;
  * the contraction runs ``i8×i8→i32`` (clang ``-march=rv64gcv`` lowers it to widening
    ``vwmacc``), reusing the original indexing maps / iterator types;
  * the i32 accumulator is requantized by ``acc * prod(operand scales)`` back to f32.

The output stays f32 so downstream (nonlinears) is unchanged — this is the contraction stage
(M1/M2). Weight zero-point must be 0 (symmetric per-channel). Only contractions whose output
map is an identity projection of the parallel iterator dims are converted (matmul + the m2m
attention/batched generics); anything else is left for the f32 fallback (``lower_quant_ext``).
"""
from __future__ import annotations


def _carry_prov(src, **roles) -> None:
    """Copy ``src``'s ``prov.*`` onto every op the rewrite split it into, each tagged with its role.

    The rewrite replaces one captured contraction with an integer contraction PLUS a requant
    epilogue. Both are that same source op, so both must keep its identity. Carrying it only to the
    epilogue — which is what this pass did — left the contraction with no ``prov.fqn`` at all, and
    since a profile's join key falls back to the MLIR op name, every contraction in an int8 model
    collapsed into a single ``linalg.generic`` bucket: measured on deepjscc, 20 of 20 contractions lost
    their key while the 383 untouched elementwise regions kept theirs.

    ``prov.role`` then distinguishes the pieces that share the key, so a consumer joining on
    ``prov.fqn`` can still separate the contraction's own cost from its epilogue's instead of reporting
    the sum as an upper bound on either. Without it, restoring the key would trade one imprecision for
    another. Nothing downstream reads ``prov.*`` for codegen, so this changes identity only.
    """
    from xdsl.dialects.builtin import StringAttr
    prov = {k: v for k, v in src.attributes.items() if k.startswith("prov.")}
    for role, dst in roles.items():
        dst.attributes.update(prov)
        dst.attributes["prov.role"] = StringAttr(role)


def _is_dequant(op) -> bool:
    # ``op`` may be a Block (a block-argument owner) — guard with getattr.
    name = getattr(op, "op_name", None)
    return (getattr(op, "name", None) == "builtin.unregistered" and name is not None
            and name.data == "quant_ext.dequantize_per_channel")


def _is_canonical_matmul(ndim: int, in_maps, out_dims, red_flags) -> bool:
    """Is this exactly ``C[m,n] += A[m,k] * B[k,n]`` -- the convention ``linalg.matmul`` asserts?

    Checked STRUCTURALLY against the contraction view, never assumed from the op's origin. A named
    op is a promise about indexing: emitting one for a contraction whose maps differ (a transposed
    B, a batched or reduced-rank form) would silently change what the op means, which is a
    correctness bug rather than a missed optimization. Everything that does not match keeps the
    generic form and simply stays out of reach of the named-op schedules.
    """
    return (ndim == 3
            and list(red_flags) == [False, False, True]
            and [list(m) for m in in_maps] == [[0, 2], [2, 1]]
            and list(out_dims) == [0, 1])


def _select_targets(targets, select, key=None):
    """Keep only the targets the caller's ``select`` predicate admits.

    ``select`` is ``(op) -> bool`` and defaults to None = admit everything, so the shipped
    datapath is byte-identical. It exists so a caller can run the SAME pass over a RESTRICTED
    set of ops (e.g. only the contractions that descend from one framework construct) and
    measure what the pass's reach — as opposed to its arithmetic — costs. The predicate is the
    caller's policy, never a fact this module assumes.
    """
    if select is None:
        return targets
    k = (lambda t: t) if key is None else key
    return [t for t in targets if select(k(t))]


# --- quantize-before-gather ---------------------------------------------------------------------
#
# WHY THIS EXISTS. model2MLIR rewrites EVERY convolution into `im2col gather + linalg.matmul`
# before merlin sees the module (`prov.conv_path = "im2col_matmul"`; measured 190 such ops in
# deepjscc int8 and 175 in lstmnetvit int8, and ZERO ops carrying `prov.op = "conv2d"` anywhere in
# either). The gather is an all-parallel `linalg.generic` with an EMPTY body -- a pure element copy
# -- whose result is `kh*kw` times larger than the activation it reads (deepjscc `enc.net.1`:
# tensor<1x3x70x70xf32> -> tensor<3x7x7x1x64x64xf32>, 6.55 MB, ~41x the input).
#
# `lower_contraction_int8` then dynamically quantizes whichever contraction operand is f32 -- which
# for an im2col matmul IS that expanded matrix. Per contraction it emits an abs-max reduction and a
# quantize map over 602112 elements instead of over the 14700 the activation actually holds, and
# the gather keeps moving f32. A trip-weighted instruction model of `forward` put 44.4% of deepjscc
# in the gather and 31.1% in activation-quantize+amax, against 18.2% in the vectorized contraction.
#
# THE ALGEBRA. Quantization is ELEMENTWISE, so it commutes exactly with any pure gather:
# `quantize_s(G(A)) == G(quantize_s(A))` for a single shared scale `s`. What blocks the commutation
# today is not the gather, it is the SCALE: the per-parallel-row scheme gives one element of A a
# different scale in every im2col column it appears in, so there is no single `s` to push through.
# A PER-TENSOR scale unblocks it -- which is exactly the scheme `lower_conv_int8`'s own docstring
# argues for ("a conv pixel feeds many outputs, so a per-output-row act scale is ill-defined").
# With it, all three costs move at once: the amax reads A, the quantize writes A-sized i8, and the
# gather moves i8 instead of f32 (4x less traffic for the same trip count).
#
# THE SCALE HAS TO BE amax(G(A)), NOT amax(A), and that is what the coverage analysis is for.
# `amax(A) >= amax(G(A))` whenever the gather skips elements -- deepjscc's stride-2 3x3 convs reach
# column 64 of their 66-wide padded input and never column 65 -- so substituting it would coarsen the
# scale without saying so. Two exact ways to get it without materializing `G(A)`:
#   * FULL COVERAGE PROVEN -- the indexing map and iteration bounds together read every element of A
#     at least once, so `amax(A) == amax(G(A))` and the reduction runs over |A|.
#   * COVERED BOX -- the read set is a gap-free box strictly inside A, so the reduction runs over
#     that `tensor.extract_slice` and is still exactly `amax(G(A))`. Read-only, so bufferization
#     takes the slice as a subview.
# What does NOT work, and was tried: reducing over A THROUGH the gather's own indexing map. It is
# algebraically right and MLIR REJECTS IT -- an iteration dim appearing only inside a compound
# result (`d4 * 2 + d1`) binds no extent, so the op is non-invertible and the module fails to parse
# ("invalid indexing maps are non-invertible", measured on deepjscc `enc.net.4`). A gather whose read
# set has holes (dilation stepping past the window's reach) or whose maps cannot be priced is
# REFUSED, with the reason counted -- never approximated.
#
# WHAT IS NOT EXACT: the per-tensor activation scale is a GENUINE numeric change against the
# shipped per-parallel-row scheme. This rewrite is NOT bit-identical to today's build and must
# never be reported as such. It is default-off behind the `quantize_before_gather` feature.

#: Shape-only ops between a gather and its consumer. Both are pure metadata on the same elements,
#: so an elementwise quantization commutes through them exactly as it does through the gather.
_RESHAPE_OPS = ("tensor.collapse_shape", "tensor.expand_shape")

#: A reshape chain produced by an im2col rewrite is two ops (collapse + expand). The bound exists so
#: an unexpected shape keeps the walk terminating rather than following an arbitrary def chain.
_MAX_RESHAPE_CHAIN = 8


def _bump(report, key: str) -> None:
    if report is not None:
        report[key] = report.get(key, 0) + 1


def _live_uses(val) -> int:
    """Uses held by ops still attached to a block.

    The rewrite DETACHES the contraction it replaces without erasing it, so the detached op keeps
    holding its operands. Counting those would make every chain look live and the dead f32
    expansion would survive -- which is the whole cost this pass exists to remove.
    """
    return sum(1 for u in val.uses if u.operation.parent_block() is not None)


def _yields_only_input(op) -> bool:
    """Body is a pure element copy: exactly ``linalg.yield %arg0``.

    Checked STRUCTURALLY against the block, never inferred from the op's provenance: a body that
    computes anything at all does not commute with quantization, and a `prov` tag is a claim about
    where an op came from, not about what its region does.
    """
    blocks = op.body.blocks
    if len(blocks) != 1:
        return False
    body = list(blocks[0].ops)
    if len(body) != 1 or body[0].name != "linalg.yield":
        return False
    if len(body[0].operands) != 1 or not blocks[0].args:
        return False
    return body[0].operands[0] is blocks[0].args[0]


def _is_pure_gather(op) -> bool:
    """All-parallel, single-input, single-result f32 ``linalg.generic`` that only copies elements."""
    from xdsl.dialects.builtin import TensorType, f32
    from xdsl.dialects.linalg import ops as L
    if getattr(op, "name", None) != "linalg.generic":
        return False
    if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        return False
    if not op.body.blocks:
        return False
    iters = [getattr(a, "data", a) for a in op.iterator_types]
    if not iters or any(i != L.IteratorType.PARALLEL for i in iters):
        return False
    src_t, res_t = op.inputs[0].type, op.results[0].type
    if not (isinstance(src_t, TensorType) and isinstance(res_t, TensorType)):
        return False
    if src_t.element_type != f32 or res_t.element_type != f32:
        return False
    try:
        shp = list(src_t.get_shape()) + list(res_t.get_shape())
    except Exception:
        return False
    if any(d < 0 for d in shp):
        return False                              # dynamic extent: nothing here is priceable
    return _yields_only_input(op)


def _gather_chain(value):
    """``((gather, reshapes), "ok")`` for a contraction operand produced by a pure gather.

    ``reshapes`` is producer-to-consumer ordered. Every op on the path must have exactly ONE live
    use, because the rewrite's win is that the f32 expansion stops existing -- a shared
    intermediate would keep it alive and the rewrite would only ADD an i8 copy. Returns
    ``(None, reason)`` and leaves the operand alone whenever that cannot be established.
    """
    from xdsl.ir import Operation
    cur = value
    reshapes = []
    for _ in range(_MAX_RESHAPE_CHAIN):
        owner = getattr(cur, "owner", None)
        if not isinstance(owner, Operation):
            return None, "block_argument"
        if _live_uses(cur) != 1:
            return None, "shared_value"
        if owner.name in _RESHAPE_OPS:
            if len(owner.results) != 1:
                return None, "multi_result_reshape"
            reshapes.append(owner)
            cur = owner.operands[0]
            continue
        if _is_pure_gather(owner):
            reshapes.reverse()
            return (owner, reshapes), "ok"
        return None, "producer_not_gather"
    return None, "chain_too_long"


def _affine_terms(expr):
    """``([(dim, coeff), ...], const)`` for a sum of ``dim`` / ``dim * const`` terms; else ``(None, 0)``.

    Parsed structurally off the affine expression tree -- no pattern is assumed about how the
    frontend SPELLS a stride (``d4 * 2 + d1`` and ``d1 + 2 * d4`` are the same map). Anything this
    cannot decompose (mod, floordiv, symbol, a product of two dims) returns ``None``, and the caller
    refuses the rewrite rather than guessing what the gather reads.

    A NEGATIVE coefficient is deliberately left unpriced. It appears here only as a weight FLIP
    (`(-d2) + 2`, the 3x3 kernel reversal a transposed convolution needs; two such gathers in
    deepjscc int8, on `dec.model.0` and `dec.model.3`). Teaching this to price it would make the
    coverage proof succeed and put a PER-TENSOR scale on a WEIGHT whose per-row scale is per output
    channel -- a precision regression dressed as a win. The refusal is counted, not silent.
    """
    from xdsl.ir.affine import (AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr,
                                AffineDimExpr)
    terms, const = [], 0
    stack = [(expr, 1)]
    while stack:
        e, mul = stack.pop()
        if isinstance(e, AffineDimExpr):
            terms.append((e.position, mul))
            continue
        if isinstance(e, AffineConstantExpr):
            const += mul * e.value
            continue
        if isinstance(e, AffineBinaryOpExpr) and e.kind == AffineBinaryOpKind.Add:
            stack.append((e.lhs, mul))
            stack.append((e.rhs, mul))
            continue
        if isinstance(e, AffineBinaryOpExpr) and e.kind == AffineBinaryOpKind.Mul:
            if isinstance(e.rhs, AffineConstantExpr):
                stack.append((e.lhs, mul * e.rhs.value))
                continue
            if isinstance(e.lhs, AffineConstantExpr):
                stack.append((e.rhs, mul * e.lhs.value))
                continue
        return None, 0
    return terms, const


def _gather_coverage(gather) -> "tuple[str | None, list[int], str]":
    """Which elements of its source does the gather read? ``(kind, covered_extents, reason)``.

    ``kind`` is ``"full"`` (every element of A is read at least once, so ``amax(A) == amax(G(A))``),
    ``"slice"`` (the read set is a gap-free box strictly inside A, so ``amax`` over that box is still
    EXACTLY ``amax(G(A))``), or ``None`` (cannot be established -- the caller must not rewrite).

    The distinction is a correctness one, not a tuning one: a stride-2 3x3 window over a 66-wide
    padded input reaches column 64 and never column 65, so ``amax(A) >= amax(G(A))`` there and using
    it would silently coarsen the scale. Reducing over the gather's OWN indexing map would be the
    obvious alternative and is NOT AVAILABLE: an iteration dim that appears only inside a compound
    result binds no extent, and MLIR rejects the op outright ("invalid indexing maps are
    non-invertible") -- measured, that lowering fails to parse. Hence the explicit box.

    Fails closed everywhere: any expression, bound or rank this cannot price returns ``None``.
    """
    from xdsl.ir.affine import AffineDimExpr
    maps = list(gather.indexing_maps)
    if len(maps) != 2:
        return None, [], "unexpected_map_count"
    in_map, out_map = maps[0].data, maps[-1].data
    ndim = in_map.num_dims
    out_shape = list(gather.results[0].type.get_shape())
    if len(out_map.results) != len(out_shape):
        return None, [], "output_rank_mismatch"
    bounds: list = [None] * ndim
    for pos, r in enumerate(out_map.results):
        if not isinstance(r, AffineDimExpr):
            return None, [], "output_map_not_projection"
        prev = bounds[r.position]
        if prev is not None and prev != out_shape[pos]:
            return None, [], "inconsistent_bound"
        bounds[r.position] = out_shape[pos]
    if any(b is None or b < 1 for b in bounds):
        return None, [], "unbounded_iteration_dim"
    src_shape = list(gather.inputs[0].type.get_shape())
    if len(in_map.results) != len(src_shape):
        return None, [], "input_rank_mismatch"
    extents = []
    used: set = set()
    for pos, r in enumerate(in_map.results):
        terms, const = _affine_terms(r)
        if terms is None or const != 0 or not terms or any(c <= 0 for _, c in terms):
            return None, [], "unpriceable_index_expr"
        if any(d >= ndim for d, _ in terms):
            return None, [], "index_expr_out_of_range"
        # EACH ITERATION DIM MAY DRIVE AT MOST ONE SOURCE AXIS. The per-axis reasoning below is only
        # valid when the read set is a product of per-axis ranges; a dim shared between two axes ties
        # them together and the set is a diagonal, not a box. `(d0, d0)` over a square source reads
        # only the diagonal, yet every axis would price as fully covered -- which would substitute
        # `amax(A)` for a strictly smaller `amax(G(A))` and coarsen the scale with nothing to show
        # for it. A dim repeated inside ONE axis (`d0 + d0`) breaks the same reach arithmetic.
        dims = [d for d, _ in terms]
        if len(set(dims)) != len(dims) or used & set(dims):
            return None, [], "coupled_iteration_dim"
        used |= set(dims)
        # gap-free? sweep the terms cheapest-coefficient first: a term of coefficient c extends a
        # contiguous prefix only if c is within one of the reach already established. A dilation that
        # steps past that reach leaves holes, and the box below would then be a superset of what the
        # gather reads -- a coarser scale, so it is refused instead.
        reach = 0
        for d, c in sorted(terms, key=lambda tc: tc[1]):
            if c > reach + 1:
                return None, [], "strided_holes"
            reach += c * (bounds[d] - 1)
        if reach > src_shape[pos] - 1:
            return None, [], "index_exceeds_extent"
        extents.append(reach + 1)
    if extents == src_shape:
        return "full", extents, "full_coverage"
    return "slice", extents, "covered_box"


def _emit_prequant_gather(gather, reshapes):
    """Build ``quantize(A) -> i8 gather -> i8 reshapes`` and the per-tensor scale that prices it.

    Returns ``(i8_value, scale_ssa, pre_ops, dead_ops, mode)``. ``pre_ops`` are to be inserted before
    the contraction; ``dead_ops`` are the f32 chain this replaces, erased once nothing live reads it.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, TensorType, f32, i8
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    src = gather.inputs[0]
    ash = list(src.type.get_shape())
    r = len(ash)
    sc_t = TensorType(f32, [])
    kind, extents, _why = _gather_coverage(gather)
    if kind is None:
        return None
    mode = "source_amax" if kind == "full" else "slice_amax"
    pre_amax = []
    # The abs-max reads exactly what the gather reads. When the gather does not reach every element
    # (a stride-2 window over a 66-wide padded input stops at column 64), reduce over the covered BOX
    # instead of over A -- `amax(A) >= amax(G(A))` there, and taking the larger one would coarsen the
    # scale without saying so. Read-only, so bufferization takes it as a subview rather than a copy.
    amax_src = src
    if kind == "slice":
        sl = tensor.ExtractSliceOp.from_static_parameters(src, [0] * r, extents, [1] * r)
        sl.attributes.update({k: v for k, v in gather.attributes.items() if k.startswith("prov.")})
        pre_amax.append(sl)
        amax_src = sl.results[0]

    # --- per-TENSOR abs-max (rank-0 result) ---
    amx_e = tensor.EmptyOp((), sc_t)
    zero_f = arith.ConstantOp(FloatAttr(0.0, f32))
    amx_f = L.FillOp(inputs=[zero_f.results[0]], outputs=[amx_e.results[0]], res=[sc_t])
    rb = Block(arg_types=[f32, f32]); a_in, acc_in = rb.args
    ab = mathd.AbsFOp(a_in); mx = arith.MaximumfOp(ab.result, acc_in)
    rb.add_ops([ab, mx, L.YieldOp(mx.result)])
    in_map = amap(r, AffineMap.identity(r).results)
    amx = L.GenericOp(inputs=(amax_src,), outputs=(amx_f.results[0],), body=Region(rb),
                      indexing_maps=ArrayAttr([in_map, amap(r, [])]),
                      iterator_types=ArrayAttr([L.IteratorTypeAttr(red)] * r),
                      result_types=(sc_t,))
    # --- s = amax / 127 ---
    sc_e = tensor.EmptyOp((), sc_t); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
    sb = Block(arg_types=[f32, f32]); s_in, _unused = sb.args
    sd = arith.DivfOp(s_in, c127.results[0]); sb.add_ops([sd, L.YieldOp(sd.result)])
    s_a = L.GenericOp(inputs=(amx.results[0],), outputs=(sc_e.results[0],), body=Region(sb),
                      indexing_maps=ArrayAttr([amap(0, []), amap(0, [])]),
                      iterator_types=ArrayAttr([]), result_types=(sc_t,))
    # --- quantize the SOURCE (not the expansion): q = fptosi(clamp(roundeven(x/s), +-127)) ---
    # Quantizes ALL of A, including elements outside the covered box, which may therefore saturate at
    # +-127. That is sound because the gather never reads them: the only consumer of this i8 tensor is
    # the gather rebuilt below, whose read set is exactly the box the scale was derived from.
    ident_r = AffineMap.identity(r).results
    i8_src_t = TensorType(i8, ash)
    q_e = tensor.EmptyOp((), i8_src_t); c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
    qb = Block(arg_types=[f32, f32, i8]); xv, sv, _q = qb.args
    q1 = arith.DivfOp(xv, sv); q2 = mathd.RoundEvenOp(q1.result)
    q3 = arith.MinimumfOp(q2.result, c127.results[0])
    q4 = arith.MaximumfOp(q3.result, c127n.results[0]); q5 = arith.FPToSIOp(q4.result, i8)
    qb.add_ops([q1, q2, q3, q4, q5, L.YieldOp(q5.result)])
    q = L.GenericOp(inputs=(src, s_a.results[0]), outputs=(q_e.results[0],), body=Region(qb),
                    indexing_maps=ArrayAttr([amap(r, ident_r), amap(r, []), amap(r, ident_r)]),
                    iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * r),
                    result_types=(i8_src_t,))
    # --- the SAME gather, now moving i8 (identical maps, iterators and empty body) ---
    g_t = TensorType(i8, list(gather.results[0].type.get_shape()))
    g_e = tensor.EmptyOp((), g_t)
    gb = Block(arg_types=[i8, i8]); gb.add_ops([L.YieldOp(gb.args[0])])
    g8 = L.GenericOp(inputs=(q.results[0],), outputs=(g_e.results[0],), body=Region(gb),
                     indexing_maps=gather.indexing_maps, iterator_types=gather.iterator_types,
                     result_types=(g_t,))
    g8.attributes.update({k: v for k, v in gather.attributes.items() if k.startswith("prov.")})
    pre = pre_amax + [amx_e, zero_f, amx_f, amx, sc_e, c127, s_a, q_e, c127n, q, g_e, g8]
    # The quantization ops belong to the region the gather came from, so they carry ITS identity with
    # a role of their own. Without this a profile joining on `prov.fqn` falls back to the MLIR op name
    # and every activation quantize in the model collapses into one `linalg.generic` bucket.
    _carry_prov(gather, act_amax=amx, act_scale=s_a, act_quantize=q, gather=g8)

    # --- the same shape metadata, on i8 ---
    cur = g8.results[0]
    for rs in reshapes:
        rt = TensorType(i8, list(rs.results[0].type.get_shape()))
        new = type(rs).create(operands=[cur, *list(rs.operands[1:])], result_types=[rt],
                              properties=dict(rs.properties), attributes=dict(rs.attributes))
        pre.append(new)
        cur = new.results[0]

    dead = [gather, *reshapes]
    out_owner = getattr(gather.outputs[0], "owner", None)
    if getattr(out_owner, "name", None) == "tensor.empty":
        dead.insert(0, out_owner)
    return cur, s_a.results[0], pre, dead, mode


def lower_contraction_int8(module, *, named_contraction: bool = False,
                           select=None, prequant_gather: bool = False,
                           report_out: "dict | None" = None) -> int:
    """Rewrite f32 contractions into i8×i8→i32 + dynamic act-quant + requant. Returns count.

    ``prequant_gather`` (default False, so the shipped datapath is byte-identical) turns on the
    ``quantize_before_gather`` feature: when a contraction's f32 activation operand is produced by a
    pure data-movement op, quantize BEFORE that op with a per-tensor scale instead of after it with a
    per-parallel-row one. See the block comment above ``_RESHAPE_OPS`` for the algebra, the exactness
    argument for both abs-max modes, and what this does NOT preserve (it is a genuine numeric change
    against the per-row scheme -- not bit-identical, and it must not be reported as such).

    ``report_out``, when given, is filled with counters: ``prequant_gather_rewrites``, one
    ``prequant_gather_mode_*`` per abs-max mode taken, one ``prequant_gather_refused_*`` per reason an
    operand was left alone, and ``prequant_gather_erased_ops`` for the f32 expansion chain removed.
    A rewrite that fires zero times has to be VISIBLE rather than silent, which is the whole reason
    the refusal reasons are counted instead of being an early ``continue``.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, TensorType,
                                       f32, i8, i32)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap, AffineDimExpr

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    def contraction_view(op):
        """(ndim, in_maps[list of dim-index lists], out_dims, iters[bool reduction])
        for a linalg.matmul or a mul(f|i)+add(f|i)-bodied reduction generic; else None."""
        if op.name == "linalg.matmul":
            # standard 2-D: (m,n,k); lhs (m,k), rhs (k,n), out (m,n)
            return 3, [[0, 2], [2, 1]], [0, 1], [False, False, True]
        if op.name != "linalg.generic" or len(op.inputs) != 2:
            return None
        # conv / non-projection contractions (input map carries a stride/dilation affine
        # expression like (d2*16+d5)) are handled by lower_conv_int8, which preserves the
        # exact maps — the matmul rebuild below would drop the compound terms.
        if any(not isinstance(r, AffineDimExpr)
               for m in op.indexing_maps for r in m.data.results):
            return None
        iters = [getattr(a, "data", a) for a in op.iterator_types]
        if not any(i == red for i in iters):
            return None
        bn = [b.name for b in op.body.blocks[0].ops]
        if not (("arith.mulf" in bn and "arith.addf" in bn)
                or ("arith.muli" in bn and "arith.addi" in bn)):
            return None
        maps = list(op.indexing_maps)
        ndim = maps[0].data.num_dims
        def dimlist(m):
            return [r.position for r in m.data.results if isinstance(r, AffineDimExpr)]
        in_maps = [dimlist(maps[0]), dimlist(maps[1])]
        out_dims = dimlist(maps[-1])
        red_flags = [it == red for it in iters]
        # only identity-projection outputs (out = the parallel dims, in order)
        par_dims = [d for d in range(ndim) if not red_flags[d]]
        if out_dims != par_dims:
            return None
        return ndim, in_maps, out_dims, red_flags

    targets = []
    for op in module.walk():
        view = contraction_view(op)
        if view is None:
            continue
        out_t = op.results[0].type
        if not (isinstance(out_t, TensorType) and out_t.element_type == f32):
            continue
        if not all(isinstance(i.type, TensorType) for i in op.inputs):
            continue
        # at least one f32 operand to quantize (else nothing to do)
        if not any(i.type.element_type == f32 or _is_dequant(i.owner) for i in op.inputs):
            continue
        targets.append((op, view))
    targets = _select_targets(targets, select, key=lambda t: t[0])

    n = 0
    dead_chain_ops: list = []
    for op, (ndim, in_maps, out_dims, red_flags) in targets:
        block = op.parent_block()
        out_t = op.results[0].type
        P = len(out_dims)                                # output / parallel rank
        d_par = AffineMap.identity(P).results
        # output-position of each iterator dim (parallel dims only)
        pos_of = {d: out_dims.index(d) for d in out_dims}
        pre = []                                         # ops to insert before `op`

        i8_inputs = []
        scale_vals = []                                  # (ssa, [output-positions]) per operand
        for oi, operand in enumerate(op.inputs):
            dims = in_maps[oi]                            # iterator dims this operand indexes
            red_pos = [p for p, d in enumerate(dims) if red_flags[d]]   # positions in operand
            par_pos = [p for p, d in enumerate(dims) if not red_flags[d]]
            par_outpos = [pos_of[dims[p]] for p in par_pos]             # -> requant out dims
            shp = list(operand.type.get_shape())
            sc_shape = [shp[p] for p in par_pos]
            sc_t = TensorType(f32, sc_shape)
            r = len(shp)

            if _is_dequant(operand.owner):
                # weight: use i8 source directly; per-channel scale (already over par dims)
                deq = operand.owner
                i8_inputs.append(deq.operands[0])
                scale_vals.append((deq.operands[1], par_outpos))
                continue
            if operand.type.element_type != f32:
                i8_inputs.append(operand)                # already integer
                scale_vals.append((None, par_outpos))
                continue

            # QUANTIZE BEFORE THE GATHER, when the operand is one. `quantize(G(A)) == G(quantize(A))`
            # holds for a pure gather and a single shared scale, so the abs-max, the quantize and the
            # data movement all move off the expanded matrix and onto the activation itself. Refuses
            # (counting why) rather than approximating; default-off.
            if prequant_gather:
                chain, why = _gather_chain(operand)
                if chain is None:
                    _bump(report_out, f"prequant_gather_refused_{why}")
                    built = None
                else:
                    g_op, g_reshapes = chain
                    built = _emit_prequant_gather(g_op, g_reshapes)
                    if built is None:
                        # coverage unprovable: leave the operand on the per-row path rather than
                        # quantize it against a scale derived from elements the gather never reads.
                        _bump(report_out,
                              f"prequant_gather_refused_{_gather_coverage(g_op)[2]}")
                if built is not None:
                    i8_val, s_ssa, g_pre, g_dead, g_mode = built
                    pre += g_pre
                    dead_chain_ops.extend(g_dead)
                    i8_inputs.append(i8_val)
                    scale_vals.append((s_ssa, []))          # PER-TENSOR: no output dim to index by
                    _bump(report_out, "prequant_gather_rewrites")
                    _bump(report_out, f"prequant_gather_mode_{g_mode}")
                    continue

            # dynamic per-(parallel-row) quant of an f32 activation
            ident_r = AffineMap.identity(r).results
            red_iters = ArrayAttr([L.IteratorTypeAttr(red if p in red_pos else par)
                                   for p in range(r)])
            sc_map_in = amap(r, ident_r)
            sc_map_out = amap(r, [ident_r[p] for p in par_pos])
            # abs-max
            amx_e = tensor.EmptyOp((), sc_t)
            zero_f = arith.ConstantOp(FloatAttr(0.0, f32))
            amx_f = L.FillOp(inputs=[zero_f.results[0]], outputs=[amx_e.results[0]], res=[sc_t])
            rb = Block(arg_types=[f32, f32]); a_in, acc_in = rb.args
            ab = mathd.AbsFOp(a_in); mx = arith.MaximumfOp(ab.result, acc_in)
            rb.add_ops([ab, mx, L.YieldOp(mx.result)])
            amx = L.GenericOp(inputs=(operand,), outputs=(amx_f.results[0],), body=Region(rb),
                              indexing_maps=ArrayAttr([sc_map_in, sc_map_out]),
                              iterator_types=red_iters, result_types=(sc_t,))
            # s = amax / 127
            sc_e = tensor.EmptyOp((), sc_t); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
            ident_p = AffineMap.identity(len(sc_shape)).results
            sb = Block(arg_types=[f32, f32]); s_in, _ = sb.args
            sd = arith.DivfOp(s_in, c127.results[0]); sb.add_ops([sd, L.YieldOp(sd.result)])
            sc = L.GenericOp(inputs=(amx.results[0],), outputs=(sc_e.results[0],), body=Region(sb),
                             indexing_maps=ArrayAttr([amap(len(sc_shape), ident_p),
                                                      amap(len(sc_shape), ident_p)]),
                             iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * len(sc_shape)),
                             result_types=(sc_t,))
            # quantize: q = fptosi(clamp(roundeven(x/s), -127, 127))
            i8_t = TensorType(i8, shp)
            q_e = tensor.EmptyOp((), i8_t); c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
            qb = Block(arg_types=[f32, f32, i8]); xv, sv, _ = qb.args
            q1 = arith.DivfOp(xv, sv); q2 = mathd.RoundEvenOp(q1.result)
            q3 = arith.MinimumfOp(q2.result, c127.results[0])
            q4 = arith.MaximumfOp(q3.result, c127n.results[0]); q5 = arith.FPToSIOp(q4.result, i8)
            qb.add_ops([q1, q2, q3, q4, q5, L.YieldOp(q5.result)])
            q = L.GenericOp(inputs=(operand, sc.results[0]), outputs=(q_e.results[0],),
                            body=Region(qb),
                            indexing_maps=ArrayAttr([amap(r, ident_r), sc_map_out, amap(r, ident_r)]),
                            iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * r),
                            result_types=(i8_t,))
            pre += [amx_e, zero_f, amx_f, amx, sc_e, c127, sc, q_e, c127n, q]
            i8_inputs.append(q.results[0])
            scale_vals.append((sc.results[0], par_outpos))

        # i8×i8→i32 contraction (reuse maps/iters)
        acc_t = TensorType(i32, list(out_t.get_shape()))
        acc_e = tensor.EmptyOp((), acc_t); zi = arith.ConstantOp.from_int_and_width(0, 32)
        acc_f = L.FillOp(inputs=[zi.results[0]], outputs=[acc_e.results[0]], res=[acc_t])
        mm_maps = ArrayAttr([amap(ndim, [AffineDimExpr(d) for d in in_maps[0]]),
                             amap(ndim, [AffineDimExpr(d) for d in in_maps[1]]),
                             amap(ndim, [AffineDimExpr(d) for d in out_dims])])
        mm_iters = ArrayAttr([L.IteratorTypeAttr(red if red_flags[d] else par) for d in range(ndim)])
        mb = Block(arg_types=[i8, i8, i32]); av, bv, acc = mb.args
        ea = arith.ExtSIOp(av, i32); eb = arith.ExtSIOp(bv, i32)
        pm = arith.MuliOp(ea.result, eb.result); pa = arith.AddiOp(pm.result, acc)
        mb.add_ops([ea, eb, pm, pa, L.YieldOp(pa.result)])
        # NAMED OP vs GENERIC — this choice decides whether the schedule levers exist at all.
        #
        # Building the contraction as a `linalg.generic` erases the named form the whole transform
        # layer keys on: `impr_features` matches `linalg.matmul` / `linalg.batch_matmul` in 39
        # places, and `transform.structured.match` on a name nothing carries yields an EMPTY handle,
        # which makes every op downstream of it a vacuous no-op. Measured on small_llama_int8:
        # 15 linalg.matmul before this pass, 0 after -- so the entire register-blocking /
        # accumulator-resident family silently did nothing on int8, while still reporting as
        # "applied". An 87-fork beam search over those levers emitted only 21 distinct binaries and
        # could not improve on the two generic-level levers.
        #
        # A mixed-type `linalg.matmul` (ins i8, i8 / outs i32) is legal and carries exactly these
        # semantics -- its region is the extsi/muli/addi body built above -- so for the CANONICAL
        # 2-D contraction we emit the named op and the levers become reachable. Anything else
        # (batch, conv, transposed or otherwise non-canonical maps) keeps the generic: a named op
        # asserts an indexing convention, and claiming one the op does not have would be a
        # correctness bug, not a missed optimization.
        if named_contraction and _is_canonical_matmul(ndim, in_maps, out_dims, red_flags):
            i8mm = L.MatmulOp(inputs=tuple(i8_inputs), outputs=(acc_f.results[0],),
                              res=(acc_t,))
        else:
            i8mm = L.GenericOp(inputs=tuple(i8_inputs), outputs=(acc_f.results[0],), body=Region(mb),
                               indexing_maps=mm_maps, iterator_types=mm_iters, result_types=(acc_t,))

        # requant: out_f32 = sitofp(acc) * scale_lhs * scale_rhs
        sc_inputs, sc_maps = [], [amap(P, d_par)]
        for ssa, outpos in scale_vals:
            if ssa is None:
                continue
            sc_inputs.append(ssa)
            sc_maps.append(amap(P, [d_par[p] for p in outpos]))
        out_e = tensor.EmptyOp((), out_t)
        wb = Block(arg_types=[i32] + [f32] * len(sc_inputs) + [f32])
        accv = wb.args[0]; svs = wb.args[1:1 + len(sc_inputs)]
        cur = arith.SIToFPOp(accv, f32); ops_w = [cur]
        for sv in svs:
            m = arith.MulfOp(cur.result, sv); ops_w.append(m); cur = m
        wb.add_ops(ops_w + [L.YieldOp(cur.result)])
        requant = L.GenericOp(inputs=(i8mm.results[0], *sc_inputs), outputs=(out_e.results[0],),
                              body=Region(wb),
                              indexing_maps=ArrayAttr(sc_maps + [amap(P, d_par)]),
                              iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * P),
                              result_types=(out_t,))
        _carry_prov(op, contraction=i8mm, requant=requant)

        for new in pre + [acc_e, zi, acc_f, i8mm, out_e, requant]:
            block.insert_op_before(new, op)
        op.results[0].replace_all_uses_with(requant.results[0])
        block.detach_op(op)
        n += 1

    # ERASE the f32 expansion the rewrite replaced. Not cosmetic: the ENTIRE point of quantizing
    # before the gather is that the f32 matrix stops being materialized, and nothing else in this
    # pipeline deletes it -- leaving it would keep every byte of the traffic while ADDING an i8 copy,
    # i.e. the lever would measure as a regression and read as a bad idea. Consumer-first, and only
    # once no op still attached to a block reads it, so a shared producer survives untouched.
    for cand in reversed(dead_chain_ops):
        if not cand.results or cand.parent_block() is None:
            continue
        if any(_live_uses(res) for res in cand.results):
            continue
        cand.detach()
        cand.erase(safe_erase=False)
        _bump(report_out, "prequant_gather_erased_ops")
    return n


# Back-compat alias (M1 name).
lower_matmul_int8 = lower_contraction_int8


def lower_conv_int8(module, *, select=None, report_out: "dict | None" = None) -> int:
    """Rewrite an f32 conv (``linalg.generic`` whose input map carries stride/dilation affine
    expressions, ``prov.op = conv2d``) into an i8×i8→i32 conv: the activation is dynamically
    quantized **per-tensor** (a conv pixel feeds many outputs, so a per-output-row act scale is
    ill-defined — one scalar ``s_a = max|x|/127`` over the whole activation), the weight is used
    as i8 with its per-output-channel scale, the contraction keeps the **exact** original
    indexing maps + iterator types (so the RVV schedule can tile/vectorize it), and the i32
    accumulator is requantized by ``acc * s_a * s_w[out_channel]``. Returns count.

    **THIS PASS IS UNREACHABLE ON EVERY CONVOLUTIONAL MODEL IN THIS FLEET, AND SAYS SO.** Its
    predicate wants a 2-input ``linalg.generic`` whose indexing maps carry a compound affine term.
    model2MLIR never produces one: it rewrites every conv into ``im2col gather + linalg.matmul``
    ahead of merlin (``prov.conv_path = "im2col_matmul"``; measured 190 such ops in deepjscc int8,
    175 in lstmnetvit int8, and ZERO ops tagged ``prov.op = "conv2d"`` in either). The compound term
    that this pass keys on lives in the GATHER, which has one input and an empty body, so nothing
    matches and the pass has never fired on any conv here. ``contraction_view``'s comment that convs
    "are handled by lower_conv_int8" describes an intent, not a fact.

    It is kept rather than deleted because its ARITHMETIC is the correct scheme and is the thing
    ``quantize_before_gather`` reuses (a per-tensor activation scale computed before the expansion);
    a frontend that hands us a fused conv would put it straight back in reach. But a pass that cannot
    fire must not be indistinguishable from one that had nothing to do, so it now COUNTS what it
    scanned into ``report_out`` (``generics_scanned``, ``compound_map_generics``, ``conv_prov_ops``,
    ``lowered``) and prints one ``INERT`` line when a module full of convolution provenance yields
    zero candidates. ``merlin/tests/ir/test_quantize_before_gather.py`` asserts that on a real bundle,
    so the deadness is a gated fact instead of a comment."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, TensorType,
                                       f32, i8, i32)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap, AffineDimExpr

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    targets = []
    n_generics = n_compound = n_windowed = n_conv_prov = 0
    for op in module.walk():
        # Counted only on ops the FRONTEND tagged. `prov.role` is added exclusively by this file's
        # own rewrites, so excluding it keeps the diagnostic reporting the module as captured rather
        # than inflating with the ops an earlier quant pass just minted (measured: 190 -> 228 on
        # deepjscc once `quantize_before_gather` had run).
        if "prov.role" not in op.attributes and (
                "prov.conv_path" in op.attributes
                or getattr(op.attributes.get("prov.op"), "data", "").startswith("conv")):
            n_conv_prov += 1
        if op.name != "linalg.generic" or len(op.inputs) != 2 or not op.body.blocks:
            continue
        n_generics += 1
        maps = list(op.indexing_maps)
        if not any(not isinstance(r, AffineDimExpr)
                   for m in maps for r in m.data.results):
            continue                                     # not a conv (no compound map term)
        n_compound += 1
        # "compound" is not the same as "windowed". The predicate above also admits a BROADCAST,
        # whose non-dim map result is a bare constant index (`(d0, d1, 0, 0)`) with no stride in it
        # -- on deepjscc int8 all four of its candidates are exactly that (elementwise mul/add on
        # `*.res_list.1`), and each is then dropped for having a one-op body. Counting them as
        # near-misses would make this pass look like it merely missed by a hair on a model where it
        # has no candidate at all, so the window count is kept separately.
        # Fails OPEN on the diagnostic side: an expression `_affine_terms` cannot decompose (mod,
        # floordiv) counts as windowed, so an unrecognised window can never make this pass claim it
        # was inert when it was merely unlucky.
        if any(not isinstance(r, AffineDimExpr) and _affine_terms(r)[0] != []
               for m in maps for r in m.data.results):
            n_windowed += 1
        out_t = op.results[0].type
        if not (isinstance(out_t, TensorType) and out_t.element_type == f32):
            continue
        bn = [b.name for b in op.body.blocks[0].ops]
        if not (("arith.mulf" in bn and "arith.addf" in bn)
                or ("arith.muli" in bn and "arith.addi" in bn)):
            continue
        # operand 0 = activation (compound map); operand 1 = weight (dequant i8, or plain f32 —
        # torchao leaves conv weights f32, so we dynamically per-output-channel quantize them).
        act, wt = op.inputs[0], op.inputs[1]
        if not (isinstance(act.type, TensorType) and act.type.element_type == f32):
            continue
        if not (_is_dequant(wt.owner)
                or (isinstance(wt.type, TensorType) and wt.type.element_type == f32)):
            continue
        targets.append(op)
    targets = _select_targets(targets, select)

    n = 0
    for op in targets:
        block = op.parent_block()
        maps = list(op.indexing_maps)
        iters = [getattr(a, "data", a) for a in op.iterator_types]
        ndim = maps[0].data.num_dims
        red_flags = [it == red for it in iters]
        act, wt = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        ash = list(act.type.get_shape()); r = len(ash)
        pre = []

        # --- weight -> i8 + per-output-channel scale s_w ---
        wt_dims = [rr.position for rr in maps[1].data.results]   # iterator dim per weight axis
        wt_keep = [ax for ax, d in enumerate(wt_dims) if not red_flags[d]]   # parallel (out-ch) axes
        if _is_dequant(wt.owner):
            deq = wt.owner
            wt_i8, s_w = deq.operands[0], deq.operands[1]        # already i8 + per-channel scale
        else:
            # dynamically per-output-channel quantize the f32 weight (reduce over its kernel axes)
            wsh = list(wt.type.get_shape()); wr = len(wsh)
            ws_shape = [wsh[ax] for ax in wt_keep]
            ws_t = TensorType(f32, ws_shape)
            id_w = AffineMap.identity(wr).results
            keep_res = [id_w[ax] for ax in wt_keep]
            w_amx_e = tensor.EmptyOp((), ws_t); wzero = arith.ConstantOp(FloatAttr(0.0, f32))
            w_amx_f = L.FillOp(inputs=[wzero.results[0]], outputs=[w_amx_e.results[0]], res=[ws_t])
            wrb = Block(arg_types=[f32, f32]); wa, wacc = wrb.args
            wab = mathd.AbsFOp(wa); wmx = arith.MaximumfOp(wab.result, wacc)
            wrb.add_ops([wab, wmx, L.YieldOp(wmx.result)])
            w_amx = L.GenericOp(inputs=(wt,), outputs=(w_amx_f.results[0],), body=Region(wrb),
                                indexing_maps=ArrayAttr([amap(wr, id_w), amap(wr, keep_res)]),
                                iterator_types=ArrayAttr(
                                    [L.IteratorTypeAttr(par if ax in wt_keep else red)
                                     for ax in range(wr)]),
                                result_types=(ws_t,))
            ws_e = tensor.EmptyOp((), ws_t); wc127 = arith.ConstantOp(FloatAttr(127.0, f32))
            id_k = AffineMap.identity(len(ws_shape)).results
            wsb = Block(arg_types=[f32, f32]); ws_in, _ = wsb.args
            wsd = arith.DivfOp(ws_in, wc127.results[0]); wsb.add_ops([wsd, L.YieldOp(wsd.result)])
            s_w_g = L.GenericOp(inputs=(w_amx.results[0],), outputs=(ws_e.results[0],),
                                body=Region(wsb),
                                indexing_maps=ArrayAttr([amap(len(ws_shape), id_k),
                                                         amap(len(ws_shape), id_k)]),
                                iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * len(ws_shape)),
                                result_types=(ws_t,))
            s_w = s_w_g.results[0]
            wi8_t = TensorType(i8, wsh); wq_e = tensor.EmptyOp((), wi8_t)
            wc127n = arith.ConstantOp(FloatAttr(-127.0, f32))
            wqb = Block(arg_types=[f32, f32, i8]); wxv, wsv, _ = wqb.args
            wq1 = arith.DivfOp(wxv, wsv); wq2 = mathd.RoundEvenOp(wq1.result)
            wq3 = arith.MinimumfOp(wq2.result, wc127.results[0])
            wq4 = arith.MaximumfOp(wq3.result, wc127n.results[0]); wq5 = arith.FPToSIOp(wq4.result, i8)
            wqb.add_ops([wq1, wq2, wq3, wq4, wq5, L.YieldOp(wq5.result)])
            wq = L.GenericOp(inputs=(wt, s_w_g.results[0]), outputs=(wq_e.results[0],),
                             body=Region(wqb),
                             indexing_maps=ArrayAttr([amap(wr, id_w), amap(wr, keep_res),
                                                      amap(wr, id_w)]),
                             iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * wr),
                             result_types=(wi8_t,))
            wt_i8 = wq.results[0]
            pre += [w_amx_e, wzero, w_amx_f, w_amx, ws_e, wc127, s_w_g, wq_e, wc127n, wq]

        # --- per-tensor activation scale: s_a = max|act| / 127 (rank-0 tensor<f32>) ---
        sc_t = TensorType(f32, [])
        ident_r = AffineMap.identity(r).results
        amx_e = tensor.EmptyOp((), sc_t); zero_f = arith.ConstantOp(FloatAttr(0.0, f32))
        amx_f = L.FillOp(inputs=[zero_f.results[0]], outputs=[amx_e.results[0]], res=[sc_t])
        rb = Block(arg_types=[f32, f32]); a_in, acc_in = rb.args
        ab = mathd.AbsFOp(a_in); mx = arith.MaximumfOp(ab.result, acc_in)
        rb.add_ops([ab, mx, L.YieldOp(mx.result)])
        amx = L.GenericOp(inputs=(act,), outputs=(amx_f.results[0],), body=Region(rb),
                          indexing_maps=ArrayAttr([amap(r, ident_r), amap(r, [])]),
                          iterator_types=ArrayAttr([L.IteratorTypeAttr(red)] * r),
                          result_types=(sc_t,))
        sc_e = tensor.EmptyOp((), sc_t); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
        sb = Block(arg_types=[f32, f32]); s_in, _ = sb.args
        sd = arith.DivfOp(s_in, c127.results[0]); sb.add_ops([sd, L.YieldOp(sd.result)])
        s_a = L.GenericOp(inputs=(amx.results[0],), outputs=(sc_e.results[0],), body=Region(sb),
                          indexing_maps=ArrayAttr([amap(0, []), amap(0, [])]),
                          iterator_types=ArrayAttr([]), result_types=(sc_t,))
        # --- quantize activation: q = fptosi(clamp(roundeven(x/s_a), ±127)) ---
        i8_t = TensorType(i8, ash); q_e = tensor.EmptyOp((), i8_t)
        c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
        qb = Block(arg_types=[f32, f32, i8]); xv, sv, _ = qb.args
        q1 = arith.DivfOp(xv, sv); q2 = mathd.RoundEvenOp(q1.result)
        q3 = arith.MinimumfOp(q2.result, c127.results[0])
        q4 = arith.MaximumfOp(q3.result, c127n.results[0]); q5 = arith.FPToSIOp(q4.result, i8)
        qb.add_ops([q1, q2, q3, q4, q5, L.YieldOp(q5.result)])
        q = L.GenericOp(inputs=(act, s_a.results[0]), outputs=(q_e.results[0],), body=Region(qb),
                        indexing_maps=ArrayAttr([amap(r, ident_r), amap(r, []), amap(r, ident_r)]),
                        iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * r),
                        result_types=(i8_t,))
        pre += [amx_e, zero_f, amx_f, amx, sc_e, c127, s_a, q_e, c127n, q]

        # --- i8×i8→i32 conv: EXACT original maps + iterators preserved ---
        acc_t = TensorType(i32, list(out_t.get_shape()))
        acc_e = tensor.EmptyOp((), acc_t); zi = arith.ConstantOp.from_int_and_width(0, 32)
        acc_f = L.FillOp(inputs=[zi.results[0]], outputs=[acc_e.results[0]], res=[acc_t])
        mb = Block(arg_types=[i8, i8, i32]); av, bv, acc = mb.args
        ea = arith.ExtSIOp(av, i32); eb = arith.ExtSIOp(bv, i32)
        pm = arith.MuliOp(ea.result, eb.result); pa = arith.AddiOp(pm.result, acc)
        mb.add_ops([ea, eb, pm, pa, L.YieldOp(pa.result)])
        i8cv = L.GenericOp(inputs=(q.results[0], wt_i8), outputs=(acc_f.results[0],),
                           body=Region(mb), indexing_maps=ArrayAttr(maps),
                           iterator_types=op.iterator_types, result_types=(acc_t,))

        # --- requant: out_f32[..] = sitofp(acc) * s_a * s_w[out_channel] ---
        # output map is a plain identity projection of the parallel dims.
        out_dims = [rr.position for rr in maps[-1].data.results]
        P = len(out_dims); d_par = AffineMap.identity(P).results
        # weight's (single) parallel iterator dim -> output channel position
        wt_dims = [rr.position for rr in maps[1].data.results]
        wt_par = [d for d in wt_dims if not red_flags[d]]
        wpos = out_dims.index(wt_par[0])
        out_e = tensor.EmptyOp((), out_t)
        wb = Block(arg_types=[i32, f32, f32, f32]); accv, sav, swv, _ = wb.args
        cur = arith.SIToFPOp(accv, f32); m1 = arith.MulfOp(cur.result, sav)
        m2 = arith.MulfOp(m1.result, swv)
        wb.add_ops([cur, m1, m2, L.YieldOp(m2.result)])
        requant = L.GenericOp(inputs=(i8cv.results[0], s_a.results[0], s_w),
                              outputs=(out_e.results[0],), body=Region(wb),
                              indexing_maps=ArrayAttr([amap(P, d_par), amap(P, []),
                                                       amap(P, [d_par[wpos]]), amap(P, d_par)]),
                              iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * P),
                              result_types=(out_t,))
        _carry_prov(op, contraction=i8cv, requant=requant)

        for new in pre + [acc_e, zi, acc_f, i8cv, out_e, requant]:
            block.insert_op_before(new, op)
        op.results[0].replace_all_uses_with(requant.results[0])
        block.detach_op(op)
        n += 1
    if report_out is not None:
        report_out.update({"generics_scanned": n_generics, "compound_map_generics": n_compound,
                           "windowed_map_generics": n_windowed, "conv_prov_ops": n_conv_prov,
                           "lowered": n})
    # SAY SO when a module is full of convolutions and this pass found none of them. Silence here is
    # exactly how a dead pass passes for a live one: "0 lowered" reads as "nothing to do" whether the
    # module had no convs or had 190 the predicate cannot see.
    if n == 0 and n_windowed == 0 and n_conv_prov:
        print(f"[quant] conv_int8 INERT: {n_conv_prov} ops carry convolution provenance but 0 "
              f"linalg.generic of {n_generics} carries a WINDOWED (stride/dilation) indexing map "
              f"({n_compound} carry a constant-index one, which is a broadcast, not a conv) -- the "
              f"frontend already expanded every conv into im2col + matmul, so this pass cannot "
              f"reach one")
    return n


# I-BERT integer-exp constants (2nd-order poly exp(p) ~ a(p+b)^2 + c on p in [-ln2,0]).
_IEXP_A, _IEXP_B, _IEXP_C = 0.35815147, 1.35330989, 0.34401959
_IEXP_SH = 30      # fixed-point fraction bits for the poly
_IEXP_K = 8        # exponent-quantization step is ln2 / 2**K  (see _IEXP_S below)
_IEXP_CLAMP = -30.0  # exp(-30) ~ 1e-13: the value a masked/saturated input decays to

# The exponent grid is FIXED, not derived from the row's dynamic range.
#
# It used to be dynamic: ``s = max(-x)/127`` per row, with everything below (qln2, b, A and a
# reciprocal of qln2) recomputed per row from it. That is wrong for the op this pass exists to
# serve. Softmax's input is ``scores - rowmax``, and an attention mask puts -inf at every masked
# position, which the clamp above turns into -30. So the row max is the MASK SENTINEL, not the
# data: a causally-masked row whose real scores span ~2 got ``s = 30/127 = 0.236``, quantizing the
# exponent of the entries that actually matter in steps of 0.236 (up to 11.8% per-term exp error)
# while spending 122 of its 127 levels on positions whose exp is 0 to any precision. Measured on
# the small_llama W8A8 recapture, that one choice roughly DOUBLED the whole model's deviation from
# the host W8A8 reference (rel 0.0077 -> 0.0148 against golden_w8a8.npy).
#
# A fixed step has no such coupling: ln2/2**K is the same for every row, masked or not, and it
# makes ``qln2`` an exact power of two, so ``z = floor(-q / qln2)`` is a shift and the per-row
# reciprocal disappears entirely. K = 8 puts the exponent step at 0.0027, which lands the i-exp on
# the 2nd-order polynomial's OWN accuracy floor (max rel err 0.404% over |x| < 10, vs 0.40% as
# K -> inf), i.e. refining further buys nothing. Six per-row generics (max-reduction, scale, qln2,
# b, A, reciprocal) are deleted along with the dynamic scale.
_IEXP_S = 0.6931471805599453 / (1 << _IEXP_K)          # ln2 / 2**K
_IEXP_BQ = int(round(_IEXP_B / _IEXP_S))               # b in exponent-grid units
_IEXP_AQ = int(round(_IEXP_A * (1 << _IEXP_SH) * _IEXP_S * _IEXP_S))
_IEXP_CQ = int(round(_IEXP_C * (1 << _IEXP_SH)))
_IEXP_QMAX = int(round(-_IEXP_CLAMP / _IEXP_S))        # |q| ceiling => z <= QMAX >> K


def _iexp_body(arith, i32, i64, q):
    """The integer I-BERT exp on the fixed exponent grid, as a flat op list.

    ``q`` is the exponent in grid units (i32, ``<= 0``, ``|q| <= _IEXP_QMAX``), i.e. ``x ~ q * S``.
    Returns the ops in emission order; the LAST one yields ``exp(x)`` as f32. Softmax and the SiLU
    logistic both call this, so the two integer exps cannot drift apart — they used to be two
    copies of the same twenty lines, and a fix to one would have silently missed the other.
    """
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32

    zc = arith.ConstantOp.from_int_and_width(0, 32)
    nq = arith.SubiOp(zc.result, q)                      # -q >= 0
    ck = arith.ConstantOp(IntegerAttr(_IEXP_K, i32))
    z = arith.ShRSIOp(nq.result, ck.results[0])          # z = floor(-q / 2**K)
    zs = arith.ShLIOp(z.result, ck.results[0])
    r = arith.AddiOp(q, zs.result)                       # r = q + z*2**K, in (-2**K, 0]
    cb = arith.ConstantOp(IntegerAttr(_IEXP_BQ, i32))
    t = arith.AddiOp(r.result, cb.results[0])            # t = r + b, all in grid units
    t64 = arith.ExtSIOp(t.result, i64)
    tt = arith.MuliOp(t64.result, t64.result)
    ca = arith.ConstantOp(IntegerAttr(_IEXP_AQ, i64))
    att = arith.MuliOp(ca.results[0], tt.result)
    cc = arith.ConstantOp(IntegerAttr(_IEXP_CQ, i64))
    ep = arith.AddiOp(att.result, cc.results[0])         # A*t^2 + C, at 2**SH fixed point
    z64 = arith.ExtSIOp(z.result, i64)
    e = arith.ShRSIOp(ep.result, z64.result)             # ... >> z  (the 2**-z range reduction)
    ef = arith.SIToFPOp(e.result, f32)
    inv = arith.ConstantOp(FloatAttr(1.0 / (1 << _IEXP_SH), f32))
    exf = arith.MulfOp(ef.result, inv.results[0])
    return [zc, nq, ck, z, zs, r, cb, t, t64, tt, ca, att, cc, ep, z64, e, ef, inv, exf]


def lower_softmax_int(module, *, select=None) -> int:
    """Replace softmax's ``math.exp`` with an integer (I-BERT) exp: a fixed-point 2nd-order
    polynomial + power-of-two shift evaluated in integer arithmetic on the FIXED exponent grid
    ``_IEXP_S``, producing the same f32 exp values the downstream sum/divide consume. The
    transcendental ``math.exp`` is gone. Anchors on each exp generic (no fragile region
    detection). Returns count.

    The exponent grid is deliberately NOT derived from the row (see ``_IEXP_S``): softmax's input
    carries the attention mask's -inf sentinels, so a per-row dynamic scale is set by the mask
    rather than by the scores. The rewrite is therefore purely element-wise — one generic, no row
    reduction — and every row is quantized identically whether it is masked or not.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr,
                                       TensorType, f32, i32, i64)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par = L.IteratorType.PARALLEL
    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    def _is_softmax_exp(op):
        # a bare exp generic whose input is the max-subtraction (S - rowmax) — the softmax
        # signature. Guards against rewriting non-softmax exps (whose input may be > 0, which
        # the <=0 i-exp + [-30,0] clamp would corrupt).
        if not (op.name == "linalg.generic" and len(op.inputs) == 1 and op.body.blocks
                and [b.name for b in op.body.blocks[0].ops] == ["math.exp", "linalg.yield"]):
            return False
        src = op.inputs[0].owner
        return (getattr(src, "name", None) == "linalg.generic" and src.body.blocks
                and any(b.name == "arith.subf" for b in src.body.blocks[0].ops))

    targets = [op for op in module.walk() if _is_softmax_exp(op)]
    targets = _select_targets(targets, select)
    n = 0
    for op in targets:
        xs = op.inputs[0]                          # x = scores - rowmax, f32 [.., L], <= 0
        st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape()); R = len(shp)
        if R < 1:
            continue
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR)
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        new = []

        # exp(x) for x <= 0, integer I-BERT i-exp on the FIXED exponent grid (_IEXP_S). Element-
        # wise and self-contained: no row reduction, no per-row constants, no reciprocal.
        #   xc = max(x, -30)                 masked positions are -inf; exp(-30) ~ 1e-13 ~ 0, and
        #                                    an unclamped -inf would make q = -inf -> poison
        #   q  = roundeven(xc / S)           exponent in grid units, in [-QMAX, 0]
        #   z  = (-q) >> K ; r = q + (z << K)    x = -z*ln2 + r*S with r*S in (-ln2, 0]
        #   e  = (A*(r+b)^2 + C) >> z        the poly, then the power-of-two range reduction
        ee = tensor.EmptyOp((), st)
        eb = Block(arg_types=[f32, f32]); xv, _ = eb.args
        cclamp = arith.ConstantOp(FloatAttr(_IEXP_CLAMP, f32))
        xc = arith.MaximumfOp(xv, cclamp.results[0])
        cs = arith.ConstantOp(FloatAttr(_IEXP_S, f32))
        qd = arith.DivfOp(xc.result, cs.results[0]); qr = mathd.RoundEvenOp(qd.result)
        qi = arith.FPToSIOp(qr.result, i32)
        ops_e = [cclamp, xc, cs, qd, qr, qi]
        ops_e += _iexp_body(arith, i32, i64, qi.result)
        exf = ops_e[-1]
        eb.add_ops(ops_e + [L.YieldOp(exf.result)])
        eg = L.GenericOp(inputs=(xs,), outputs=(ee.results[0],), body=Region(eb),
                         indexing_maps=ArrayAttr([full, full]), iterator_types=all_par,
                         result_types=(st,))
        new += [ee, eg]

        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(eg.results[0])
        block.detach_op(op)
        n += 1
    return n


# I-BERT i-GELU constants: erf(z) ~ sgn(z)[a(min(|z|,-b)+b)^2 + 1], z = x/sqrt(2).
_IGELU_A, _IGELU_B = -0.2888, -1.769


def lower_gelu_int(module, *, select=None) -> int:
    """Replace GELU's ``math.erf`` with an integer (I-BERT) i-GELU: per-row dynamic scale, the
    erf approximated by a fixed-point 2nd-order polynomial in integer arithmetic, then
    ``0.5*x*(1+erf)``. The transcendental ``math.erf`` is gone. Anchors on each erf generic.
    Numerically validated vs exact GELU: cos > 0.9999. Returns count."""
    import math as _m
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr,
                                       TensorType, f32, i32, i64)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par = L.IteratorType.PARALLEL; red = L.IteratorType.REDUCTION
    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))
    sqrt2 = _m.sqrt(2.0)
    SH = 24

    targets = [op for op in module.walk()
               if op.name == "linalg.generic" and len(op.inputs) == 1 and op.body.blocks
               and any(b.name == "math.erf" for b in op.body.blocks[0].ops)]
    targets = _select_targets(targets, select)
    n = 0
    for op in targets:
        x = op.inputs[0]; st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape()); R = len(shp)
        if R < 1:
            continue
        Rt = TensorType(f32, shp[:-1])
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR); row = amap(R, idR[:R - 1]); idp = AffineMap.identity(R - 1).results
        red_it = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1) + [L.IteratorTypeAttr(red)])
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        par_rowit = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1))
        new = []

        # sx[..] = max|x| over last / 127
        ae = tensor.EmptyOp((), Rt); z0 = arith.ConstantOp(FloatAttr(0.0, f32))
        af = L.FillOp(inputs=[z0.results[0]], outputs=[ae.results[0]], res=[Rt])
        rb = Block(arg_types=[f32, f32]); a_in, acc = rb.args
        ab = mathd.AbsFOp(a_in); mx = arith.MaximumfOp(ab.result, acc)
        rb.add_ops([ab, mx, L.YieldOp(mx.result)])
        amax = L.GenericOp(inputs=(x,), outputs=(af.results[0],), body=Region(rb),
                           indexing_maps=ArrayAttr([full, row]), iterator_types=red_it, result_types=(Rt,))
        sre = tensor.EmptyOp((), Rt); c127 = arith.ConstantOp(FloatAttr(127.0, f32))
        sb = Block(arg_types=[f32, f32]); nm, _ = sb.args
        sd = arith.DivfOp(nm, c127.results[0]); eps = arith.ConstantOp(FloatAttr(1e-12, f32))
        sfl = arith.MaximumfOp(sd.result, eps.results[0]); sb.add_ops([sd, eps, sfl, L.YieldOp(sfl.result)])
        sx = L.GenericOp(inputs=(amax.results[0],), outputs=(sre.results[0],), body=Region(sb),
                         indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                         iterator_types=par_rowit, result_types=(Rt,))
        new += [ae, z0, af, amax, sre, c127, sx]

        def per_row(fn, elem):
            ot = TensorType(elem, shp[:-1]); e = tensor.EmptyOp((), ot)
            bb = Block(arg_types=[f32, elem]); s_in, _ = bb.args
            cops, res = fn(s_in); bb.add_ops(cops + [L.YieldOp(res)])
            g = L.GenericOp(inputs=(sx.results[0],), outputs=(e.results[0],), body=Region(bb),
                            indexing_maps=ArrayAttr([amap(R - 1, idp), amap(R - 1, idp)]),
                            iterator_types=par_rowit, result_types=(ot,))
            new.extend([e, g]); return g.results[0]

        # qb = round(-B*sqrt2 / sx) ; Aq = round(A * sx^2/2 * 2^SH)
        def _qb(s):
            c = arith.ConstantOp(FloatAttr(-_IGELU_B * sqrt2, f32)); d = arith.DivfOp(c.results[0], s)
            r = mathd.RoundEvenOp(d.result); ci = arith.FPToSIOp(r.result, i32)
            return [c, d, r, ci], ci.result
        def _A(s):
            c = arith.ConstantOp(FloatAttr(_IGELU_A * 0.5 * (1 << SH), f32))
            s2 = arith.MulfOp(s, s); m2 = arith.MulfOp(c.results[0], s2.result)
            r = mathd.RoundEvenOp(m2.result); ci = arith.FPToSIOp(r.result, i32)
            return [c, s2, m2, r, ci], ci.result
        qb = per_row(_qb, i32); Aq = per_row(_A, i32)

        # per-element integer i-GELU
        ge = tensor.EmptyOp((), st)
        eb = Block(arg_types=[f32, f32, i32, i32, f32])   # x, sx, qb, Aq, out
        xv, sv, qbv, av, _ = eb.args
        c127f = arith.ConstantOp(FloatAttr(127.0, f32)); c127n = arith.ConstantOp(FloatAttr(-127.0, f32))
        qd = arith.DivfOp(xv, sv); qr = mathd.RoundEvenOp(qd.result)
        qcl = arith.MinimumfOp(qr.result, c127f.results[0]); qcl2 = arith.MaximumfOp(qcl.result, c127n.results[0])
        q = arith.FPToSIOp(qcl2.result, i32)
        # |q| via arith (math.absi has no LLVM lowering in this pipeline): maxsi(q, -q)
        zi0 = arith.ConstantOp.from_int_and_width(0, 32); nq = arith.SubiOp(zi0.result, q.result)
        aq = arith.MaxSIOp(q.result, nq.result)
        qc = arith.MinSIOp(aq.result, qbv); t = arith.SubiOp(qc.result, qbv)
        t64 = arith.ExtSIOp(t.result, i64); a64 = arith.ExtSIOp(av, i64)
        tt = arith.MuliOp(t64.result, t64.result); att = arith.MuliOp(a64.result, tt.result)
        Ci = arith.ConstantOp(IntegerAttr(1 << SH, i64)); poly = arith.AddiOp(att.result, Ci.results[0])
        polyf = arith.SIToFPOp(poly.result, f32)
        inv = arith.ConstantOp(FloatAttr(1.0 / (1 << SH), f32)); pf = arith.MulfOp(polyf.result, inv.results[0])
        # sgn(x): x<0 -> -1 else +1
        zc = arith.ConstantOp(FloatAttr(0.0, f32)); lt = arith.CmpfOp(xv, zc.results[0], "olt")
        nf = arith.ConstantOp(FloatAttr(-1.0, f32)); pf1 = arith.ConstantOp(FloatAttr(1.0, f32))
        sgn = arith.SelectOp(lt.result, nf.results[0], pf1.results[0])
        erf = arith.MulfOp(sgn.result, pf.result)
        one = arith.ConstantOp(FloatAttr(1.0, f32)); ope = arith.AddfOp(one.results[0], erf.result)
        half = arith.ConstantOp(FloatAttr(0.5, f32)); hx = arith.MulfOp(half.results[0], xv)
        g = arith.MulfOp(hx.result, ope.result)
        eb.add_ops([c127f, c127n, qd, qr, qcl, qcl2, q, zi0, nq, aq, qc, t, t64, a64, tt, att, Ci, poly,
                    polyf, inv, pf, zc, lt, nf, pf1, sgn, erf, one, ope, half, hx, g, L.YieldOp(g.result)])
        gelu = L.GenericOp(inputs=(x, sx.results[0], qb, Aq), outputs=(ge.results[0],), body=Region(eb),
                           indexing_maps=ArrayAttr([full, row, row, row, full]),
                           iterator_types=all_par, result_types=(st,))
        new += [ge, gelu]
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(gelu.results[0])
        block.detach_op(op)
        n += 1
    return n


def lower_silu_int(module, *, select=None) -> int:
    """Replace the logistic ``sigmoid`` generic (``1/(1+exp(-x))``, the SiLU/swish nonlinear)
    with an integer (I-BERT) version: the shared integer-exp (``_iexp_body``, the same fixed-point
    poly + power-of-two shift softmax uses, on the same fixed exponent grid) evaluated on
    ``-|x| <= 0``, then the stable logistic ``sig = (x>=0 ? 1 : e) / (1 + e)`` in f32 (the divide
    is RVV ``vfdiv`` — no integer per-lane divide). The ``math.exp`` is gone. Anchors on each
    ``prov.op = sigmoid`` generic; the downstream ``x * sigmoid`` multiply stays. Returns count."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr,
                                       TensorType, f32, i32, i64)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par = L.IteratorType.PARALLEL
    def amap(n, dims):
        return AffineMapAttr(AffineMap(n, 0, tuple(dims)))

    def _is_sigmoid(op):
        if not (op.name == "linalg.generic" and len(op.inputs) == 1 and op.body.blocks):
            return False
        hint = op.attributes.get("prov.op")
        return (getattr(hint, "data", None) == "sigmoid"
                and any(b.name == "math.exp" for b in op.body.blocks[0].ops))

    targets = [op for op in module.walk() if _is_sigmoid(op)]
    targets = _select_targets(targets, select)
    n = 0
    for op in targets:
        x = op.inputs[0]; st = op.results[0].type
        if not (isinstance(st, TensorType) and st.element_type == f32):
            continue
        shp = list(st.get_shape()); R = len(shp)
        if R < 1:
            continue
        block = op.parent_block()
        idR = AffineMap.identity(R).results
        full = amap(R, idR)
        all_par = ArrayAttr([L.IteratorTypeAttr(par)] * R)
        new = []

        # sigmoid body: e = iexp(-|x|); sig = (x>=0 ? 1 : e)/(1+e). Element-wise on the SAME fixed
        # exponent grid softmax uses (_iexp_body) — the scale used to be a per-row max|x|/127,
        # which made the logistic's resolution depend on the largest activation sharing its row.
        #   q = -min(roundeven(|x| / S), QMAX)     QMAX caps z so the >> z below stays in range
        ee = tensor.EmptyOp((), st)
        eb = Block(arg_types=[f32, f32]); xv, _ = eb.args
        axv = mathd.AbsFOp(xv)
        cs = arith.ConstantOp(FloatAttr(_IEXP_S, f32))
        qd = arith.DivfOp(axv.result, cs.results[0]); qr = mathd.RoundEvenOp(qd.result)
        cqm = arith.ConstantOp(FloatAttr(float(_IEXP_QMAX), f32))
        qcl = arith.MinimumfOp(qr.result, cqm.results[0])
        qa = arith.FPToSIOp(qcl.result, i32)
        c0 = arith.ConstantOp.from_int_and_width(0, 32); qv = arith.SubiOp(c0.result, qa.result)
        ops_s = [axv, cs, qd, qr, cqm, qcl, qa, c0, qv]
        ops_s += _iexp_body(arith, i32, i64, qv.result)
        exf = ops_s[-1]
        # sig = num/denom, num = (x>=0 ? 1 : e), denom = 1+e
        one = arith.ConstantOp(FloatAttr(1.0, f32)); denom = arith.AddfOp(one.results[0], exf.result)
        zc = arith.ConstantOp(FloatAttr(0.0, f32)); ge0 = arith.CmpfOp(xv, zc.results[0], "oge")
        num = arith.SelectOp(ge0.result, one.results[0], exf.result)
        sig = arith.DivfOp(num.result, denom.result)
        eb.add_ops(ops_s + [one, denom, zc, ge0, num, sig, L.YieldOp(sig.result)])
        sg = L.GenericOp(inputs=(x,), outputs=(ee.results[0],), body=Region(eb),
                         indexing_maps=ArrayAttr([full, full]), iterator_types=all_par,
                         result_types=(st,))
        new += [ee, sg]
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(sg.results[0])
        block.detach_op(op)
        n += 1
    return n


# Fast inverse square root magic constant (Quake): y0 = bitcast(0x5f3759df - (bits(v)>>1)).
_RSQRT_MAGIC = 0x5f3759df


def lower_rsqrt_int(module, *, select=None) -> int:
    """Replace RMSNorm/LayerNorm's transcendental ``math.rsqrt`` with the fast inverse square
    root: a bit-hack initial guess (an integer subtract + shift on the f32 bit pattern) refined
    by Newton steps in f32 (``y = y*(1.5 - 0.5*v*y*y)``). No libm call, no integer per-lane
    divide — only ``arith`` integer ops + f32 ``vfmul``/``vfsub`` (all RVV-vectorizable, the
    plan's sanctioned f32-normalization path). Operates in-place on each ``math.rsqrt`` op inside
    its host generic body. Validated vs 1/sqrt: rmsnorm cos ~ 1.0 (4 Newton iters). Returns count."""
    from xdsl.dialects import arith
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32, i32

    NEWTON = 4
    rsqrts = [op for op in module.walk() if op.name == "math.rsqrt" and op.operands[0].type == f32]
    rsqrts = _select_targets(rsqrts, select)
    n = 0
    for op in rsqrts:
        v = op.operands[0]
        block = op.parent_block()
        new = []
        # y0 = bitcast(magic - (bitcast(v) >> 1))
        bi = arith.BitcastOp(v, i32)
        one = arith.ConstantOp(IntegerAttr(1, i32)); sh = arith.ShRSIOp(bi.result, one.results[0])
        magic = arith.ConstantOp(IntegerAttr(_RSQRT_MAGIC, i32)); sub = arith.SubiOp(magic.results[0], sh.result)
        y = arith.BitcastOp(sub.result, f32)
        new += [bi, one, sh, magic, sub, y]
        half = arith.ConstantOp(FloatAttr(0.5, f32)); threehalf = arith.ConstantOp(FloatAttr(1.5, f32))
        new += [half, threehalf]
        cur = y.result
        for _ in range(NEWTON):
            yy = arith.MulfOp(cur, cur); vyy = arith.MulfOp(v, yy.result)
            hvyy = arith.MulfOp(half.results[0], vyy.result); s = arith.SubfOp(threehalf.results[0], hvyy.result)
            ny = arith.MulfOp(cur, s.result)
            new += [yy, vyy, hvyy, s, ny]; cur = ny.result
        for nop in new:
            block.insert_op_before(nop, op)
        op.results[0].replace_all_uses_with(cur)
        block.detach_op(op)
        n += 1
    return n
