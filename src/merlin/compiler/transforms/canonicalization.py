"""Erase dead tensor cones only after complete typed purity and validity checks.

Preserve source ordering, numerical/effect obligations and explicit selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from xdsl.dialects import builtin, tensor
from xdsl.rewriter import Rewriter

from merlin.compiler.ir.provenance import carry_provenance
from merlin.compiler.transforms.certificates import ir_digest

# ---- dead --------------------------------------------------------------------------------------


def _dead_tensor_operation_is_pure(op) -> bool:
    """Require typed value semantics, including every structured scalar body.

    Dialect names and unused results confer no execution-effect information.
    In particular unregistered quantization ops survive until their separate
    explicit lowering provides typed semantics. Physical allocation effects
    remain outside this tensor cleanup.
    """
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import MemRefType, NoneAttr, TensorType, UnrankedMemRefType, UnregisteredOp
    from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.irdl import IRDLOperation
    from xdsl.traits import get_effects, is_speculatable

    if not isinstance(op, IRDLOperation) or isinstance(op, UnregisteredOp) or op.successors:
        return False
    definition = op.get_irdl_definition()
    if any(key not in definition.attributes and not key.startswith("prov.") for key in op.attributes) or any(
        key not in definition.properties for key in op.properties
    ):
        return False
    if any(
        isinstance(value.type, MemRefType | UnrankedMemRefType)
        or isinstance(value.type, TensorType)
        and value.type.encoding != NoneAttr()
        for value in (*op.operands, *op.results)
    ):
        return False
    if isinstance(op, LinalgStructuredOperation):
        if (
            not op.results
            or any(not isinstance(value.type, TensorType) for value in op.results)
            or not op.regions
            or any(len(region.blocks) != 1 for region in op.regions)
            or op.properties.get("library_call") is not None
        ):
            return False
        # The structured tensor operation is value-only only when its actual
        # scalar body is known pure. Its generic container has no such trait.
        return all(_dead_tensor_operation_is_pure(child) for region in op.regions for child in region.block.ops)
    if op.regions:
        return False
    if isinstance(op, YieldOp):
        return True
    if isinstance(op, tensor.EmptyOp):
        return get_effects(op) == set()
    return get_effects(op) == set() and is_speculatable(op)


def prune_dead_pure_tensor_ops(module) -> int:
    """Erase dead top-level pure tensor-expression cones; return the number of erased ops.

    The pass is deliberately conservative: it works within each ``func.func`` entry block and
    only removes result-producing operations with known value-only effects and nontrapping
    semantics. Calls, stores, terminators, unregistered ops and unknown scalar bodies survive.
    It therefore provides a common cleanup seam for scalar/RVV and generated target dialects
    without teaching the frontend anything about a particular accelerator.
    """
    from xdsl.utils.exceptions import VerifyException

    erased = 0
    for func in [op for op in module.walk() if op.name == "func.func"]:
        if not func.regions or not func.regions[0].blocks:
            continue
        block = func.regions[0].blocks[0]
        # Actual SSA uses include captures inside live/unknown scalar regions
        # and uses in other blocks; a local dialect-name liveness approximation
        # must never hide those edges. Reverse erasure releases dead consumers
        # before considering their producers.
        for op in reversed(list(block.ops)):
            if op.results and all(not result.uses for result in op.results) and _dead_tensor_operation_is_pure(op):
                # DCE must not turn malformed/missing source semantics into an
                # apparently valid program by erasing the invalid operation.
                try:
                    op.verify()
                except VerifyException:
                    continue
                block.erase_op(op)
                erased += 1
    return erased


# ---- matmul ------------------------------------------------------------------------------------
# Normalize tensor contraction ranks and explicitly selected accumulator precision.
#
# Preserve source ordering, numerical/effect obligations and explicit selection.


def collapse_overrank_matmul(module) -> int:
    """Fix `linalg.matmul` ops with a >2-D LHS + 2-D RHS (model2MLIR emits `aten.linear` on a
    3-D activation as a 2-D-map matmul over 3-D operands, which is invalid linalg). Rebuild
    each as a `linalg.generic` batched matmul with a broadcast RHS, using proper-rank maps:

        out[b.., n] = sum_k lhs[b.., k] * rhs[k, n]

    Iteration dims = (R-1 leading parallel) + N(parallel) + K(reduction). Returns count.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr, IntegerType, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION
    targets = [
        op
        for op in module.walk()
        if op.name == "linalg.matmul"
        and isinstance(op.results[0].type, TensorType)
        and len(op.results[0].type.get_shape()) > 2
        and len(op.inputs[1].type.get_shape()) == 2
    ]
    for op in targets:
        lhs, rhs = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        et = out_t.element_type
        R = len(out_t.get_shape())  # out rank (== lhs rank)
        ndim = R + 1  # leading(R-1) + N + K
        idm = AffineMap.identity(ndim)
        dims = idm.results
        lead = dims[: R - 1]
        N, K = dims[R - 1], dims[R]
        lhs_map = AffineMap(ndim, 0, (*lead, K))
        rhs_map = AffineMap(ndim, 0, (K, N))
        out_map = AffineMap(ndim, 0, (*lead, N))
        iters = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1) + [L.IteratorTypeAttr(par), L.IteratorTypeAttr(red)])
        block = op.parent_block()

        empty = tensor.EmptyOp((), out_t)
        zero = arith.ConstantOp(IntegerAttr(0, et) if isinstance(et, IntegerType) else FloatAttr(0.0, et))
        fill = L.FillOp(inputs=[zero.results[0]], outputs=[empty.results[0]], res=[out_t])
        body = Block(arg_types=[et, et, et])
        a, b, acc = body.args
        prod = arith.MulfOp(a, b)
        summ = arith.AddfOp(acc, prod.result)
        body.add_ops([prod, summ, L.YieldOp(summ.result)])
        gen = L.GenericOp(
            inputs=(lhs, rhs),
            outputs=(fill.results[0],),
            body=Region(body),
            indexing_maps=ArrayAttr([AffineMapAttr(lhs_map), AffineMapAttr(rhs_map), AffineMapAttr(out_map)]),
            iterator_types=iters,
            result_types=(out_t,),
        )
        carry_provenance(gen, op, "collapse_overrank_matmul")
        for new_op in (empty, zero, fill, gen):
            block.insert_op_before(new_op, op)
        op.results[0].replace_all_uses_with(gen.results[0])
        block.detach_op(op)
    return len(targets)


def lower_bf16_matmul_f32acc(module) -> int:
    """Rewrite every 16-bit-float (bf16 OR fp16) contraction to accumulate in f32; returns count.

    A half-precision (bf16/fp16) ``linalg.matmul`` accumulates in its 16-bit output type, rounding
    every partial sum to the small mantissa — lossy over long contractions, and for fp16 (5-bit
    exponent, max ~65504) the accumulator also OVERFLOWS to inf on real-LLM-sized reductions.
    Hardware/torch instead accumulate in f32 and round only the final result. This rewrites

        C_bf16 = matmul(A_bf16, B_bf16, init=C0)     # bf16 accumulate

    into a ``linalg.generic`` matmul whose body extends each operand to f32, multiplies and
    accumulates in f32 starting from ``extf(C0)``, followed by a ``truncf`` back to bf16 —
    matching the reference numerics. (``extf``/``truncf`` here are scalar ops inside the
    generic body, so the earlier "extf is tensor-rank-polymorphic" limitation does not apply.)

    Also handles bf16 ``linalg.generic`` contractions (e.g. ``batch_matmul``, ``aten.bmm``);
    these otherwise accumulate in bf16 too (attention scores/context), the dominant precision
    loss on VLAs. Either kind qualifies only when its body is exactly ``yield(acc + a * b)`` over
    its own block arguments; any other body computes something else and is left untouched. The
    op's own indexing maps and iterator types are reused (a ``linalg.matmul`` may carry
    transposing maps); only the accumulation dtype is promoted.
    """
    import math

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.arith import FastMathFlag, FastMathFlagsAttr
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, BFloat16Type, Float16Type, FloatAttr, TensorType, f32
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    # FP-CONTRACTION LICENSE on the accumulate. The mul/add below are a multiply-accumulate; without
    # `contract` LLVM must keep them as a separate fmul + fadd and CANNOT form an FMA, so the RVV
    # backend emits vfwmul.vf + vfadd.vv instead of the single fused vfwmacc.vf -- measured: zero
    # vfwmacc in the emitted fp16 kernel until this flag was set. The f32 path never needed it
    # because linalg.matmul -> vector.contract -> vector.fma already lowers to llvm.fmuladd; a
    # MIXED-precision contraction (16-bit operands, f32 accumulator) cannot use vector.contract
    # (which is same-type only) and so arrives here as explicit mulf/addf.
    #
    # This is a strict-accuracy WIN, not a fast-math relaxation: contraction removes the
    # intermediate rounding of the product, which is exactly the single-rounding semantics the
    # hardware FMA and the f32 reference path already have. No other fast-math flag is set (no
    # reassoc/nnan/ninf), so the reduction order is unchanged and NaN/Inf behavior is preserved.
    _CONTRACT = FastMathFlagsAttr([FastMathFlag.ALLOW_CONTRACT])

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def is_half_tensor(t):
        return isinstance(t, TensorType) and isinstance(t.element_type, (BFloat16Type, Float16Type))

    def is_mul_add(block):
        """``yield(acc + a * b)`` over the block's two inputs and accumulator, in either operand order."""
        if len(block.args) != 3 or [o.name for o in block.ops] != ["arith.mulf", "arith.addf", "linalg.yield"]:
            return False
        x, y, acc = block.args
        mul, add, out = block.ops
        return (
            {*mul.operands} == {x, y}
            and {*add.operands} == {acc, mul.results[0]}
            and tuple(out.operands) == (add.results[0],)
        )

    def is_contraction(op):
        """A two-input, one-result 16-bit structured reduction whose body is ``acc + a * b``."""
        return (
            isinstance(op, (L.MatmulOp, L.GenericOp))
            and len(op.inputs) == 2
            and len(op.results) == 1
            and all(is_half_tensor(v.type) for v in (*op.inputs, *op.results))
            and any(it.data == red for it in op.get_iterator_types().data)
            and is_mul_add(op.regions[0].blocks[0])
        )

    def is_zero(value):
        """A splat or fill of a ``+0.0`` constant, so the f32 accumulator can start at ``+0.0``."""
        if not isinstance(value.owner, (tensor.SplatOp, L.FillOp)):
            return False
        constant = value.owner.operands[0].owner
        return (
            isinstance(constant, arith.ConstantOp)
            and isinstance(constant.value, FloatAttr)
            and math.copysign(1.0, constant.value.value.data) == 1.0
            and constant.value.value.data == 0.0
        )

    def convert(value, element, cast):
        """Elementwise ``cast`` of a static tensor into ``element``: ``[tensor.empty, linalg.generic]``."""
        shape = value.type.get_shape()
        result_t = TensorType(element, shape)
        empty = tensor.EmptyOp((), result_t)
        body = Block(arg_types=[value.type.element_type, element])
        converted = cast(body.args[0], element)
        body.add_ops([converted, L.YieldOp(converted.result)])
        identity = AffineMapAttr(AffineMap.identity(len(shape)))
        generic = L.GenericOp(
            inputs=(value,),
            outputs=(empty.results[0],),
            body=Region(body),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * len(shape)),
            result_types=(result_t,),
        )
        return [empty, generic]

    targets = [op for op in module.walk() if is_contraction(op)]
    for op in targets:
        a, b = op.inputs
        out_t = op.results[0].type
        f32_t = TensorType(f32, out_t.get_shape())
        block = op.parent_block()

        # The f32 accumulator starts from the op's own destination (a bias, a running sum, ...).
        # A +0.0 destination needs no conversion pass: start from an f32 +0.0 fill directly.
        if is_zero(op.outputs[0]):
            empty_f = tensor.EmptyOp((), f32_t)
            zero = arith.ConstantOp(FloatAttr(0.0, f32))
            start = [empty_f, zero, L.FillOp(inputs=[zero.results[0]], outputs=[empty_f.results[0]], res=[f32_t])]
        else:
            start = convert(op.outputs[0], f32, arith.ExtFOp)

        # Preserve each operand's own 16-bit type (bf16 or f16); truncate back to the result's.
        a_e, b_e = a.type.element_type, b.type.element_type
        body = Block(arg_types=[a_e, b_e, f32])
        x, y, acc = body.args
        xf, yf = arith.ExtFOp(x, f32), arith.ExtFOp(y, f32)
        prod = arith.MulfOp(xf.result, yf.result, _CONTRACT)
        summ = arith.AddfOp(acc, prod.result, _CONTRACT)
        body.add_ops([xf, yf, prod, summ, L.YieldOp(summ.result)])
        acc_f32 = L.GenericOp(
            inputs=(a, b),
            outputs=(start[-1].results[0],),
            body=Region(body),
            indexing_maps=op.get_indexing_maps(),
            iterator_types=op.get_iterator_types(),
            result_types=(f32_t,),
        )
        finish = convert(acc_f32.results[0], out_t.element_type, arith.TruncFOp)

        # Tag by the ACTUAL operand width -- this pass covers bf16 and fp16, and a breadcrumb
        # that says "bf16" on an fp16 kernel sends the next reader to the wrong datapath.
        carry_provenance(acc_f32, op, "fp16_f32acc" if isinstance(a_e, Float16Type) else "bf16_f32acc")
        for new_op in (*start, acc_f32, *finish):
            block.insert_op_before(new_op, op)
        op.results[0].replace_all_uses_with(finish[-1].results[0])
        block.detach_op(op)
    return len(targets)


# ---- tensor_views ------------------------------------------------------------------------------
# Same-block CSE of static immutable tensor views with explicit source lineage.
#
# This is a structural transform, not a physical alias/residency permission or a
# numerical qualification. Ordinary pipeline verification remains caller-owned.
# No arithmetic, quantization, operation order or tensor payload is changed.


@dataclass(frozen=True)
class TensorViewWitness:
    operation_index: int
    retained_index: int
    operation: str
    before_sha256: str
    result_type: str


def _key(op):
    allowed = {
        tensor.CollapseShapeOp: {"reassociation"},
        tensor.ExpandShapeOp: {"reassociation", "static_output_shape"},
        tensor.CastOp: set(),
    }
    if (
        type(op) not in allowed
        or len(op.operands) != 1
        or len(op.results) != 1
        or op.regions
        or op.successors
        or set(op.properties) != allowed[type(op)]
        or any(not name.startswith("prov.") for name in op.attributes)
    ):
        return None
    source, result = op.operands[0].type, op.results[0].type
    if any(
        not isinstance(t, builtin.TensorType) or t.encoding != builtin.NoneAttr() or any(d <= 0 for d in t.get_shape())
        for t in (source, result)
    ):
        return None
    if source.element_type != result.element_type or prod(source.get_shape()) != prod(result.get_shape()):
        return None
    # Static same-type casts and explicit reassociation views have immutable
    # tensor semantics. An encoded/dynamic/layout-dependent view is excluded.
    if type(op) is tensor.CastOp and source != result:
        return None
    return (type(op), tuple(op.operands), tuple(op.result_types), tuple(sorted(op.properties.items())))


def _lineage(op):
    previous = op.attributes.get("prov.view_cse_sources")
    if previous is not None:
        if not isinstance(previous, builtin.ArrayAttr) or any(
            not isinstance(item, builtin.DictionaryAttr) for item in previous
        ):
            raise ValueError("tensor view CSE lineage must contain exact attribute dictionaries")
        return tuple(previous)
    return (builtin.DictionaryAttr(dict(op.attributes)),)


def canonicalize_tensor_views(module: builtin.ModuleOp) -> tuple[TensorViewWitness, ...]:
    """Reuse a dominating identical view, preserving every use and source record.

    Caches are local to each block; nothing moves across CFG/region boundaries.
    Unknown operations are retained, including side effects between two views.
    This is legal only for the closed immutable tensor-view domain above: it
    does not merge loads or mutable memref views. Arithmetic and visible outputs
    remain in their original order. Use the ordinary host verification boundary
    when consuming the transformed program; direct calls are structural planning.
    """
    module.verify()
    indices = {op: i for i, op in enumerate(module.walk())}
    witnesses = []
    blocks = [block for op in module.walk() for region in op.regions for block in region.blocks]
    for block in blocks:
        known = {}
        for op in tuple(block.ops):
            key = _key(op)
            if key is None:
                continue
            previous = known.get(key)
            if previous is None:
                known[key] = op
                continue
            witness = TensorViewWitness(indices[op], indices[previous], op.name, ir_digest(op), str(op.results[0].type))
            lineage = (*_lineage(previous), *_lineage(op))
            # Metadata is preserved, not used to justify view equivalence.
            previous.attributes["prov.view_cse_sources"] = builtin.ArrayAttr(lineage)
            for name in ("prov.source_node_ids", "prov.origin_node_ids"):
                entries = [record.data.get(name) for record in lineage]
                if all(value is None or isinstance(value, builtin.ArrayAttr) for value in entries):
                    union = tuple(dict.fromkeys(item for value in entries if value is not None for item in value))
                    if union:
                        previous.attributes[name] = builtin.ArrayAttr(union)
            Rewriter.replace_op(op, [], new_results=list(previous.results))
            witnesses.append(witness)
    module.verify()
    return tuple(witnesses)


def fold_complete_matrix_slice_concats(module: builtin.ModuleOp) -> int:
    """View a complete ordered matrix-slice concat through its original aggregate.

    A rank-three tensor ``[h,m,n]`` sliced at every ``h`` coordinate,
    collapsed to ``[m,n]`` and concatenated along dimension zero is exactly
    its row-major ``[h*m,n]`` collapse. Slice observers remain intact; only
    the concat is replaced. Static unencoded tensors and exact unit strides
    exclude partial, reordered, duplicated and layout-dependent alternatives.
    """
    module.verify()
    count = 0
    for concat in tuple(module.walk()):
        if (
            type(concat) is not tensor.ConcatOp
            or concat.dim.value.data != 0
            or set(concat.properties) != {"dim"}
            or any(not key.startswith("prov.") for key in concat.attributes)
        ):
            continue
        aggregate = None
        for index, value in enumerate(concat.inputs):
            collapse = value.owner
            if (
                type(collapse) is not tensor.CollapseShapeOp
                or len(collapse.operands) != 1
                or set(collapse.properties) != {"reassociation"}
                or any(not key.startswith("prov.") for key in collapse.attributes)
                or tuple(tuple(i.value.data for i in group) for group in collapse.reassociation) != ((0, 1), (2,))
            ):
                break
            view = collapse.operands[0].owner
            if (
                type(view) is not tensor.ExtractSliceOp
                or len(view.operands) != 1
                or set(view.properties) != {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
                or any(not key.startswith("prov.") for key in view.attributes)
            ):
                break
            typ = view.source.type
            if (
                not isinstance(typ, builtin.TensorType)
                or typ.encoding != builtin.NoneAttr()
                or len(typ.get_shape()) != 3
                or any(d <= 0 for d in typ.get_shape())
            ):
                break
            h, m, n = typ.get_shape()
            if (
                h != len(concat.inputs)
                or (aggregate is not None and view.source is not aggregate)
                or tuple(view.static_offsets.get_values()) != (index, 0, 0)
                or tuple(view.static_sizes.get_values()) != (1, m, n)
                or tuple(view.static_strides.get_values()) != (1, 1, 1)
                or view.result.type != builtin.TensorType(typ.element_type, (1, m, n))
                or value.type != builtin.TensorType(typ.element_type, (m, n))
                or concat.result.type != builtin.TensorType(typ.element_type, (h * m, n))
            ):
                break
            aggregate = view.source
        else:
            if aggregate is None:
                continue
            replacement = tensor.CollapseShapeOp.create(
                operands=[aggregate],
                result_types=[concat.result.type],
                properties={"reassociation": collapse.reassociation},
                attributes=dict(concat.attributes),
            )
            Rewriter.replace_op(concat, replacement)
            count += 1
    module.verify()
    return count


# ---- casts -------------------------------------------------------------------------------------
# Legalize scalar Boolean/float conversion without introducing poison.
#
# Preserve source ordering, numerical/effect obligations and explicit selection.


def fix_bool_sitofp(module) -> int:
    """Rewrite ``arith.sitofp`` of an ``i1`` (a bool) to ``arith.uitofp``; returns the count.

    PyTorch casts a bool tensor to float as ``True -> 1.0`` (unsigned). model2MLIR, however,
    emits a *signed* ``arith.sitofp %b : i1 to fN`` for these casts. ``sitofp`` interprets the
    i1 as a signed 1-bit integer, whose ``true`` bit pattern is the value ``-1`` (two's
    complement), so the compiled kernel produces ``-1.0`` instead of ``+1.0``. This is silent
    on the driver's scalar interpreter (Python ``float(True) == 1.0``) but wrong in every
    *compiled* kernel (host clang and RVV alike).

    The canonical place it bites: the eager-attention causal mask, built as
    ``mask_f32 * sitofp(bool)``; the sign flip turns the ``-FLT_MAX`` fill into ``+FLT_MAX``,
    so softmax attends to *future* tokens (molmoact decoder cos 0.10). Rewriting to
    ``uitofp`` (``true -> 1.0``) matches torch. ``module.walk()`` recurses into linalg.generic
    bodies, so casts inside outlined kernels are covered. Apply before outlining/lowering.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType

    n = 0
    for op in list(module.walk()):
        if op.name != "arith.sitofp":
            continue
        src = op.operands[0]
        st = src.type
        if not (isinstance(st, IntegerType) and st.width.data == 1):
            continue
        new = arith.UIToFPOp(src, op.results[0].type)
        carry_provenance(new, op, "bool_sitofp_uitofp")
        op.parent_block().insert_op_before(new, op)
        op.results[0].replace_all_uses_with(new.results[0])
        op.parent_block().detach_op(op)
        n += 1
    return n


def fix_bool_fptosi(module) -> int:
    """Rewrite a float->``i1`` ``arith.fptosi``/``arith.fptoui`` into ``x != 0``; returns the count.

    The mirror image of :func:`fix_bool_sitofp`, and far worse than a sign flip. model2MLIR emits
    a cast to a bool tensor (``aten._to_copy`` with ``prov.orig_dtype = "bool"``) as
    ``arith.fptosi %x : f32 to i1``. Signed ``i1`` holds only ``{-1, 0}``, so *every* float whose
    truncation is not one of those two values — ``1.0`` included — is **poison** in LLVM, not a
    wrong number. mlir-translate emits ``fptosi float 1.0 to i1``, instcombine folds it to
    ``poison``, and the poison propagates out of the bool tensor into whatever consumes it.

    Where that lands is not local. In smolvla the poisoned mask feeds a masked-select whose result
    count sizes a ``malloc`` and bounds a data-dependent loop, so ``br i1 poison`` becomes a
    self-branch, ``simplifycfg`` deletes every block after it, and ``forward`` compiles to 3,654
    bytes with a call set of ``malloc``/``memset``/``roundevenf`` — a whole 500M-parameter model
    erased while the link succeeds and the build reports success.

    PyTorch's bool cast is ``x != 0`` (NaN included, which is why the predicate is the *unordered*
    ``une``), so that is what replaces it. ``module.walk()`` recurses into linalg.generic bodies.
    Apply before outlining/lowering.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import AnyFloat, FloatAttr, IntegerType

    n = 0
    for op in list(module.walk()):
        if op.name not in ("arith.fptosi", "arith.fptoui"):
            continue
        res_t = op.results[0].type
        if not (isinstance(res_t, IntegerType) and res_t.width.data == 1):
            continue
        src = op.operands[0]
        if not isinstance(src.type, AnyFloat):
            continue
        zero = arith.ConstantOp(FloatAttr(0.0, src.type))
        # `une` = unordered-or-not-equal: NaN compares TRUE, matching torch's `bool(nan) is True`.
        cmp = arith.CmpfOp(src, zero.results[0], "une")
        carry_provenance(cmp, op, "bool_fptosi_cmpf_une")
        block = op.parent_block()
        block.insert_op_before(zero, op)
        block.insert_op_before(cmp, op)
        op.results[0].replace_all_uses_with(cmp.results[0])
        block.detach_op(op)
        n += 1
    return n
