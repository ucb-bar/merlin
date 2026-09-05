"""Decompose torchao's affine activation-quantization calls into linalg.

WHY THIS EXISTS. A model2MLIR capture of a torchao *dynamic-activation* scheme leaves two
operations as opaque ``func.call``s to private, body-less externs::

    func.func private @torchao_choose_qparams_affine_default(tensor<1x2048xf32>) -> tensor<1xf32>
    func.func private @torchao_quantize_affine_default(tensor<1x2048xf32>, tensor<1xf32>)
                                                      -> tensor<1x2048xi8>

They are opaque because m2m's FX importer has a decomposition for ``torchao.dequantize_affine``
only; everything else falls through to its opaque ``func.call`` fallback. Two consequences of that
fallback matter here and are the reason the semantics cannot simply be read off the call:

  * the fallback keeps only the args that are FX *nodes*. ``block_size``, ``mapping_type``,
    ``target_dtype``, ``quant_min``, ``quant_max`` and ``eps`` are Python constants in the graph,
    so they are dropped and never reach the MLIR;
  * ``choose_qparams_affine`` returns ``(scale, zero_point)``; the importer maps ``getitem(x, 0)``
    to the call's single result and DROPS ``getitem(x, 1)``. So the zero point disappears too, and
    ``quantize_affine`` arrives with two operands instead of four.

Nothing in merlin defined those symbols, so the link failed with
``undefined reference to torchao_quantize_affine_default`` and the interpreter path failed with
``OutlineError: @forward calls 2 symbol(s) this module never defines``.

WHAT IS DERIVED, AND FROM WHERE. Nothing here is assumed:

  * the block layout (which axes share one qparam) comes from the call's OWN types — the input
    shape against the scale shape;
  * the target integer type and its element width come from the ``quantize_affine`` RESULT type;
  * the mapping type, ``quant_min``/``quant_max`` and ``eps`` come from the torchao scheme the
    bundle records in its module attribute ``prov.quantization``, resolved through
    :data:`ACTIVATION_QUANT` — a transcription of torchao's own ``quant_api`` source, cited
    per entry.

Everything that cannot be derived FAILS CLOSED with :class:`TorchAOAffineError` rather than
substituting a default: a silently wrong rounding rule or eps produces a model that still gates at
cos 0.99 and is quietly wrong.

ARITHMETIC. Transcribed op-for-op from ``torchao.quantization.quant_primitives`` so the emitted
linalg is bit-identical to torchao's own eager result, not merely close:

``_choose_qparams_affine`` (SYMMETRIC)::

    max_abs = max(-amin(x), amax(x))            # == max(|x|) elementwise-exactly
    scale   = max_abs / (float(qmax - qmin) / 2)
    scale   = clamp(scale, min=eps)

``_quantize_affine``::

    q = clamp(round_half_to_even(x * (1.0 / scale)) + zero_point, qmin, qmax).to(int8)

Three details are load-bearing and each was checked against the source rather than guessed:

  * it is ``x * (1.0 / scale)``, NOT ``x / scale`` — the reciprocal is taken first, and the two
    differ in the last ulp;
  * ``_Round`` forwards to ``torch.round``, i.e. round-half-to-EVEN — ``math.roundeven``, not
    ``math.round`` (which is ties-away-from-zero);
  * ``scale`` is divided by ``(qmax - qmin) / 2`` and only then clamped to ``eps``.

The dropped zero point is safe for a SYMMETRIC scheme and only for it: torchao sets it to
``int((qmax + qmin + 1) / 2)`` which is 0 whenever ``qmin == -qmax`` or ``qmin == -qmax - 1``, and
it is added as an f32 zero, whose only effect on the result is turning ``-0.0`` into ``0.0`` — and
both convert to integer 0. An ASYMMETRIC scheme has a data-dependent zero point that the capture
threw away, so it is refused here rather than approximated.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The torchao op each opaque callee stands for, keyed by the callee's base name. m2m derives the
#: symbol from the FX target (``torchao.choose_qparams_affine.default`` ->
#: ``torchao_choose_qparams_affine_default``) and appends ``_<n>`` when one target needs several
#: signatures, so the base name is recovered by stripping a trailing numeric token.
CHOOSE_QPARAMS_SYMBOL = "torchao_choose_qparams_affine_default"
QUANTIZE_SYMBOL = "torchao_quantize_affine_default"

#: Module attribute a model2MLIR bundle records its torchao scheme under.
SCHEME_ATTR = "prov.quantization"


class TorchAOAffineError(RuntimeError):
    """A torchao affine-quant call whose semantics could not be derived. Never approximated."""


@dataclass(frozen=True)
class ActivationQuant:
    """The activation-quantization parameters one torchao scheme applies.

    Attributes:
        mapping: torchao ``MappingType`` name. Only ``SYMMETRIC`` can be honoured — see module docs.
        quant_min / quant_max: the integer range torchao passes explicitly (NOT the dtype's range).
        eps: the floor ``scale`` is clamped to.
        granularity: the block shape the scheme asks for, as a name checked against the shape
            actually derived from the call's types. A cross-check, not an input.
        source: the torchao function this entry transcribes, so it can be re-verified.
    """

    mapping: str
    quant_min: int
    quant_max: int
    eps: float
    granularity: str
    source: str


#: Scheme -> activation quant, transcribed from torchao's own source. Add an entry only after
#: reading the function it cites; an unknown scheme fails closed.
ACTIVATION_QUANT: dict[str, ActivationQuant] = {
    "int8_dyn_act_int8_weight": ActivationQuant(
        mapping="SYMMETRIC", quant_min=-127, quant_max=127, eps=1e-5, granularity="per_token",
        source="torchao.quantization.quant_api._int8_symm_per_token_reduced_range_quant, "
               "selected by Int8DynamicActivationInt8WeightConfig's default "
               "act_mapping_type=MappingType.SYMMETRIC",
    ),
}


@dataclass(frozen=True)
class BlockLayout:
    """How the scale tensor tiles the input: which axes are shared, which are reduced.

    ``block_size`` is torchao's own parameter — the extent of the tensor region sharing one qparam,
    one entry per input axis.
    """

    in_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    block_size: tuple[int, ...]

    @property
    def granularity(self) -> str:
        """The torchao granularity name this block shape spells, or ``"other"``."""
        rank = len(self.in_shape)
        if self.block_size == self.in_shape:
            # Every axis in one block. Per-token and per-tensor coincide when the leading axes are
            # all 1, and then the arithmetic is identical either way; name it per_token so a
            # batch-1 per-token capture is not rejected against its scheme.
            return "per_token" if all(d == 1 for d in self.in_shape[:-1]) else "per_tensor"
        if self.block_size == (1,) * (rank - 1) + (self.in_shape[-1],):
            return "per_token"
        return "other"


def derive_block_layout(in_shape, scale_shape) -> BlockLayout:
    """Recover torchao's ``block_size`` from the input and scale shapes alone.

    torchao computes qparams by reducing the axes whose ``block_size`` entry is not 1, with
    ``keepdim=False`` by default. Two shapes of result are therefore possible and both are handled:

      * rank-reduced (``keepdim=False``): the scale keeps the LEADING axes and drops the reduced
        trailing ones, so ``scale_shape == in_shape[:len(scale_shape)]`` and every dropped axis is
        one whole block;
      * rank-preserving (``keepdim=True``): ranks match and ``block_size[i] = in_shape[i] //
        scale_shape[i]``.

    Anything else is ambiguous — a rank-1 scale against a rank-3 input could describe several
    tilings that compute different numbers — and raises instead of picking one.
    """
    in_shape = tuple(int(d) for d in in_shape)
    scale_shape = tuple(int(d) for d in scale_shape)
    if any(d < 0 for d in in_shape) or any(d < 0 for d in scale_shape):
        raise TorchAOAffineError(
            f"dynamic dims in a torchao affine quant (input {in_shape}, scale {scale_shape}); "
            "the block layout is not derivable from a symbolic shape")

    if len(scale_shape) == len(in_shape):
        block: list[int] = []
        for dim, nblocks in zip(in_shape, scale_shape):
            if nblocks == 0 or dim % nblocks:
                raise TorchAOAffineError(
                    f"scale shape {scale_shape} does not tile input shape {in_shape} evenly")
            block.append(dim // nblocks)
        return BlockLayout(in_shape, scale_shape, tuple(block))

    if len(scale_shape) < len(in_shape) and scale_shape == in_shape[:len(scale_shape)]:
        kept = len(scale_shape)
        return BlockLayout(in_shape, scale_shape,
                           (1,) * kept + in_shape[kept:])

    raise TorchAOAffineError(
        f"cannot derive the torchao block_size from input shape {in_shape} and scale shape "
        f"{scale_shape}: the scale is neither a rank-preserving tiling nor the leading axes of "
        "the input. The capture dropped block_size, so there is nothing else to read it from.")


def _base_symbol(callee: str) -> str:
    """Strip m2m's ``_<n>`` per-signature suffix (``..._default_7`` -> ``..._default``)."""
    head, sep, tail = callee.rpartition("_")
    return head if sep and tail.isdigit() and head else callee


def scheme_of(module) -> str | None:
    """The torchao scheme name the bundle records, or None when it records none."""
    from xdsl.dialects.builtin import StringAttr

    attr = module.attributes.get(SCHEME_ATTR)
    return attr.data if isinstance(attr, StringAttr) else None


def _f32_const(value: float):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import Float32Type, FloatAttr

    f32 = Float32Type()
    return arith.ConstantOp(FloatAttr(value, f32), f32)


def _maps(rank: int, kept: int, block_size, *, in_map_first: bool = True):
    """(identity map over the input, projection map onto the scale) for a block layout."""
    from xdsl.ir.affine import AffineMap

    identity = AffineMap.identity(rank)
    if kept == rank:
        # Rank-preserving: axis i of the scale indexes block (d_i floordiv block_size[i]).
        results = []
        for i in range(rank):
            results.append(identity.results[i] if block_size[i] == 1
                           else identity.results[i] // int(block_size[i]))
        scale_map = AffineMap(rank, 0, tuple(results))
    else:
        scale_map = AffineMap(rank, 0, tuple(identity.results[:kept]))
    return identity, scale_map


def lower_torchao_affine_quant(module, *, report_out: "dict | None" = None) -> int:
    """Rewrite every opaque torchao affine activation-quant call into linalg; returns the count.

    A module carrying none of these calls is left untouched (and returns 0), so adding this to a
    pipeline cannot perturb a bundle that never had them.
    """
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, Float32Type, IntegerType,
                                       TensorType)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region

    from .passes_xdsl import carry_provenance

    choose_calls: list = []
    quant_calls: list = []
    for op in module.walk():
        if op.name != "func.call":
            continue
        base = _base_symbol(op.callee.string_value())
        if base == CHOOSE_QPARAMS_SYMBOL:
            choose_calls.append(op)
        elif base == QUANTIZE_SYMBOL:
            quant_calls.append(op)

    if report_out is not None:
        report_out.update(choose_qparams=len(choose_calls), quantize=len(quant_calls))
    if not choose_calls and not quant_calls:
        return 0

    scheme = scheme_of(module)
    if scheme is None:
        raise TorchAOAffineError(
            f"the module carries {len(choose_calls) + len(quant_calls)} torchao affine-quant "
            f"call(s) but no {SCHEME_ATTR!r} attribute, so the mapping type, quant range and eps "
            "cannot be derived. The capture dropped them from the call itself.")
    spec = ACTIVATION_QUANT.get(scheme)
    if spec is None:
        raise TorchAOAffineError(
            f"torchao scheme {scheme!r} has no activation-quant entry in ACTIVATION_QUANT; add one "
            "after reading the quant_api function it uses (known: "
            + ", ".join(sorted(ACTIVATION_QUANT)) + ")")
    if spec.mapping != "SYMMETRIC":
        raise TorchAOAffineError(
            f"torchao scheme {scheme!r} quantizes activations with mapping {spec.mapping}, whose "
            "zero point is data-dependent — and the capture's opaque fallback DROPPED the zero "
            "point operand (it maps getitem(x, 0) only). There is nothing to reconstruct it from, "
            "so this is refused rather than approximated with zero.")

    f32 = Float32Type()
    # torchao: scale = max_abs / (float(quant_max - quant_min) / 2), then clamp(min=eps).
    half_range = float(spec.quant_max - spec.quant_min) / 2.0

    n = 0
    for call in choose_calls:
        (src,) = call.operands
        in_t, scale_t = src.type, call.results[0].type
        if not isinstance(in_t, TensorType) or not isinstance(scale_t, TensorType):
            raise TorchAOAffineError(f"@{call.callee.string_value()} is not tensor-typed")
        if in_t.element_type != f32 or scale_t.element_type != f32:
            raise TorchAOAffineError(
                f"@{call.callee.string_value()} operates on {in_t.element_type}/"
                f"{scale_t.element_type}; torchao computes qparams in the input dtype and this "
                "lowering has only been verified bit-exact for f32")
        layout = derive_block_layout(in_t.get_shape(), scale_t.get_shape())
        if layout.granularity != spec.granularity:
            raise TorchAOAffineError(
                f"@{call.callee.string_value()}: the call's types imply block_size "
                f"{layout.block_size} ({layout.granularity}) but scheme {scheme!r} quantizes "
                f"activations {spec.granularity} ({spec.source})")
        rank, kept = len(layout.in_shape), len(layout.scale_shape)
        identity, scale_map = _maps(rank, kept, layout.block_size)
        red_iters = [L.IteratorTypeAttr(L.IteratorType.PARALLEL) for _ in range(rank)]
        if kept < rank:
            for i in range(kept, rank):
                red_iters[i] = L.IteratorTypeAttr(L.IteratorType.REDUCTION)
        else:
            for i in range(rank):
                if layout.block_size[i] != 1:
                    red_iters[i] = L.IteratorTypeAttr(L.IteratorType.REDUCTION)

        parent = call.parent_block()
        # max(|x|) over the block. Seeded with 0.0, which cannot change the result: |x| >= 0, so
        # max(0, max|x|) == max|x| (and a NaN input still propagates through arith.maximumf).
        acc_t = TensorType(f32, list(layout.scale_shape))
        empty_acc = tensor.EmptyOp((), acc_t)
        zero = _f32_const(0.0)
        seed = L.FillOp(inputs=[zero.result], outputs=[empty_acc.tensor], res=[acc_t])
        rbody = Block(arg_types=[f32, f32])
        xv, accv = rbody.args
        absf = math.AbsFOp(xv)
        mx = arith.MaximumfOp(accv, absf.result)
        rbody.add_ops([absf, mx, L.YieldOp(mx.result)])
        reduce_gen = L.GenericOp(
            inputs=(src,), outputs=(seed.results[0],), body=Region(rbody),
            indexing_maps=ArrayAttr([AffineMapAttr(identity), AffineMapAttr(scale_map)]),
            iterator_types=ArrayAttr(red_iters), result_types=(acc_t,))

        # scale = clamp(max_abs / half_range, min=eps) — divide first, clamp second.
        empty_scale = tensor.EmptyOp((), scale_t)
        sbody = Block(arg_types=[f32, f32])
        mval, _ = sbody.args
        denom = _f32_const(half_range)
        epsc = _f32_const(spec.eps)
        div = arith.DivfOp(mval, denom.result)
        clamped = arith.MaximumfOp(div.result, epsc.result)
        sbody.add_ops([denom, epsc, div, clamped, L.YieldOp(clamped.result)])
        par_iters = ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL) for _ in range(kept)])
        sid = _maps(kept, kept, (1,) * kept)[0]
        scale_gen = L.GenericOp(
            inputs=(reduce_gen.results[0],), outputs=(empty_scale.tensor,), body=Region(sbody),
            indexing_maps=ArrayAttr([AffineMapAttr(sid), AffineMapAttr(sid)]),
            iterator_types=par_iters, result_types=(scale_t,))

        for new in (reduce_gen, scale_gen):
            carry_provenance(new, call, "torchao_choose_qparams_affine")
        for new in (empty_acc, zero, seed, reduce_gen, empty_scale, scale_gen):
            parent.insert_op_before(new, call)
        call.results[0].replace_all_uses_with(scale_gen.results[0])
        parent.detach_op(call)
        n += 1

    for call in quant_calls:
        if len(call.operands) != 2:
            raise TorchAOAffineError(
                f"@{call.callee.string_value()} has {len(call.operands)} operands; this lowering "
                "handles the capture's (input, scale) shape, whose dropped zero point is provably "
                "zero for a SYMMETRIC scheme")
        src, scale = call.operands
        in_t, scale_t, out_t = src.type, scale.type, call.results[0].type
        if in_t.element_type != f32:
            raise TorchAOAffineError(
                f"@{call.callee.string_value()} quantizes a {in_t.element_type} input; only f32 "
                "has been verified bit-exact against torchao")
        out_elem = out_t.element_type
        if not isinstance(out_elem, IntegerType):
            raise TorchAOAffineError(
                f"@{call.callee.string_value()} returns {out_elem}, not an integer type")
        width = out_elem.width.data
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if not (lo <= spec.quant_min <= spec.quant_max <= hi):
            raise TorchAOAffineError(
                f"scheme {scheme!r} clamps to [{spec.quant_min}, {spec.quant_max}], outside the "
                f"range of the call's own result type {out_elem}")
        layout = derive_block_layout(in_t.get_shape(), scale_t.get_shape())
        if layout.granularity != spec.granularity:
            raise TorchAOAffineError(
                f"@{call.callee.string_value()}: types imply {layout.granularity} "
                f"(block_size {layout.block_size}) but scheme {scheme!r} is {spec.granularity}")
        rank, kept = len(layout.in_shape), len(layout.scale_shape)
        identity, scale_map = _maps(rank, kept, layout.block_size)

        parent = call.parent_block()
        empty = tensor.EmptyOp((), out_t)
        body = Block(arg_types=[f32, f32, out_elem])
        xv, sv, _ = body.args
        one = _f32_const(1.0)
        qmin = _f32_const(float(spec.quant_min))
        qmax = _f32_const(float(spec.quant_max))
        recip = arith.DivfOp(one.result, sv)                 # 1.0 / scale, THEN multiply
        prod = arith.MulfOp(xv, recip.result)
        rnd = math.RoundEvenOp(prod.result)                  # torch.round == ties-to-even
        clo = arith.MaximumfOp(rnd.result, qmin.result)
        chi = arith.MinimumfOp(clo.result, qmax.result)
        cast = arith.FPToSIOp(chi.result, out_elem)
        body.add_ops([one, qmin, qmax, recip, prod, rnd, clo, chi, cast, L.YieldOp(cast.result)])
        gen = L.GenericOp(
            inputs=(src, scale), outputs=(empty.tensor,), body=Region(body),
            indexing_maps=ArrayAttr([AffineMapAttr(identity), AffineMapAttr(scale_map),
                                     AffineMapAttr(identity)]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)
                                      for _ in range(rank)]),
            result_types=(out_t,))
        carry_provenance(gen, call, "torchao_quantize_affine")
        parent.insert_op_before(empty, call)
        parent.insert_op_before(gen, call)
        call.results[0].replace_all_uses_with(gen.results[0])
        parent.detach_op(call)
        n += 1

    # The private externs the calls referenced are now unused. Leaving them behind is not cosmetic:
    # every func in the module is compiled as a kernel, and a body-less one fails much later and
    # much further from its cause.
    for op in list(module.walk()):
        if op.name != "func.func":
            continue
        base = _base_symbol(op.sym_name.data)
        if base in (CHOOSE_QPARAMS_SYMBOL, QUANTIZE_SYMBOL) and not op.body.blocks:
            op.parent_block().detach_op(op)
    return n
