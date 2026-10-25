"""What a compute group DEMANDS of a backend, in the words a capsule corpus is built from.

Group formation (:mod:`.compute_groups`) decides which operations run together on which unit. This
states one of those groups as an entry a backend can be asked for: the operation, its extents, and
the epilogue in the command-buffer's own stage names. A model route and a capsule route lower the
same pattern only if they are asked in the same words, so the capsule that CERTIFIES a pattern and
the model that NEEDS it are built from this one statement and cannot drift apart.

A group the vocabulary cannot state raises :class:`NoCapsuleForm` with the reason, and
:func:`.compute_group_demand.demand` counts those rather than dropping them: a model must not read as
fully demanded because the part nobody could state was quietly skipped. Nothing here names a target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.common import mlir_query as mq
from merlin.kernels import shapes as KS

from .compute_groups import (
    CAST,
    CLAMP,
    CONTRACTION,
    DEQUANTIZE,
    MOVEMENT,
    PAD,
    ROUND,
    VIEW,
    Group,
    _has_device_form,
    _is_windowed,
    _product,
    _stride_of,
    epilogue_stage_names,
)


class NoCapsuleForm(ValueError):
    """A group has a stage the capsule vocabulary cannot state, so no backend can be asked for it."""


def capsule_entry(
    group: Group, *, name: str | None = None, extents: Mapping[int, tuple] | None = None
) -> dict[str, Any]:
    """The capsule-corpus entry that demands exactly this group from a backend.

    A model route and a capsule route lower the same pattern only if they are asked in the same
    words. This states a group in the entry vocabulary the capsule builders already consume
    (``op``, extents, ``epilogue`` in the command-buffer stage names), so the capsule that
    certifies a pattern and the model that needs it cannot drift apart.
    """
    if not _has_device_form(group):
        raise NoCapsuleForm(
            f"a region placed on {group.placement!r} has no contraction, operand sum or window mean to demand it as"
        )
    if group.operand_sum is not None:
        return _operand_sum_entry(group, name=name)
    if group.window_mean is not None:
        return {
            "name": name or f"group_{group.index}",
            "kind": "op",
            "op": "matmul",
            "M": int(group.window_mean["rows"]),
            "K": int(group.window_mean["window"]),
            "N": 1,
            "epilogue": ["acc_scale"],
            "acc_scale": float(group.window_mean["multiplier"]),
            "scale_granularity": "tensor",
            "operand_dtype": group.in_dtype,
        }
    stage_name = epilogue_stage_names()
    # STAGES A READOUT PERFORMS NO ARITHMETIC FOR. Each is either the contraction itself, a
    # relocation of elements, or a piece of the quantization arithmetic that is folded into a
    # recorded scale rather than issued as a command.
    #
    # CAST is here on evidence, not by analogy. Measured on the M2 capture: every cast a group
    # carries is a `linalg.generic` whose body is exactly `arith.sitofp` + `yield`, with no scale
    # operand and no body constant -- a signed integer widened into a float box, values unchanged.
    # A readout that commits those integers computes the same function.
    #
    # IT IS SILENT AS ARITHMETIC AND LOAD-BEARING AS A TYPE, and the difference matters: the group
    # then commits an integer while the capture's consumer reads a float, so the program has to SAY
    # that the two are the same numbers. `llvmlower.whole_program` records it as a readout of
    # divisor 1.0 -- the same mechanism a classifier's dequantize uses -- and refuses a float escape
    # it cannot state that way. Making this silent WITHOUT that record would emit a buffer whose i8
    # output is read as f32 bit patterns.
    silent = {DEQUANTIZE, PAD, MOVEMENT, VIEW, CONTRACTION, ROUND, CLAMP, CAST}
    epilogue: list[str] = []
    for kind in group.stages:
        if kind in silent:
            continue
        if kind not in stage_name:
            raise NoCapsuleForm(f"stage {kind!r} has no name in the capsule epilogue vocabulary")
        if stage_name[kind] not in epilogue:
            epilogue.append(stage_name[kind])

    root = group.root
    out_shape, _ = mq.type_shape_dtype(root.results[0].type)
    entry: dict[str, Any] = {
        "name": name or f"group_{group.index}",
        "kind": "op",
        "epilogue": epilogue,
        "scale_granularity": group.scale_granularity,
        "operand_dtype": group.in_dtype,
    }
    if _is_windowed(root):
        maps = KS.indexing_maps(root)
        in_shape, _ = mq.type_shape_dtype(root.operands[0].type)
        w_shape, _ = mq.type_shape_dtype(root.operands[1].type)
        if not maps or len(in_shape) != 4 or len(w_shape) != 4:
            raise NoCapsuleForm("a windowed contraction whose geometry is not a 2-D convolution")
        strides = [_stride_of(expr) for expr in maps[0][2:4]]
        if None in strides:
            raise NoCapsuleForm("the convolution's strides could not be read from its index maps")
        entry.update(
            {
                "op": "conv2d",
                "ci": int(in_shape[1]),
                "N": int(w_shape[0]),
                "Himg": int(in_shape[2]),
                "Wimg": int(in_shape[3]),
                "kh": int(w_shape[2]),
                "kw": int(w_shape[3]),
                "stride": strides,
            }
        )
        return entry
    from merlin.targetgen import model_coverage

    def walked() -> Mapping[int, tuple]:
        return model_coverage._contraction_extents(root.parent_op().parent_op()) if root.parent_op() is not None else {}

    # One walk of the whole module. A caller stating many groups passes the table in, because
    # recomputing it per group is quadratic in the model; a root the table does not hold is read afresh.
    found = (extents if extents is not None else walked()).get(id(root))
    if not found and extents is not None:
        found = walked().get(id(root))
    if not found or None in found[:3]:
        raise NoCapsuleForm("the contraction's M/K/N extents could not be read")
    m, k, n, _rank = found
    entry.update({"op": "matmul", "M": int(m), "K": int(k), "N": int(n)})
    return entry


#: WHERE AN ACTIVATION KEEPS ITS FEATURES, AS THE CAPTURE'S OWN OPERATORS FIX IT.
#:
#: An activation of rank 3 or more has no self-evident axis order: ``[1, 256, 56, 56]`` is a
#: 256-channel 56x56 plane under one reading and a 56-channel 256x56 plane under another, and the
#: two differ by a permutation that no reshape performs. The capture states which it is, not
#: through a layout attribute -- it carries none -- but through the ATen operators it was exported
#: from. Each key below is an operator whose PUBLISHED SIGNATURE fixes the axis order of an
#: activation operand, mapped from that operand's RANK to where the features sit in it:
#: ``aten.conv2d`` takes ``(N, Cin, H, W)`` so a rank-4 activation has features at axis 1, while
#: ``aten.bmm`` takes ``(B, n, m)`` so a rank-3 one has them last. The rank matters because the
#: same capture can carry both, and an operator says nothing about a rank its signature does not
#: mention. These are the frameworks' contracts, readable in their own documentation, and they say
#: nothing about any target.
_ATEN_FEATURE_AXIS: Mapping[str, Mapping[int, int]] = {
    "aten.conv2d.default": {4: 1},
    "aten.convolution.default": {4: 1},
    "aten.max_pool2d.default": {4: 1},
    "aten.adaptive_avg_pool2d.default": {4: 1},
    "aten.avg_pool2d.default": {4: 1},
    "aten.batch_norm.default": {4: 1},
    "aten.bmm.default": {3: 2},
    "aten.baddbmm.default": {3: 2},
    "aten._softmax.default": {3: 2},
    "aten.linear.default": {3: 2},
}

#: The attribute a capture records an operation's originating framework operator under.
_PROVENANCE_ATTRIBUTE = "prov.aten"


def _aten_tag(op: Any) -> str:
    """The framework operator ``op`` was exported from, or ``""`` when it records none."""
    try:
        attributes = op.attributes
    except AttributeError:
        return ""
    stated = attributes.get(_PROVENANCE_ATTRIBUTE) if hasattr(attributes, "get") else None
    if stated is None:
        return ""
    return str(getattr(stated, "data", stated))


def _module_of(op: Any) -> Any:
    """The outermost operation containing ``op``."""
    outer = op
    while True:
        parent = outer.parent_op()
        if parent is None:
            return outer
        outer = parent


def feature_axis(op: Any, rank: int) -> int:
    """Where a rank-``rank`` activation keeps its features in the capture ``op`` belongs to.

    DERIVED, NOT ASSUMED. The answer comes from the ATen operators the capture records against its
    own operations: each of those fixes the axis order of an operand of some rank by its published
    signature, so a module containing them states its layout whether or not it carries a layout
    attribute. Only operators that speak about THIS rank are consulted -- a convolution says
    nothing about a rank-3 activation -- and operators indifferent to the order (an elementwise
    ``aten.add``) state nothing at any rank. That is why the tag on the operation being stated is
    the wrong place to look, and the module is the right one.

    WHEN NO OPERATOR STATES IT -- a capture whose convolutions were rewritten as an im2col of slices,
    transposes and reshapes feeding an integer matmul records none -- the axis is DERIVED from the
    dataflow instead (:func:`.feature_axis_trace.derive_feature_axis`): ``op``'s own rank-``rank``
    tensors are followed to the contractions they reach, where a reduction's extent is an activation's
    input features and an output column its output features. Every path that answers must agree.

    Raises :class:`NoCapsuleForm` when neither the capture's operators nor its dataflow state this
    rank's feature axis, or when either disagrees with itself. Guessing here is not a near miss: a
    wrong axis binds an operand of the right size in the wrong order, and produces a numerically wrong
    program that every shape check still passes.
    """
    axes: set[int] = set()
    witnesses: dict[int, str] = {}
    for inner in _module_of(op).walk():
        stated = _ATEN_FEATURE_AXIS.get(_aten_tag(inner))
        axis = stated.get(rank) if stated else None
        if axis is not None:
            axes.add(axis)
            witnesses.setdefault(axis, _aten_tag(inner))
    if not axes:
        from .feature_axis_trace import derive_feature_axis

        own = [
            value
            for value in (*getattr(op, "results", ()), *getattr(op, "operands", ()))
            if len(mq.type_shape_dtype(value.type)[0] or ()) == rank
        ]
        derived = derive_feature_axis(own)
        if derived.axis is not None:
            return derived.axis
        raise NoCapsuleForm(
            f"this capture records no framework operator whose signature fixes a rank-{rank} "
            f"activation's axis order, and its dataflow does not state it either: {derived.why}"
        )
    if len(axes) > 1:
        disagreement = ", ".join(f"{witnesses[axis]} puts them at {axis}" for axis in sorted(axes))
        raise NoCapsuleForm(f"the capture's own operators disagree on the feature axis: {disagreement}")
    return axes.pop()


def _activation_plane(shape: Sequence[int], op: Any) -> tuple[int, int]:
    """``shape`` as the ``[positions, features]`` matrix a device commits it in.

    A rank-2 tensor is already that matrix. A rank-4 activation is folded to it by collapsing every
    axis that is not the feature axis, which is where the positions live. The result is the shape
    the REST of the program speaks -- a convolution's operands, a prepacked weight and a spliced
    package's declared tensors are all stated in it -- so an elementwise group that flattened the
    tensor its own way would state the same bytes under a shape nothing else could bind.
    """
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) < 2:
        raise NoCapsuleForm(f"an elementwise group of rank {len(shape)} has no stated device plane")
    axis = feature_axis(op, len(shape))
    features = int(shape[axis])
    positions = _product([extent for index, extent in enumerate(shape) if index != axis])
    return int(positions), features


def _operand_sum_entry(group: Group, *, name: str | None) -> dict[str, Any]:
    """An integer sum of two tensors in the ``residual_add`` builder's words.

    An elementwise operation has no rows and columns of its own, so it is stated in the SAME
    ``[positions, features]`` matrix every other group states its tensors in (:func:`_activation_plane`).
    The arithmetic is indifferent to the folding -- contiguous storage means any flattening sums the
    same numbers -- but the BINDING is not: a neighbour commits ``[3136, 256]`` and a package declares
    its operand in that shape, so a sum that folded the same bytes to ``[14336, 56]`` states a tensor
    of the right size that nothing in the program can bind, and is refused as a permutation.
    """
    shape, _ = mq.type_shape_dtype(group.root.results[0].type)
    if not shape or any(not isinstance(extent, int) or extent <= 0 for extent in shape):
        raise NoCapsuleForm("the sum's extents are not static")
    positions, features = _activation_plane(shape, group.root)
    return {
        "name": name or f"group_{group.index}",
        "kind": "op",
        "op": "residual_add",
        "M": positions,
        "N": features,
        "lhs_scale": float(group.operand_sum["lhs_scale"]),
        "rhs_scale": float(group.operand_sum["rhs_scale"]),
        **({"source_scales": list(group.operand_sum["source_scales"])}
           if "source_scales" in group.operand_sum else {}),
        "bound_lsb": int(group.operand_sum["bound_lsb"]),
        "epilogue": ["relu"] if group.operand_sum.get("relu") else [],
        "scale_granularity": "tensor",
        "operand_dtype": group.in_dtype,
    }
