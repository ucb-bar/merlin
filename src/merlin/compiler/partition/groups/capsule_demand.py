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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.compiler.ir import contraction_shapes as KS
from merlin.compiler.ir import mlir_query as mq
from merlin.compiler.partition.groups.compute_groups import (
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

# ---- group_demand ------------------------------------------------------------------------------


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

    def walked() -> Mapping[int, tuple]:
        return KS.contraction_extents(root.parent_op().parent_op()) if root.parent_op() is not None else {}

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
        **({"source_scales": list(group.operand_sum["source_scales"])} if "source_scales" in group.operand_sum else {}),
        "bound_lsb": int(group.operand_sum["bound_lsb"]),
        "epilogue": ["relu"] if group.operand_sum.get("relu") else [],
        "scale_granularity": "tensor",
        "operand_dtype": group.in_dtype,
    }


# ---- feature_axis_trace ------------------------------------------------------------------------
# Where a rank-N activation keeps its features, DERIVED from the contractions its data flows through.
#
# :func:`merlin.compiler.partition.groups.capsule_demand.feature_axis` first reads the answer off the
# framework operators a capture records (a convolution's published signature fixes its activation's
# axis order). A capture whose convolutions were rewritten before export -- as an im2col of slices,
# transposes and reshapes feeding an integer ``matmul`` -- records no such operator, and then nothing
# it TAGS says where a rank-4 activation's features are. Its DATAFLOW still does, and this module reads
# it there:
#
# * a contraction's REDUCTION extent, on the operand that is not the stored one, is that activation's
#   input features -- an im2col packs each window's channels into it, with the window's taps;
# * a contraction's OUTPUT column -- the extent the stored operand contributes and the activation does
#   not -- is its output features, which the program's reshape and transpose then put back into a
#   rank-4 tensor.
#
# So the query tensor's axes are followed, as row-major FACTORS, through every op that only moves or
# relabels data (transposes, reshapes, slices, pads, concatenations, and elementwise ops whose maps are
# permutations) to the contractions it reaches in either direction. At each one the feature axis is the
# query axis whose whole non-unit extent lands in the feature extent and nowhere else: a window's taps
# split the image's spatial axes between the positions and the reduction, so only the channel axis lands
# wholly in the reduction; the positions of an output gather every spatial axis, so only its channel
# axis lands wholly in the column.
#
# FAIL CLOSED, NEVER GUESS. A step this module cannot follow exactly (a broadcast, a reduction, a
# reshape whose factors do not align, a slice that drops an axis) ends that path with no answer. A
# contraction where no axis -- or more than one -- lands wholly in the feature extent is AMBIGUOUS and
# answers nothing. Two paths that answer differently are a DISAGREEMENT and the caller refuses. Nothing
# here names a target, an operator spelling from a framework, or a layout.

#: Ops that re-chunk a tensor in row-major order without moving a byte.
_RESHAPES = ("tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape")
#: Ops that keep every axis and change only some extents (a window, a padded border).
_SAME_AXES = ("tensor.extract_slice", "tensor.pad", "tensor.cast")
#: Elementwise ops outside ``linalg`` whose FIRST operand is the data: a scale applied per tensor.
_ELEMENTWISE_ON_FIRST = ("quant_ext.quantize_per_tensor", "quant_ext.dequantize_per_tensor")
#: How many tensors a derivation will visit before it gives up and says so.
_MAX_VISITS = 20000


@dataclass
class _Dim:
    """One dim of a tensor on the path: the query axes it is made of, outermost first.

    ``factors`` are ``(query axis, extent)`` -- ``None`` for an extent that is not the query's (another
    concatenated operand's). ``exact`` is False once the dim's extent is no longer the product of its
    factors (a concatenation): it can still be read at a contraction but not reshaped. ``mixed`` marks
    a dim whose composition is unknown; no answer is read from a tensor that has one."""

    factors: list[tuple[int | None, int]] = field(default_factory=list)
    exact: bool = True
    mixed: bool = False

    def axes(self) -> set[int]:
        return {axis for axis, _ in self.factors if axis is not None}


@dataclass
class Derivation:
    """What the walk found: ``axis`` when every answering path agrees on one, with each answer's
    witness; ``why`` when it found none or several."""

    axis: int | None
    witnesses: dict[int, list[str]]
    ambiguous: list[str]
    why: str


def _shape(value: Any) -> list[int] | None:
    shape, _ = mq.type_shape_dtype(value.type)
    if not shape or any(not isinstance(e, int) or e < 0 for e in shape):
        return None
    return [int(e) for e in shape]


def _start(shape: Sequence[int]) -> list[_Dim]:
    return [_Dim([(axis, int(extent))] if int(extent) != 1 else []) for axis, extent in enumerate(shape)]


def _reshape(dims: Sequence[_Dim], to: Sequence[int]) -> list[_Dim] | None:
    """``dims`` re-chunked row-major into extents ``to``; None when the factors do not align."""
    if any(d.mixed or not d.exact for d in dims):
        return None
    queue = [list(f) for d in dims for f in d.factors if f[1] != 1]
    out: list[_Dim] = []
    for extent in to:
        need, taken = int(extent), []
        while need > 1:
            if not queue:
                return None
            axis, size = queue[0]
            if size <= need and need % size == 0:
                taken.append((axis, size))
                need //= size
                queue.pop(0)
            elif size > need and size % need == 0:
                # The OUTER part of this factor fills the dim; its inner part stays for the next.
                taken.append((axis, need))
                queue[0][1] = size // need
                need = 1
            else:
                return None
        out.append(_Dim(taken))
    return out if not queue else None


def _same_axes(dims: Sequence[_Dim], to: Sequence[int]) -> list[_Dim] | None:
    """The same axes at new extents (a slice, a pad): a dim of one factor keeps its axis; a dim of
    several cannot say how a changed extent splits among them, and becomes unknown."""
    if len(dims) != len(to):
        return None
    out = []
    for dim, extent in zip(dims, to, strict=True):
        if int(extent) == dim_extent(dim):
            out.append(_Dim(list(dim.factors), dim.exact, dim.mixed))
        elif len(dim.factors) == 1 and dim.exact:
            out.append(_Dim([(dim.factors[0][0], int(extent))]) if int(extent) != 1 else _Dim([]))
        elif not dim.factors and int(extent) == 1:
            out.append(_Dim([]))
        else:
            out.append(_Dim([], mixed=True))
    return out


def dim_extent(dim: _Dim) -> int:
    total = 1
    for _, size in dim.factors:
        total *= int(size)
    return total


def _permutation(results: Sequence[Any], loops: int) -> list[int] | None:
    """The loop dim each operand dim names, when the map is a permutation of every loop dim."""
    positions = [KS._dim_position(expr) for expr in results]
    if len(positions) != loops or any(p is None for p in positions) or len(set(positions)) != loops:
        return None
    return [int(p) for p in positions]


def _iterators(op: Any) -> list[str] | None:
    its = KS._iterator_types(op)
    return [str(it) for it in its] if its is not None else None


def _is_contraction(op: Any) -> bool:
    from merlin.compiler.partition.groups.compute_groups import CONTRACTION, classify

    stage = classify(op)
    return stage is not None and stage.kind == CONTRACTION


def _n_inputs(op: Any) -> int:
    return len(getattr(op, "inputs", ())) or max(len(op.operands) - len(op.results), 0)


def _arguments_behind(value: Any, *, stop_at_contraction: bool) -> set[int] | None:
    """The function arguments ``value`` is computed from; None when ``stop_at_contraction`` and a
    contraction or a reduction lies behind it (it is then an activation, not a stored operand)."""
    from xdsl.ir import BlockArgument

    found: set[int] = set()
    seen: set[int] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BlockArgument):
            found.add(id(current))
            continue
        owner = current.owner
        if stop_at_contraction and (_is_contraction(owner) or "reduction" in (_iterators(owner) or [])):
            return None
        stack.extend(owner.operands)
    return found


def _weight_like(op: Any, index: int) -> bool:
    """Whether a contraction's input ``index`` is a STORED operand: computed from function arguments
    alone, through no contraction or reduction, and sharing none of them with the other input."""
    inputs = list(op.operands)[: _n_inputs(op)]
    mine = _arguments_behind(inputs[index], stop_at_contraction=True)
    other = _arguments_behind(inputs[1 - index], stop_at_contraction=False) or set()
    return mine is not None and not (mine & other)


def _feature_dim(op: Any, *, operand: int | None) -> tuple[int | None, str]:
    """The dim of ``op``'s operand ``operand`` (or of its result, when None) that carries features.

    Reached through an operand, the path IS an activation; the other input must be stored, or this is
    a product of two activations (attention's scores), whose reduction need not be anyone's features.
    Reached through the result, exactly one input must be stored: its extent the activation does not
    share is the output's features. Both inputs reading only arguments (a first layer, whose activation
    is the model's own input) leaves neither side named, and answers nothing."""
    maps = KS.indexing_maps(op)
    iterators = _iterators(op)
    if maps is None or iterators is None or _n_inputs(op) != 2:
        return None, "a contraction whose maps cannot be read"
    reduction = {i for i, it in enumerate(iterators) if "reduction" in it}
    if operand is not None:
        if not _weight_like(op, 1 - operand):
            return None, "the other operand is not a stored one, so this reduction need not be features"
        dims = [j for j, expr in enumerate(maps[operand]) if KS._dim_position(expr) in reduction]
        return (dims[0], "reduction") if len(dims) == 1 else (None, "the reduction is spread over several dims")
    stored = [index for index in (0, 1) if _weight_like(op, index)]
    if len(stored) != 1:
        return None, "neither input, or both, is a stored operand, so no output extent is the weight's"
    stored_loops = {KS._dim_position(expr) for expr in maps[stored[0]]} - reduction
    other_loops = {KS._dim_position(expr) for expr in maps[1 - stored[0]]}
    columns = stored_loops - other_loops
    dims = [j for j, expr in enumerate(maps[-1]) if KS._dim_position(expr) in columns]
    return (dims[0], "output column") if len(dims) == 1 else (None, "the stored operand spans several output dims")


def _answer(dims: Sequence[_Dim], feature: int) -> int | None:
    """The query axis whose whole non-unit extent lands in dim ``feature`` and nowhere else."""
    if any(d.mixed for d in dims):
        return None
    landed = [
        axis
        for axis in dims[feature].axes()
        if not any(axis in dim.axes() for index, dim in enumerate(dims) if index != feature)
    ]
    return landed[0] if len(landed) == 1 else None


def _forward(op: Any, index: int, dims: list[_Dim]) -> Iterable[tuple[Any, list[_Dim]]]:
    """Where the query's axes go when ``op`` reads them as operand ``index``."""
    name = mq.op_name(op)
    if not op.results:
        return
    result = op.results[0]
    to = _shape(result)
    if to is None:
        return
    if name == "linalg.transpose" and index == 0:
        from merlin.compiler.partition.groups.compute_groups import _int_array

        permutation = _int_array(op, "permutation")
        if permutation and len(permutation) == len(dims):
            yield result, [dims[p] for p in permutation]
    elif name in _RESHAPES and index == 0:
        moved = _reshape(dims, to)
        if moved is not None:
            yield result, moved
    elif (name in _SAME_AXES and index == 0) or (name == "tensor.insert_slice" and index == 0):
        moved = _same_axes(dims, to)
        if moved is not None:
            yield result, moved
    elif name in _ELEMENTWISE_ON_FIRST and index == 0 and _shape(op.operands[0]) == to:
        yield result, [_Dim(list(d.factors), d.exact, d.mixed) for d in dims]
    elif name == "tensor.concat":
        axis = _concat_dim(op)
        if axis is not None and len(dims) == len(to):
            moved = [_Dim(list(d.factors), d.exact, d.mixed) for d in dims]
            moved[axis].exact = False  # the other operands' extent joins it; it is no longer a product
            yield result, moved
    elif name == "linalg.generic" and index < _n_inputs(op):
        iterators = _iterators(op)
        maps = KS.indexing_maps(op)
        if iterators is None or maps is None or any("parallel" not in it for it in iterators):
            return
        mine = _permutation(maps[index], len(iterators))
        theirs = _permutation(maps[-1], len(iterators))
        if mine is None or theirs is None:
            return
        by_loop = {loop: dims[j] for j, loop in enumerate(mine)}
        yield result, [_Dim(list(by_loop[loop].factors), by_loop[loop].exact, by_loop[loop].mixed) for loop in theirs]


def _backward(value: Any, dims: list[_Dim]) -> Iterable[tuple[Any, list[_Dim]]]:
    """Where the query's axes come from: the operands of the op that produced ``value``."""
    from xdsl.ir import BlockArgument

    if isinstance(value, BlockArgument):
        return
    op = value.owner
    name = mq.op_name(op)
    source = op.operands[0] if op.operands else None
    if source is None:
        return
    to = _shape(source)
    if to is None:
        return
    if name == "linalg.transpose":
        from merlin.compiler.partition.groups.compute_groups import _int_array

        permutation = _int_array(op, "permutation")
        if permutation and len(permutation) == len(dims):
            moved: list[_Dim] = [_Dim() for _ in dims]
            for out_axis, in_axis in enumerate(permutation):
                moved[in_axis] = dims[out_axis]
            yield source, moved
    elif name in _RESHAPES:
        moved_r = _reshape(dims, to)
        if moved_r is not None:
            yield source, moved_r
    elif name in _SAME_AXES or name == "tensor.insert_slice":
        moved_s = _same_axes(dims, to)
        if moved_s is not None:
            yield source, moved_s
    elif name in _ELEMENTWISE_ON_FIRST and _shape(value) == to:
        yield source, [_Dim(list(d.factors), d.exact, d.mixed) for d in dims]
    elif name == "linalg.generic":
        iterators = _iterators(op)
        maps = KS.indexing_maps(op)
        if iterators is None or maps is None or any("parallel" not in it for it in iterators):
            return
        theirs = _permutation(maps[_n_inputs(op) + list(op.results).index(value)], len(iterators))
        if theirs is None:
            return
        by_loop = {loop: dims[j] for j, loop in enumerate(theirs)}
        for index, operand in enumerate(list(op.operands)[: _n_inputs(op)]):
            mine = _permutation(maps[index], len(iterators))
            if mine is not None:
                yield operand, [_Dim(list(by_loop[k].factors), by_loop[k].exact, by_loop[k].mixed) for k in mine]


def _concat_dim(op: Any) -> int | None:
    for table in mq._attr_tables(op):
        raw = table.get("dim")
        if raw is None:
            continue
        value = getattr(raw, "value", raw)
        data = getattr(value, "data", value)
        if isinstance(data, int):
            return int(data)
    return None


def derive_feature_axis(values: Sequence[Any]) -> Derivation:
    """The feature axis shared by ``values`` (tensors of one rank, one layout -- an elementwise op's
    result and operands), read off every contraction their data reaches. See the module docstring."""
    rank = None
    pending: list[tuple[Any, list[_Dim]]] = []
    for value in values:
        shape = _shape(value)
        if shape is None:
            continue
        if rank is None:
            rank = len(shape)
        if len(shape) == rank:
            pending.append((value, _start(shape)))
    witnesses: dict[int, list[str]] = {}
    ambiguous: list[str] = []
    seen: set[int] = set()
    visits = 0
    while pending:
        value, dims = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        visits += 1
        if visits > _MAX_VISITS:
            return Derivation(None, witnesses, ambiguous, f"the dataflow walk passed {_MAX_VISITS} tensors")
        owner = getattr(value, "owner", None)
        if owner is not None and hasattr(owner, "results") and value in list(owner.results) and _is_contraction(owner):
            # Reached from below: the query was computed from this contraction's result.
            feature, how = _feature_dim(owner, operand=None)
            axis = _answer(dims, feature) if feature is not None else None
            (witnesses.setdefault(axis, []) if axis is not None else ambiguous).append(
                f"{mq.op_name(owner)} {how}" if axis is not None else f"{mq.op_name(owner)}: {how}"
            )
            continue
        for use in value.uses:
            user, index = use.operation, use.index
            if _is_contraction(user):
                feature, how = _feature_dim(user, operand=index)
                axis = _answer(dims, feature) if feature is not None else None
                if axis is not None:
                    witnesses.setdefault(axis, []).append(f"{mq.op_name(user)} {how} of operand {index}")
                else:
                    ambiguous.append(f"{mq.op_name(user)} operand {index}: {how}")
                continue
            pending.extend(_forward(user, index, dims))
        pending.extend(_backward(value, dims))
    if len(witnesses) == 1:
        (axis,) = witnesses
        return Derivation(axis, witnesses, ambiguous, "")
    if not witnesses:
        why = "no contraction this tensor's data reaches states, without ambiguity, which of its axes is features"
        if ambiguous:
            why += f" ({'; '.join(sorted(set(ambiguous))[:4])})"
        return Derivation(None, witnesses, ambiguous, why)
    disagreement = ", ".join(f"{len(w)} path(s) put them at {a}" for a, w in sorted(witnesses.items()))
    return Derivation(None, witnesses, ambiguous, f"the contractions this tensor reaches disagree: {disagreement}")


__all__ = [
    "Derivation",
    "derive_feature_axis",
]
