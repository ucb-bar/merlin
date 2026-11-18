"""Compute groups: which operations of a model run together on which unit, decided once.

A model is lowered onto an accelerator one contraction at a time, and everything that is not a
contraction is left where it falls. That is where a model's time goes: the bias, the activation,
the requantization and the pool that follow every contraction are per-element work, the hardware
that ran the contraction can usually apply them on its way out, and nothing asked it to.

This pass forms dispatch regions in the sense of a dataflow compiler's dispatch formation, on
linalg-on-tensors, before any routing:

* a **root** is a contraction;
* the root grows a **consumer closure**: each operation that consumes the group's single live
  value is absorbed while the target admits that stage attached to a contraction, at the
  granularity the target's readout was derived to hold;
* a root enclosed by dequantize producers is an **integer region**: its operands are the integer
  tensors behind the dequantizes, which is what decides whether an integer unit can take it;
* every operation that is absorbed by no group becomes an **explicit host group**, carrying the
  refusal that put it there and the class of owner that can change it.

The result is a partition. Every computation-carrying operation is in exactly one group, so "fell
back to the host" is a row with a reason and never an absence.

Nothing here names a target, an instruction or a register, and nothing here names a WORKLOAD either.
What a stage IS is read from the operation's structure; which stages a contraction's group may carry
out at all is asked of the target's declared readout (:meth:`TargetOracle.absorbs`, over
``readout_epilogue_capability`` reached through :mod:`merlin.target.capabilities.readout_facet`); and whether
this instance of one is admitted is asked of the capability contract
(:mod:`merlin.target.capabilities.eligibility`). That first question used to be a module constant listing one
workload's conv-block epilogue, which short-circuited the oracle: a stage outside the list stopped
growth with no refusal to show, and the ``None`` was stringified into a gap class.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.compiler.ir import contraction_shapes as KS
from merlin.compiler.ir import mlir_query as mq
from merlin.compiler.lower.options.quantization_scope import value_preserving_input
from merlin.compiler.partition import outline as CC
from merlin.compiler.partition.groups.generic_body import (
    _CMPF_ORDERED,
    _CMPI_SIGNED,
    _ZERO_CASTS,
    _body_constants,
    _body_ops,
    _is_rectifier,
    _is_zero,
)
from merlin.compiler.partition.groups.group_composition import readout_absorbs
from merlin.compiler.partition.groups.group_target_admission import Admission, TargetOracle
from merlin.compiler.partition.groups.scale_geometry import (
    _axis_through,
    _granularity,
    _int_array,
    _join,
    _root_axes,
)

# ---- compute_groups ----------------------------------------------------------------------------

SCHEMA = "compute_groups_v1"


HOST = "host"


INSIDE_REDUCTION = "inside_reduction"


UNREAD_SCALE = "scale_extent_unread"


OPERAND_SUM = "operand_sum"


WINDOW_MEAN = "window_mean"


READOUT_DOES_NOT_APPLY = "readout_does_not_apply"


READOUT_UNDECLARED = "readout_epilogue_undeclared"


READOUT_REQUIRES_SCALE = "readout_requires_scale"


FLOAT_EPILOGUE_REASSOCIATION = "float_epilogue_reassociation"


ACCUMULATE_UNPROVED = "accumulate_exactness_unproved"


STRUCTURAL = "graph_structure"


NO_DEVICE_FORM = "no_device_form"


UNRECORDED_REFUSAL = "refusal_not_recorded"


CONTRACTION = "contraction"


BIAS_ADD = "bias_add"


RESIDUAL_ADD = "residual_add"


SCALE = "scale"


ROUND = "round"


CLAMP = "clamp"


RELU = "relu"


CAST = "cast"


POOL = "pool"


PAD = "pad"


QUANTIZE = "quantize"


DEQUANTIZE = "dequantize"


MOVEMENT = "movement"


VIEW = "view"


REDUCTION = "reduction"


ELEMENTWISE = "elementwise"


FAMILY_OF_STAGE: dict[str, str] = {
    CONTRACTION: "contraction",
    BIAS_ADD: "elementwise_map",
    RESIDUAL_ADD: "elementwise_map",
    SCALE: "elementwise_map",
    ROUND: "elementwise_map",
    CLAMP: "elementwise_map",
    RELU: "elementwise_map",
    CAST: "elementwise_map",
    QUANTIZE: "elementwise_map",
    DEQUANTIZE: "elementwise_map",
    ELEMENTWISE: "elementwise_map",
    POOL: "reduction",
    REDUCTION: "reduction",
    PAD: "movement",
    MOVEMENT: "movement",
    VIEW: "movement",
}


_VIEW_OPS = ("tensor.expand_shape", "tensor.collapse_shape", "tensor.cast", "tensor.reshape")


CONVERSION_OF_SCALED_STORE = frozenset({ROUND, CLAMP, CAST})


_NOT_ARITHMETIC = frozenset({VIEW, PAD, MOVEMENT, CONTRACTION})


SCALED = frozenset({SCALE, QUANTIZE})


def epilogue_stage_names() -> dict[str, str]:
    """Stage kind -> the name a readout DECLARES it under, where the two spell it differently.

    A declaration is written in the command-buffer ABI's vocabulary
    (:data:`merlin.runtime.engine.command_buffer.EPILOGUE_STAGES`), which spells a quantizing scale
    ``acc_scale`` and a window max ``maxpool``. ONE table, so what a group DEMANDS of a readout
    (:func:`readout_absorbs`) and what a capsule entry ASKS for (:func:`capsule_entry`) cannot drift.
    A kind ABSENT from it is asked about under its OWN name, which is what keeps this
    target-agnostic: a target declaring a stage this module never heard of is answered from its own
    declaration, so adding a capability is a change to the target and never to the table.
    """
    from merlin.runtime.engine.command_buffer import BIAS_STAGES

    return {BIAS_ADD: BIAS_STAGES[0], RELU: "relu", QUANTIZE: "acc_scale", SCALE: "acc_scale", POOL: "maxpool"}


def declared_stage_name(kind: str) -> str:
    """The name ``kind`` is declared under, including the conversion stages carried by the scale."""
    names = epilogue_stage_names()
    return names[SCALE] if kind in CONVERSION_OF_SCALED_STORE else names.get(kind, kind)


_ADD = {"arith.addf", "arith.addi"}


_MUL = {"arith.mulf", "arith.muli"}


_DIV = {"arith.divf"}


_MAX = {"arith.maximumf", "arith.maxnumf", "arith.maxsi", "arith.maxf"}


_MIN = {"arith.minimumf", "arith.minnumf", "arith.minsi", "arith.minf"}


_ROUND = {"math.roundeven", "math.round", "math.rint"}


_CAST = {
    "arith.sitofp",
    "arith.fptosi",
    "arith.trunci",
    "arith.extsi",
    "arith.extf",
    "arith.truncf",
    "arith.uitofp",
    "arith.fptoui",
}


_BODY_NOISE = {"linalg.yield", "arith.constant"}


_PRODUCERS = ("arith.constant", "tensor.empty", "tensor.splat", "linalg.fill")


@dataclass(frozen=True)
class Stage:
    """One operation's role, with the facts admission needs."""

    kind: str
    #: Iteration dims a scale/bias operand varies over; ``()`` for a scalar, ``None`` when unread.
    varies_over: tuple[int, ...] | None = None
    #: The per-axis quantize/dequantize axis of the operand tensor, when the op declares one.
    axis: int | None = None
    detail: str = ""


@dataclass
class Group:
    index: int
    placement: str  # a unit name, or HOST
    root: Any | None  # the contraction op, or None for a host group
    members: list[Any] = field(default_factory=list)  # ops in program order
    stages: list[str] = field(default_factory=list)  # stage kinds in program order
    in_dtype: str | None = None
    weight_dtype: str | None = None
    scale_granularity: str | None = None
    reason: str | None = None  # why this is on the host, or why growth stopped
    refusal: str | None = None
    stopped_by: str | None = None  # the stage kind that ended an accelerator group's growth
    stopped_at: Any | None = None  # the operation growth stopped at; it opens a host group
    gap: str | None = None  # owner class of a host placement
    macs: int | None = None
    elements: int = 0  # output elements of the group's non-root stages
    #: Set when the group is an integer SUM of two quantized tensors and not a contraction: the two
    #: multipliers into the output's domain, the unit's bound, and whether an activation follows.
    #: ``root`` is then the add.
    operand_sum: dict[str, Any] | None = None
    #: Set when the group is a MEAN over a trailing window, legalized as a contraction against a
    #: constant one: the rows kept, the window summed, and the one readout multiplier that carries
    #: both scales and the reciprocal of the count. ``root`` is then the reduction.
    window_mean: dict[str, Any] | None = None
    window_pool: dict[str, Any] | None = None
    #: The output-fusion pass's decision for this root (:mod:`.output_fusion`): the cut taken, the
    #: stages absorbed, the proof and every longer cut's refusal. ``None`` when the pass did not run.
    fusion: dict[str, Any] | None = None

    def key(self) -> dict[str, Any]:
        """What makes two groups the same program: stages, numerics, extents. Not position."""
        shape = None
        if self.root is not None and self.root.results:
            shape = mq.type_shape_dtype(self.root.results[0].type)[0]
        return {
            "stages": list(self.stages),
            "in_dtype": self.in_dtype,
            "weight_dtype": self.weight_dtype,
            "scale_granularity": self.scale_granularity,
            "root_result_shape": shape,
            **({"operand_sum": dict(self.operand_sum)} if self.operand_sum else {}),
            **({"window_mean": dict(self.window_mean)} if self.window_mean else {}),
            **({"window_pool": dict(self.window_pool)} if self.window_pool else {}),
            **({"fusion_cut": self.fusion["cut"]} if self.fusion else {}),
        }


def _dims_of(expr) -> set[int]:
    position = KS._dim_position(expr)
    if position is not None:
        return {position}
    found: set[int] = set()
    for side in ("lhs", "rhs"):
        inner = getattr(expr, side, None)
        if inner is not None:
            found |= _dims_of(inner)
    return found


def _operand_dims(op) -> list[set[int]] | None:
    maps = KS.indexing_maps(op)
    if maps is None:
        return None
    return [set().union(*[_dims_of(expr) for expr in results]) if results else set() for results in maps]


def _stride_of(expr) -> int | None:
    """``s`` in an index expression ``d_out * s + d_window``; 1 when the output dim is unscaled."""
    if KS._dim_position(expr) is not None:
        return 1
    sides = [getattr(expr, "lhs", None), getattr(expr, "rhs", None)]
    if None in sides:
        return None
    strides = []
    for side in sides:
        if KS._dim_position(side) is not None:
            strides.append(1)
            continue
        factors = [getattr(side, "lhs", None), getattr(side, "rhs", None)]
        constants = [getattr(f, "value", None) for f in factors if isinstance(getattr(f, "value", None), int)]
        if len(constants) != 1 or not any(KS._dim_position(f) is not None for f in factors):
            return None
        strides.append(constants[0])
    # One side is the scaled output dim and the other the window dim, which is never scaled.
    return max(strides) if 1 in strides else None


def _is_windowed(op) -> bool:
    maps = KS.indexing_maps(op) or []
    return any(KS._dim_position(expr) is None and _dims_of(expr) for results in maps for expr in results)


def _n_inputs(op) -> int:
    return len(getattr(op, "inputs", ())) or max(len(op.operands) - len(op.results), 0)


def classify(op) -> Stage | None:
    """The stage an operation is, or ``None`` when it carries no computation of its own."""
    name = mq.op_name(op)
    short = name.rpartition(".")[2]
    if name in _PRODUCERS or name in ("func.return", "linalg.yield", "linalg.index"):
        return None
    if short.startswith("dequantize_"):
        return Stage(DEQUANTIZE, axis=_axis_of(op), varies_over=None if "per_tensor" in short else (), detail=short)
    if short.startswith("quantize_"):
        return Stage(QUANTIZE, axis=_axis_of(op), detail=short)
    if name == "tensor.insert_slice":
        return Stage(PAD)
    if name in _VIEW_OPS:
        return Stage(VIEW)
    if name.startswith("tensor.") or name in ("linalg.transpose", "linalg.broadcast"):
        return Stage(MOVEMENT)
    if not name.startswith("linalg."):
        return None
    if name != "linalg.generic":
        named = name.removeprefix("linalg.")
        if "matmul" in named or named.startswith("conv") or named in ("vecmat", "matvec", "dot"):
            return Stage(CONTRACTION, detail=named)
        if named.startswith("pooling"):
            return Stage(POOL, detail=named)
        return Stage(ELEMENTWISE, detail=named)

    structure = CC.classify_generic(op)
    if structure == "contraction":
        return Stage(CONTRACTION)
    if structure in ("max-reduction", "sum-reduction"):
        return Stage(POOL if _is_windowed(op) else REDUCTION, detail=structure)
    if structure in ("absmax", "other-reduction"):
        return Stage(REDUCTION, detail=structure)
    if structure == "movement":
        return Stage(MOVEMENT)

    body = [b for b in _body_ops(op) if b not in _BODY_NOISE]
    kinds = set(body)
    dims = _operand_dims(op)
    n_in = _n_inputs(op)
    out_dims = dims[-1] if dims else None
    narrow = None
    if dims and out_dims is not None:
        narrowed = [d for d in dims[:n_in] if d != out_dims]
        if len(narrowed) == 1:
            narrow = tuple(sorted(narrowed[0]))
    if kinds and kinds <= _ADD:
        if n_in == 2 and narrow is not None:
            return Stage(BIAS_ADD, varies_over=narrow)
        return Stage(RESIDUAL_ADD if n_in == 2 else ELEMENTWISE)
    if kinds and kinds <= (_MUL | _DIV):
        if n_in == 1:
            return Stage(SCALE, varies_over=())  # by a constant inside the body
        if n_in == 2 and narrow is not None:
            return Stage(SCALE, varies_over=narrow)
        if n_in == 2:
            # A scale is often materialised to the full shape first. It is still a scale: what it
            # varies over is what its broadcast was built from.
            spread = [_broadcast_dims(operand) for operand in list(op.operands)[:2]]
            known = [dims for dims in spread if dims is not None]
            if len(known) == 1:
                return Stage(SCALE, varies_over=known[0])
        return Stage(ELEMENTWISE, detail="product of two full tensors")
    if kinds and kinds <= (_MAX | _MIN):
        constants = _body_constants(op)
        if kinds <= _MAX and constants == [0.0]:
            return Stage(RELU)
        return Stage(CLAMP if n_in == 1 else ELEMENTWISE)
    if n_in == 1 and _is_rectifier(op):
        return Stage(RELU)
    if kinds and kinds <= _ROUND:
        return Stage(ROUND)
    if kinds and kinds <= _CAST:
        return Stage(CAST)
    return Stage(ELEMENTWISE, detail=" ".join(sorted(kinds)))


def _is_scalar_fill(owner) -> bool:
    """``tensor.splat`` / ``linalg.fill`` whose filled value is a rank-0 scalar."""
    operands = list(owner.operands)
    if not operands:
        return False
    shape, _ = mq.type_shape_dtype(operands[0].type)
    return not shape


def _broadcast_dims(value) -> tuple[int, ...] | None:
    """Output dims a broadcast's SOURCE varies over, or ``None`` when ``value`` is not a broadcast."""
    owner = getattr(value, "owner", None)
    if owner is None or not hasattr(owner, "operands"):
        return None
    if mq.op_name(owner) in ("tensor.splat", "linalg.fill") and _is_scalar_fill(owner):
        # A scalar spread to the full shape varies over no dim: it is a per-tensor value.
        return ()
    if mq.op_name(owner) != "linalg.generic":
        return None
    if CC.classify_generic(owner) != "movement" or _n_inputs(owner) != 1:
        return None
    dims = _operand_dims(owner)
    if not dims or len(dims) < 2 or not dims[0] < dims[-1]:
        return None
    return tuple(sorted(dims[0]))


def _axis_of(op) -> int | None:
    for table in mq._attr_tables(op):
        raw = table.get("axis")
        data = getattr(getattr(raw, "value", None), "data", None)
        if isinstance(data, int):
            return data
    return None


_NAMED_DIMS: dict[str, tuple[set[int], set[int], set[int]]] = {
    "linalg.matmul": ({0, 2}, {2, 1}, {0, 1}),
    "linalg.batch_matmul": ({0, 1, 3}, {0, 3, 2}, {0, 1, 2}),
}


_NAMED_OPERAND_AXES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "linalg.matmul": ((0, 2), (2, 1)),
    "linalg.batch_matmul": ((0, 1, 3), (0, 3, 2)),
}


def _operand_axis_dims(root, operand_index: int, axis: int) -> set[int] | None:
    """The iteration dims axis ``axis`` of the root's operand ``operand_index`` is indexed by."""
    name = mq.op_name(root)
    if name in _NAMED_OPERAND_AXES:
        axes = _NAMED_OPERAND_AXES[name][operand_index]
        return {axes[axis]} if axis < len(axes) else None
    maps = KS.indexing_maps(root)
    if not maps or operand_index >= len(maps) or axis >= len(maps[operand_index]):
        return None
    return _dims_of(maps[operand_index][axis])


_product = CC._product


def _dequantize_granularity(dequantize, adapters: Sequence[Any], root, operand_index: int) -> str | None:
    """Granularity of the scale a dequantize carries into operand ``operand_index`` of ``root``."""
    stage = classify(dequantize)
    if stage is None:
        return None
    if stage.axis is None:
        return "tensor" if "per_tensor" in stage.detail else None
    shape, _ = mq.type_shape_dtype(dequantize.results[0].type)
    axis = _axis_through(adapters, stage.axis, shape)
    if axis is None:
        return None
    dims = _operand_axis_dims(root, operand_index, axis)
    axes = _root_axes(root)
    if dims is not None and axes is not None and not dims <= (axes[0] | axes[1]):
        return INSIDE_REDUCTION
    return _granularity(dims, axes)


def _users(value) -> list[Any]:
    return [use.operation for use in value.uses]


def _elements(op) -> int:
    if not op.results:
        return 0
    shape, _ = mq.type_shape_dtype(op.results[0].type)
    total = 1
    for extent in shape:
        total *= max(int(extent), 0)
    return total if shape else 0


def _dtype_token(mlir_dtype: str | None) -> str | None:
    """An MLIR element type in the quant-format registry's spelling, or itself when unknown."""
    if not mlir_dtype:
        return None
    from merlin.common import target_data as qf

    for candidate in (
        mlir_dtype,
        "int" + mlir_dtype[1:] if mlir_dtype.startswith("i") else "",
        {"f32": "float32", "f16": "float16", "f64": "float64"}.get(mlir_dtype, ""),
    ):
        if candidate and qf.has(candidate):
            return qf.get(candidate).name
    return mlir_dtype


def _input_chain(value) -> tuple[list[Any], Any | None, str | None]:
    """``(adapters, dequantize, integer dtype)`` behind one root operand.

    Adapters are the padding and movement between a dequantize and the root: they change where
    elements sit and not what they are, and a unit that pads or strides on the way in takes them
    with the contraction. The walk ends at the first operation that is neither.
    """
    adapters: list[Any] = []
    for _ in range(8):
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "operands"):
            break
        stage = classify(owner)
        if stage is not None and stage.kind == DEQUANTIZE:
            _, dtype = mq.type_shape_dtype(owner.operands[0].type)
            return adapters, owner, dtype
        passes = stage is not None and stage.kind in (PAD, MOVEMENT, VIEW)  # or a copy / exact widening
        if not (passes or value_preserving_input(owner) is not None) or not owner.operands:
            break
        adapters.append(owner)
        value = owner.operands[0]
    return [], None, None


def _read_format(op) -> tuple[list[int], str | None]:
    """``(shape, element type)`` of the value ``op`` COMPUTES ON -- its first tensor operand.

    The scale and zero-point beside a quantize are rank-0 and are not what it reads; an operation
    that reads no tensor at all is described by its own result, the only format it has.
    """
    for operand in getattr(op, "operands", ()):
        shape, dtype = mq.type_shape_dtype(operand.type)
        if shape:
            return shape, dtype
    return mq.type_shape_dtype(op.results[0].type) if op.results else ([], None)


def _has_device_form(group: Group) -> bool:
    """Can a group be STATED as something to ask a device for? (:mod:`.group_demand` states it.)

    The three forms are the three that module holds: a contraction root, an integer operand sum, a
    mean over a window. This is the ONE place that enumerates them, so a fourth is added by teaching
    this function about it and not by a placement that discovers its absence four passes later.
    """
    return group.root is not None or group.operand_sum is not None or group.window_mean is not None


def _sole_user_is(op, consumers: set[int]) -> bool:
    return all(len(_users(result)) == 1 and id(_users(result)[0]) in consumers for result in op.results)
