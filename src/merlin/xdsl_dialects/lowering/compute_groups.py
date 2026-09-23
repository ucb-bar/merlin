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
``readout_epilogue_capability`` reached through :mod:`merlin.targetgen.readout_facet`); and whether
this instance of one is admitted is asked of the capability contract
(:mod:`merlin.targetgen.eligibility`). That first question used to be a module constant listing one
workload's conv-block epilogue, which short-circuited the oracle: a stage outside the list stopped
growth with no refusal to show, and the ``None`` was stringified into a gap class.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.common import mlir_query as mq
from merlin.kernels import shapes as KS
from merlin.xdsl_dialects.lowering import contraction_coverage as CC

SCHEMA = "compute_groups_v1"
HOST = "host"
#: A scale that varies along a dim the contraction REDUCES. It cannot be factored out of the sum,
#: so the integers behind it are not the contraction's operands and no integer region exists.
INSIDE_REDUCTION = "inside_reduction"
#: Refusal of a scaled stage whose scale extent the capture does not express.
UNREAD_SCALE = "scale_extent_unread"
#: Refusal of an integer sum of two tensors the unit's scaled load cannot compute within a bound.
OPERAND_SUM = "operand_sum"
#: Refusal of a window mean whose integer form the unit cannot hold.
WINDOW_MEAN = "window_mean"
#: Refusal of a stage the target DECLARES its readouts do not apply. A group that carried it out
#: anyway would compute something the hardware discards on the way to memory.
READOUT_DOES_NOT_APPLY = "readout_does_not_apply"
#: Refusal of a stage on a target that declares no readout epilogue capability at all. UNKNOWN, and
#: never "applies everything": assuming would put a stage on a store path nobody established applies
#: it, which is the silent-discard defect :mod:`merlin.verify.epilogue_applicability` exists for.
READOUT_UNDECLARED = "readout_epilogue_undeclared"
#: Growth stopped by a fact about the GRAPH rather than about the target: a value read outside the
#: group, or a pad no pool consumes. Not a capability question and not a capability gap.
STRUCTURAL = "graph_structure"
#: The clause label of a stop that recorded no refusal. Every stop above names one, so nothing should
#: reach this -- and if something does it says so rather than naming a plausible cause, as its
#: predecessor ``"not_absorbable"`` did: that stood in for a capability answer nobody had asked for,
#: and was filed as a capability gap.
UNRECORDED_REFUSAL = "refusal_not_recorded"

# --- stage kinds: what an operation does, read from its structure ------------------------------
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

#: Stage kind -> the semantic family a target must provide for it.
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
#: Shape metadata: no element is read or written, so nothing is placed and nothing is owed.
_VIEW_OPS = ("tensor.expand_shape", "tensor.collapse_shape", "tensor.cast", "tensor.reshape")
#: Stages a readout declaration never names separately because they ARE the scaled store's own
#: conversion: the readout that applies the accumulator scale is the one that rounds, saturates and
#: narrows on the way out, and the one that dumps the accumulator raw does none of the three. So they
#: are ASKED about under the scale's name and EMITTED under none (see :func:`capsule_entry`'s
#: ``silent``).
CONVERSION_OF_SCALED_STORE = frozenset({ROUND, CLAMP, CAST})
#: Stages that touch no element's VALUE. They are placed, and they are not per-element arithmetic,
#: which is the work the element share measures.
_NOT_ARITHMETIC = frozenset({VIEW, PAD, MOVEMENT, CONTRACTION})
#: Stages that carry a scale, and so a granularity the readout must hold.
SCALED = frozenset({SCALE, QUANTIZE})


def epilogue_stage_names() -> dict[str, str]:
    """Stage kind -> the name a readout DECLARES it under, where the two spell it differently.

    A declaration is written in the command-buffer ABI's vocabulary
    (:data:`merlin.runtime.commandbuffer.EPILOGUE_STAGES`), which spells a quantizing scale
    ``acc_scale`` and a window max ``maxpool``. ONE table, so what a group DEMANDS of a readout
    (:func:`readout_absorbs`) and what a capsule entry ASKS for (:func:`capsule_entry`) cannot drift.
    A kind ABSENT from it is asked about under its OWN name, which is what keeps this
    target-agnostic: a target declaring a stage this module never heard of is answered from its own
    declaration, so adding a capability is a change to the target and never to the table.
    """
    from merlin.runtime.commandbuffer import BIAS_STAGES

    return {BIAS_ADD: BIAS_STAGES[0], RELU: "relu", QUANTIZE: "acc_scale", SCALE: "acc_scale", POOL: "maxpool"}


def declared_stage_name(kind: str) -> str:
    """The name ``kind`` is declared under, including the conversion stages carried by the scale."""
    names = epilogue_stage_names()
    return names[SCALE] if kind in CONVERSION_OF_SCALED_STORE else names.get(kind, kind)


def readout_absorbs(kind: str, readout: Any, *, target: str) -> "Admission":
    """Can ``target``'s DECLARED readout carry stage ``kind`` out of a contraction with it?

    The question a module constant used to answer. It listed one workload's conv-block epilogue --
    bias, scale, round, clamp, relu, cast, quantize, pool -- and every target got that list whether
    its store path applied those stages or not. A stage outside it stopped a group with no refusal to
    show, so the capability oracle was never reached and the missing refusal was stringified into a
    gap class. The target already declares the truth per readout (``readout_epilogue_capability``,
    reached through :meth:`merlin.targetgen.readout_facet.TargetReadout.applies_stage`); this asks
    it, and says no in two ways because they are two different facts: :data:`READOUT_UNDECLARED`,
    the target describes no readout so the answer is UNKNOWN and absorbing would assume it; and
    :data:`READOUT_DOES_NOT_APPLY`, the target described its readouts and none lists this stage.
    """
    stage = declared_stage_name(kind)
    applies = readout.applies_stage(stage) if readout is not None else None
    if applies is None:
        return Admission(
            False,
            READOUT_UNDECLARED,
            f"{target!r} declares no readout epilogue capability, so whether its store path applies "
            f"{stage!r} is unknown; absorbing {kind!r} would assume it",
        )
    if not applies:
        return Admission(
            False,
            READOUT_DOES_NOT_APPLY,
            f"no readout {target!r} declares applies {stage!r}, so a group carrying {kind!r} out would "
            f"compute something its store path discards",
        )
    return Admission(True)


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
        }


# --- structural classification -----------------------------------------------------------------
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


def _is_windowed(op) -> bool:
    maps = KS.indexing_maps(op) or []
    return any(KS._dim_position(expr) is None and _dims_of(expr) for results in maps for expr in results)


def _body_ops(op) -> list[str]:
    if not op.regions or not op.regions[0].blocks:
        return []
    return [mq.op_name(inner) for inner in op.regions[0].blocks[0].ops]


def _body_constants(op) -> list[float]:
    values: list[float] = []
    if not op.regions or not op.regions[0].blocks:
        return values
    for inner in op.regions[0].blocks[0].ops:
        if mq.op_name(inner) != "arith.constant":
            continue
        attribute = inner.properties.get("value")
        if attribute is None:  # not `or`: a zero-valued attribute is falsy
            attribute = inner.attributes.get("value")
        raw = getattr(attribute, "value", None)
        data = getattr(raw, "data", None)
        if isinstance(data, (int, float)):
            values.append(float(data))
    return values


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
    if kinds and kinds <= _ROUND:
        return Stage(ROUND)
    if kinds and kinds <= _CAST:
        return Stage(CAST)
    return Stage(ELEMENTWISE, detail=" ".join(sorted(kinds)))


def _broadcast_dims(value) -> tuple[int, ...] | None:
    """Output dims a broadcast's SOURCE varies over, or ``None`` when ``value`` is not a broadcast."""
    owner = getattr(value, "owner", None)
    if owner is None or not hasattr(owner, "operands") or mq.op_name(owner) != "linalg.generic":
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


# --- granularity: how a group's scale varies, relative to its root ------------------------------
#: Iteration dims of the named two-operand contractions, which carry no indexing maps of their own.
_NAMED_DIMS: dict[str, tuple[set[int], set[int], set[int]]] = {
    "linalg.matmul": ({0, 2}, {2, 1}, {0, 1}),
    "linalg.batch_matmul": ({0, 1, 3}, {0, 3, 2}, {0, 1, 2}),
}
_NAMED_OPERAND_AXES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "linalg.matmul": ((0, 2), (2, 1)),
    "linalg.batch_matmul": ((0, 1, 3), (0, 3, 2)),
}


def _root_axes(root) -> tuple[set[int], set[int]] | None:
    """``(row, column)`` output dims of a contraction: those only its first / second operand carries.

    One scale per ``row`` dim is a scale per accumulator row, per ``column`` dim per column, both
    is rank-1. Read from the indexing maps, so it holds for a matmul and a convolution alike.
    """
    name = mq.op_name(root)
    if name in _NAMED_DIMS:
        lhs, rhs, out = _NAMED_DIMS[name]
        return (lhs & out) - rhs, (rhs & out) - lhs
    dims = _operand_dims(root)
    iterators = KS._iterator_types(root)
    if not dims or len(dims) < 3 or not iterators:
        return None
    parallel = {i for i, kind in enumerate(iterators) if kind == "parallel"}
    lhs, rhs = dims[0] & parallel, dims[1] & parallel
    return lhs - rhs, rhs - lhs


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


def _granularity(varies: set[int] | None, axes: tuple[set[int], set[int]] | None) -> str | None:
    if varies is None:
        return None
    if not varies:
        return "tensor"
    if axes is None:
        return None
    row, column = axes
    if varies <= column:
        return "column"
    if varies <= row:
        return "row"
    if varies <= (row | column):
        return "rank1"
    return None


def _join(scales: Sequence[str | None]) -> str | None:
    """The one granularity that expresses every scale of a group; ``None`` if any is unknown.

    An unknown scale is not a per-tensor scale. Treating it as one admits a group whose
    arithmetic the hardware cannot hold, which is the defect this pass exists to name.
    """
    if not scales or any(scale is None for scale in scales):
        return None if scales else "tensor"
    kinds = set(scales) - {"tensor"}
    if not kinds:
        return "tensor"
    if kinds == {"row", "column"} or "rank1" in kinds:
        return "rank1" if kinds <= {"row", "column", "rank1"} else None
    return next(iter(kinds)) if len(kinds) == 1 else None


def _product(extents: Sequence[int]) -> int:
    total = 1
    for extent in extents:
        total *= int(extent)
    return total


def _axis_through(adapters: Sequence[Any], axis: int, source_shape: Sequence[int]) -> int | None:
    """Where tensor axis ``axis`` sits after the views and permutations in ``adapters``.

    ``adapters`` run from the root back toward the dequantize, so they are applied in reverse. A
    reshape keeps row-major order, so an axis is identified across it by its extent and by how many
    elements one step along it spans; it may vanish into a flattened dim and reappear, and is
    resolved only where an index is needed. A reshape that splits or merges the axis itself loses
    it, and so does a pad that changes its extent: the answer is then ``None``.
    """
    shape = [int(v) for v in source_shape]
    extent, inner = shape[axis], _product(shape[axis + 1 :])

    def resolve(current: Sequence[int]) -> int | None:
        for index, value in enumerate(current):
            if int(value) == extent and _product(current[index + 1 :]) == inner:
                return index
        return None

    for op in reversed(list(adapters)):
        out_shape = [int(v) for v in mq.type_shape_dtype(op.results[0].type)[0]]
        name = mq.op_name(op)
        if name in _VIEW_OPS:
            if _product(out_shape) != _product(shape):
                return None
        else:
            here = resolve(shape)
            if here is None:
                return None
            if name == "linalg.transpose":
                permutation = _int_array(op, "permutation") or []
                if here not in permutation:
                    return None
                here = permutation.index(here)
            elif len(out_shape) != len(shape) or out_shape[here] != extent:
                return None
            inner = _product(out_shape[here + 1 :])
        shape = out_shape
    return resolve(shape)


def _int_array(op, key: str) -> list[int] | None:
    for table in mq._attr_tables(op):
        raw = table.get(key)
        if raw is None:
            continue
        values = getattr(raw, "get_values", None)
        if callable(values):
            return [int(v) for v in values()]
        data = getattr(raw, "data", None)
        if isinstance(data, (list, tuple)):
            return [int(getattr(getattr(v, "value", v), "data", v)) for v in data]
    return None


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


# --- admission -----------------------------------------------------------------------------------
@dataclass
class Admission:
    """What the target says about one stage attached to a contraction."""

    admitted: bool
    refusal: str | None = None
    reason: str = ""
    units: tuple[str, ...] = ()


class TargetOracle:
    """The target's answers, asked through the same oracles the rest of the system uses."""

    def __init__(self, target: str, *, readout: Any | None = None):
        from merlin.targetgen import eligibility as E
        from merlin.targetgen import readout_facet

        self.target = target
        self._E = E
        self.cap_map = E.capability_map_for_target(target)
        self.undetermined = E.undetermined_families_for_target(target)
        self.providers = E.providers_for_target(target)
        self.readout = (
            readout if readout is not None else readout_facet.TargetReadout(tuple(readout_facet.for_target(target)))
        )

    def unit_for(
        self, in_dtype: str | None, weight_dtype: str | None, rank: int | None, providers: Sequence[str]
    ) -> str:
        """The unit the target's own router gives a contraction of these formats."""
        from merlin.targetgen import routing as R

        try:
            routed = R.route_plan(
                [
                    R.OpDemand(
                        op="matmul", in_fmt=in_dtype or "", weight_fmt=weight_dtype, rank=rank, family="contraction"
                    )
                ],
                self.target,
            )["results"]
            unit = getattr(routed[0], "unit", None) if routed else None
        except Exception:  # noqa: BLE001 -- an unroutable demand leaves the declared provider
            unit = None
        return unit or (providers[0] if providers else "accelerator")

    def absorbs(self, kind: str) -> Admission:
        """Does a readout this target DECLARES carry ``kind`` out of a contraction with it?"""
        return readout_absorbs(kind, self.readout, target=self.target)

    def ask(
        self,
        *,
        op: str,
        family: str,
        in_dtype: str | None,
        weight_dtype: str | None = None,
        rank: int | None = None,
        attached: bool,
        granularity: str | None = None,
    ) -> Admission:
        region = self._E.RegionDescriptor(
            op=op, family=family, in_dtype=in_dtype, weight_dtype=weight_dtype, rank=rank, scale_granularity=granularity
        )
        verdict = self._E.is_eligible(
            region,
            self.cap_map,
            undetermined=self.undetermined,
            providers=self.providers,
            fused_with=("contraction",) if attached else None,
            readout=self.readout if granularity is not None else None,
        )
        return Admission(verdict.eligible, verdict.refusal, verdict.reason, verdict.units)


# --- formation -----------------------------------------------------------------------------------
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
    from merlin.common import quant_formats as qf

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
        if stage is None:
            break
        if stage.kind == DEQUANTIZE:
            _, dtype = mq.type_shape_dtype(owner.operands[0].type)
            return adapters, owner, dtype
        if stage.kind not in (PAD, MOVEMENT, VIEW) or not owner.operands:
            break
        adapters.append(owner)
        value = owner.operands[0]
    return [], None, None


def _sole_user_is(op, consumers: set[int]) -> bool:
    return all(len(_users(result)) == 1 and id(_users(result)[0]) in consumers for result in op.results)


def form_groups(module, target: str, *, oracle: TargetOracle | None = None, function: str | None = None) -> list[Group]:
    """Partition ``function``'s computation-carrying operations into compute groups."""
    oracle = oracle or TargetOracle(target)
    functions = [op for op in module.walk() if op.name == "func.func" and op.body.blocks]
    if function is not None:
        functions = [fn for fn in functions if fn.sym_name.data == function]
    if not functions:
        return []
    ops = list(functions[0].body.blocks[0].ops)
    stage_of = {id(op): classify(op) for op in ops}
    position = {id(op): index for index, op in enumerate(ops)}
    taken: dict[int, Group] = {}
    groups: list[Group] = []

    extents = _macs_by_op(module)
    for root in ops:
        stage = stage_of[id(root)]
        if stage is None or stage.kind != CONTRACTION or id(root) in taken:
            continue
        group = _grow(root, oracle, stage_of, taken)
        group.macs = extents.get(id(root))
        for member in group.members:
            taken[id(member)] = group
        groups.append(group)

    groups += _host_regions(ops, stage_of, taken, oracle)
    for group in groups:
        group.members.sort(key=lambda member: position[id(member)])
        group.stages = [stage_of[id(member)].kind for member in group.members]
        absorbed = group.placement != HOST
        group.elements = sum(
            _elements(m)
            for m in group.members
            if m is not group.root
            and stage_of[id(m)].kind not in _NOT_ARITHMETIC
            and not (absorbed and stage_of[id(m)].kind == DEQUANTIZE)
        )
    groups.sort(key=lambda g: position[id(g.members[-1])])
    for index, group in enumerate(groups):
        group.index = index
    return groups


def _host_regions(ops, stage_of, taken: dict[int, Group], oracle: TargetOracle) -> list[Group]:
    """Every operation no accelerator group took, fused into explicit host regions.

    Two host operations joined by a value nothing else reads are one region: on the host that is
    one loop nest instead of two and one buffer fewer. Joining only along single-use values makes
    every region a tree with one sink, which is what lets it be outlined as one function at the
    sink's position. A host contraction stays a region of its own, so the work that was refused is
    not hidden inside the glue around it.
    """
    parent: dict[int, int] = {}

    def find(key: int) -> int:
        while parent.setdefault(key, key) != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    loose = [op for op in ops if stage_of[id(op)] is not None and id(op) not in taken]
    loose_ids = {id(op) for op in loose}
    for op in loose:
        if stage_of[id(op)].kind == CONTRACTION or len(op.results) != 1:
            continue
        users = _users(op.results[0])
        if len(users) != 1 or id(users[0]) not in loose_ids or stage_of[id(users[0])].kind == CONTRACTION:
            continue
        if stage_of[id(op)].kind == QUANTIZE and stage_of[id(users[0])].kind == DEQUANTIZE:
            # A quantized tensor handed from one integer region to the next. Joining across it
            # would make one region of two that a unit can each take whole, and it is the one edge
            # where the value between them is already the integers a unit reads.
            continue
        parent[find(id(op))] = find(id(users[0]))

    regions: dict[int, list[Any]] = {}
    for op in loose:
        regions.setdefault(find(id(op)), []).append(op)

    out: list[Group] = []
    for members in regions.values():
        # What the region IS: its first stage that is not an adapter on the way in. A region of
        # adapters around a dequantize IS the dequantize: that is where elements are computed, and
        # asking a unit about the transpose beside it would place floating-point work by its glue.
        primary = next(
            (m for m in members if stage_of[id(m)].kind not in (DEQUANTIZE, PAD, MOVEMENT, VIEW)),
            next(
                (m for m in members if stage_of[id(m)].kind == DEQUANTIZE),
                next((m for m in members if stage_of[id(m)].kind != VIEW), members[0]),
            ),
        )
        kind = stage_of[id(primary)].kind
        dequantized = [m for m in members if stage_of[id(m)].kind == DEQUANTIZE]
        if dequantized and primary is not members[0]:
            _, dtype = mq.type_shape_dtype(dequantized[0].operands[0].type)
        else:
            _, dtype = mq.type_shape_dtype(primary.results[0].type) if primary.results else ([], None)
        feeder = _accelerator_feeder(members, taken)
        group = Group(index=0, placement=HOST, root=None, members=list(members), in_dtype=_dtype_token(dtype))
        working = [m for m in members if stage_of[id(m)].kind != VIEW]
        if not working:
            group.reason = "shape metadata only: no element is read or written"
            out.append(group)
            continue
        if all(stage_of[id(m)].kind in (PAD, MOVEMENT) for m in working):
            group.reason = "data movement only: elements are relocated, none is computed"
            out.append(group)
            continue
        mean, unread_mean = _window_mean_of(working, stage_of)
        if mean is not None or unread_mean is not None:
            refusal = unread_mean or _window_mean_refusal(mean, oracle, _dtype_token(dtype))
            if refusal is None:
                group.placement = oracle.unit_for(_dtype_token(dtype), _dtype_token(dtype), 2, mean["units"])
                group.root = mean["reduce"]
                group.scale_granularity = "tensor"
                group.window_mean = {k: mean[k] for k in ("rows", "window", "multiplier", "bound_lsb", "exactness")}
                out.append(group)
                continue
            if mean is None or mean.get("asked"):
                group.refusal, group.reason = WINDOW_MEAN, refusal
                group.gap = "OG7" if mean is None else gap_class(mean.get("refusal")) or "OG1"
                out.append(group)
                continue
        summed, unread = _operand_sum_of(working, stage_of)
        readout = getattr(oracle, "readout", None)
        asked = getattr(readout, "operand_sum", None)
        # Asked only where some unit's load multiplies: on any other target the region is what it
        # always was, an elementwise map with no standalone provider.
        if callable(asked) and any(getattr(f, "operand_sum", None) for f in getattr(readout, "facets", ())):
            facet, refusal = asked(summed["multipliers"]) if summed is not None else (None, unread)
            if facet is not None:
                group.placement = facet.unit or "accelerator"
                group.root = summed["add"]
                group.operand_sum = {
                    "lhs_scale": summed["multipliers"][0],
                    "rhs_scale": summed["multipliers"][1],
                    "bound_lsb": facet.operand_sum_bound(summed["multipliers"]),
                    # The factor the readout's scale carries when a multiplier exceeds what a
                    # saturating load can take; one when both go through the load as they are.
                    "readout_factor": max(1.0, *summed["multipliers"]),
                    "relu": summed["relu"],
                }
                out.append(group)
                continue
            if refusal is not None:
                # A unit here CAN sum operands and cannot take this one: the owner is the design's
                # limit, or the capture's unread numbers, and naming it is what separates a gap
                # from a fallback.
                group.refusal, group.reason = OPERAND_SUM, refusal
                group.gap = "OG7" if summed is None else "OG0"
                out.append(group)
                continue
        # Every stage has to be admitted standalone for a unit to take the region; the first
        # refusal is the region's.
        verdict = None
        for member in working:
            member_kind = stage_of[id(member)].kind
            if member_kind in (DEQUANTIZE, PAD, MOVEMENT) and member is not primary:
                continue
            verdict = oracle.ask(
                op=member_kind, family=FAMILY_OF_STAGE[member_kind], in_dtype=_dtype_token(dtype), attached=False
            )
            if not verdict.admitted:
                break
        if feeder is not None and feeder.stopped_at is not None and any(m is feeder.stopped_at for m in members):
            # The group before it would have taken this stage and was refused: that refusal, not
            # the standalone one, is why it is here.
            group.refusal, group.reason = (
                feeder.refusal or UNRECORDED_REFUSAL,
                feeder.reason
                or (
                    f"a {stage_of[id(feeder.stopped_at)].kind} ended the group before it and no "
                    f"refusal was recorded for it"
                ),
            )
            group.gap = gap_class(group.refusal)
        elif verdict.admitted:
            group.placement = verdict.units[0] if verdict.units else "accelerator"
        else:
            group.refusal, group.reason = verdict.refusal, verdict.reason
            group.gap = gap_class(verdict.refusal)
            if verdict.refusal == "fused_only":
                # Available only attached to a contraction. With a contraction's group feeding it
                # the route failed to attach it; with none, no capability covers it standalone.
                group.gap = "OG4" if feeder is not None else "OG1"
        out.append(group)
    return out


def _window_mean_of(working: Sequence[Any], stage_of: Mapping[int, Stage | None]) -> tuple[dict | None, str | None]:
    """``(facts, None)`` when the region is a mean over a trailing window between a dequantize and
    a quantize, ``(None, why)`` when it is one whose numbers cannot be read, else ``(None, None)``.

    The shape is exact: one dequantize, one sum-reduction over the LAST dims from a zero, one
    division by a constant, one quantize, and views. Its integer form is
    ``q(sum(x) * s_in / (count * s_out))``: each kept position is a row, the window is what the
    contraction reduces over, the stored operand is a constant one, and one readout scale carries
    everything else. Trailing dims only, because only then is a window contiguous in a row.
    """
    from . import group_numerics as GN  # noqa: PLC0415 -- it imports this module

    by_name: dict[str, list[Any]] = {}
    for member in working:
        by_name.setdefault(mq.op_name(member), []).append(member)
    kinds = Counter(stage_of[id(member)].kind for member in working)
    reduces = by_name.get("linalg.reduce") or []
    if len(reduces) != 1 or kinds[DEQUANTIZE] != 1 or kinds[QUANTIZE] != 1 or len(working) != 4:
        return None, None
    reduce = reduces[0]
    dequantize = next(m for m in working if stage_of[id(m)].kind == DEQUANTIZE)
    quantize = next(m for m in working if stage_of[id(m)].kind == QUANTIZE)
    divide = next(m for m in working if m not in (reduce, dequantize, quantize))
    bodies = [[name for name in _body_ops(op) if name not in _BODY_NOISE] for op in (reduce, divide)]
    if bodies != [["arith.addf"], ["arith.divf"]] or _n_inputs(divide) != 2:
        return None, None

    def through_views(value):
        owner = getattr(value, "owner", None)
        while owner is not None and mq.op_name(owner) in _VIEW_OPS and getattr(owner, "operands", None):
            value = owner.operands[0]
            owner = getattr(value, "owner", None)
        return owner

    if through_views(reduce.operands[0]) is not dequantize or through_views(divide.operands[0]) is not reduce:
        return None, None
    if through_views(quantize.operands[0]) is not divide:
        return None, None
    in_shape, _ = mq.type_shape_dtype(reduce.operands[0].type)
    out_shape, _ = mq.type_shape_dtype(reduce.results[0].type)
    reduced = _int_array(reduce, "dimensions")
    if not in_shape or reduced is None or any(not isinstance(e, int) or e <= 0 for e in in_shape):
        return None, "the reduction's extents or dimensions are not static"
    kept = len(in_shape) - len(reduced)
    if sorted(reduced) != list(range(kept, len(in_shape))):
        return None, (
            f"the mean reduces dims {sorted(reduced)} of a rank-{len(in_shape)} tensor; only a trailing "
            f"window is contiguous in a row, and a gather to make it so is not stated here"
        )
    window, start, count = (
        _product(in_shape[kept:]),
        GN._constant_of(reduce.operands[1]),
        GN._constant_of(divide.operands[1]),
    )
    if start != 0 or not isinstance(count, (int, float)) or not count:
        return None, "the reduction does not start from zero, or its divisor is not a compile-time number"
    try:
        scales = [GN._scale_source(op) for op in (dequantize, quantize)]
    except GN.GroupNumericsError as exc:
        return None, str(exc)
    if any(scale.value is None for scale in scales) or any(scale.zero_point for scale in scales) or not scales[1].value:
        return None, "a scale of the mean is not a compile-time number, or a zero point is carried"
    return {
        "reduce": reduce,
        "rows": _product(in_shape[:kept]),
        "window": window,
        "multiplier": scales[0].value / (float(count) * scales[1].value),
        # The capture divides a float sum and rounds; the unit multiplies an exact integer sum and
        # rounds. The same real number computed two ways can land on either side of a tie.
        "bound_lsb": 1,
        "exactness": "float_reassociation",
        "out_elements": _product(out_shape),
    }, None


def _window_mean_refusal(mean: dict[str, Any], oracle: TargetOracle, dtype: str | None) -> str | None:
    """Why no unit takes the mean as a contraction, or ``None`` with ``mean['units']`` set.

    Two questions, both the ones a fake-quantized layer is asked: can a unit contract these
    integers, and does its readout hold the one scale the mean needs. ``mean['asked']`` records
    whether a contracting unit exists at all, so a target with none plans the region as before.
    """
    contraction = oracle.ask(
        op=CONTRACTION, family="contraction", in_dtype=dtype, weight_dtype=dtype, rank=2, attached=False
    )
    if not contraction.admitted:
        mean["asked"], mean["refusal"] = False, contraction.refusal
        return contraction.reason or "no unit contracts these operands"
    mean["asked"] = True
    scaled = oracle.ask(
        op=QUANTIZE, family=FAMILY_OF_STAGE[QUANTIZE], in_dtype=dtype, attached=True, granularity="tensor"
    )
    if not scaled.admitted:
        mean["refusal"] = scaled.refusal
        return scaled.reason or "the readout does not hold the scale a mean needs"
    readout = getattr(oracle, "readout", None)
    for facet in getattr(readout, "facets", ()) or ():
        fits = getattr(facet, "sum_fits_accumulator", None)
        if callable(fits) and fits(mean["window"]) is False:
            mean["refusal"] = "scale_granularity"
            return (
                f"a window of {mean['window']} elements of {facet.element_dtype} can exceed the "
                f"{facet.accumulator_dtype} accumulator"
            )
    mean["units"] = contraction.units
    return None


def _operand_sum_of(working: Sequence[Any], stage_of: Mapping[int, Stage | None]) -> tuple[dict | None, str | None]:
    """``(facts, None)`` when the region is an integer sum of two quantized tensors, ``(None, why)``
    when it is one whose numbers cannot be read, ``(None, None)`` when it is something else.

    The shape is exact and nothing looser is accepted: two dequantizes feeding one two-tensor add,
    optionally an activation, one quantize at the sink. Its integer form is
    ``q(a*sa + b*sb)`` with the multipliers ``sa/so`` and ``sb/so``, which exists only when every
    scale is one compile-time number per tensor and no zero point is carried.
    """
    kinds = Counter(stage_of[id(member)].kind for member in working)
    if kinds[RESIDUAL_ADD] != 1 or kinds[DEQUANTIZE] != 2 or kinds[QUANTIZE] != 1 or kinds[RELU] > 1:
        return None, None
    if set(kinds) - {RESIDUAL_ADD, DEQUANTIZE, QUANTIZE, RELU}:
        return None, None
    by_kind = {kind: [m for m in working if stage_of[id(m)].kind == kind] for kind in kinds}
    add, quantize = by_kind[RESIDUAL_ADD][0], by_kind[QUANTIZE][0]

    def through_views(value):
        owner = getattr(value, "owner", None)
        while owner is not None and mq.op_name(owner) in _VIEW_OPS and getattr(owner, "operands", None):
            value = owner.operands[0]
            owner = getattr(value, "owner", None)
        return owner

    sources = [through_views(operand) for operand in list(add.operands)[:2]]
    if {id(source) for source in sources} != {id(d) for d in by_kind[DEQUANTIZE]}:
        return None, None
    tail = through_views(quantize.operands[0])
    if kinds[RELU]:
        if tail is not by_kind[RELU][0]:
            return None, None
        tail = through_views(tail.operands[0])
    if tail is not add:
        return None, None
    shapes = {tuple(mq.type_shape_dtype(v.type)[0] or ()) for v in (*list(add.operands)[:2], add.results[0])}
    if len(shapes) != 1:
        return None, None  # a broadcast is a different operation from a sum of two tensors

    from . import group_numerics as GN  # noqa: PLC0415 -- it imports this module

    try:
        scales = [GN._scale_source(op) for op in (*sources, quantize)]
    except GN.GroupNumericsError as exc:
        return None, str(exc)
    if any(scale.value is None for scale in scales):
        return None, "a scale of the sum is a model argument, not a compile-time number, so no multiplier exists"
    if any(scale.zero_point for scale in scales):
        return None, "an operand or the result of the sum carries a zero point, which a scaled load cannot apply"
    out = scales[2].value
    if not out:
        return None, "the sum's output scale is zero"
    return {
        "add": add,
        "multipliers": (scales[0].value / out, scales[1].value / out),
        "relu": bool(kinds[RELU]),
    }, None


def _accelerator_feeder(members, taken: Mapping[int, Group]) -> Group | None:
    inside = {id(m) for m in members}
    for member in members:
        for operand in member.operands:
            owner = getattr(operand, "owner", None)
            group = taken.get(id(owner))
            if group is not None and group.placement != HOST and id(owner) not in inside:
                return group
    return None


def _macs_by_op(module) -> dict[int, int]:
    try:
        from merlin.targetgen import model_coverage

        raw = model_coverage._contraction_extents(module)
    except Exception:  # noqa: BLE001 -- extents are a weight, never a condition of grouping
        return {}
    out: dict[int, int] = {}
    for key, (m, k, n, _rank) in raw.items():
        if None not in (m, k):
            out[key] = int(m) * int(k) * int(n or 1)
    return out


def _grow(root, oracle: TargetOracle, stage_of: Mapping[int, Stage | None], taken: Mapping[int, Group]) -> Group:
    group = Group(index=0, placement=HOST, root=root, members=[root])

    # The operands behind the dequantizes are what an integer unit is asked about.
    chains = [_input_chain(operand) for operand in list(root.operands)[:2]]
    integer_region = len(chains) == 2 and all(dq is not None for _, dq, _ in chains)
    if integer_region and any(
        _dequantize_granularity(dq, adapters, root, index) == INSIDE_REDUCTION
        for index, (adapters, dq, _) in enumerate(chains)
    ):
        # sum_k x[k] * w[k] * s[k] is not s * sum_k x[k] * w[k]: the dequantize has to happen
        # before the contraction, so the contraction is what the capture says it is, a float one.
        integer_region = False
    _, native = mq.type_shape_dtype(root.operands[0].type) if root.operands else ([], None)
    group.in_dtype = _dtype_token(chains[0][2] if integer_region else native)
    group.weight_dtype = _dtype_token(chains[1][2]) if integer_region else None
    shape, _ = mq.type_shape_dtype(root.results[0].type) if root.results else ([], None)

    verdict = oracle.ask(
        op=CONTRACTION,
        family="contraction",
        in_dtype=group.in_dtype,
        weight_dtype=group.weight_dtype,
        rank=len(shape) or None,
        attached=False,
    )
    if not verdict.admitted:
        group.reason, group.refusal, group.gap = (verdict.reason, verdict.refusal, gap_class(verdict.refusal))
        return group
    group.placement = oracle.unit_for(group.in_dtype, group.weight_dtype, len(shape) or None, verdict.units)
    members = [root]
    scales: list[str | None] = []
    if integer_region:
        for index, (adapters, dequantize, _dtype) in enumerate(chains):
            chain = [*adapters, dequantize]
            inside = {id(root), *(id(op) for op in chain)}
            # An adapter or dequantize something else also reads has to exist outside the group.
            if all(id(op) not in taken and _sole_user_is(op, inside) for op in chain):
                members += chain
            scales.append(_dequantize_granularity(dequantize, adapters, root, index))
        group.scale_granularity = _join(scales)

    axes = _root_axes(root)
    value = root.results[0] if root.results else None
    pending_pad: Any | None = None
    views: list[Any] = []  # looked through; kept only if a stage follows them
    while value is not None:
        users = _users(value)
        if len(users) != 1:
            if users:
                # A fact about the GRAPH, not a capability question: no oracle is asked and no gap
                # class is owed. It is labelled so it cannot be counted as one.
                group.stopped_by, group.refusal = "fan_out", STRUCTURAL
                group.reason = (
                    f"the group's value has {len(users)} consumers, and a value read "
                    f"outside the group has to exist outside it"
                )
            break
        user = users[0]
        stage = stage_of.get(id(user))
        if stage is None or id(user) in taken:
            break
        if stage.kind == VIEW:
            views.append(user)
            value = user.results[0]
            continue
        if stage.kind == PAD and pending_pad is None:
            pending_pad, value = user, user.results[0]  # kept only if a pool follows
            continue
        absorbable = oracle.absorbs(stage.kind)
        if not absorbable.admitted:
            group.stopped_by, group.stopped_at = stage.kind, pending_pad or user
            group.reason, group.refusal = absorbable.reason, absorbable.refusal
            break
        granularity = group.scale_granularity
        if stage.kind in SCALED:
            own = (
                "tensor"
                if stage.kind == QUANTIZE and stage.axis is None
                else _granularity(set(stage.varies_over) if stage.varies_over is not None else None, axes)
            )
            granularity = _join([*scales, own])
            if granularity is None:
                # Not a per-tensor scale and not a refusal by the hardware: the capture does not
                # say what this scale varies over, so nothing can be asked.
                group.stopped_by, group.stopped_at = stage.kind, pending_pad or user
                group.refusal = UNREAD_SCALE
                group.reason = (
                    "the extent of a scale in this group could not be read from the "
                    "capture, so no readout can be asked to hold it"
                )
                break
        asked = oracle.ask(
            op=stage.kind,
            family=FAMILY_OF_STAGE[stage.kind],
            in_dtype=group.in_dtype,
            attached=True,
            granularity=granularity if stage.kind in SCALED else None,
        )
        if not asked.admitted:
            group.stopped_by, group.stopped_at = stage.kind, pending_pad or user
            group.reason, group.refusal = asked.reason, asked.refusal
            break
        if pending_pad is not None:
            if stage.kind != POOL:
                # Also a fact about the graph: the pad was kept only for a pool, and none followed.
                group.stopped_by, group.stopped_at, group.refusal = PAD, pending_pad, STRUCTURAL
                group.reason = "the padding is followed by no pooling, so nothing in the group consumes it"
                break
            members.append(pending_pad)
            pending_pad = None
        members += views
        views = []
        members.append(user)
        if stage.kind in SCALED:
            scales.append(own)
            group.scale_granularity = granularity
        value = user.results[0] if user.results else None
        if stage.kind == QUANTIZE:
            break  # the region is closed: integers out
    group.members = members
    return group


# --- the record ------------------------------------------------------------------------------------
def gap_class(refusal: str | None) -> str | None:
    from merlin.perf.placement_census import GAP_CLASS_OF_REFUSAL, UNCLASSIFIED

    if refusal is None:
        return None
    if refusal == "unplaced":
        return "OG6"
    if refusal == UNREAD_SCALE:
        return "OG7"
    if refusal == STRUCTURAL:
        # The graph, not the target. Nothing about the hardware is established, so no gap is owed.
        return None
    if refusal in (READOUT_DOES_NOT_APPLY, READOUT_UNDECLARED):
        # No readout DECLARES this stage as something a contraction's group can take (a residual
        # accumulated into the result, for one) -- either none lists it, or the target described no
        # readout at all. Declaring it is the first move, which is what OG1 means. Deliberately NOT
        # OG0: a stage absent from a declaration is a gap in what was declared, and calling it a
        # limit of the design would claim hardware evidence nobody produced.
        return "OG1"
    return GAP_CLASS_OF_REFUSAL.get(refusal, UNCLASSIFIED)


def plan(module, target: str, *, oracle: TargetOracle | None = None, function: str | None = None) -> dict[str, Any]:
    """The group plan of a model on a target, with its denominators."""
    groups = form_groups(module, target, oracle=oracle, function=function)
    rows = []
    for group in groups:
        rows.append(
            {
                "index": group.index,
                "placement": group.placement,
                "stages": list(group.stages),
                "operations": len(group.members),
                "key": group.key(),
                "macs": group.macs,
                "elements": group.elements,
                "reason": group.reason,
                "refusal": group.refusal,
                "gap_class": (group.gap or gap_class(group.refusal)) if group.placement == HOST else None,
                "stopped_by": group.stopped_by,
                "stop_gap_class": (gap_class(group.refusal) if group.placement != HOST and group.refusal else None),
                "region_ids": [rid for rid in (mq.attr_str(m, "prov.region_id") for m in group.members) if rid],
            }
        )
    device = [row for row in rows if row["placement"] != HOST]
    host = [row for row in rows if row["placement"] == HOST]
    # WHY growth stopped, per group, by the clause that decided it. Growth stops at its FIRST
    # refusal, which is the census's own caveat: lifting the top clause does not close these
    # groups, it reveals the next stage's verdict.
    from merlin.perf import capability_refusal

    absorption = capability_refusal.census(
        "epilogue_absorption",
        [
            capability_refusal.RefusalSite(
                site=(row["region_ids"][0] if row["region_ids"] else f"group_{row['index']}"),
                admitted=row["stopped_by"] is None or QUANTIZE in row["stages"],
                clause=(
                    capability_refusal.SELECTED
                    if row["stopped_by"] is None or QUANTIZE in row["stages"]
                    else f"{row['stopped_by']}:{row['refusal'] or UNRECORDED_REFUSAL}"
                ),
                detail={
                    "stages": row["stages"],
                    "scale_granularity": row["key"]["scale_granularity"],
                    "reason": row["reason"],
                },
            )
            for row in device
        ],
    )
    elementwise_host = sum(row["elements"] for row in host)
    elementwise_device = sum(row["elements"] for row in device)
    total = elementwise_host + elementwise_device
    return {
        "schema": SCHEMA,
        "target": target,
        "groups": rows,
        "absorption_refusals": absorption,
        "summary": {
            "groups": len(rows),
            "accelerator_groups": len(device),
            "host_groups": len(host),
            "operations": sum(row["operations"] for row in rows),
            "operations_on_accelerator": sum(row["operations"] for row in device),
            "macs_on_accelerator": sum(row["macs"] or 0 for row in device),
            "macs_on_host": sum(row["macs"] or 0 for row in host),
            # Per-element work is where a whole model's host time goes, so it has its own ratio.
            "elements_in_accelerator_groups": elementwise_device,
            "elements_on_host": elementwise_host,
            "element_share_on_accelerator": (round(elementwise_device / total, 6) if total else None),
            "host_by_gap_class": dict(sorted(Counter(row["gap_class"] for row in host if row["gap_class"]).items())),
            "growth_stopped_by": dict(
                sorted(
                    Counter(
                        f"{row['stopped_by']}:{row['refusal'] or UNRECORDED_REFUSAL}"
                        for row in device
                        if row["stopped_by"]
                    ).items()
                )
            ),
            "closed_integer_regions": sum(1 for row in device if QUANTIZE in row["stages"]),
            # Host groups that compute something and carry no owner. Empty is the invariant: a
            # host placement nobody can explain is the silent fallback this pass exists to end.
            "unexplained_host_groups": [
                row["index"]
                for row in host
                if _computes(row["stages"]) and (not row["reason"] or row["gap_class"] in (None, "UNCLASSIFIED"))
            ],
        },
    }


def _computes(stages: Sequence[str]) -> bool:
    return any(stage not in (VIEW, PAD, MOVEMENT) for stage in stages)


class SilentHostPlacement(RuntimeError):
    """A group that computes is on the host and nothing says why or whose move it is."""


def require_explained(report: Mapping[str, Any]) -> None:
    unexplained = report["summary"]["unexplained_host_groups"]
    if unexplained:
        raise SilentHostPlacement(
            f"{len(unexplained)} host group(s) on {report['target']} carry no reason or no owner: {unexplained[:8]}"
        )


def annotate(groups: Sequence[Group]) -> None:
    """Stamp each member with its group and placement, so the decision survives into later passes."""
    from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64

    for group in groups:
        for member in group.members:
            member.attributes["merlin.group"] = IntegerAttr(group.index, i64)
            member.attributes["merlin.placement"] = StringAttr(group.placement)


# --- the demand a group places on a backend --------------------------------------------------------
class NoCapsuleForm(ValueError):
    """A group has a stage the capsule vocabulary cannot state, so no backend can be asked for it."""


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


def capsule_entry(
    group: Group, *, name: str | None = None, extents: Mapping[int, tuple] | None = None
) -> dict[str, Any]:
    """The capsule-corpus entry that demands exactly this group from a backend.

    A model route and a capsule route lower the same pattern only if they are asked in the same
    words. This states a group in the entry vocabulary the capsule builders already consume
    (``op``, extents, ``epilogue`` in the command-buffer stage names), so the capsule that
    certifies a pattern and the model that needs it cannot drift apart.
    """
    if group.root is None:
        raise NoCapsuleForm("a host region has no contraction root to demand")
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
    silent = {DEQUANTIZE, PAD, MOVEMENT, VIEW, CONTRACTION, ROUND, CLAMP}
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
    if extents is None:
        # One walk of the whole module. A caller stating many groups passes the table in, because
        # recomputing it per group is quadratic in the model.
        from merlin.targetgen import model_coverage

        extents = (
            model_coverage._contraction_extents(root.parent_op().parent_op()) if root.parent_op() is not None else {}
        )
    found = extents.get(id(root))
    if not found or None in found[:3]:
        raise NoCapsuleForm("the contraction's M/K/N extents could not be read")
    m, k, n, _rank = found
    entry.update({"op": "matmul", "M": int(m), "K": int(k), "N": int(n)})
    return entry


def _operand_sum_entry(group: Group, *, name: str | None) -> dict[str, Any]:
    """An integer sum of two tensors in the ``residual_add`` builder's words.

    An elementwise operation has no rows and columns of its own, so the tensor is stated as the
    matrix its last axis makes of it: contiguous storage means any flattening is the same program.
    """
    shape, _ = mq.type_shape_dtype(group.root.results[0].type)
    if not shape or any(not isinstance(extent, int) or extent <= 0 for extent in shape):
        raise NoCapsuleForm("the sum's extents are not static")
    return {
        "name": name or f"group_{group.index}",
        "kind": "op",
        "op": "residual_add",
        "M": _product(shape[:-1]),
        "N": int(shape[-1]),
        "lhs_scale": float(group.operand_sum["lhs_scale"]),
        "rhs_scale": float(group.operand_sum["rhs_scale"]),
        "bound_lsb": int(group.operand_sum["bound_lsb"]),
        "epilogue": ["relu"] if group.operand_sum.get("relu") else [],
        "scale_granularity": "tensor",
        "operand_dtype": group.in_dtype,
    }


def demand(groups: Sequence[Group], *, weight_args: Collection[int] | None = None) -> dict[str, Any]:
    """Every distinct accelerator group of a model as capsule entries, with what could not be stated.

    Stated in DEVICE form (:mod:`.group_command`): the stored tensor on the right, a gathered
    convolution as the unit's own convolution, the stages in the readout's order. A demand in the
    capture's form asks a backend for the framework's lowering, not for the layer.
    """
    from . import group_command

    entries: dict[str, dict[str, Any]] = {}
    unstated: Counter = Counter()
    for group in groups:
        if group.placement == HOST:
            continue
        if group.root is None:
            # A unit took a region with no contraction in it. The capsule vocabulary states
            # contraction groups only, so nothing can demand this one; say so rather than skip it.
            unstated["a region placed on a unit has no contraction root to state"] += 1
            continue
        try:
            entry = group_command.program(group, weight_args=weight_args).entry
        except NoCapsuleForm as error:
            unstated[str(error)] += 1
            continue
        # A multiplier is a number the same program is run with, not a different program.
        numbers = ("name", "acc_scale", "lhs_scale", "rhs_scale")
        key = repr(sorted((k, repr(v)) for k, v in entry.items() if k not in numbers))
        entries.setdefault(key, {**entry, "count": 0})["count"] += 1
    return {"entries": sorted(entries.values(), key=lambda e: -e["count"]), "unstated": dict(unstated)}
