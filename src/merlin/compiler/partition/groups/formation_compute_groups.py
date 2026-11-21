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


def form_groups(
    module,
    target: str,
    *,
    oracle: TargetOracle | None = None,
    function: str | None = None,
    numeric_refusals: Mapping[int, str] | None = None,
    fusion: Mapping[int, Any] | None = None,
) -> list[Group]:
    """Partition ``function``'s computation-carrying operations into compute groups.

    ``fusion`` maps a contraction root's identity to the output-fusion pass's decision
    (:mod:`.output_fusion`): when one is given it alone decides what the root's group absorbs.
    """
    from merlin.compiler.partition.groups.classification import (
        _NOT_ARITHMETIC,
        CONTRACTION,
        DEQUANTIZE,
        HOST,
        Group,
        _elements,
        classify,
    )
    from merlin.compiler.partition.groups.plan import _grow, _macs_by_op

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
        group = _grow(root, oracle, stage_of, taken, numeric_refusals or {}, (fusion or {}).get(id(root)))
        group.macs = extents.get(id(root))
        for member in group.members:
            taken[id(member)] = group
        groups.append(group)

    groups += _host_regions(ops, stage_of, taken, oracle)
    for group in groups:
        group.members.sort(key=lambda member: position[id(member)])
        if group.window_pool is None:
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


def _dequantize_feeds_add(op, stage_of) -> bool:
    """A direct dequantized add input, through shape metadata only.

    Input preparation remains outside this arithmetic region. In particular,
    an integer transpose before the dequantize is still real host movement;
    the operand-sum reader must never silently absorb that operation.
    """
    from merlin.compiler.partition.groups.classification import RESIDUAL_ADD, VIEW, _users

    if len(op.results) != 1:
        return False
    value, seen = op.results[0], set()
    while True:
        users = _users(value)
        if len(users) != 1 or id(users[0]) in seen:
            return False
        user = users[0]
        seen.add(id(user))
        stage = stage_of.get(id(user))
        if stage is None:
            return False
        if stage.kind != VIEW or len(user.results) != 1:
            return stage.kind == RESIDUAL_ADD
        value = user.results[0]


def _host_regions(ops, stage_of, taken: dict[int, Group], oracle: TargetOracle) -> list[Group]:
    """Every operation no accelerator group took, fused into explicit host regions.

    Two host operations joined by a value nothing else reads are one region: on the host that is
    one loop nest instead of two and one buffer fewer. Joining only along single-use values makes
    every region a tree with one sink, which is what lets it be outlined as one function at the
    sink's position. A host contraction stays a region of its own, so the work that was refused is
    not hidden inside the glue around it.
    """
    from merlin.compiler.partition.groups.classification import (
        CONTRACTION,
        DEQUANTIZE,
        FAMILY_OF_STAGE,
        HOST,
        MOVEMENT,
        NO_DEVICE_FORM,
        OPERAND_SUM,
        PAD,
        QUANTIZE,
        RESIDUAL_ADD,
        UNRECORDED_REFUSAL,
        VIEW,
        WINDOW_MEAN,
        Group,
        _dtype_token,
        _has_device_form,
        _read_format,
        _users,
    )
    from merlin.compiler.partition.groups.plan import gap_class

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
        if stage_of[id(op)].kind == QUANTIZE:
            # Quantize closes its producer region.  Its integer result may feed
            # another Q/DQ region, or integer layout/movement on the way to a
            # contraction.  Joining across either edge lets downstream glue
            # obscure a complete quantized sum/mean on the producer side.
            continue
        if (
            stage_of[id(op)].kind in (VIEW, PAD, MOVEMENT)
            and stage_of[id(users[0])].kind == DEQUANTIZE
            and _dequantize_feeds_add(users[0], stage_of)
        ):
            # The scaled add reads the prepared integer tensor. Keep its
            # upstream layout/padding as a separate original host region;
            # its arithmetic statement contains only DQ/add/readout stages.
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
            # No dequantize inside the region names a stored format, so its elements ARE the format
            # they carry, and what a unit would have to hold is the format the region READS. The
            # RESULT is the wrong end to read it from: a quantize is the one stage whose two ends are
            # in different formats by definition, so taking its result asked whether a unit holds
            # int8 about a region whose operand is a float32 model ARGUMENT. Nothing dequantizes an
            # argument, so that float32 is not a fake-quantized stand-in for integers the way an
            # intermediate is -- it is the format the bytes are in at run time.
            _, dtype = _read_format(primary)
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
        from merlin.compiler.partition.groups.window_pool import match as match_window_pool

        pooled = match_window_pool(group, oracle)
        if pooled is not None:
            group.root, group.stages, group.placement, group.window_pool = pooled
            group.scale_granularity = "tensor"
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
        # A BARE INTEGER SUM (multiplier one): its dtype is the quantize's own result, not the outer
        # `dtype`, which for a lone quantize is the float operand it quantized from.
        from merlin.compiler.partition.groups.window_sum import window_sum_of

        wsum, unread_wsum = window_sum_of(working, stage_of)
        if wsum is not None or unread_wsum is not None:
            wsum_dtype = _dtype_token(wsum["in_dtype"]) if wsum is not None else _dtype_token(dtype)
            refusal = unread_wsum or _window_mean_refusal(wsum, oracle, wsum_dtype)
            if refusal is None:
                group.placement = oracle.unit_for(wsum_dtype, wsum_dtype, 2, wsum["units"])
                group.root = wsum["reduce"]
                group.scale_granularity = "tensor"
                group.window_mean = {k: wsum[k] for k in ("rows", "window", "multiplier", "bound_lsb", "exactness")}
                out.append(group)
                continue
            if wsum is None or wsum.get("asked"):
                group.refusal, group.reason = WINDOW_MEAN, refusal
                group.gap = "OG7" if wsum is None else gap_class(wsum.get("refusal")) or "OG1"
                out.append(group)
                continue
        summed, unread = _operand_sum_of(working, stage_of)
        readout = getattr(oracle, "readout", None)
        asked = getattr(readout, "operand_sum", None)
        # Asked only where some unit's load multiplies: on any other target the region is what it
        # always was, an elementwise map with no standalone provider.
        if callable(asked) and any(getattr(f, "operand_sum", None) for f in getattr(readout, "facets", ())):
            facet, refusal = asked(summed["multipliers"]) if summed is not None else (None, unread)
            ask_standalone = getattr(oracle, "standalone_admission", None)  # a stand-in may not have it
            standalone = ask_standalone(FAMILY_OF_STAGE[RESIDUAL_ADD]) if facet and callable(ask_standalone) else None
            if standalone is not None and not standalone["standalone"]:  # e.g. only a prohibited role runs it
                facet, refusal = None, standalone["reason"]
            if facet is not None:
                group.placement = facet.unit or "accelerator"
                group.root = summed["add"]
                group.operand_sum = {
                    "lhs_scale": summed["multipliers"][0],
                    "rhs_scale": summed["multipliers"][1],
                    "source_scales": summed["source_scales"],
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
        elif verdict.admitted and _has_device_form(group):
            group.placement = verdict.units[0] if verdict.units else "accelerator"
        elif verdict.admitted:
            # Admitted, and still no device FORM. A capability is declared per FAMILY, so
            # `elementwise_map` covers both the scaled add two separately loaded operands meet in an
            # accumulator for AND, say, a two-tensor multiply no accumulator performs: admitting the
            # family is not a claim about every member of it. The forms that can be stated are the
            # three decided above, each from its own evidence, and a region arriving here matched
            # none -- so admission alone must not place it. Placed anyway it became a group every
            # consumer refuses, and refuses by the wrong name: the statement path called it "a host
            # region" while it sat on a unit, and a whole model died naming a contraction root
            # nothing in the region was ever going to have.
            group.refusal, group.reason = (
                NO_DEVICE_FORM,
                f"{FAMILY_OF_STAGE[kind]} is admitted standalone here, and this {kind} region has "
                f"no contraction, operand sum or window mean to be stated as",
            )
            group.gap = gap_class(group.refusal)
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
    from merlin.compiler.partition.groups import group_numerics as GN  # noqa: PLC0415 -- it imports this module
    from merlin.compiler.partition.groups.classification import (
        _BODY_NOISE,
        _VIEW_OPS,
        DEQUANTIZE,
        QUANTIZE,
        Stage,
        _n_inputs,
        _product,
    )

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
    from merlin.compiler.partition.groups.classification import CONTRACTION, FAMILY_OF_STAGE, QUANTIZE

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
    from merlin.compiler.partition.groups.classification import (
        _VIEW_OPS,
        DEQUANTIZE,
        QUANTIZE,
        RELU,
        RESIDUAL_ADD,
        Stage,
    )

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

    from merlin.compiler.partition.groups import group_numerics as GN  # noqa: PLC0415 -- it imports this module

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
        "source_scales": tuple(float(scale.value) for scale in scales),
        "relu": bool(kinds[RELU]),
    }, None


def _accelerator_feeder(members, taken: Mapping[int, Group]) -> Group | None:
    from merlin.compiler.partition.groups.classification import HOST, Group

    inside = {id(m) for m in members}
    for member in members:
        for operand in member.operands:
            owner = getattr(operand, "owner", None)
            group = taken.get(id(owner))
            if group is not None and group.placement != HOST and id(owner) not in inside:
                return group
    return None
