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


def _macs_by_op(module) -> dict[int, int]:
    try:
        raw = KS.contraction_extents(module)
    except Exception:  # noqa: BLE001 -- extents are a weight, never a condition of grouping
        return {}
    out: dict[int, int] = {}
    for key, (m, k, n, _rank) in raw.items():
        if None not in (m, k):
            out[key] = int(m) * int(k) * int(n or 1)
    return out


def _grow(
    root,
    oracle: TargetOracle,
    stage_of: Mapping[int, Stage | None],
    taken: Mapping[int, Group],
    numeric_refusals: Mapping[int, str],
    fusion: Any | None = None,
) -> Group:
    from merlin.compiler.partition.groups.classification import (
        BIAS_ADD,
        CAST,
        CONTRACTION,
        CONVERSION_OF_SCALED_STORE,
        FAMILY_OF_STAGE,
        FLOAT_EPILOGUE_REASSOCIATION,
        HOST,
        INSIDE_REDUCTION,
        MOVEMENT,
        PAD,
        POOL,
        QUANTIZE,
        READOUT_REQUIRES_SCALE,
        SCALE,
        SCALED,
        STRUCTURAL,
        UNREAD_SCALE,
        VIEW,
        Group,
        Stage,
        _dequantize_granularity,
        _dtype_token,
        _input_chain,
        _sole_user_is,
        _users,
    )

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
    if fusion is not None:
        return _fused(group, members, fusion, taken)

    axes = _root_axes(root)
    value = root.results[0] if root.results else None
    pending_pad: Any | None = None
    # An integer accumulator converted to a wider float BEFORE its output scale is the first half of
    # a scaled store written as dequantize -> scale (`float(acc) * s_in * s_w`), which is how a
    # static integerization emits every contraction's readout. It is held, like a pad, and kept only
    # when an output scale is the next stage; anything else leaves it on the host as before.
    pending_cast: list[Any] | None = None
    converted_at: int | None = None  # where an accepted accumulator conversion entered ``members``
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
        if stage.kind == VIEW or (stage.kind == MOVEMENT and mq.op_name(user) == "linalg.transpose"):
            # A pure permutation only relabels the result's axes (the capture's layout convention);
            # like a reshape it is looked through and kept only if a stage follows it.
            views.append(user)
            value = user.results[0]
            continue
        if stage.kind == PAD and pending_pad is None:
            pending_pad, value = user, user.results[0]  # kept only if a pool follows
            continue
        if pending_cast is not None and stage.kind != SCALE:
            group.stopped_by, group.stopped_at = CAST, pending_cast[-1]
            group.refusal = READOUT_REQUIRES_SCALE
            group.reason = (
                f"'cast' is a conversion of a scaled store, but this group has no output scale (the "
                f"conversion is followed by {stage.kind!r}); the conversion must remain on the host"
            )
            break
        if (
            stage.kind == CAST
            and pending_cast is None
            and pending_pad is None
            and not any(stage_of[id(member)].kind == SCALE for member in members)
        ):
            pending_cast, views = [*views, user], []
            value = user.results[0] if user.results else None
            continue
        if stage.kind in CONVERSION_OF_SCALED_STORE and not any(
            stage_of[id(member)].kind == SCALE for member in members
        ):
            group.stopped_by, group.stopped_at = stage.kind, pending_pad or user
            group.refusal = READOUT_REQUIRES_SCALE
            group.reason = (
                f"{stage.kind!r} is a conversion of a scaled store, but this group has no output "
                "scale; the conversion must remain on the host"
            )
            break
        if stage.kind == BIAS_ADD and id(root) in numeric_refusals:
            # Keep a raw device contraction and its unproved float chain on the host.
            cast = (
                pending_cast[-1]
                if pending_cast is not None
                else next(member for member in members[converted_at:] if stage_of[id(member)].kind == CAST)
            )
            if converted_at is not None:
                members = members[:converted_at]
            pending_cast = None
            converted_at = None
            group.stopped_by, group.stopped_at = CAST, cast
            group.refusal = FLOAT_EPILOGUE_REASSOCIATION
            group.reason = numeric_refusals[id(root)]
            break
        absorbable = oracle.absorbs(stage.kind)
        sequence_check = getattr(oracle, "absorbs_sequence", None)
        if absorbable.admitted and sequence_check is not None:
            absorbable = sequence_check(
                [*(stage_of[id(member)].kind for member in members), stage.kind],
                unit=group.placement,
            )
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
        if pending_cast is not None:
            converted_at = len(members)
            members += pending_cast
            pending_cast = None
        members += views
        views = []
        members.append(user)
        if stage.kind in SCALED:
            scales.append(own)
            group.scale_granularity = granularity
        value = user.results[0] if user.results else None
        if stage.kind == QUANTIZE:
            break  # the region is closed: integers out
    if converted_at is not None and not any(stage_of[id(member)].kind == QUANTIZE for member in members):
        # The chain converted and scaled the accumulator and never quantized it: the result is a
        # float tensor, which no integer readout writes. The whole chain stays on the host.
        stopped = members[converted_at]
        members = members[:converted_at]
        earlier = f" (growth had stopped at {group.stopped_by!r}: {group.reason})" if group.refusal else ""
        group.stopped_by, group.stopped_at = CAST, stopped
        group.refusal = READOUT_REQUIRES_SCALE
        group.reason = (
            "'cast' is a conversion of a scaled store, but no quantize closes this group's scale, so its "
            "result is a float tensor no integer readout writes; the conversion must remain on the host" + earlier
        )
    elif pending_cast is not None and group.refusal is None:
        group.stopped_by, group.stopped_at = CAST, pending_cast[-1]
        group.refusal = READOUT_REQUIRES_SCALE
        group.reason = (
            "'cast' is a conversion of a scaled store, but this group has no output scale; "
            "the conversion must remain on the host"
        )
    group.members = members
    return group


def _fused(group: Group, members: list[Any], decision: Any, taken: Mapping[int, Group]) -> Group:
    """The group the output-fusion pass decided: its absorbed stages, its cut and every refusal."""
    from merlin.compiler.partition.groups.classification import Group

    absorbed = [op for op in decision.members if id(op) not in taken]
    group.members = [*members, *absorbed]
    group.fusion = decision.record()
    refusal = decision.first_refusal
    if refusal is not None:
        group.refusal, group.reason = f"fusion_{refusal['code']}", f"{refusal['cut']}: {refusal['reason']}"
        group.stopped_by = "fusion"
    return group


def gap_class(refusal: str | None) -> str | None:
    from merlin.analysis.optimization.candidates.placement_census import GAP_CLASS_OF_REFUSAL, UNCLASSIFIED
    from merlin.compiler.partition.groups.classification import (
        ACCUMULATE_UNPROVED,
        FLOAT_EPILOGUE_REASSOCIATION,
        NO_DEVICE_FORM,
        READOUT_DOES_NOT_APPLY,
        READOUT_REQUIRES_SCALE,
        READOUT_UNDECLARED,
        STRUCTURAL,
        UNREAD_SCALE,
    )

    if refusal is None:
        return None
    if refusal == "unplaced":
        return "OG6"
    if refusal == UNREAD_SCALE:
        return "OG7"
    if refusal == STRUCTURAL:
        # The graph, not the target. Nothing about the hardware is established, so no gap is owed.
        return None
    if refusal == FLOAT_EPILOGUE_REASSOCIATION:
        return "OG0"
    if refusal and refusal.startswith("fusion_"):
        from merlin.compiler.partition.groups.output_fusion import GAP_CLASS

        return GAP_CLASS.get(refusal.removeprefix("fusion_"), "UNCLASSIFIED")
    if refusal == ACCUMULATE_UNPROVED:
        # The hardware evidence exists and the missing piece is this compiler's exactness proof: the
        # region is unplaced by us, like a region no device form states.
        return "OG6"
    if refusal == READOUT_REQUIRES_SCALE:
        # The readout can carry a conversion only when a scale stage precedes it. This instance
        # lacks that required attachment; the owner is the same route/attachment class as fused_only.
        return "OG4"
    if refusal == NO_DEVICE_FORM:
        # The target admits it and nothing can state it. No fact about the hardware is established
        # either way, so no hardware gap is owed: the region is unplaced, by us.
        return "OG6"
    if refusal in (READOUT_DOES_NOT_APPLY, READOUT_UNDECLARED):
        # No readout DECLARES this stage as something a contraction's group can take (a residual
        # accumulated into the result, for one) -- either none lists it, or the target described no
        # readout at all. Declaring it is the first move, which is what OG1 means. Deliberately NOT
        # OG0: a stage absent from a declaration is a gap in what was declared, and calling it a
        # limit of the design would claim hardware evidence nobody produced.
        return "OG1"
    return GAP_CLASS_OF_REFUSAL.get(refusal, UNCLASSIFIED)


def plan(
    module,
    target: str,
    *,
    oracle: TargetOracle | None = None,
    function: str | None = None,
    groups: Sequence[Group] | None = None,
) -> dict[str, Any]:
    """The group plan of a model on a target, with its denominators."""
    from merlin.compiler.partition.groups.classification import HOST, QUANTIZE, SCHEMA, UNRECORDED_REFUSAL, Group
    from merlin.compiler.partition.groups.formation import form_groups

    oracle = oracle or TargetOracle(target)
    groups = groups if groups is not None else form_groups(module, target, oracle=oracle, function=function)
    from merlin.compiler.partition.groups.group_route_report import lowering_obligation

    rows = []
    for group in groups:
        rows.append(
            {
                "index": group.index,
                "placement": group.placement,
                "lowering_obligation": lowering_obligation(group),
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
                "fusion": group.fusion,
            }
        )
    device = [row for row in rows if row["placement"] != HOST]
    host = [row for row in rows if row["placement"] == HOST]
    # WHY growth stopped, per group, by the clause that decided it. Growth stops at its FIRST
    # refusal, which is the census's own caveat: lifting the top clause does not close these
    # groups, it reveals the next stage's verdict.
    from merlin.analysis.optimization.candidates import isa_utilization as capability_refusal

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
    from merlin.compiler.partition.groups.group_route_report import input_format_changes

    return {
        "schema": SCHEMA,
        "target": target,
        "groups": rows,
        "absorption_refusals": absorption,
        "summary": {
            "groups": len(rows),
            "accelerator_groups": len(device),
            "host_groups": len(host),
            "host_by_lowering_obligation": dict(sorted(Counter(row["lowering_obligation"] for row in host).items())),
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
            "device_groups_requiring_input_format_change": input_format_changes(groups),
            # WHICH READOUT FACTS NOBODY DERIVED, beside the placements they decided. A run whose
            # fact bundle is missing loses the readout entirely -- no granularity, no operand sum --
            # and every other number here comes out IDENTICAL, because an underived facet refuses
            # by never being consulted rather than by saying so. Measured on a whole model: with the
            # bundle absent the share, the group counts and the gap classes were unchanged and the
            # only difference was inside the facet. A reader has to be able to see that from the
            # summary, or a report from an unprovisioned checkout is indistinguishable from a
            # derived one.
            "readout_underived": dict(sorted(getattr(oracle.readout, "unknown", {}).items())),
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
    from merlin.compiler.partition.groups.classification import MOVEMENT, PAD, VIEW

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

    from merlin.compiler.partition.groups.classification import Group

    for group in groups:
        for member in group.members:
            member.attributes["merlin.group"] = IntegerAttr(group.index, i64)
            member.attributes["merlin.placement"] = StringAttr(group.placement)


def __getattr__(name: str):
    """``NoCapsuleForm`` and ``capsule_entry`` now live in :mod:`.group_demand`.

    They are still reached through this module because every caller asks THIS module for them, and a
    statement of a group belongs beside the formation that produced it as far as a reader is
    concerned. Resolved on attribute access, so neither module imports the other at import time.
    """
    if name in ("NoCapsuleForm", "capsule_entry"):
        from merlin.compiler.partition.groups import capsule_demand as group_demand

        return getattr(group_demand, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def demand(groups: Sequence[Group], *, weight_args: Collection[int] | None = None) -> dict[str, Any]:
    """Keep the group-formation API stable while capsule reporting lives separately."""
    from merlin.compiler.partition.groups import compute_group_demand
    from merlin.compiler.partition.groups.classification import Group

    return compute_group_demand.demand(groups, weight_args=weight_args)
