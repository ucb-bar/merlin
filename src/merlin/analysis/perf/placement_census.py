"""Where every region of a model landed, out of what, and whose move it is when it missed.

A placement records what the compiler did with the operations it was asked about. It is silent on
the ones it was never asked about, and those are the ones that cost: a whole-model placement is
built from contraction demands, so a requantize, a clamp, a residual add and a pool are not placed
on the host, they are simply absent, and the program that results spends almost all of its time in
them.

This census takes its population from the model, not from the placement: every
computation-carrying region of the module, described structurally. Each region is then asked the
two questions the rest of the system already answers, by the oracles that already answer them:

* could this target run it (`targetgen.eligibility`, the denominator), and
* did a unit take it (`targetgen.routing`, the numerator).

A region on the host carries the refusal that put it there and the class of owner that can change
it. A region the target COULD run and no unit took carries no reason at all, and that is the point:
it is the population `lowering_coverage.coverage_gate(allow_fallback=False)` refuses.

Offload is reported two ways because they disagree by an order of magnitude on real models: by
region count, and by multiply-accumulates for the regions whose extents are known. A thousand
index renamings on the host are not a thousand matmuls on the host.

Nothing here names a target, an operation or a lane. The target is a parameter; families, formats
and units come from its contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from merlin.analysis.perf import gate_phase as GP
from merlin.analysis.perf import lowering_coverage as LC
from merlin.development.targetgen import eligibility as E
from merlin.development.targetgen import routing as R

SCHEMA = "placement_census_v1"

#: Who can move a host placement, by the check that refused it. The classes are the ordered
#: questions of a missed offload: did the frontend present a form anything could lower (OG7), does
#: a capability declare it (OG1), and did the route attach a capability that exists only fused
#: (OG4). A refusal this table does not know is UNCLASSIFIED and fails the strict gate, so a new
#: eligibility check cannot quietly become an unowned fallback.
GAP_CLASS_OF_REFUSAL: dict[str, str] = {
    "unrecognized_family": "OG7",
    "input_dtype": "OG7",
    "weight_dtype": "OG7",
    "operand_pair": "OG7",
    "result_dtype": "OG7",
    # The target has not declared whether its elementwise unit can change
    # output format; absence of that evidence is not an unsupported frontend.
    "result_dtype_unknown": "OG1",
    "undetermined_family": "OG1",
    "undeclared_family": "OG1",
    "rank": "OG1",
    "batch": "OG1",
    "layout": "OG1",
    "form": "OG1",
    "engine": "OG1",
    "fused_only": "OG4",
    # The design cannot hold a scale at the region's granularity: a missing hardware feature, and
    # the quantization the model was captured under asked for it.
    "scale_granularity": "OG0",
    # Nobody derived what the readout holds, so nothing can say the hardware cannot.
    "scale_granularity_unknown": "OG1",
    "unexpressed_dtype": "OG7",
}
UNEXPRESSED_DTYPE = "unexpressed_dtype"
UNCLASSIFIED = "UNCLASSIFIED"

#: The gate whose phase governs the offload verdict this census now forms. Declared in
#: ``src/merlin/resources/contracts/gate_phases.yaml``; read, never assumed, so a rollout stays a one-line diff on a
#: tracked file rather than a literal buried here.
OFFLOAD_GATE = "offload_verdict"

#: What each refusal establishes about the target's UNIT for that class of work -- the first of the
#: two facts :func:`lowering_coverage.off_accelerator_cause` needs, and the only one the census is in
#: a position to know. ``True``: a unit for the class exists and turned this instance down for a
#: property it carries. ``False``: the target declares no unit for the class at all -- the one cause
#: that may exempt a model. ``None``: nobody established it, which is neither of those.
#:
#: A refusal this table does not name resolves to ``None``, not ``True``: a new eligibility check
#: must not be able to make a model addressable -- or exempt -- by not being listed. Nothing here
#: names a target, a family or a format; the refusal vocabulary is ``eligibility.REFUSALS``.
UNIT_EXISTS_FOR_REFUSAL: dict[str, bool | None] = {
    # Nobody could name the family, or reach evidence about it. Not a fact about the hardware.
    "unrecognized_family": None,
    "undetermined_family": None,
    "scale_granularity_unknown": None,
    "result_dtype_unknown": None,
    UNEXPRESSED_DTYPE: None,
    # The target declares no capability for this family at all.
    "undeclared_family": False,
    # A unit for the family exists and refused THIS instance -- only a unit that exists can do that.
    "input_dtype": True,
    "weight_dtype": True,
    "operand_pair": True,
    "result_dtype": True,
    "rank": True,
    "batch": True,
    "layout": True,
    "form": True,
    "engine": True,
    "fused_only": True,
    "scale_granularity": True,
}

#: Which authority decided the operand format a region was JUDGED at. Recorded per row because a
#: census that cannot say where a dtype came from cannot be audited, and the dtype is the single
#: fact that decides both halves of its ratio.
FROM_CAPTURE = "capture"
FROM_DATAPATH = "datapath"
FROM_UNEXPRESSED = "unexpressed"

Router = Callable[[Sequence[R.OpDemand]], dict]


class ZeroOffloadError(RuntimeError):
    """The target could run part of this model and the compiler put none of it there."""


class TwoDtypeAuthorities(ValueError):
    """A caller declared both a weights manifest and a datapath for the same operand format."""


def _unit_exists_for(refusal: str | None) -> bool | None:
    """Does the target declare a unit for this region's CLASS of work, per the refusal that named it?

    ``None`` refusal means nothing refused the region: it is on a unit, or it is eligible and nothing
    took it. Either way a unit for the class demonstrably exists -- which is precisely the case the
    strict gate calls a silent fallback, and it must stay addressable rather than become an exemption.
    """
    if refusal is None:
        return True
    return UNIT_EXISTS_FOR_REFUSAL.get(refusal)


def _macs(region: E.RegionDescriptor) -> int | None:
    if None in (region.m, region.k, region.n):
        return None
    return int(region.m) * int(region.k) * int(region.n) * max(int(region.batch), 1)


def census(
    regions: Sequence[E.RegionDescriptor],
    target: str,
    *,
    router: Router | None = None,
    datapath: str | None = None,
    precision_declared: bool = True,
) -> dict[str, Any]:
    """The placement census of ``regions`` on ``target``.

    ``router`` maps demands to a route plan and defaults to the target's own router. A caller that
    has already decided placement passes its own so the census reports THAT decision; the two share
    one legality function, so they agree unless the caller's cost model chose otherwise.

    ``datapath`` is the requested operand format, not evidence that a capture
    was transformed. A region's observed operand type takes precedence; only
    an untyped region may use the request as a weaker fallback. The authority
    and any request/capture mismatch are recorded per row. This is the same
    rule as model routing: an f32 contraction does not become an int8 device
    contraction because the caller requested int8.

    ``precision_declared`` says whether judged formats are authoritative for
    execution. Parsed IR with known operand types is authoritative; an
    unexpressed type remains undecided even when the flag is true.

    WHY THE HEADLINE RATIO MOVED. ``offload_of_eligible`` asks whether the router and the eligibility
    oracle AGREED; it does not ask how much of the model reached a unit, and the two are not the same
    question. Measured on a real ResNet-50 capture at its declared integer datapath: 53 convolutions
    carrying 4,087,136,256 multiply-accumulates -- 99.95% of the model -- were refused on ``rank``
    and ran on the host, while ``offload_of_eligible`` read **1.000**, because those 53 regions had
    left the denominator by the same verdict that kept them off the unit. A ratio whose denominator
    is the census's own refusal is a tautology: it reads 1.0 whenever the two oracles agree, however
    little of the model that is. The number that answers the question a reader is asking is
    ``offload["offload_of_addressable"]`` -- work on a unit over work the target HAS a unit for,
    refusals included -- and it reads 0.0005 on that same capture.
    """
    cap_map = E.capability_map_for_target(target)
    undetermined = E.undetermined_families_for_target(target)
    route = router or (lambda demands: R.route_plan(list(demands), target))

    # A requested datapath does not rewrite executable capture types. Prefer
    # the operation's own operand format; use a declared format only where the
    # capture genuinely did not express one, and keep that weaker authority
    # visible in the row.
    judged = [region.in_dtype or datapath for region in regions]
    authority = [
        FROM_CAPTURE if region.in_dtype is not None else (FROM_DATAPATH if datapath is not None else FROM_UNEXPRESSED)
        for region in regions
    ]
    # A format nobody expressed is not a stated one whatever the caller declares: there is nothing to
    # have declared. Everywhere else the caller's declaration stands.
    demands = [
        R.OpDemand(
            op=region.op or "",
            in_fmt=judged[index] or "",
            weight_fmt=(region.weight_dtype or judged[index] or "")
            if region.captured_input_formats is not None and region.resolved_family() == "contraction"
            else region.weight_dtype,
            site=str(index),
            m=region.m,
            n=region.n,
            k=region.k,
            rank=region.rank,
            family=region.resolved_family(),
            elem_fmt=region.in_dtype,
            form=region.form,
            captured_input_formats=region.captured_input_formats,
        )
        for index, region in enumerate(regions)
    ]
    stated = [
        precision_declared and authority[index] is not FROM_UNEXPRESSED and demands[index].source_formats_complete
        for index in range(len(regions))
    ]
    plan = route(demands)
    unit_of = {result.demand.site: result.unit for result in plan["results"] if getattr(result, "unit", None)}

    readout = None
    if any(region.scale_granularity is not None for region in regions):
        # Derived only when a region asks: a model with no scaled region pays nothing for it.
        from merlin.development.targetgen import readout_facet

        readout = readout_facet.TargetReadout(tuple(readout_facet.for_target(target)))

    rows: list[dict[str, Any]] = []
    placements: list[LC.Placement] = []
    for index, region in enumerate(regions):
        # ONE FORMAT, BOTH ORACLES. The demand above and this verdict are the numerator and the
        # denominator of the same ratio; asking them about different operand formats is how a
        # complete placement came to read as no coverage at all.
        asked = region if judged[index] == region.in_dtype else replace(region, in_dtype=judged[index])
        verdict = (
            E.is_eligible(asked, cap_map, undetermined=undetermined, readout=readout)
            if demands[index].source_formats_complete
            else E.EligibilityVerdict(
                False,
                region.resolved_family(),
                "captured input roles are incomplete; precision unverified",
                undetermined=True,
                refusal="input_dtype",
            )
        )
        unit = unit_of.get(str(index))
        on_unit = unit is not None
        eligible, refusal = verdict.eligible, verdict.refusal
        gap_class = reason = None
        if eligible and judged[index] is None and not on_unit:
            # The two oracles read an unexpressed precision in opposite directions: eligibility
            # treats it as "not applicable" and admits the region, the router cannot match it to a
            # unit's formats and refuses. Neither has decided anything. Counting the region as
            # eligible would report a fallback the compiler could not have avoided.
            eligible, refusal = False, UNEXPRESSED_DTYPE
            gap_class = GAP_CLASS_OF_REFUSAL[UNEXPRESSED_DTYPE]
            reason = (
                f"{gap_class}: the capture expresses no operand precision for this region, "
                f"so neither the capability check nor the router can decide it"
            )
        elif not on_unit and not eligible:
            gap_class = GAP_CLASS_OF_REFUSAL.get(refusal or "", UNCLASSIFIED)
            reason = f"{gap_class}: {verdict.reason}"
        rows.append(
            {
                "index": index,
                "op": region.op,
                "family": verdict.family,
                "dtype": judged[index],
                "captured_dtype": region.in_dtype,
                "captured_input_formats": region.captured_input_formats,
                "captured_weight_dtype": region.weight_dtype,
                "source_formats_complete": demands[index].source_formats_complete,
                "dtype_authority": authority[index],
                "requested_datapath_mismatch": bool(
                    datapath is not None
                    and region.in_dtype is not None
                    and not E._dtype_ok(region.in_dtype, (datapath,))
                ),
                "placement": LC.ACCELERATOR if on_unit else LC.HOST,
                "unit": unit,
                "eligible": eligible,
                "refusal": refusal,
                "gap_class": gap_class,
                "reason": reason,
                "macs": _macs(region),
                # The two facts that decide whether a host placement is a missed offload, an
                # exemption, or nothing anyone established -- classified by the one function that
                # classifies them, so this census and the offload gate cannot disagree about it.
                "off_accelerator_cause": LC.off_accelerator_cause(
                    class_has_unit=_unit_exists_for(refusal),
                    precision_stated=stated[index],
                ),
            }
        )
        placements.append(
            LC.Placement(
                operation=f"{index}:{region.op}",
                placement=LC.ACCELERATOR if on_unit else LC.HOST,
                family=verdict.family,
                dtype=judged[index],
                reason=reason,
            )
        )

    known = [row for row in rows if row["macs"] is not None]
    total_macs = sum(row["macs"] for row in known)
    eligible_macs = sum(row["macs"] for row in known if row["eligible"])
    on_unit_macs = sum(row["macs"] for row in known if row["placement"] == LC.ACCELERATOR)
    # NUMERATOR INSIDE DENOMINATOR. `on_unit` and `eligible` are two different populations -- one is
    # the router's, one is the oracle's -- and dividing one by the other produced ratios above 1.0
    # whenever they disagreed. The agreed set is the intersection, and each disagreement is reported
    # as itself rather than folded into a ratio that cannot express it.
    agreed_macs = sum(row["macs"] for row in known if row["eligible"] and row["placement"] == LC.ACCELERATOR)
    gate = LC.coverage_gate(placements, allow_fallback=False)
    by_class = Counter(row["gap_class"] for row in rows if row["gap_class"])
    # The offload verdict over the SAME rows, phased by its declaration. This is the number that
    # answers "did the target run what it demonstrably can run": its denominator is the work the
    # target has a unit for, so a region refused on a property stays in it instead of leaving by the
    # same verdict that kept it off the unit.
    offload = LC.offload_verdict(rows, phase=GP.configured_phase(OFFLOAD_GATE))
    # A COVERAGE FIGURE IS A BOUND WHEN PART OF ITS POPULATION IS UNDECIDED. Blanking the whole ratio
    # because one region of a hundred is unnameable would make every real model read `incomplete`
    # forever, which is as useless as the 1.0 this instrument used to print; reporting it as a point
    # value over the rest is how a denominator collapses to the regions that happened to be readable.
    # So the undecided regions stay in the denominator and move the two ENDS: the floor is what
    # demonstrably reached a unit, the ceiling is that plus everything nobody decided. A span of
    # nearly nothing is a usable number; a span of the whole interval says so on its face.
    undecided_rows = [row for row in rows if row["off_accelerator_cause"] == LC.ELIGIBILITY_UNKNOWN]
    population = len(rows)
    floor = round(gate["on_accelerator"] / population, 6) if population else None
    ceiling = round((gate["on_accelerator"] + len(undecided_rows)) / population, 6) if population else None
    return {
        "schema": SCHEMA,
        "target": target,
        # The requested format, beside the target. Per-row `dtype` and
        # `dtype_authority` record what was actually judged; a mixed capture
        # cannot be described by this single request alone.
        "datapath": datapath,
        "regions": rows,
        "coverage": {
            **{key: gate[key] for key in ("operations", "on_accelerator", "on_host", "host_reasons", "by_family")},
            # A POINT ONLY WHERE THERE IS ONE. `coverage` is the field a grader reads, and the 0.000
            # it used to carry -- beside `admitted: True`, on a model whose router places every
            # contraction -- was not a conservative measurement, it was a wrong one. It is now the
            # value only when the bound below is a point, and None otherwise, with both ends and the
            # reason stated so nothing is lost by refusing to collapse them.
            "coverage": floor if floor == ceiling else None,
            "coverage_floor": floor,
            "coverage_ceiling": ceiling,
            "coverage_undecided_regions": len(undecided_rows),
            "coverage_status": offload["status"],
            "coverage_undecided_because": offload["reason"] if undecided_rows else None,
        },
        "macs": {
            "regions_with_known_extents": len(known),
            "regions_with_unknown_extents": len(rows) - len(known),
            "total": total_macs,
            "eligible": eligible_macs,
            "on_unit": on_unit_macs,
            # The intersection, and the two ways the router and the oracle can part company. A
            # region on a unit the oracle refused is not a bonus, it is a disagreement between the
            # two halves of this report, and it is reported as one.
            "on_unit_and_eligible": agreed_macs,
            "on_unit_not_eligible": on_unit_macs - agreed_macs,
            "eligible_not_on_unit": eligible_macs - agreed_macs,
            # AGREEMENT, NOT COVERAGE. Read this as "of the work the oracle admitted, how much did
            # the router take" and nothing more; it is 1.0 whenever the two agree, including on a
            # model they agree the target can barely touch. None, not 0.0, with an empty denominator:
            # a ratio nobody could form is not a measured zero.
            "offload_of_eligible": round(agreed_macs / eligible_macs, 6) if eligible_macs else None,
            # Of the model's whole measurable weight, how much reached a unit. No oracle in the
            # denominator, so no verdict can shrink it.
            "offload_of_total": round(on_unit_macs / total_macs, 6) if total_macs else None,
        },
        # The phased verdict, whole. `status` is `offloaded`, `zero_offload`, or `incomplete` -- and
        # `incomplete` is never a pass at either phase.
        "offload": offload,
        "status": offload["status"],
        "host_by_gap_class": dict(sorted(by_class.items())),
        # Eligible, on the host, and nobody said why: the compiler's move, and the strict gate's.
        "silent_fallbacks": gate["unjustified_host_operations"],
        # AND WHETHER THAT LIST WAS IN A POSITION TO BE COMPLETE. A silent fallback is a region the
        # oracle ADMITTED that nothing took; a region judged at a format the backend will not place
        # from is refused instead, so it can never appear here. The list therefore under-reports --
        # never over-reports -- exactly when the census could not decide, and it does so as a clean
        # `0`. This field is on its way to being graded, and a gate that reads that zero would pass a
        # submission nobody measured, which is this repo's oldest failure shape. A consumer that
        # grades the list above must refuse a value here that is not `offloaded`.
        "silent_fallbacks_status": offload["status"],
        "unclassified_refusals": [row["index"] for row in rows if row["gap_class"] == UNCLASSIFIED],
        "admitted": gate["admitted"] and UNCLASSIFIED not in by_class,
        # The record three consumers already read from a command buffer's params and nothing wrote.
        "lane_placement": [{"lane": row["unit"] or LC.HOST, "family": row["family"]} for row in rows],
    }


def census_of_module(
    module: Any,
    target: str,
    *,
    router: Router | None = None,
    precisions: dict[str, str] | None = None,
    datapath: str | None = None,
) -> dict[str, Any]:
    """:func:`census` over every computation-carrying region of a parsed model module.

    ``precisions`` (a weights manifest) and ``datapath`` (a requested compile
    format) are independent declarations that must not silently override one
    another, so passing both is refused. The parsed module's actual operation
    types are the execution authority. If quantization occurs later, run this
    census on the transformed module; a request is not a substitute for that
    evidence.
    """
    from merlin.development.targetgen import model_coverage

    if precisions and datapath is not None:
        raise TwoDtypeAuthorities(
            "a weights manifest and a requested datapath both supply fallback operand formats; "
            "pass one. Ranking them would decide invisibly which program this census describes."
        )
    return census(
        model_coverage.regions_from_module(module, precisions=precisions),
        target,
        router=router,
        datapath=datapath,
        precision_declared=True,
    )


def planned_outlined_alignment(module: Any, target: str, outlined: Any) -> dict[str, Any]:
    """Check whether the runtime outline preserves accelerator groups it could form.

    This is deliberately a *static* comparison on the runtime's normalized module, not an
    execution certificate. The group's members are joined to outlined symbols by the exact source
    operation objects for the one-op outline, or by the outliner's own group index and exact stage
    sequence for a grouped outline. A missing join is unknown, never a clean placement.
    """
    from merlin.compiler.dialects.lowering import compute_groups as CG
    from merlin.compiler.dialects.lowering.outline import _is_root

    functions = [op for op in module.walk() if op.name == "func.func" and op.body.blocks]
    if not functions:
        return {"schema": "planned_outlined_alignment_v1", "status": "incomplete", "reason": "no source function"}
    groups = CG.form_groups(module, target)
    roots = [op for op in functions[0].body.blocks[0].ops if _is_root(op)]
    dispatches = list(outlined.dispatches)
    grouped = {dispatch.group: dispatch for dispatch in dispatches if dispatch.group is not None}
    if len(grouped) != sum(dispatch.group is not None for dispatch in dispatches):
        return {
            "schema": "planned_outlined_alignment_v1",
            "status": "incomplete",
            "reason": "more than one outlined dispatch claims a compute group index",
        }
    one_op_outline = not grouped
    if one_op_outline and (
        len(roots) != len(dispatches) or any(op.name != dispatch.root_op for op, dispatch in zip(roots, dispatches))
    ):
        return {
            "schema": "planned_outlined_alignment_v1",
            "status": "incomplete",
            "reason": "source compute roots do not match one-op outlined dispatches in program order",
            "source_roots": len(roots),
            "outlined_dispatches": len(dispatches),
        }
    symbol_of = {id(op): dispatch.symbol for op, dispatch in zip(roots, dispatches)} if one_op_outline else {}
    position = {id(op): index for index, op in enumerate(functions[0].body.blocks[0].ops)}
    split: list[dict[str, Any]] = []
    unjoined: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    assessed = 0
    for group in groups:
        if group.placement == CG.HOST:
            continue
        if group.refusal == CG.READOUT_UNDECLARED:
            unresolved.append({"group": group.index, "reason": group.reason})
        if group.root is None:
            continue
        root_symbol = symbol_of.get(id(group.root))
        grouped_dispatch = grouped.get(group.index)
        for member in group.members:
            if member is group.root or position.get(id(member), -1) <= position.get(id(group.root), -1):
                continue  # operand preparation is not an accelerator readout stage
            stage = CG.classify(member)
            if stage is None or stage.kind in (CG.VIEW, CG.PAD, CG.MOVEMENT):
                continue
            assessed += 1
            row = {"group": group.index, "stage": stage.kind, "root_symbol": root_symbol}
            if grouped_dispatch is not None:
                if (
                    grouped_dispatch.root_op != group.root.name
                    or grouped_dispatch.placement != group.placement
                    or grouped_dispatch.stages != group.stages
                ):
                    unjoined.append({**row, "reason": "outlined group identity, placement or stages differ"})
                continue
            symbol = symbol_of.get(id(member))
            if root_symbol is None or symbol is None:
                unjoined.append({**row, "outlined_symbol": symbol})
            elif symbol != root_symbol:
                split.append({**row, "outlined_symbol": symbol})
    return {
        "schema": "planned_outlined_alignment_v1",
        "status": "split" if split else ("incomplete" if unjoined or unresolved else "matched"),
        "evidence": "runtime_normalized_source_and_outlined_dispatches; static_only",
        "n_planned_accelerator_stages": assessed,
        "split_stages": split,
        "unjoined_stages": unjoined,
        "unresolved_groups": unresolved,
        "note": (
            "A matched outline is not evidence that its symbol executed on the accelerator; "
            "use the dynamic dispatch ledger separately."
        ),
    }


def require_offload(report: dict[str, Any]) -> None:
    """Refuse a build that could have used the accelerator and used none of it.

    An emission with eligible work and nothing on a unit used to return success with zero commands.
    A model with NO eligible work passes: the absence of an accelerator's work is only a defect when
    there was some to do.
    """
    # The phased rule first, and it blocks only where its declaration says it may. Its denominator is
    # work the target HAS a unit for, so it sees the case the clause below cannot: every region
    # refused on a property, nothing on a unit, and `eligible` therefore zero.
    if "offload" in report:
        try:
            LC.require_offload(report["offload"])
        except LC.ZeroOffloadError as refusal:
            raise ZeroOffloadError(str(refusal)) from refusal
    macs = report["macs"]
    if macs["eligible"] and not macs["on_unit"]:
        raise ZeroOffloadError(
            f"{macs['eligible']} eligible multiply-accumulates on {report['target']} and none "
            f"placed on a unit; silent fallbacks: {report['silent_fallbacks'][:8]}"
        )
