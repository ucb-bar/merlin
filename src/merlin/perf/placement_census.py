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

from merlin.perf import gate_phase as GP
from merlin.perf import lowering_coverage as LC
from merlin.targetgen import eligibility as E
from merlin.targetgen import routing as R

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
    "undetermined_family": "OG1",
    "undeclared_family": "OG1",
    "rank": "OG1",
    "batch": "OG1",
    "layout": "OG1",
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
#: ``merlin/contract/gate_phases.yaml``; read, never assumed, so a rollout stays a one-line diff on a
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
    UNEXPRESSED_DTYPE: None,
    # The target declares no capability for this family at all.
    "undeclared_family": False,
    # A unit for the family exists and refused THIS instance -- only a unit that exists can do that.
    "input_dtype": True,
    "weight_dtype": True,
    "rank": True,
    "batch": True,
    "layout": True,
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

    ``datapath`` is the operand format the compile DECLARES it will lower to, and when given it is
    the format every region is judged at. Without it this census judges a capture at the element
    types the capture happens to carry -- which is not the program the compiler is building, and on
    a dynamically quantized model it is not even close. Measured on a ResNet-50 ``int8_dyn_act``
    capture: the compiler routed 54/54 contractions onto the mesh while this census, reading ``f32``
    element types off the same module, refused all 53 of its matmuls with ``input_dtype`` and
    reported coverage 0.004 -- and reported ``offload_of_eligible`` as **1.0**, because the
    denominator had collapsed to the one region that happened to carry an ``i8`` element type. A
    perfect-looking ratio over a denominator of one is worse than the zero beside it.

    The rule is not invented here: ``capsule_source.model_op_demands`` already routes every demand at
    the compile's declared format, "because that is what legality is about: a capture is routed under
    the datapath the compiler will lower it to". Judging the denominator by a different rule than the
    numerator is what made the two disagree. The capture's own spelling is kept on every row
    (``captured_dtype``) and never discarded.

    ``precision_declared`` says whether the format each region was judged at is the one the BACKEND
    WILL PLACE FROM. A caller that built these descriptors chose their dtypes and is that authority,
    so it defaults True; :func:`census_of_module` walking a capture is not, and says so. When it is
    False every property refusal becomes :data:`lowering_coverage.ELIGIBILITY_UNKNOWN` and the
    verdict below is ``incomplete`` -- never a pass, and never the 0.000 that used to be reported
    with ``admitted: True`` beside it.

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

    judged = [region.in_dtype if datapath is None else datapath for region in regions]
    authority = [
        FROM_DATAPATH if datapath is not None else (FROM_CAPTURE if region.in_dtype is not None else FROM_UNEXPRESSED)
        for region in regions
    ]
    # A format nobody expressed is not a stated one whatever the caller declares: there is nothing to
    # have declared. Everywhere else the caller's declaration stands.
    stated = [precision_declared and authority[index] is not FROM_UNEXPRESSED for index in range(len(regions))]

    demands = [
        R.OpDemand(
            op=region.op or "",
            in_fmt=judged[index] or "",
            weight_fmt=region.weight_dtype,
            site=str(index),
            m=region.m,
            n=region.n,
            k=region.k,
            rank=region.rank,
            family=region.resolved_family(),
        )
        for index, region in enumerate(regions)
    ]
    plan = route(demands)
    unit_of = {result.demand.site: result.unit for result in plan["results"] if getattr(result, "unit", None)}

    readout = None
    if any(region.scale_granularity is not None for region in regions):
        # Derived only when a region asks: a model with no scaled region pays nothing for it.
        from merlin.targetgen import readout_facet

        readout = readout_facet.TargetReadout(tuple(readout_facet.for_target(target)))

    rows: list[dict[str, Any]] = []
    placements: list[LC.Placement] = []
    for index, region in enumerate(regions):
        # ONE FORMAT, BOTH ORACLES. The demand above and this verdict are the numerator and the
        # denominator of the same ratio; asking them about different operand formats is how a
        # complete placement came to read as no coverage at all.
        asked = region if judged[index] == region.in_dtype else replace(region, in_dtype=judged[index])
        verdict = E.is_eligible(asked, cap_map, undetermined=undetermined, readout=readout)
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
                "dtype_authority": authority[index],
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
        # The format this census JUDGED at, beside the target it judged against. Both halves of
        # every ratio below were asked at this format; a reader who does not know it cannot tell a
        # model the target cannot run from a census asked about a different program.
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

    ``precisions`` (a weights manifest) and ``datapath`` (the compile's declared operand format) are
    two different authorities over the same fact, and passing both is REFUSED rather than ranked: a
    silent precedence between two sources of a dtype is exactly the shape of the defect this
    parameter exists to fix. Pass the manifest for a mixed-precision capture whose regions really do
    differ; pass the datapath when one format is what the compiler will lower every region to.

    A MODULE DECLARES NOTHING ABOUT THE PROGRAM IT WILL BECOME. Only ``datapath`` states the operand
    format the backend will place from, so only ``datapath`` makes this census DECIDED. Walking a
    capture without one judges every region at whatever element type integer preparation has not yet
    consumed, and on a dynamically quantized model that is ``fp32`` everywhere while the compiler
    lowers all of it to the target's integer datapath: measured on a real ResNet-50, 177 of 178
    regions refused on ``input_dtype``, coverage 0.000, ``admitted: True``, and the same module at
    its declared datapath places 56. So without a datapath the verdict is ``incomplete`` and the
    coverage ratio is ``None``, with the reason attached.

    A weights manifest does not rescue it, and is not ranked into that role: ``regions_from_module``
    joins the manifest on each region's provenance and falls back to the element type where the join
    misses, inside one expression -- so no region can afterwards say which of the two set its format.
    An authority that cannot be attributed per region is not one this census will claim.
    """
    from merlin.targetgen import model_coverage

    if precisions and datapath is not None:
        raise TwoDtypeAuthorities(
            "a weights manifest and a declared datapath both claim the operand format of every "
            "region; pass one. Ranking them here would decide, per region and invisibly, which "
            "program this census is about."
        )
    return census(
        model_coverage.regions_from_module(module, precisions=precisions),
        target,
        router=router,
        datapath=datapath,
        precision_declared=datapath is not None,
    )


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
