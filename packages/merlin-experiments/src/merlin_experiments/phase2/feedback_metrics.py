"""Declared workload metrics and host-owned measurement summaries for tuning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Declared operations whose required work this stage can derive from the capsule's own shapes.
#: These are the emitted ABI's operation names, carried by the corpus rather than assumed about any
#: device. An operation outside this set yields no declared work and says so, rather than a zero.
#: The declared operation whose work depends on a GEOMETRY rather than only on operand shapes, so it
#: is priced through its own branch below.
_CONV_OPERATION = "conv2d"
#: A contraction with its bias epilogue forced on. The bias is an elementwise add on the readout path
#: and contributes no multiply-accumulate, so the declared work is the contraction's alone -- the same
#: M x K x N the plain matmul branch computes.
_FUSED_CONTRACTION = "fused_matmul_bias"
#: An ELEMENTWISE member, priced separately below because it has no MACs at all. It exists as the
#: "part" arm of the fusion comparison, and its verdict is that paired difference rather than a
#: utilization ratio, so pricing it in MACs would be a category error rather than a missing number.
_ELEMENTWISE_OPERATIONS = ("bias_add",)
#: Q @ K^T. Both operands are [queries, depth] and the right one is transposed by the operation
#: rather than by an attribute, so there is no "weight" in the matmul sense and the rank-2 weight
#: check below cannot see it. `work_volume` has counted this opcode all along; only the declared
#: price refused it, which left every attention member with no utilization, no share of achievable,
#: and no verdict.
_ATTENTION_QK_OPERATION = "attention_qk"
#: A batch of independent contractions, rank-3 on both operands. Same story: `work_volume` counts
#: BATCHED_MATMUL, and the declared price refused it for not being rank-2 -- so the one shape the
#: target contract was corrected to admit still could not be scored.
_BATCHED_OPERATION = "gemv_batched"
# A whole-program comparison with two declared contractions and an intervening host map. The
# host map contributes no MACs; both contraction stages do, regardless of which lane emits them.
_HOST_ISLAND_OPERATION = "host_island_seam"
#: The SAME contraction under another declared name: `linear` is bound to `build_matmul` in
#: `corpus_spec.BUILDERS`, declares the identical rank-2 `lhs`/`weight` attributes, and emits a plain
#: MATMUL that `work_volume` counts. Only this tuple refused it, so twelve members whose emitted
#: program IS priced by the work counter carried no declared price at all -- the two paths disagreeing
#: over a spelling.
_LINEAR_OPERATION = "linear"
_WORK_OPERATIONS = (
    "matmul",
    _LINEAR_OPERATION,
    "resident_reuse",
    _CONV_OPERATION,
    _FUSED_CONTRACTION,
    _ATTENTION_QK_OPERATION,
    _BATCHED_OPERATION,
    _HOST_ISLAND_OPERATION,
)

ARM_WORKSPACE = "m{index:03d}_{arm}"
_BASELINE_ARM_GLOB = "m*_baseline"
RECOVERABLE_RANK_LIMIT = 8


def capsule_verdict_fields(**kwargs: Any) -> dict[str, Any]:
    """Decide one member, and never let a failure to decide read as a decision."""
    try:
        from . import capsule_verdict as CV  # noqa: PLC0415

        row = CV.capsule_verdict(**kwargs)
        # CARRY THE HEADROOM, do not recompute it and do not throw it away. `capsule_verdict`
        # already derives how far this member is from the rate something on this machine actually
        # reached; keeping only the categorical verdict left the agent with "headroom_open" on a
        # member 1.02x off the ceiling and on one 30x off, which are not the same instruction. The
        # numbers ride as None when the verdict short-circuited before deriving them -- absent is a
        # different statement from zero, and a closed schema needs the key either way.
        return {
            "verdict": row.get("verdict"),
            "verdict_reason": row.get("reason"),
            "factor_to_achievable": row.get("factor_to_achievable"),
            "ideal_cycles_at_achievable": row.get("ideal_cycles_at_achievable"),
            "cycles_saved": row.get("cycles_saved"),
            "gap_closed": row.get("gap_closed"),
        }
    except Exception as exc:  # noqa: BLE001 - an undecidable member is refused, never assumed
        return {
            "verdict": "refused",
            "verdict_reason": f"the verdict could not be computed: {type(exc).__name__}",
            "factor_to_achievable": None,
            "ideal_cycles_at_achievable": None,
            "cycles_saved": None,
            "gap_closed": None,
        }


def recoverable_cycles(
    cells: Sequence[Mapping[str, Any]], achievable_macs_per_cycle: float | None = None
) -> dict[str, Any]:
    """Which members hold the cycles, ranked, and what fraction of the objective each one is.

    ⚠️ WHY THIS EXISTS, measured on a completed campaign. The corpus total was 171,739 cycles and the
    search converged at ~0.2%. It was not a search failure: the agent improved 9-14 members a trial
    with ZERO regressions. It was an Amdahl problem nobody had told it about. Seven deep-K residency
    members were **92.4% of all cycles**, already at 0.59-0.94 of the achievable rate, while the 18
    members with real headroom were **7.2% of the total** -- so perfecting every member the agent
    could reach was worth at most 4.64%. One member, the deepest spilling one, held 21,500 recoverable
    cycles by itself: **12.5% of the entire objective**, more than every small member combined.

    Every number above was already derivable from the cells: the agent was given `declared_macs`,
    `baseline_gsim_cycles` and `share_of_achievable` per member and would have had to multiply,
    subtract and rank across 38 rows to find it. It never did, and three trials of search went into
    7% of the objective. Reporting a share of the ACHIEVABLE rate tells a member how it is doing;
    reporting recoverable CYCLES tells the corpus where its time is. Those are different questions and
    only the second one orders the work.

    Recoverable is measured against the achievable rate, never the structural peak: no program on this
    machine has reached the peak (31.3% is the best observed), so pricing headroom against it would
    hand back a number that does not exist. A member already at or past the achievable rate recovers
    nothing rather than a negative amount.
    """
    rates = [row.get("achievable_macs_per_cycle", achievable_macs_per_cycle) for row in cells]
    if not any(isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0 for rate in rates):
        return {
            "status": "unavailable",
            "reason": ("no member-matched achievable rate was derived, so headroom cannot be priced in cycles"),
            "ranked": [],
            "corpus_total_cycles": None,
        }
    priced, total = [], 0.0
    for row in cells:
        macs, cycles = row.get("declared_macs"), row.get("baseline_gsim_cycles")
        rate = row.get("achievable_macs_per_cycle", achievable_macs_per_cycle)
        if (
            not row.get("measured")
            or not isinstance(macs, int)
            or not isinstance(cycles, int)
            or not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or rate <= 0
        ):
            continue
        total += float(cycles)
        ideal = float(macs) / float(rate)
        priced.append(
            {
                "family": row["family"],
                "capsule": row["capsule"],
                "baseline_cycles": int(cycles),
                "recoverable_cycles": max(0.0, float(cycles) - ideal),
            }
        )
    if not priced or total <= 0:
        return {
            "status": "unavailable",
            "reason": "no measured member declares the work its headroom would be priced from",
            "ranked": [],
            "corpus_total_cycles": None,
        }
    for row in priced:
        row["share_of_corpus_cycles"] = row["baseline_cycles"] / total
        row["recoverable_share_of_corpus"] = row["recoverable_cycles"] / total
    ranked = sorted(priced, key=lambda row: -row["recoverable_cycles"])[:RECOVERABLE_RANK_LIMIT]
    recoverable_total = sum(row["recoverable_cycles"] for row in priced)
    return {
        "status": "derived",
        "corpus_total_cycles": total,
        "total_recoverable_cycles": recoverable_total,
        "total_recoverable_share": recoverable_total / total,
        "ranked": ranked,
        "ranked_members": len(ranked),
        "priced_members": len(priced),
        "basis": (
            "baseline cycles minus the cycles this member's own declared work would take at "
            "the best host-owned measured rate with the same reduction-depth signature"
        ),
        "licence": (
            "where the objective's cycles are, not a prediction that they are reachable; a member's lever may not exist"
        ),
    }


def derived_peak_macs_per_cycle(rtl_facts_path: Path, target: str) -> tuple[int | None, str]:
    """The machine's structural MAC ceiling, DERIVED from its own RTL facts.

    This is the denominator of utilization, and it must never be a literal: it comes from
    ``merlin.perf.contract``, which reads the discovered array's geometry (rows x cols x the
    multipliers per element that the mac_idiom fact states) and refuses rather than inventing a peak
    when no array grounds the unit. If more than one compute resource carries a peak, this refuses
    too -- picking one would be choosing which machine the number describes.
    """
    try:
        from merlin.perf.contract import derive_contract  # noqa: PLC0415

        facts = json.loads(Path(rtl_facts_path).read_text(encoding="utf-8"))
        contract = derive_contract(target, facts=facts)
    except Exception as exc:  # noqa: BLE001 - an underivable ceiling is reported, never guessed
        return None, f"peak is not derivable from this target's RTL facts ({type(exc).__name__})"
    peaks: list[tuple[str, int]] = []
    for resource in contract.resources:
        term = (resource.terms or {}).get("peak_macs_per_cycle")
        value = getattr(term, "value", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            peaks.append((resource.name, value))
    if not peaks:
        return None, "this target's RTL facts evidence no compute array, so it has no derived peak"
    if len(peaks) > 1:
        names = ", ".join(sorted(name for name, _ in peaks))
        return None, f"this target evidences several compute units ({names}); utilization needs one"
    name, value = peaks[0]
    return value, f"facts-derived peak of compute unit {name!r}"


def declared_capsule_macs(descriptor: Mapping[str, Any]) -> tuple[int | None, str]:
    """The MACs the capsule's own declaration REQUIRES, independent of how a compiler emits them.

    Utilization must be priced against the work the spec demands, not the work the program happens to
    perform: dividing emitted MACs by cycles would reward a candidate for doing redundant arithmetic.
    Shapes come from the capsule's declared operands, so this stays a statement about the workload.
    """
    operation = descriptor.get("operation")
    if not isinstance(operation, Mapping) or operation.get("op") not in _WORK_OPERATIONS:
        return None, (
            f"declared work is derived for {sorted(_WORK_OPERATIONS)} only, not {(operation or {}).get('op')!r}"
        )
    attributes = operation.get("attributes")
    if not isinstance(attributes, Mapping):
        return None, "the declared operation carries no operand attributes"
    shapes: dict[str, list[int]] = {}
    for row in descriptor.get("inputs") or ():
        if not isinstance(row, Mapping):
            continue
        shape = row.get("shape")
        if (
            isinstance(shape, Sequence)
            and not isinstance(shape, (str, bytes))
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape)
        ):
            shapes[str(row.get("name"))] = [int(v) for v in shape]
    # BOTH OF THESE PRECEDE THE RANK-2 WEIGHT CHECK, because neither has a rank-2 weight to find:
    # attention names its operands q/k and transposes the second by definition, and a batched
    # contraction is rank-3 on both sides. Reaching the check below would refuse them for the shape
    # of a field they do not declare.
    if operation.get("op") == _ATTENTION_QK_OPERATION:
        q = shapes.get(str(attributes.get("q")))
        k = shapes.get(str(attributes.get("k")))
        if q is None or k is None or len(q) != 2 or len(k) != 2:
            return None, "the declared attention operands are not two rank-2 shapes"
        if q[1] != k[1]:
            return None, (f"the declared attention operands do not share a depth: q {q} against k {k}")
        # [queries, depth] @ [keys, depth]^T -> [queries, keys], so queries x depth x keys.
        return q[0] * q[1] * k[0], "declared attention operand shapes (queries x depth x keys)"

    if operation.get("op") == _BATCHED_OPERATION:
        lhs = shapes.get(str(attributes.get("lhs")))
        rhs = shapes.get(str(attributes.get("weight")))
        if lhs is None or rhs is None or len(lhs) != 3 or len(rhs) != 3:
            return None, "the declared batched operands are not two rank-3 shapes"
        if lhs[0] != rhs[0]:
            return None, (f"the declared batched operands describe different batches: {lhs} against {rhs}")
        if lhs[2] != rhs[1]:
            return None, (f"the declared batched operands do not contract: {lhs} against {rhs}")
        return lhs[0] * lhs[1] * lhs[2] * rhs[2], (
            "declared batched operand shapes (batch x M x K x N), one independent contraction per batch slice"
        )

    if operation.get("op") == _HOST_ISLAND_OPERATION:
        lhs = shapes.get(str(attributes.get("lhs")))
        first = shapes.get(str(attributes.get("weight0")))
        second = shapes.get(str(attributes.get("weight1")))
        if any(shape is None or len(shape) != 2 for shape in (lhs, first, second)):
            return None, "the declared host-island operands are not three rank-2 shapes"
        assert lhs is not None and first is not None and second is not None
        if lhs[1] != first[0] or first[1] != second[0]:
            return None, f"the declared host-island stages do not contract: {lhs}, {first}, {second}"
        extents = (lhs[0], lhs[1], first[1], second[1])
        if any(
            type(attributes.get(name)) is not int or attributes[name] != value
            for name, value in zip(("M", "K", "H", "N"), extents)
        ):
            return None, "the declared host-island extents disagree with its operand shapes"
        if attributes.get("accelerator_contractions") != 2:
            return None, "the declared host-island program does not name exactly two contractions"
        return lhs[0] * lhs[1] * first[1] + lhs[0] * first[1] * second[1], (
            "declared two-stage contraction shapes (M x K x H plus M x H x N); "
            "the intervening host map contributes no MACs"
        )

    weight = shapes.get(str(attributes.get("weight")))
    if weight is None or len(weight) != 2:
        return None, "the declared weight operand is not a rank-2 shape"

    # A CONVOLUTION'S WORK IS ITS OUTPUT EXTENT, WHICH ITS OPERAND SHAPES DO NOT CARRY. The other
    # operations read M from an activation row count; a conv's output rows are Ho x Wo, a function of
    # the image, the window, the stride, the padding and the dilation -- so a member whose geometry
    # differs does different work at identical operand shapes. Priced as None until now, which cost
    # the whole conv family its utilization, its share of the achievable rate and its verdict, and
    # disabled the corpus-wide attainment stop condition for every other member too (one unpriced
    # member is enough).
    #
    # The extent is DERIVED through the same helper the golden and the harness use, never recomputed
    # here: a second implementation of this arithmetic is a second thing to keep in sync, and it would
    # be wrong in exactly the padded and strided cases this pricing was added to reach.
    if operation.get("op") == _CONV_OPERATION:
        ifm = shapes.get(str(attributes.get("ifm")))
        if ifm is None or len(ifm) != 4:
            return None, "the declared convolution input is not a rank-4 NHWC shape"
        for field in ("ci", "kh", "kw"):
            if not isinstance(attributes.get(field), int) or isinstance(attributes.get(field), bool):
                return None, f"the declared convolution carries no integer {field}"
        ci, kh, kw = int(attributes["ci"]), int(attributes["kh"]), int(attributes["kw"])
        if ifm[3] != ci:
            return None, (f"the declared input channel count {ifm[3]} disagrees with the declared ci {ci}")
        if weight[0] != kh * kw * ci:
            return None, (
                f"the packed weight's {weight[0]} rows are not the {kh}x{kw}x{ci} window "
                f"the declaration names, so the two do not describe one convolution"
            )
        from merlin.runtime.commandbuffer import conv_out_dims  # noqa: PLC0415

        try:
            rows, cols = conv_out_dims(
                int(ifm[1]),
                int(ifm[2]),
                kh,
                kw,
                list(attributes.get("stride") or [1, 1]),
                list(attributes.get("padding") or [0, 0, 0, 0]),
                list(attributes.get("dilation") or [1, 1]),
            )
        except Exception as exc:  # noqa: BLE001 - an underivable extent refuses, never defaults
            return None, f"the declared convolution geometry has no output extent ({exc})"
        if rows <= 0 or cols <= 0:
            return None, (f"the declared convolution geometry leaves no output position ({rows}x{cols})")
        # THE BATCH IS PART OF THE WORK, and reading the extent while ignoring N is exactly how the
        # two pricing paths come to disagree over the same program. `work_volume._conv_work` multiplies
        # by the NHWC batch; every conv capsule in the tree today declares N=1, so this factor changes
        # no existing price -- which is the point: it is added while it is provably inert, rather than
        # discovered as an N-fold under-price by the first member that carries a batch.
        batch = int(ifm[0])
        return batch * rows * cols * weight[0] * weight[1], (
            f"declared convolution geometry: {batch} image(s) x {rows}x{cols} output positions x "
            f"{weight[0]} window taps x {weight[1]} output channels"
        )

    # A REUSED WEIGHT IS STILL DECLARED WORK. Twelve of the thirty-eight corpus members declare one
    # resident weight and a LIST of activations sharing it, and reading only a single `lhs` left
    # every one of them with no declared work: no utilization, no share of the achievable rate, no
    # verdict -- a third of the corpus with no headroom signal at all. It also left the corpus-wide
    # attainable total UNKNOWN, which silently disabled the attainment stop condition. The rule is
    # the same rule, summed: each reuse contracts the same weight, so each contributes its own
    # M x K x N and the total is what the specification demands however a compiler emits it.
    reuses = attributes.get("matmuls")
    if isinstance(reuses, Sequence) and not isinstance(reuses, (str, bytes)):
        total = 0
        for index, row in enumerate(reuses):
            if not isinstance(row, Mapping):
                return None, f"reuse {index} of the declared operation is not a mapping"
            lhs = shapes.get(str(row.get("lhs")))
            if lhs is None or len(lhs) != 2:
                return None, f"reuse {index} declares no rank-2 activation shape"
            if lhs[1] != weight[0]:
                return None, (f"reuse {index} does not contract: lhs {lhs} against weight {weight}")
            total += lhs[0] * lhs[1] * weight[1]
        if not total:
            return None, "the declared operation reuses the weight zero times"
        return total, (
            f"declared operand shapes, summed over the {len(reuses)} reuse(s) of one resident weight (M x K x N each)"
        )

    lhs = shapes.get(str(attributes.get("lhs")))
    if lhs is None or len(lhs) != 2:
        return None, "the declared matmul operands are not two rank-2 shapes"
    if lhs[1] != weight[0]:
        return None, (f"the declared operand shapes do not contract: lhs {lhs} against weight {weight}")
    basis = "declared matmul operand shapes (M x K x N)"
    if operation.get("op") == _FUSED_CONTRACTION:
        basis = (
            "declared contraction operand shapes (M x K x N); the fused bias epilogue is an "
            "elementwise add on the readout path and contributes no multiply-accumulate"
        )
    return lhs[0] * lhs[1] * weight[1], basis


def declared_reduction_depths(descriptor: Mapping[str, Any]) -> tuple[tuple[int, ...] | None, str]:
    """Contraction-depth signature of a frozen member, from its declared operand geometry.

    This mirrors the operation families accepted by :func:`declared_capsule_macs`, but returns only
    the axis that controls fixed-cost amortisation.  It intentionally does not infer a target tile
    size or bucket nearby depths: exact equality is the conservative notion of "could resemble".
    """
    operation = descriptor.get("operation")
    inputs = descriptor.get("inputs")
    if not isinstance(operation, Mapping) or not isinstance(inputs, Sequence):
        return None, "the member has no declared operation and inputs"
    attributes = operation.get("attributes")
    if not isinstance(attributes, Mapping):
        return None, "the member operation has no declared attributes"
    shapes: dict[str, tuple[int, ...]] = {}
    for row in inputs:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            continue
        raw = row.get("shape")
        if (
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes))
            and raw
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in raw)
        ):
            shapes[str(row["name"])] = tuple(int(v) for v in raw)

    op = operation.get("op")
    if op == _ATTENTION_QK_OPERATION:
        lhs = shapes.get(str(attributes.get("q")))
        return (
            ((lhs[1],), "declared attention reduction depth")
            if lhs and len(lhs) == 2
            else (None, "the declared attention query is not rank 2")
        )
    if op == _BATCHED_OPERATION:
        lhs = shapes.get(str(attributes.get("lhs")))
        return (
            ((lhs[2],), "declared batched-contraction reduction depth")
            if lhs and len(lhs) == 3
            else (None, "the declared batched lhs is not rank 3")
        )

    if op == _HOST_ISLAND_OPERATION:
        macs, basis = declared_capsule_macs(descriptor)
        if macs is None:
            return None, basis
        lhs = shapes.get(str(attributes.get("lhs")))
        first = shapes.get(str(attributes.get("weight0")))
        second = shapes.get(str(attributes.get("weight1")))
        if not lhs or not first or not second or any(len(shape) != 2 for shape in (lhs, first, second)):
            return None, "the declared host-island operands are not three rank-2 shapes"
        if lhs[1] != first[0] or first[1] != second[0]:
            return None, "the declared host-island stages do not contract"
        return (lhs[1], first[1]), "declared reduction depths of both host-island contractions"

    weight = shapes.get(str(attributes.get("weight")))
    if weight is None or len(weight) != 2:
        return None, "the declared weight is not rank 2"
    if op == _CONV_OPERATION:
        return (weight[0],), "declared packed convolution-window depth"
    reuses = attributes.get("matmuls")
    if isinstance(reuses, Sequence) and not isinstance(reuses, (str, bytes)):
        if not reuses:
            return None, "the declared resident operation has no reuses"
        return tuple(sorted(weight[0] for _ in reuses)), (
            "declared reduction depth for every reuse of the resident weight"
        )
    lhs = shapes.get(str(attributes.get("lhs")))
    if lhs is None or len(lhs) != 2 or lhs[1] != weight[0]:
        return None, "the declared matmul operands do not form a rank-2 contraction"
    return (lhs[1],), "declared matmul reduction depth"


def unmeasured_cell(member: Any, *, reason: str) -> dict[str, Any]:
    return {
        "family": member.family,
        "capsule": member.capsule,
        "baseline_correct": None,
        "candidate_correct": None,
        "baseline_gsim_cycles": None,
        "candidate_gsim_cycles": None,
        "candidate_minus_baseline_cycles": None,
        "baseline_over_candidate": None,
        "comparable": False,
        "declared_macs": None,
        "declared_work_basis": None,
        "ideal_cycles_at_peak": None,
        "baseline_utilization": None,
        "candidate_utilization": None,
        "baseline_share_of_achievable": None,
        "candidate_share_of_achievable": None,
        "achievable_macs_per_cycle": None,
        "achievable_basis": None,
        "verdict": "refused",
        "verdict_reason": reason,
        "factor_to_achievable": None,
        "ideal_cycles_at_achievable": None,
        "cycles_saved": None,
        "gap_closed": None,
        "measured": False,
        "skip_reason": reason,
    }


def harvest_baseline_points(work_root: Path) -> list:
    """Measured points from the FROZEN-BASELINE arm of this run, and nothing else.

    The baseline package is phase 1's submission, sha-pinned before the first member runs and
    re-executed unchanged for every one, so these points cannot depend on anything the agent wrote.
    CANDIDATE arms are excluded by construction and that exclusion is the whole safety argument: a
    ceiling the agent can raise is a target the agent authors, and `share_of_achievable` would then
    pin its own candidate near 1.0 while pushing every other member down.

    Scoped to ONE run's work root on purpose. Harvesting across stages would make the objective
    depend on what happens to be on disk at launch -- two trials of one configuration would be
    scored differently, and clearing `out/` would silently move the target.

    The filter is the workspace directory THIS module creates, not a parse of the runner's run-id:
    that name is minted in another module and matching it would be exactly the brittleness the
    no-regex rule exists to prevent.
    """
    from . import calibration as PMODEL  # noqa: PLC0415

    root = Path(work_root)
    if not root.is_dir():
        return []
    found: dict[str, Any] = {}
    for arm_dir in sorted(root.glob(f"round_*/call_*/{_BASELINE_ARM_GLOB}")):
        for point in PMODEL.harvest_measured_points(arm_dir)[0]:
            found.setdefault(point.capsule, point)
    return sorted(found.values(), key=lambda p: p.capsule)


def harvest_member_cost(roots: Sequence[Path]) -> dict[str, float]:
    """Median measured simulation seconds per capsule, from runs already on disk.

    ORDER THE SWEEP BY WHAT IT COSTS, and derive that from measurement rather than from a proxy.
    Declared MACs are the obvious proxy and they are WRONG here: measured on this corpus, a
    262144-MAC deep-K member simulates in 74.9 s while a 65536-MAC wide-M/N member takes 178.5 s --
    the proxy inverts on exactly the pair it would need to get right. Simulation cost tracks the
    shape of the program, not the size of its arithmetic, so it is read from prior runs.

    Absent history yields an empty table and the caller keeps the declared order, saying so. An
    empty table is never a claim that every member costs the same.
    """
    seconds: dict[str, list[float]] = {}
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        for path in Path(root).rglob("capsule_result.json"):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - an unreadable result contributes no cost, not a zero
                continue
            capsule = str(document.get("capsule") or "")
            tier = (document.get("tiers") or {}).get("L3") or {}
            active = (tier.get("timing") or {}).get("sim_active_s")
            # A WALL TIME MEASURED UNDER FAN-OUT DOES NOT PRICE THE MEMBER. Cycles are
            # concurrency-invariant on this stack; wall times are emphatically not -- the same query
            # measured 3.7 s serial and 23.4 s at 16 workers, a 6.3x spread. Mixing those into the
            # median would inflate exactly the members that happened to run beside others and
            # silently corrupt the cheapest-first ordering the early stop depends on. The runner
            # stamps the fan-out it ran at; a row that does not say, or says more than one, is not
            # a price.
            fanout = (tier.get("concurrency") or {}).get("workers")
            if fanout is not None and fanout != 1:
                continue
            if capsule and isinstance(active, (int, float)) and not isinstance(active, bool) and active > 0:
                seconds.setdefault(capsule, []).append(float(active))
    table: dict[str, float] = {}
    for capsule, values in seconds.items():
        values.sort()
        middle = len(values) // 2
        table[capsule] = values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])
    return table


def order_members_by_cost(members: Sequence[Any], cost: Mapping[str, float]) -> tuple[tuple, str]:
    """Cheapest measured member first; unpriced members last, so absence never looks cheap.

    Returns the ordering and the basis, because an ordering nobody can account for is one nobody
    can check. A member with no recorded cost sorts AFTER every priced one: it might be the most
    expensive in the corpus, and guessing it is cheap would put the slowest member first.
    """
    if not cost:
        return tuple(members), "declared order; no measured simulation cost is on record"
    ordered = sorted(
        members, key=lambda m: (cost.get(m.capsule) is None, cost.get(m.capsule, 0.0), m.family, m.capsule)
    )
    priced = sum(1 for m in members if m.capsule in cost)
    return tuple(ordered), (
        f"ascending median measured simulation seconds ({priced}/{len(members)} members priced; unpriced sort last)"
    )
