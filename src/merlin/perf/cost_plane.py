"""A cycle count, and the two things that have to be true before it is a COST.

WHY THIS EXISTS. Every tier in this repo can report cycles and nothing gates on them. A submission
may pass every correctness tier while the program it emitted takes a hundred times longer than the
machine needs, and the grade says `pass`. The counts are on disk -- `tier_cycles` rides the verdict
the agent reads -- and no rule anywhere consumes them.

The two things, and why each is a refusal rather than a number:

**1. A cycle count is about a DESIGN.** Two engines at the same declared fidelity differ by more
than an order of magnitude on the same capsule, and the same RTL built from a different
configuration is a different machine. Comparing counts across those is not an approximation, it is a
category error. :func:`require_same_design` refuses it, on the same keys
``perf.firesim_checkpoint.compare`` refuses on (``substrate``, ``hw_config``,
``hwdb_config_artifact_sha256``) plus the two this harness varies (``engine``, ``tier``) -- and, on
top of that, refuses a design it cannot NAME. Two records that both say ``None`` compare equal,
which would make a missing identity read as a matching one; that is the flattering direction.

**2. A bound has to come from somewhere nobody invented.** There are exactly two here.

  * A DECLARED ceiling: an author's ``{max: N}`` on the capsule's own ``performance.cost``
    projection, evaluated with :func:`merlin.perf.functional_gate.compare_field` -- the same
    comparator that bounds every other declared quantity, so the bound's grammar has one spelling.
  * A DERIVED FLOOR: the cycles the array must spend issuing this program's own tiles, from
    :func:`merlin.perf.mesh_occupancy.tile_issue_cycles` through
    :func:`merlin.perf.envelope.array_issue_time`. A floor needs no slack constant, which is exactly
    why it is the derived half: ``measured / floor`` is reported as efficiency, and a count BELOW
    the floor is decidable with nothing invented -- the program cannot have issued its own tiles in
    fewer cycles than the sequencer's loop takes, so such a count is not about that program.

A slack factor over the floor is what turns the floor into an "is this fast enough" verdict, and
there is still no MEASURED one. Inventing one in this file would silently decide which submissions
pass -- so it is not invented here, it is DECLARED: ``policies.cost_plane`` in
``merlin/contract/gate_phases.yaml``, beside the phase this gate runs in, where changing it is a
one-line reviewable diff. :func:`declared_slack` reads it and never defaults; undeclared, the plane
goes back to reporting the ratio and saying it cannot decide.

That is what makes the capsule corpus's own word mean something. 113 capsules wrote
``derived_at_preflight`` in their one ceiling slot and nothing resolved it, so all 113 read as "no
ceiling"; :func:`resolve_ceiling` now derives their ceiling from the floor above times that declared
slack, through :func:`merlin.perf.preflight.projected_cycle_ceiling` -- the module the word is named
after.

**PHASED, DELIBERATELY.** This landed at :data:`PHASE_REPORT` -- computing the whole verdict, recording
it beside the grade, blocking nothing -- because a plane that flips to ``fail`` on day one is
indistinguishable from a regression. It moved to :data:`PHASE_FAIL` once a ceiling could be resolved
at all, and the phase is read from the tracked declaration, never passed as a literal. The middle
state is neither -- :data:`STATUS_INCOMPLETE` means the plane could not decide, and it is never a pass
at either phase, because "not measured" reading as "measured fine" is the failure this repo keeps
rediscovering.

**AND SOMETHING HAS TO READ THE VERDICT.** Every verdict carries ``blocking``, and for as long as
nothing consumed it the phase was decorative: flipping it changed what an artifact said and not what
happened. :func:`require_within` raises on that field and :func:`apply_gate` is the enforcement call
site -- it attaches the verdict to a result row and, at a blocking phase, sets ``status`` and
``failure`` in the shape ``targetgen.epilogue_store_path`` already writes, so a reader of a failed row
meets one vocabulary.

Target-neutral: the array geometry, the tiles, the design identity and the ladder are all
PARAMETERS, derived by the caller from the target's own facts. This module names no target, no
engine and no geometry of its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from merlin.targetgen import tier_policy as _tier_policy

from . import cycle_bound
from . import gate_phase as _gate_phase_module
from . import preflight as preflight_module
from .decompose import UNKNOWN, is_unknown
from .gate_phase import PHASE_FAIL, PHASE_REPORT, PHASES, STATUS_INCOMPLETE, blocks

__all__ = [
    "PHASE_REPORT",
    "PHASE_FAIL",
    "PHASES",
    "STATUS_WITHIN",
    "STATUS_OVER",
    "STATUS_BELOW_FLOOR",
    "STATUS_INCOMPLETE",
    "FAILING_STATUSES",
    "CYCLES_FIELD",
    "TIMING_CEILING_FIELD",
    "AXIS_TIMING",
    "DESIGN_KEYS",
    "CrossDesignComparison",
    "CostCeilingExceeded",
    "GATE",
    "PLANE",
    "CATEGORY",
    "design_of",
    "require_same_design",
    "declared_slack",
    "declared_ceiling",
    "resolve_ceiling",
    "derived_floor",
    "assess",
    "require_within",
    "apply_gate",
]

#: The name this plane is declared under in ``merlin/contract/gate_phases.yaml`` -- both for its phase
#: (``gates``) and for the numbers it decides with (``policies``). One spelling, because a gate whose
#: policy is filed under a different name than its phase is a gate with no policy and no error.
GATE = "cost_plane"
#: How a failure of this plane names itself in a result row, matching the shape
#: ``epilogue_store_path`` writes so a reader of ``failure`` sees one vocabulary.
PLANE = "cost_plane"
CATEGORY = "performance"

#: The measurement is within its declared ceiling (or has none and cleared the floor).
STATUS_WITHIN = "within"
#: The measurement exceeded a ceiling its own capsule declared.
STATUS_OVER = "over"
#: The measurement is BELOW the cycles the array must spend issuing this program's tiles, so it is
#: not a measurement of this program. Decidable with no invented constant.
STATUS_BELOW_FLOOR = "below_floor"

#: The DECIDED failures of this plane. Everything else that is not :data:`STATUS_WITHIN` is
#: :data:`STATUS_INCOMPLETE` and blocks at no phase.
FAILING_STATUSES = (STATUS_OVER, STATUS_BELOW_FLOOR)

#: The quantity this plane bounds, spelled as the result field a harness prints it under.
CYCLES_FIELD = "cycles"

#: The capsule field that excludes a member from the timing measurement matrix, and the axis name a
#: tier is bought on. RE-EXPORTED from the tier policy rather than spelled again: those slots were
#: cut for exactly this plane and a second copy of either would drift from the policy that owns them.
TIMING_CEILING_FIELD = _tier_policy.TIMING_CEILING_FIELD
AXIS_TIMING = _tier_policy.AXIS_TIMING

#: What identifies the design a cycle count is about. The first three are the keys
#: ``firesim_checkpoint.compare`` refuses on; ``engine`` and ``tier`` are the two this harness
#: varies between grades of the same submission.
DESIGN_KEYS = ("substrate", "hw_config", "hwdb_config_artifact_sha256", "engine", "tier")


class CrossDesignComparison(ValueError):
    """Two cycle counts were about different machines, or about machines that could not be named."""


class CostCeilingExceeded(ValueError):
    """A measured cycle count was a DECIDED failure of this plane, at a phase that blocks."""


@lru_cache(maxsize=1)
def declared_slack() -> tuple[float | None, str]:
    """``(slack, basis)`` -- how many times the derived issue floor a submission may spend, or why not.

    CACHED, like :func:`merlin.perf.gate_phase._declared` and for the same reason: :func:`assess` runs
    once per graded row, and a grade walks hundreds of them. Re-reading and re-parsing the declaration
    for every capsule is a file read per row to answer a question whose answer cannot change inside one
    process -- the file is a tracked contract, not a live control.

    Read from ``merlin/contract/gate_phases.yaml`` under ``policies.cost_plane``, beside the phase this
    gate runs in, and never defaulted. The module docstring's own objection stands and this is the
    answer to it: a slack over the floor DOES decide which submissions pass, so it may not be invented
    at the point of use -- it lives in a tracked file where changing it is a one-line reviewable diff,
    exactly like the phase.

    Absent, unparseable, or below 1 resolves to ``None`` and a reason, which makes the plane report
    ``incomplete``. Deleting the declaration stops the plane deciding; it never starts it admitting.
    """
    import yaml

    from ..common.paths import merlin_dir

    path = merlin_dir() / "contract" / "gate_phases.yaml"
    if not path.is_file():
        return None, f"no gate declaration at {path}, so no slack over the issue floor is declared"
    try:
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as broken:
        return None, f"{path} could not be parsed ({broken}), so no slack is declared"
    policy = ((body.get("policies") or {}).get(GATE) or {}) if isinstance(body.get("policies"), dict) else {}
    value = policy.get("ceiling_slack_over_issue_floor") if isinstance(policy, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, (
            f"policies.{GATE}.ceiling_slack_over_issue_floor is {value!r} in {path.name}: this plane "
            f"derives a FLOOR with nothing invented and needs a declared multiple of it to decide "
            f"'fast enough'. Absent, it reports rather than guessing"
        )
    if value < 1:
        return None, (
            f"policies.{GATE}.ceiling_slack_over_issue_floor is {value!r}, below 1 -- that would "
            f"demand a program finish in fewer cycles than its own sequencer's loop takes"
        )
    return float(value), f"policies.{GATE}.ceiling_slack_over_issue_floor = {value} in {path.name}"


def design_of(record: Mapping[str, Any] | None, *, tier: str | None = None) -> dict[str, Any]:
    """The design identity carried by one tier record, with every key present.

    Present-and-``None`` rather than absent on purpose: an absent key reads to a consumer as "this
    axis does not apply", while an explicit ``None`` reads as "nobody said", and only the second is
    true of a record that omitted its engine.
    """
    record = record if isinstance(record, Mapping) else {}
    out = {key: record.get(key) for key in DESIGN_KEYS}
    if tier is not None:
        out["tier"] = str(tier)
    return out


def require_same_design(current: Mapping[str, Any], previous: Mapping[str, Any]) -> None:
    """Raise unless both counts are about the SAME, NAMED design.

    Inherited from ``perf.firesim_checkpoint.compare``, which raises on a substrate / hw_config /
    hwdb mismatch, and extended with the refusal that module does not need: an identity nobody
    stated. There, a checkpoint always names its design; here a tier record may omit its engine, and
    two omissions compare equal -- so an unnamed design would silently license the comparison it
    exists to prevent.
    """
    unnamed = sorted(
        {key for key in DESIGN_KEYS for side in (current, previous) if not str(side.get(key) or "").strip()}
    )
    if unnamed:
        raise CrossDesignComparison(
            "cycle counts cannot be compared: " + ", ".join(unnamed) + " is not stated on both "
            "sides, and two unstated identities compare equal -- which would read as the same "
            "design rather than as an unknown one"
        )
    for key in DESIGN_KEYS:
        if current.get(key) != previous.get(key):
            raise CrossDesignComparison(
                f"cycle counts differ in {key} ({previous.get(key)!r} -> {current.get(key)!r}); "
                f"cycles from different designs are not comparable"
            )


def declared_ceiling(capsule: Mapping[str, Any]) -> tuple[int | None, str]:
    """``(max_cycles, basis)`` from the capsule's own cost projection, read and never inferred.

    The corpus already carries the slot (``performance.cost.projected_cycles``) and most members fill
    it with a word naming when the number would arrive. A word is not a bound, so it resolves to no
    ceiling and says so -- rather than being parsed into one.

    The vocabulary lives in :mod:`merlin.perf.cycle_bound`, which is where the capsule schema and the
    corpus generator read it from too, so the three cannot drift into admitting different words. This
    reader used to walk the field itself and treat ANY non-integer as "no ceiling", which made a typo
    indistinguishable from a declaration -- a capsule that meant to owe a number quietly owed nothing.
    A value the vocabulary does not know is now a REFUSAL carried in the basis, not silence.
    """
    try:
        declared = cycle_bound.declared_cycles(capsule)
    except cycle_bound.CycleBoundError as malformed:
        return None, str(malformed)
    if declared is not None:
        return declared, "declared in performance.cost.projected_cycles"
    node: Any = capsule
    for key in ("performance", "cost", "projected_cycles"):
        node = node.get(key) if isinstance(node, Mapping) else None
    if node is None:
        return None, "the capsule declares no performance.cost.projected_cycles"
    return None, (
        f"performance.cost.projected_cycles is {node!r}: {cycle_bound.NO_CYCLE_BOUND[node]}"
        if node in cycle_bound.NO_CYCLE_BOUND
        else f"performance.cost.projected_cycles is {node!r}, which states when a number "
        f"would be derived rather than stating one"
    )


def _no_bound_word(capsule: Mapping[str, Any]) -> str | None:
    """The word the capsule wrote in its ceiling slot, or ``None`` when it wrote a count or nothing."""
    node: Any = capsule
    for key in ("performance", "cost", "projected_cycles"):
        node = node.get(key) if isinstance(node, Mapping) else None
    return node if isinstance(node, str) else None


def resolve_ceiling(capsule: Mapping[str, Any], floor: Mapping[str, Any]) -> tuple[int | None, str]:
    """``(max_cycles, basis)`` -- the ceiling this capsule is actually held to, literal OR derived.

    THE PROMISSORY NOTE, HONOURED. 113 capsules wrote ``derived_at_preflight`` in their one ceiling
    slot and nothing anywhere resolved it, so every one of them read to the plane as "no ceiling" --
    which is how a submission a hundred times slower than the machine needs graded exactly like one
    that was not. The word says a number is OWED and derived rather than typed, and it is derived
    here, from three things a capsule cannot carry and a run can:

      * the ISSUE FLOOR, from the array the target's own facts declare and the work this capsule's own
        command buffer counted (:func:`derived_floor`) -- every machine fact lives here;
      * the SLACK, declared in ``gate_phases.yaml`` beside this gate's phase (:func:`declared_slack`);
      * the arithmetic, in :func:`merlin.perf.preflight.projected_cycle_ceiling` -- the module the
        word is named after, which until now had no caller outside its own tests.

    A LITERAL COUNT STILL WINS. A capsule that states an integer is held to that integer and the
    derivation does not run: an author's explicit number is a stronger statement than a policy
    multiple, and quietly overriding it would make the field decorative in the other direction.

    FAIL CLOSED. Every step that cannot be taken returns ``None`` with the reason, and ``None`` makes
    the plane ``incomplete`` -- never a pass. An unresolvable floor, an undeclared slack and a word
    that means "no bound at all" are three different sentences and each is carried as its own.
    """
    literal, basis = declared_ceiling(capsule)
    if literal is not None:
        return literal, basis
    word = _no_bound_word(capsule)
    if word not in cycle_bound.DERIVED_AT_ASSESSMENT:
        return None, basis
    # A LAW MEMBER IS NOT HELD TO A CEILING, whatever its slot says. Its cycles are the OBSERVATIONS a
    # law about the machine is fitted to -- a reduction-depth sweep exists to be measured at every
    # depth, including the slow ones -- so budgeting it would refuse the very points that identify the
    # machine's own rate. Read off the capsule's own ``performance.member_class`` rather than from a
    # naming convention, and structural rather than declarative on purpose: the shared template says
    # ``unbounded`` for these families now, and a member that predates that edit (or a family added
    # without it) must not acquire a budget by omission.
    member_class = str(
        ((capsule.get("performance") or {}) if isinstance(capsule, Mapping) else {}).get("member_class") or ""
    )
    if member_class.upper() == "LAW":
        return None, (
            f"performance.member_class is {member_class!r}: this member's cycles are the observations "
            f"a law about the machine is fitted to, not a schedule anyone is held to, so no ceiling is "
            f"derived for it (the shared template spells this {sorted(cycle_bound.NO_BOUND_AT_ALL)!r})"
        )
    floor_cycles = floor.get("cycles")
    if is_unknown(floor_cycles):
        return None, (
            f"performance.cost.projected_cycles is {word!r}, so the ceiling is derived -- and the "
            f"issue floor it is derived from is UNKNOWN: {floor.get('reason') or 'no reason recorded'}"
        )
    slack, slack_basis = declared_slack()
    if slack is None:
        return None, (
            f"performance.cost.projected_cycles is {word!r}, so the ceiling is derived -- and the "
            f"slack it is derived with is not declared: {slack_basis}"
        )
    ceiling, why = preflight_module.projected_cycle_ceiling(float(floor_cycles), slack=slack)
    if ceiling is None:
        return None, f"performance.cost.projected_cycles is {word!r}, and {why}"
    lower_bound = (
        " (over work that is itself a LOWER bound, so the ceiling is loose)" if floor.get("is_lower_bound_work") else ""
    )
    return ceiling, (
        f"derived from performance.cost.projected_cycles = {word!r}: {why}, floor basis "
        f"{floor.get('basis')}{lower_bound}; slack from {slack_basis}"
    )


def derived_floor(
    *,
    tiles: Sequence[Mapping[str, Any]] = (),
    macs: int | None = None,
    array_rows: int | None = None,
    array_cols: int | None = None,
    provenance: str = "",
) -> dict[str, Any]:
    """The cycles the array cannot finish this program in fewer of, or why that is UNKNOWN.

    Two derivations, strongest first, and both are floors:

    * from TILES, through :func:`merlin.perf.envelope.array_issue_time` -- the sequencer's own loop,
      including the partial-block waste a division cannot see. This is the tight one;
    * from MACS alone, divided by the slots the array retires per cycle. Looser, and still a floor:
      no schedule retires more than every slot every cycle. A MAC count that is itself a lower bound
      (some command had no work-counting rule) keeps this a valid floor -- a smaller numerator can
      only lower it -- and that is recorded so a reader knows how loose it is.

    Neither invents a rate. When the array geometry is not resolved, or no work could be counted,
    the floor is UNKNOWN with the reason, because a floor of zero admits every measurement.
    """
    from .envelope import array_issue_time

    if tiles:
        time = array_issue_time(
            list(tiles),
            array_rows=array_rows,
            array_cols=array_cols,
            resource="array_issue",
            provenance=provenance or "declared tiles",
        )
        if time.known:
            return {
                "cycles": float(time.cycles),
                "basis": "array_issue_cycles_over_declared_tiles",
                "is_lower_bound_work": False,
                "reason": "",
            }
        return {
            "cycles": UNKNOWN,
            "basis": "array_issue_cycles_over_declared_tiles",
            "is_lower_bound_work": False,
            "reason": time.reason,
        }

    unresolved = [
        name
        for name, value in (("array_rows", array_rows), ("array_cols", array_cols))
        if not isinstance(value, int) or isinstance(value, bool) or value < 1
    ]
    if unresolved:
        return {
            "cycles": UNKNOWN,
            "basis": "macs_over_array_slots",
            "is_lower_bound_work": None,
            "reason": (
                f"the array geometry is UNKNOWN: {', '.join(unresolved)} was not "
                f"resolved from the target's own facts, and no default array exists"
            ),
        }
    if not isinstance(macs, int) or isinstance(macs, bool) or macs < 0:
        return {
            "cycles": UNKNOWN,
            "basis": "macs_over_array_slots",
            "is_lower_bound_work": None,
            "reason": (
                "no tile geometry and no counted MAC work, so there is nothing to divide;"
                " a floor of zero would admit any measurement at all"
            ),
        }
    slots = int(array_rows) * int(array_cols)
    return {"cycles": float(macs) / slots, "basis": "macs_over_array_slots", "is_lower_bound_work": True, "reason": ""}


def _cycles_of(tiers: Mapping[str, Any] | None, tier: str) -> tuple[int | None, Mapping[str, Any]]:
    record = (tiers or {}).get(tier)
    record = record if isinstance(record, Mapping) else {}
    value = record.get(CYCLES_FIELD)
    if isinstance(value, bool) or not isinstance(value, int):
        return None, record
    return value, record


def assess(
    capsule: Mapping[str, Any],
    *,
    tiers: Mapping[str, Any] | None,
    tiles: Sequence[Mapping[str, Any]] = (),
    macs: int | None = None,
    array_rows: int | None = None,
    array_cols: int | None = None,
    design: Mapping[str, Any] | None = None,
    phase: str = PHASE_REPORT,
) -> dict[str, Any]:
    """The cost verdict for one capsule, on the TIMING axis its own acceptance block declares.

    Which rung owes the cycle count is read from the capsule
    (:func:`merlin.targetgen.tier_policy.declared_axes`), and whether the member was excluded from
    the measurement matrix at all is read from the same policy's timing ceiling
    (``max_timing_tier``). Neither is guessed from a naming convention: a capsule that owes its
    timing evidence on a different rung, or owes none, is covered by the same code.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    def verdict(status: str, reason: str, **extra: Any) -> dict[str, Any]:
        blocking = blocks(phase, status, failing=FAILING_STATUSES)
        return {
            "schema": "cost_plane_verdict_v1",
            "axis": _tier_policy.AXIS_TIMING,
            "phase": phase,
            "status": status,
            "reason": reason,
            "timing_tier": timing_tier,
            _tier_policy.TIMING_CEILING_FIELD: timing_cap,
            "design": dict(design_identity),
            "enforced": phase == PHASE_FAIL,
            "blocking": blocking,
            "admitted": status == STATUS_WITHIN,
            "would_fail_at": (
                "phase=fail blocks a measurement over a declared ceiling or below the array's own "
                "issue floor; an undecided plane stays incomplete at every phase"
            ),
            **extra,
        }

    _correctness, timing_tier = _tier_policy.declared_axes(capsule)
    timing_cap, _cap_source = _tier_policy.declared_ceiling(capsule, _tier_policy.AXIS_TIMING)
    design_identity = design_of(design)

    if not timing_tier:
        return verdict(
            STATUS_INCOMPLETE,
            "the capsule declares no performance.acceptance.evidence.timing_tier, so no "
            "rung owes a cycle count and there is nothing on this axis to bound",
        )
    if (
        timing_cap
        and _tier_policy.tier_depth_order([timing_cap, timing_tier])[0] == timing_cap
        and timing_cap != timing_tier
    ):
        return verdict(
            STATUS_INCOMPLETE,
            f"{_tier_policy.TIMING_CEILING_FIELD}={timing_cap} caps this member below "
            f"the {timing_tier} rung that owes its timing evidence, so it was excluded "
            f"from the measurement matrix deliberately and is not a cell that failed",
        )

    measured, record = _cycles_of(tiers, timing_tier)
    design_identity = design_of(record if design is None else design, tier=timing_tier)
    if measured is None:
        return verdict(
            STATUS_INCOMPLETE,
            f"the {timing_tier} tier reported no cycle count, so there is no measurement "
            f"on this axis; an absent count is not a fast one",
            measured_cycles=None,
        )

    floor = derived_floor(
        tiles=tiles,
        macs=macs,
        array_rows=array_rows,
        array_cols=array_cols,
        provenance=f"{timing_tier} tier of this capsule",
    )
    ceiling, ceiling_basis = resolve_ceiling(capsule, floor)
    efficiency = (
        round(float(floor["cycles"]) / measured, 6) if not is_unknown(floor["cycles"]) and measured > 0 else None
    )
    common = {
        "measured_cycles": measured,
        "floor_cycles": None if is_unknown(floor["cycles"]) else floor["cycles"],
        "floor_basis": floor["basis"],
        "floor_reason": floor["reason"],
        "floor_work_is_lower_bound": floor["is_lower_bound_work"],
        "array_efficiency": efficiency,
        "ceiling_cycles": ceiling,
        "ceiling_basis": ceiling_basis,
    }

    if not is_unknown(floor["cycles"]) and measured < floor["cycles"]:
        return verdict(
            STATUS_BELOW_FLOOR,
            f"{measured} cycles is below the {floor['cycles']:.0f} this array must spend issuing "
            f"this program's own tiles. A program cannot finish faster than its sequencer's loop, "
            f"so the count is not a measurement of this program",
            **common,
        )

    if ceiling is None:
        return verdict(
            STATUS_INCOMPLETE,
            f"{ceiling_basis}. Without a ceiling the derived half of this plane is a FLOOR -- it can "
            f"refute an impossible count but cannot say whether {measured} cycles is fast enough, and "
            f"the missing ingredient is stated above rather than filled in: a slack nobody declared, "
            f"or a floor nothing could derive, is reported as undecided and never as a pass",
            **common,
        )

    from .functional_gate import FunctionalGateSpec, compare_field

    spec = FunctionalGateSpec(expectations={CYCLES_FIELD: {"max": ceiling}})
    row = compare_field(CYCLES_FIELD, spec.expectations[CYCLES_FIELD], str(measured))
    if row["ok"]:
        return verdict(
            STATUS_WITHIN, f"{measured} cycles is within the declared ceiling of {ceiling}", comparison=row, **common
        )
    return verdict(
        STATUS_OVER,
        f"{measured} cycles exceeds this capsule's ceiling of {ceiling} ({ceiling_basis}): {row['reason']}",
        comparison=row,
        **common,
    )


def require_within(verdict: Mapping[str, Any]) -> None:
    """Raise :class:`CostCeilingExceeded` for a verdict that BLOCKS.

    Blocking is the verdict's own decision, so the phase is honoured here rather than re-derived: at
    ``report`` this raises for nothing, which is the point of landing there first, and an
    :data:`STATUS_INCOMPLETE` verdict never raises at either phase.

    The exact shape of :func:`merlin.perf.lowering_coverage.require_offload`, deliberately. Before
    this existed, ``cost_plane`` computed ``blocking`` on every verdict and NOTHING in the tree read
    it -- the field was the only evidence that a rollout had been thought about, and a phase flip over
    a field nobody reads changes what an artifact says and not what happens.
    """
    if verdict.get("blocking"):
        raise CostCeilingExceeded(str(verdict.get("reason")))


def apply_gate(
    results: Sequence[Mapping[str, Any]],
    capsules: Sequence[Mapping[str, Any]],
    *,
    array_rows: int | None = None,
    array_cols: int | None = None,
    macs_of=None,
    phase: str | None = None,
) -> list[dict[str, Any]]:
    """Attach the cost verdict to every row, and FAIL it at the declared phase.

    The enforcement call site. It mirrors
    :func:`merlin.targetgen.epilogue_store_path.apply_gate` field for field -- the verdict lands on
    the row under its plane's name, and a DECIDED failure at a blocking phase sets ``status`` and
    ``failure`` -- so a reader of a result row meets one vocabulary rather than one per plane.

    ``macs_of`` maps a result row to the MAC count its own command buffer counted, defaulting to the
    ``work_volume.known_macs`` the grader already records. It is a parameter because the floor must be
    derived from the work THIS program did, and no rule here may assume where a harness files it.

    Returns every row it JUDGED, not only the ones it failed, so a caller can tell a gate that ran and
    found nothing from one that did not run -- the distinction this repo keeps paying for when it is
    missing. A row whose capsule is not on this axis at all is not judged and not returned.
    """
    resolved = _gate_phase_module.configured_phase(GATE) if phase is None else phase
    by_name = {str(capsule.get("name")): capsule for capsule in capsules}
    getter = macs_of if macs_of is not None else (lambda row: (row.get("work_volume") or {}).get("known_macs"))
    judged: list[dict[str, Any]] = []
    for result in results:
        name = str(result.get("capsule") or result.get("name") or "")
        capsule = by_name.get(name)
        if capsule is None:
            continue
        outcome = assess(
            capsule,
            tiers=result.get("tiers"),
            macs=getter(result),
            array_rows=array_rows,
            array_cols=array_cols,
            phase=resolved,
        )
        if not outcome.get("timing_tier"):
            # Not a row on this axis. Recording a verdict here would pad it with a field that does not
            # apply, which a structural gate on result shape refuses -- correctly.
            continue
        result[PLANE] = outcome
        judged.append({"capsule": name, **outcome})
        try:
            require_within(outcome)
        except CostCeilingExceeded as decided:
            result["status"] = "fail"
            result["failure"] = {
                "plane": PLANE,
                "category": CATEGORY,
                "detail": str(decided),
                "measured_cycles": outcome.get("measured_cycles"),
                "ceiling_cycles": outcome.get("ceiling_cycles"),
                "floor_cycles": outcome.get("floor_cycles"),
            }
    return judged


def compare(current: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, Any]:
    """Cycle change between two cost verdicts ON THE SAME DESIGN, or a refusal.

    Raises :class:`CrossDesignComparison` rather than returning a ratio nobody can read.
    """
    require_same_design(current.get("design") or {}, previous.get("design") or {})
    now, before = current.get("measured_cycles"), previous.get("measured_cycles")
    if (
        not isinstance(now, int)
        or not isinstance(before, int)
        or isinstance(now, bool)
        or isinstance(before, bool)
        or before <= 0
    ):
        return {
            "status": STATUS_INCOMPLETE,
            "reason": "one side reports no cycle count, so there is no change to state",
        }
    return {
        "status": "measured",
        "cycles": now,
        "previous_cycles": before,
        "ratio": round(now / before, 6),
        "design": dict(current.get("design") or {}),
    }
