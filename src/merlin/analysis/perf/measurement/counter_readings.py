"""Joint occupancy from the target's OWN hardware counters, when it counts overlap itself.

Realised overlap normally needs a per-cycle trace, and a per-cycle trace normally needs a waveform
build or a co-simulation model. On a target whose RTL carries performance counters it needs neither:
the hardware can already count the cycles in which each combination of engines was busy, and a
combination counter IS a joint-occupancy reading.

Measured on the interlocked target here, from its own shipped counter header — seven counters that
between them partition busy time over three engines:

    <prefix>_LD_CYCLES  <prefix>_ST_CYCLES  <prefix>_EX_CYCLES          the three singles
    <prefix>_LD_ST_CYCLES  <prefix>_LD_EX_CYCLES  <prefix>_ST_EX_CYCLES the three pairs
    <prefix>_LD_ST_EX_CYCLES                                            all three at once

⚠️ **The engine tokens are FACTORED OUT of the counter names, not typed here.** A target that spells
its engines differently, or has two of them, or five, is served by the same derivation: find the
counters whose names share a prefix and a suffix and differ only in a set of tokens, then the tokens
ARE the engines and the token-set size is the combination order. Writing this target's spellings into
this module would make it that target's counter reader, which is the overfit the repo's cardinal rule
exists to prevent.

**What this does NOT do.** It reads the header to learn what the hardware can count, and it computes η
from a set of counter VALUES. It does not run a program, emit the counter-read commands, or claim a
measurement: a caller supplies the values, and an absent value is UNKNOWN rather than zero. Wiring the
reads into a capsule's command stream is a separate, invasive step in the runner.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

# ---- hw_counters -------------------------------------------------------------------------------


def eta_from_counters(
    values: dict,
    counters: OccupancyCounters,
    *,
    hw_text: str = "",
    codes: Mapping[str, int] | None = None,
    module: str = "",
    counter_module: str = "",
    exclusivity: str = PROVED_FROM_ARTIFACT,
    measurement_cycles: int | None = None,
    source: str | None = None,
) -> dict:
    """η from a set of counter READINGS, with every refusal carrying its reason.

    ``realised`` is the total cycles in which two or more engines were busy at once. ``available`` is
    the second-largest per-engine busy total — deliberately the same quantity ``headroom`` and the
    falsifier use, so this η and theirs are one number rather than two that share a name.

    A per-engine busy total must include the cycles that engine spent overlapping, so a single counter
    is summed with every combination containing it. Reading the singles as whole-engine totals instead
    understates the busiest engine and inflates η.

    ⚠️ THAT SUM IS EXACT, NOT AN APPROXIMATION, and it is worth knowing why. Verified in the pinned RTL
    of the target measured here: the increment conditions are mutually exclusive by construction --
    each counter fires only on its own combination, with the other engines' busy bits explicitly
    negated (``ld.busy && !st.busy && !ex.busy`` for the load single, and so on up to
    ``ld.busy && st.busy && ex.busy``). So the seven counters PARTITION busy time rather than
    overlapping in their own accounting, no cycle is counted twice, and adding a single to the
    combinations containing it recovers that engine's true busy total.

    A target whose counters are NOT mutually exclusive would double-count, and this arithmetic would
    silently over-report both the per-engine totals and the realised overlap. Nothing here can detect
    that from the header alone -- the exclusivity is a property of the RTL, not of the ``#define`` list
    -- so a new target's counter block should be read once before its η is cited.
    """
    from merlin.analysis.perf.measurement.counter_events import (
        DECLARED_BY_PRODUCER,
        PROVED_FROM_ARTIFACT,
        OccupancyCounters,
        prove_occupancy_partition_from_circt,
    )

    if exclusivity not in (PROVED_FROM_ARTIFACT, DECLARED_BY_PRODUCER):
        raise ValueError(
            f"exclusivity must be {PROVED_FROM_ARTIFACT!r} or {DECLARED_BY_PRODUCER!r}, not "
            f"{exclusivity!r}; there is no third way to establish that these counters partition "
            f"busy time, and defaulting to the weaker one is how a declared property comes to be "
            f"cited as a proved one"
        )
    if exclusivity == DECLARED_BY_PRODUCER:
        # The weaker evidence, and LABELLED as such rather than silently substituted. It is
        # accepted because refusing it outright would leave a target with no elaborated artifact
        # unable to report an eta its hardware genuinely measured; it is labelled because the
        # difference decides whether the number may be cited.
        proof = {
            "status": "declared",
            "method": exclusivity,
            "source": source,
            "why": (
                "the producer declared these counters mutually exclusive and exhaustive; "
                "no elaborated artifact was supplied, so nothing verified it. Counter "
                "exclusivity is a property of the RTL, not of the header's #define list."
            ),
        }
    else:
        if not hw_text or codes is None or not module or not counter_module:
            return {
                "state": "unknown",
                "eta": None,
                "complete": counters.complete(),
                "partition_proof": {"status": "unknown", "method": PROVED_FROM_ARTIFACT},
                "why": (
                    "proving counter exclusivity from the artifact needs hw_text, codes, "
                    "module and counter_module, and at least one was not supplied; pass "
                    f"exclusivity={DECLARED_BY_PRODUCER!r} to report the producer's own "
                    "declaration instead, which is recorded as the weaker evidence it is"
                ),
            }
        proof = prove_occupancy_partition_from_circt(
            hw_text, counters, codes, module=module, counter_module=counter_module, source=source
        )
        if proof.get("status") != "proved":
            return {
                "state": "unknown",
                "eta": None,
                "complete": counters.complete(),
                "partition_proof": proof,
                "why": "counter exclusivity/exhaustiveness was not proved from CIRCT: "
                + str(proof.get("why", "unknown proof failure")),
            }
    required = set(counters.by_combination.values())
    supplied = set(values or {})
    missing = sorted(required - supplied)
    if missing:
        return {
            "state": "unknown",
            "eta": None,
            "partition_proof": proof,
            "why": f"{len(missing)} counter reading(s) absent ({missing[:4]}); a missing counter "
            f"is UNKNOWN, and treating it as zero would report overlap that was never "
            f"measured as overlap that did not happen",
        }
    extra = sorted(supplied - required)
    if extra:
        return {
            "state": "unknown",
            "eta": None,
            "partition_proof": proof,
            "why": f"unexpected counter reading(s) {extra[:4]}; the occupancy pass must contain "
            "exactly its proved partition so readings from another window cannot be mixed",
        }
    invalid = sorted(
        name
        for name in required
        if (isinstance(values[name], bool) or not isinstance(values[name], int) or values[name] < 0)
    )
    if invalid:
        return {
            "state": "unknown",
            "eta": None,
            "partition_proof": proof,
            "why": f"counter reading(s) {invalid[:4]} are not non-negative integers",
        }
    if isinstance(measurement_cycles, bool) or not isinstance(measurement_cycles, int) or measurement_cycles <= 0:
        return {
            "state": "unknown",
            "eta": None,
            "partition_proof": proof,
            "why": "a positive integer cycle measurement for the identical counter window is "
            "required to exclude impossible or wrapped occupancy readings",
        }
    partition_cycles = sum(values[name] for name in required)
    if partition_cycles > measurement_cycles:
        return {
            "state": "unknown",
            "eta": None,
            "partition_proof": proof,
            "why": f"the exclusive occupancy partition totals {partition_cycles} cycles, more "
            f"than its measured {measurement_cycles}-cycle window; the readings are "
            "mixed, corrupt, or wrapped",
        }
    busy: dict = {e: 0 for e in counters.engines}
    realised = 0
    for combo, name in counters.by_combination.items():
        v = values[name]
        for e in combo:
            busy[e] = busy.get(e, 0) + v
        if len(combo) >= 2:
            realised += v
    # HOW MUCH OVERLAP WAS AVAILABLE, for any number of engines.
    #
    # The second-largest per-engine total is the right bound for TWO engines and is wrong for three:
    # with three engines overlapping in disjoint pairs the numerator counts every pair while the
    # denominator only admits the top pair's ceiling, and η comes out above 1. Measured on the first
    # real run of this instrument -- a bit-exact A/B on the pinned RTL reported η of 1.1726 and 1.0253,
    # which is not a fraction and cannot be quoted as one.
    #
    # Two facts bound it, and the binding one is whichever is smaller. Every cycle in which two or more
    # engines are busy consumes at least TWO engine-busy-cycles, so overlap <= floor(total / 2). And in
    # every such cycle at least one engine other than the busiest is busy, so overlap <= total minus
    # the busiest engine's own total.
    #
    # For two engines this REDUCES to the second-largest exactly (checked: [100,60] -> 60, [80,80] ->
    # 80, [500,7] -> 7), so it generalises the existing convention rather than replacing it, and the
    # falsifier's denominator and this one stay the same quantity wherever they were already equal.
    ordered = sorted(busy.values(), reverse=True)
    total = sum(ordered)
    available = min(total - ordered[0], total // 2) if len(ordered) >= 2 else 0
    if available <= 0:
        return {
            "state": "unknown",
            "eta": None,
            "busy_cycles": busy,
            "realised_cycles": realised,
            "partition_proof": proof,
            "why": "the second-busiest engine has no busy cycles, so no overlap was AVAILABLE; "
            "0/0 is undefined, not 0.0",
        }
    return {
        "state": "measured",
        "eta": realised / float(available),
        "busy_cycles": busy,
        "realised_cycles": realised,
        "available_cycles": available,
        "measurement_cycles": measurement_cycles,
        "partition_cycles": partition_cycles,
        "complete": counters.complete(),
        "partition_proof": proof,
        "note": (
            "realised counts every cycle with two or more engines busy; available is the "
            "most overlap the per-engine totals admit -- min(total - busiest, total // 2) -- "
            "which equals the second-largest total when there are two engines, the case the "
            "falsifier and headroom were written for"
        ),
    }


COUNTER_MARKER = "MERLIN_HWCOUNTER"


COUNTER_SCHEMA_MARKER = "MERLIN_COUNTER_SCHEMA"


def counter_bracket_for_names(
    names: list[str] | tuple[str, ...], codes: dict, *, slots: int, padding_code: int | None = None
) -> dict:
    """C for configuring and reading an explicit derived counter set around a kernel.

    The names are supplied by another structural selector (for example
    :func:`derive_occupancy_counters` or :func:`counters_with_unit`).  This function assigns physical
    slots and refuses a partial set; it does not know any target event name or code.
    """
    ordered = list(dict.fromkeys(str(name) for name in names))
    missing = [name for name in ordered if name not in (codes or {})]
    if missing:
        raise ValueError(
            f"no event code for {missing}; a counter read at an unconfigured slot returns whatever "
            f"that slot last held, which would be reported as this run's measurement"
        )
    selected_codes = [codes[name] for name in ordered]
    if any(isinstance(code, bool) or not isinstance(code, int) or code < 0 for code in selected_codes):
        raise ValueError("counter event codes must be non-negative integers derived from the header")
    if len(set(selected_codes)) != len(selected_codes):
        raise ValueError(
            "selected counter names share an event code; assigning both to slots would label one "
            "hardware event as two different measurements"
        )
    if len(ordered) > int(slots):
        raise ValueError(
            f"{len(ordered)} counter(s) need slots but the target exposes {slots}; refusing to emit a "
            f"partial bracket, because omitted events would turn an exact measurement into an "
            f"unlabelled lower bound"
        )
    if padding_code is not None and (
        isinstance(padding_code, bool) or not isinstance(padding_code, int) or padding_code < 0
    ):
        raise ValueError("counter padding code must be a non-negative integer derived from the header")
    slot_of = {name: index for index, name in enumerate(ordered)}
    pro = ["  // merlin: counters configured from this target's own event codes.", "  counter_reset();"]
    for name in ordered:
        pro.append(f"  counter_configure({slot_of[name]}, {int(codes[name])});  // {name}")
    # Multi-pass measurements must execute the same number of pre-window counter commands.  Otherwise
    # selecting two events in one pass and seven in another perturbs instruction/cache state before
    # the timer starts, even though both brackets surround the same kernel.  A target may provide its
    # own header-derived disabled-event code to fill the remaining physical slots; absence means no
    # padding rather than an assumed zero code.
    if padding_code is not None:
        for slot in range(len(ordered), int(slots)):
            pro.append(f"  counter_configure({slot}, {padding_code});  // padding: disabled event")
    epi = [
        "  // merlin: read the counters back. Marker-prefixed so a reader attributes by NAME, not",
        "  // by position -- a simulator console interleaves writers.",
        "  counter_snapshot_take();",
    ]
    for name in ordered:
        epi.append(f'  printf("{COUNTER_MARKER} {name} %u\\n", counter_read({slot_of[name]}));')
    return {
        "prologue": "\n".join(pro) + "\n",
        "epilogue": "\n".join(epi) + "\n",
        "slot_of": slot_of,
        "configured_slots": int(slots) if padding_code is not None else len(ordered),
    }


def occupancy_partition_bracket(
    header_text: str, *, slots: int, additional=(), padding_code: int | None = None
) -> dict:
    """The bracket that makes accelerator-busy CLOSED rather than a lower bound.

    WHY THIS EXISTS AS A DERIVED SELECTOR. A whole-model run on this tree measured
    ``main_ld + main_st + main_ex = 58,659,625`` against ``1,383,906,735`` cycles and reported
    accelerator-busy as **4.24%** -- correctly labelled a LOWER bound, because the three singles are
    3 of a 7-member exhaustive partition and the 4 absent members are exactly the overlap terms. With
    the partition complete, busy is the SUM and host residue is ``rdcycle - sum``: a closed
    decomposition rather than two one-sided bounds. Nothing downstream
    (:mod:`merlin.analysis.perf.bounds.attribution`, :mod:`merlin.analysis.perf.bounds.envelope`) can price a bucket it only has a
    bound for, which is why the whole feedback stack had no data.

    The selection is DERIVED here, from the target's own counter header, and refuses a partition the
    header does not prove exhaustive. That is the point: the harness that produced the 3-counter run
    picked its eight slots by hand, so the partition silently stopped being one. ``additional`` names
    extra counters for the slots the partition leaves free; they are the caller's declared choice and
    are never allowed to displace a partition member.
    """
    from merlin.analysis.perf.measurement.counter_events import derive_occupancy_counters

    occupancy = derive_occupancy_counters(header_text)
    if not occupancy.complete():
        raise ValueError(
            "this target's counter header does not derive a complete occupancy partition "
            f"({len(occupancy.by_combination)} of the combinations over engines "
            f"{list(occupancy.engines)}); a partial set makes accelerator-busy a lower bound, and "
            "emitting it as though it were a total is the defect this selector exists to prevent"
        )
    partition = [
        occupancy.by_combination[key] for key in sorted(occupancy.by_combination, key=lambda combo: sorted(combo))
    ]
    extra = [str(name) for name in additional if str(name) not in partition]
    if len(partition) > int(slots):
        raise ValueError(
            f"the occupancy partition needs {len(partition)} slots but the target exposes {slots}; "
            "there is no subset of a partition that is still a partition"
        )
    ordered = partition + extra[: max(0, int(slots) - len(partition))]
    dropped = extra[len(ordered) - len(partition) :]
    bracket = counter_bracket_for_names(ordered, event_codes(header_text), slots=slots, padding_code=padding_code)
    bracket["partition_names"] = tuple(partition)
    bracket["additional_names"] = tuple(ordered[len(partition) :])
    # Named rather than silently truncated: a caller that asked for a byte counter and did not get one
    # would otherwise read its absence as "this run moved no bytes".
    bracket["additional_declined"] = tuple(dropped)
    bracket["engines"] = tuple(occupancy.engines)
    bracket["closes_accelerator_busy"] = True
    return bracket


def counter_bracket_c(counters: OccupancyCounters, codes: dict, *, slots: int, padding_code: int | None = None) -> dict:
    """C for configuring, then reading, this target's combination counters around a kernel.

    Returns ``{"prologue": str, "epilogue": str, "slot_of": {...}}`` — text a harness generator can
    place before and after the work, plus which slot each counter was assigned so the epilogue's
    output can be attributed. Emitting text rather than editing a harness is deliberate: nothing on
    the graded path changes until a caller chooses to place it.

    ``codes`` maps a counter NAME to the numeric event code the target's own header declares, and
    ``slots`` is how many counter slots the hardware exposes — on the target measured here the config
    register masks the index with three bits and the RTL's counter bundle carries an
    ``external_values`` array, so eight, and the seven combination counters fit exactly. Both are
    supplied rather than assumed: a target with fewer slots than counters must be told so it can
    FAIL rather than silently read whichever ones happened to fit.
    """
    from merlin.analysis.perf.measurement.counter_events import OccupancyCounters

    names = [counters.by_combination[key] for key in sorted(counters.by_combination, key=lambda combo: sorted(combo))]
    try:
        return counter_bracket_for_names(names, codes, slots=slots, padding_code=padding_code)
    except ValueError as exc:
        if len(names) > int(slots):
            raise ValueError(
                str(exc) + "; a missing combination makes realised overlap a lower "
                "bound that could be reported as the total"
            ) from exc
        raise


def parse_counter_output(console: str) -> dict:
    """``counter name -> value`` from a bracketed run's console output.

    Attributes by the NAME the line carries, so an interleaved or reordered console still reads
    correctly, and a line whose value is not an integer is skipped rather than coerced -- a truncated
    console is a missing reading, which η already refuses on, and never a zero.
    """
    out: dict = {}
    ambiguous: set[str] = set()
    for raw in (console or "").splitlines():
        line = raw.strip()
        at = line.find(COUNTER_MARKER)
        if at == -1:
            continue
        parts = line[at + len(COUNTER_MARKER) :].split()
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[1]
        if value.isdigit():
            if name in out:
                ambiguous.add(name)
            else:
                out[name] = int(value)
    for name in ambiguous:
        out.pop(name, None)
    return out


def parse_counter_schema(console: str) -> str | None:
    """Return the unique header digest embedded in the measured ELF, or UNKNOWN."""
    values: list[str] = []
    for raw in (console or "").splitlines():
        parts = raw.strip().split()
        if len(parts) != 2 or parts[0] != COUNTER_SCHEMA_MARKER:
            continue
        digest = parts[1]
        if len(digest) == 64 and all(char in "0123456789abcdefABCDEF" for char in digest):
            values.append(digest.lower())
    return values[0] if values and len(set(values)) == 1 else None


def event_codes(text: str) -> dict:
    """``NAME -> code`` for every integer ``#define`` in a counter header, for :func:`counter_bracket_c`."""
    from merlin.analysis.perf.measurement.counter_events import _defines

    return _defines(text)


def observations_from_counters(
    values: Mapping[str, int],
    counters: "OccupancyCounters",
    *,
    total_cycles: int | None = None,
    source: str | None = None,
    kind_of: Mapping[str, str] | None = None,
) -> dict:
    """A :mod:`merlin.analysis.perf.measurement.observations` timing block from one bracketed run's counter readings.

    This is the hop that was missing. The bracket emitter, the console parser, the wire contract and
    every consumer of a per-unit activity vector all existed; nothing turned a console full of
    ``<PREFIX>_<engines>_CYCLES`` readings into the block the tier record carries, so a target that
    counts overlap in hardware still reported ``missing: ['at least one activity source']`` and no
    composition operator, headroom or η could resolve from it.

    Three rules, each of which changes the numbers:

    **A per-engine busy total is the single counter PLUS every combination containing it.** An engine
    is busy during the cycles it shares with another engine, so reading the singles as whole-engine
    totals understates the busiest engine — which is η's denominator, so it inflates η. The sum is
    exact rather than approximate because the increment conditions are mutually exclusive by
    construction: every cycle is charged to exactly one subset.

    **The emitted totals therefore do NOT partition the timeline**, and the block says so. That is
    what licenses the overlap reading: ``partitioned=True`` exists to reject an overlap inferred from
    buckets that charge each cycle once and so report zero overlap whether or not the hardware
    overlaps. Here the overlap is not inferred at all — the multi-engine combinations measure it
    directly, which is the independent observation :func:`~merlin.analysis.perf.bounds.headroom.composition_operator`
    requires.

    **A counter the bracket configured but whose reading did not come back is UNMEASURED, never
    zero.** Its engines lose their busy entry and are named in ``unmeasured_units``, because a total
    computed from a partial combination set is a lower bound that would be read as the total.
    ``alias_collisions`` is deliberately omitted rather than reported as 0: it is a property of the
    address span a program touched, and this instrument does not establish it.

    ``kind_of`` maps each engine to its resource KIND and unlocks the second overlap quantity. A unit's
    kind cannot be derived from a counter name, so it is declared or it is absent -- and the distinction
    is load-bearing rather than cosmetic. ``overlap_cycles.observed`` counts any two engines busy
    together; the eta a kind-axis consumer needs counts only cycles spanning two DIFFERENT kinds,
    because two movement engines running together is not movement/compute overlap. On a machine whose
    engines are not all distinct kinds the two numbers differ, and reporting the first where the second
    is meant overstates the overlap that a compute/movement pairing achieved.
    """
    from merlin.analysis.perf.measurement.observations import (
        BUSY_PREFIX,
        IDLE_QUANTITY,
        IN_PROGRAM_SUFFIX,
        OVERLAP_ACROSS_KINDS,
        OVERLAP_OBSERVED,
        PARTITIONED_KEY,
        TIMING_OBSERVATIONS_KEY,
        UNMEASURED_UNITS_KEY,
    )

    by_combo = dict(counters.by_combination)
    missing = sorted(name for name in by_combo.values() if name not in values)
    # An engine is unreadable if ANY combination naming it is missing: its total would be a lower bound.
    unmeasured: set[str] = set()
    for combo, name in by_combo.items():
        if name in missing:
            unmeasured.update(combo)

    prov = source or "hardware combination counters"
    entries: list[dict] = []
    for engine in counters.engines:
        if engine in unmeasured:
            continue
        busy = sum(int(values[name]) for combo, name in by_combo.items() if engine in combo)
        entry = {
            "quantity": f"{BUSY_PREFIX}{engine}{IN_PROGRAM_SUFFIX}",
            "value": busy,
            "unit": "cycles",
            "source": prov,
        }
        # Carry the declared kind ON the entry. `ObservationBlock.kinds()` reads it from here, and a
        # consumer that cannot see it refuses the whole source ("the producer stated no kind for
        # unit(s) ..."): computing `overlap_cycles.across_kinds` from `kind_of` while dropping the
        # per-unit kind left the block internally inconsistent -- it classified the overlap by kind
        # and then could not say which unit had which kind.  Absent when undeclared, never defaulted.
        if kind_of and engine in kind_of:
            entry["kind"] = str(kind_of[engine])
        entries.append(entry)

    # Overlap: the cycles two or more engines were busy at once, read off the combination counters
    # themselves. Refused outright if any combination is missing -- a partial sum is a lower bound.
    if not missing:
        realised = sum(int(values[name]) for combo, name in by_combo.items() if len(combo) >= 2)
        entries.append({"quantity": OVERLAP_OBSERVED, "value": realised, "unit": "cycles", "source": prov})
        if kind_of:
            missing_kinds = sorted(e for e in counters.engines if e not in kind_of)
            if not missing_kinds:  # a partial kind map cannot classify every combination
                across = sum(
                    int(values[name]) for combo, name in by_combo.items() if len({kind_of[e] for e in combo}) >= 2
                )
                entries.append({"quantity": OVERLAP_ACROSS_KINDS, "value": across, "unit": "cycles", "source": prov})
        if total_cycles is not None:
            charged = sum(int(values[name]) for name in by_combo.values())
            idle = int(total_cycles) - charged
            if idle >= 0:  # a negative idle means the window and the counters disagree: drop it
                entries.append({"quantity": IDLE_QUANTITY, "value": idle, "unit": "cycles", "source": prov})

    return {TIMING_OBSERVATIONS_KEY: entries, UNMEASURED_UNITS_KEY: sorted(unmeasured), PARTITIONED_KEY: False}


# ---- counter_trust -----------------------------------------------------------------------------
# Whether an execution engine's accelerator counters are measurements or noise.
#
# WHY THIS EXISTS, precisely. A functional ISS can implement an accelerator's counter-read instruction
# by returning a random number, and one does: the RoCC model this tree runs increments every counter
# with `rand()` on every accelerator instruction. An optimization loop read those values for hours,
# reported them in a receipt beside genuine cycle counts, and had no signal at all -- the giveaway was
# a byte counter reading `7` after a whole-model inference, which nothing checked. The values were not
# wrong in a way that looked wrong; they looked like small measurements.
#
# So this module refuses rather than annotates. :func:`values_or_refusal` returns counter values only
# when the engine that produced them is DECLARED to wire them to hardware events, with the evidence
# cited in the engine's own sources (``src/merlin/resources/contracts/counter_trust.yaml``). An engine that is not
# declared is ``UNKNOWN`` and is refused too -- fail closed, because the whole failure mode is a
# plausible number arriving from an unexamined source.
#
# **Not a simulator ranking and not a target fact.** Nothing here names a target, an opcode or a
# counter; it is keyed on the ENGINE that executed the program, and the per-engine verdict is reviewed
# data, not a heuristic. `merlin.analysis.perf.measurement.counter_events` derives what a target can count; this says whether
# what came back means anything.

#: Counters are wired to hardware event signals; values are measurements.
REAL = "real"
#: The engine synthesises values; any number it reports is noise.
FABRICATED = "fabricated"
#: The engine has no cycle model, so a cycle counter is meaningless even if not fabricated.
UNTIMED = "untimed"
#: The engine is not declared. Refused, like a fabricated one -- an unexamined source is not a
#: trusted one, and this is the state that let the original failure through.
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Verdict:
    """One engine's verdict plus the citation that justifies it."""

    engine: str
    verdict: str
    evidence: str = ""
    evidence_source: str = ""
    verified: str = ""

    @property
    def trusted(self) -> bool:
        return self.verdict == REAL

    def refusal(self) -> str | None:
        """Why these values must not be reported, or ``None`` when they may be."""
        if self.trusted:
            return None
        if self.verdict == UNKNOWN:
            return (
                f"engine {self.engine!r} is not declared in counter_trust.yaml, so whether its "
                f"accelerator counters carry real values is UNKNOWN; refusing to report them"
            )
        detail = f" ({self.evidence_source})" if self.evidence_source else ""
        return (
            f"engine {self.engine!r} accelerator counters are {self.verdict}{detail}: "
            f"{self.evidence.strip() or 'no evidence recorded'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "evidence_source": self.evidence_source,
            "verified": self.verified,
        }


@lru_cache(maxsize=1)
def _declared() -> dict[str, Verdict]:
    """The reviewed table. Absent or malformed means every engine is UNKNOWN, never trusted."""
    import yaml

    from merlin.common.paths import contract_dir

    path = contract_dir() / "counter_trust.yaml"
    try:
        body = yaml.safe_load(path.read_text())
    except (OSError, ValueError):
        return {}
    engines = (body or {}).get("engines")
    if not isinstance(engines, Mapping):
        return {}
    out: dict[str, Verdict] = {}
    for name, spec in engines.items():
        if not isinstance(spec, Mapping):
            continue
        v = str(spec.get("verdict") or "")
        if v not in (REAL, FABRICATED, UNTIMED):
            continue  # an unrecognised verdict is not a licence to trust
        out[str(name)] = Verdict(
            str(name),
            v,
            str(spec.get("evidence") or ""),
            str(spec.get("evidence_source") or ""),
            str(spec.get("verified") or ""),
        )
    return out


def declared_engines() -> tuple[str, ...]:
    return tuple(sorted(_declared()))


def verdict_for(engine: Any) -> Verdict:
    """``engine``'s verdict, or an UNKNOWN verdict when it is not declared."""
    name = str(engine or "").strip().lower()
    hit = _declared().get(name)
    return hit if hit is not None else Verdict(name or "<unnamed>", UNKNOWN)


def is_trusted(engine: Any) -> bool:
    return verdict_for(engine).trusted


def values_or_refusal(engine: Any, values: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """``(values, None)`` when ``engine``'s counters are real; ``(None, why)`` otherwise.

    Callers should record the refusal STRING in the receipt where the numbers would have gone, so a
    reader sees why the field is absent instead of finding it silently missing -- an absent field
    reads as "not collected", which is the same shape as the original defect.
    """
    v = verdict_for(engine)
    why = v.refusal()
    return (dict(values), None) if why is None else (None, why)


def require_trusted(engine: Any) -> Verdict:
    """The verdict, raising ``ValueError`` unless ``engine``'s counters are real."""
    v = verdict_for(engine)
    why = v.refusal()
    if why is not None:
        raise ValueError(why)
    return v


__all__ = [
    "FABRICATED",
    "REAL",
    "UNKNOWN",
    "UNTIMED",
    "Verdict",
    "declared_engines",
    "is_trusted",
    "require_trusted",
    "values_or_refusal",
    "verdict_for",
]
