"""The wire contract for the per-tier timing block an oracle MAY carry, and the rules it must keep.

An oracle that grades a program for correctness usually also *sees* where the time went. The tier
record has carried a slot for that since the harvest layer was built
(:data:`~merlin.perf.harvest.TIMING_OBSERVATIONS_KEY`) and, across the whole corpus on disk, nothing
ever filled it. This module is the contract that a producer fills it against and that a consumer
validates before believing a single number in it.

WHY A VALIDATOR AND NOT JUST A SCHEMA
-------------------------------------
Every rule here exists because its absence produces a number that is wrong in the flattering
direction, silently:

* **Absent is not zero.** A producer with no timing capability emits *nothing* -- not the key, not a
  list of zeros. :func:`validate_block` returns ``None`` for that case, and ``None`` is not an empty
  block. An entry whose ``value`` is null is dropped with a refusal, never recorded as 0.
* **``unmeasured_units`` is required and may not be defaulted.** A block that does not say which
  units it failed to read is claiming completeness it has not earned, so it is refused whole. An
  empty list is a *claim*, and it is only believable when the producer wrote it.
* **A per-unit count from a running program is contended, forever.** Its quantity must be spelled
  ``busy_cycles.<unit>.in_program``. The namespace is the enforcement: a bare ``busy_cycles.<unit>``
  could be paired with an isolation-probe constant of the same name, and the two are not the same
  measurement. An unnamespaced per-unit count is refused.
* **Overlap is a joint observation, never a partition.** Buckets that partition a timeline report
  zero overlap by construction, whether or not the hardware overlaps
  (:func:`merlin.perf.headroom.composition_operator` refuses them for exactly this reason). So a
  block carrying an ``overlap_cycles.*`` entry must assert ``partitioned: false`` explicitly. If it
  is ever reduced to marginal counts, the assertion fails loudly instead of the overlap quietly
  reading zero.
* **Alias accounting travels with every run.** A wrapping memory window turns two addresses into one
  byte. Zero collisions is a property of the span a particular program touched, not of the harness,
  so it has to be re-established per run and carried next to the numbers it licenses.

Nothing here names a target, a unit, an opcode or a geometry: the unit NAMES and their kinds come
from the producer, which is the only party that knows them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ALIAS_COLLISIONS_KEY", "BUSY_PREFIX", "CONCURRENCY_UNRECORDED", "IDLE_QUANTITY",
    "IN_PROGRAM_SUFFIX", "ObservationBlock", "OVERLAP_ACROSS_KINDS", "OVERLAP_OBSERVED",
    "OVERLAP_PREFIX", "PARTITIONED_KEY", "SAMPLED_QUANTITY", "TIMING_CAPABILITY_KEY",
    "TIMING_OBSERVATIONS_KEY", "UNMEASURED_UNITS_KEY", "block_from_tier_record", "concurrency_of",
    "validate_block",
]

#: The tier-record key the block rides on. Shared with :mod:`merlin.perf.harvest`, which reads it.
TIMING_OBSERVATIONS_KEY = "timing_observations"
#: Which units the producer did NOT read. Required; may not be defaulted to an empty list.
UNMEASURED_UNITS_KEY = "unmeasured_units"
#: Whether the buckets partition the timeline. Required whenever an overlap entry is present.
PARTITIONED_KEY = "partitioned"
#: Accesses that landed on a byte owned by another page of a wrapping window. Required on every run.
ALIAS_COLLISIONS_KEY = "alias_collisions"
#: On a TIER RECORD the observations and the instrument's self-description are stored apart -- the
#: list stays a list of measurements and the capability record sits beside it under this key. A
#: reader recomposes them with :func:`block_from_tier_record`.
TIMING_CAPABILITY_KEY = "timing_capability"

#: ``busy_cycles.<unit>.in_program`` -- the ONLY legal spelling for a per-unit count taken from a
#: running program. The suffix is what keeps it from being paired with an isolation-probe constant.
BUSY_PREFIX = "busy_cycles."
IN_PROGRAM_SUFFIX = ".in_program"
#: Cycles in which no unit was busy. First-class: a schema that reports only occupancy cannot see it,
#: and on the measured corpus it is the largest single quantity there is.
IDLE_QUANTITY = "idle_cycles.no_unit_busy"
#: Joint occupancy. Not a partition, not derivable from the per-unit counts.
OVERLAP_PREFIX = "overlap_cycles."
OVERLAP_OBSERVED = "overlap_cycles.observed"
#: Overlap between DISTINCT declared kinds -- the reading that matches a by-kind resource grouping.
OVERLAP_ACROSS_KINDS = "overlap_cycles.across_kinds"
#: How many cycles the instrument actually sampled. Reported next to the run's own cycle count
#: because the two need not be equal, and the buckets reconcile against this one.
SAMPLED_QUANTITY = "sampled_cycles.dbg_tap"

#: What a run that predates the concurrency stamp gets called. It is NOT retro-labelled on disk: the
#: concurrency those runs were taken at is simply not recoverable, and saying so is the whole point.
CONCURRENCY_UNRECORDED = (
    "concurrency unrecorded: this run predates the concurrency stamp. Its CYCLE counts are still "
    "comparable (cycles are concurrency-invariant); its WALL times are not comparable with any "
    "other run's, and the concurrency it ran at cannot be recovered")


def concurrency_of(record: Any) -> "dict[str, Any] | str":
    """The concurrency a tier record was measured at, or :data:`CONCURRENCY_UNRECORDED`.

    Read at consumption time so no artifact on disk has to be rewritten -- and so a historical run is
    marked rather than given a plausible-looking stamp it never had.
    """
    if isinstance(record, Mapping):
        stamp = record.get("concurrency")
        if isinstance(stamp, Mapping) and stamp:
            return dict(stamp)
    return CONCURRENCY_UNRECORDED


@dataclass(frozen=True)
class ObservationBlock:
    """A validated timing block: what may be believed, and what was refused and why."""

    #: Entries that passed every rule, in producer order.
    observations: tuple[dict[str, Any], ...]
    #: The units the producer states it did not read. NEVER defaulted -- see the module docstring.
    unmeasured_units: tuple[str, ...]
    #: The producer's explicit assertion. Only ``False`` licenses an overlap reading.
    partitioned: bool
    #: Accesses that collided inside a wrapping memory window. ``None`` = the producer did not say.
    alias_collisions: int | None
    #: Everything dropped, with the reason. Never silent.
    refusals: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return bool(self.observations)

    def busy_by_unit(self) -> dict[str, int]:
        """``{unit: in-program busy cycles}`` -- contended counts, and only ever contended."""
        out: dict[str, int] = {}
        for e in self.observations:
            q = str(e.get("quantity") or "")
            if q.startswith(BUSY_PREFIX) and q.endswith(IN_PROGRAM_SUFFIX):
                out[q[len(BUSY_PREFIX):-len(IN_PROGRAM_SUFFIX)]] = int(e["value"])
        return out

    def kinds(self) -> dict[str, str]:
        """``{unit: kind}`` as the PRODUCER declared it. A unit whose producer declared no kind is
        absent here rather than defaulted: a role read out of a spelling is how a register load once
        became "DMA"."""
        out: dict[str, str] = {}
        for e in self.observations:
            q = str(e.get("quantity") or "")
            kind = str(e.get("kind") or "")
            if kind and q.startswith(BUSY_PREFIX) and q.endswith(IN_PROGRAM_SUFFIX):
                out[q[len(BUSY_PREFIX):-len(IN_PROGRAM_SUFFIX)]] = kind
        return out

    def quantity(self, name: str) -> int | None:
        """One named quantity, or ``None`` for "the instrument did not report it" -- never 0."""
        for e in self.observations:
            if str(e.get("quantity") or "") == name:
                return int(e["value"])
        return None

    def overlap_cycles(self, *, across_kinds: bool = True) -> int | None:
        """The joint-occupancy reading, or ``None`` when the block cannot license one.

        ``across_kinds`` selects the reading that matches a by-KIND resource grouping (two engines of
        one kind being busy together is not compute/movement overlap, and counting it as such is the
        error this vector exists to avoid). Falls back to the raw joint count only when the producer
        emitted no by-kind reading.
        """
        if self.partitioned:
            return None
        if across_kinds:
            v = self.quantity(OVERLAP_ACROSS_KINDS)
            if v is not None:
                return v
        return self.quantity(OVERLAP_OBSERVED)

    def to_dict(self) -> dict[str, Any]:
        return {UNMEASURED_UNITS_KEY: list(self.unmeasured_units),
                PARTITIONED_KEY: self.partitioned,
                ALIAS_COLLISIONS_KEY: self.alias_collisions,
                "n_observations": len(self.observations),
                "refusals": list(self.refusals)}


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def validate_block(raw: Any) -> "ObservationBlock | None":
    """Validate a producer's timing block. ``None`` means the producer carries no timing capability.

    ``None`` and an empty :class:`ObservationBlock` are different answers on purpose: the first is
    "this oracle does not report timing", the second is "it reports timing and none of it survived
    validation". Collapsing them is how a missing instrument becomes a measurement of zero.
    """
    if not isinstance(raw, Mapping):
        return None
    entries = raw.get(TIMING_OBSERVATIONS_KEY)
    if entries is None:
        return None                     # no capability: emit nothing, not a block of zeros
    refusals: list[str] = []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ObservationBlock((), (), True, None,
                                (f"{TIMING_OBSERVATIONS_KEY!r} is not a list of entries, so nothing "
                                 "in it can be placed",))

    unmeasured = raw.get(UNMEASURED_UNITS_KEY)
    if not isinstance(unmeasured, Sequence) or isinstance(unmeasured, (str, bytes)):
        return ObservationBlock(
            (), (), True, None,
            (f"the producer emitted {TIMING_OBSERVATIONS_KEY!r} without {UNMEASURED_UNITS_KEY!r}. "
             "That field is REQUIRED and may not be defaulted to an empty list: a block that does "
             "not say which units it failed to read is claiming a completeness it has not earned, "
             "and an unread signal is UNMEASURED, never zero. The whole block is refused",))

    partitioned = raw.get(PARTITIONED_KEY)
    if not isinstance(partitioned, bool):
        refusals.append(
            f"the producer did not state {PARTITIONED_KEY!r}, so no overlap reading is licensed "
            "from this block; a bucket set that partitions the timeline reports zero overlap by "
            "construction and cannot be told apart from one that genuinely observed none. Treated "
            "as partitioned (fail closed)")
        partitioned = True

    alias = raw.get(ALIAS_COLLISIONS_KEY)
    alias_n = None if _numeric(alias) is None else int(alias)
    if alias_n is None:
        refusals.append(
            f"the producer did not carry {ALIAS_COLLISIONS_KEY!r}. A wrapping memory window maps two "
            "addresses onto one byte, so zero collisions is a property of the span this program "
            "touched and has to be re-established per run. Recorded as UNKNOWN, not as zero")

    kept: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            refusals.append(f"entry {i} is not a mapping")
            continue
        quantity = str(entry.get("quantity") or "")
        if not quantity:
            refusals.append(f"entry {i} names no quantity")
            continue
        value = _numeric(entry.get("value"))
        if value is None:
            refusals.append(f"{quantity}: the instrument did not report a value, so it is dropped "
                            "rather than recorded as 0 ('not reported' is not 'cost nothing')")
            continue
        if value < 0:
            refusals.append(f"{quantity}: negative value {value!r}")
            continue
        if quantity.startswith(BUSY_PREFIX) and not quantity.endswith(IN_PROGRAM_SUFFIX):
            refusals.append(
                f"{quantity}: a per-unit busy count must be namespaced "
                f"'{BUSY_PREFIX}<unit>{IN_PROGRAM_SUFFIX}'. A count taken from a running program is "
                "CONTENDED, and an unnamespaced one can be paired with an isolation-probe constant "
                "of the same name -- which is a different measurement")
            continue
        if quantity.startswith(OVERLAP_PREFIX) and partitioned:
            refusals.append(
                f"{quantity}: the producer asserts {PARTITIONED_KEY}=true, so its buckets charge "
                "every cycle to exactly one owner and report zero overlap whether or not the "
                "hardware overlaps. An overlap reading from a partitioned source is not evidence")
            continue
        kept.append(dict(entry))
    return ObservationBlock(tuple(kept), tuple(str(u) for u in unmeasured), bool(partitioned),
                            alias_n, tuple(refusals))


def block_from_tier_record(record: Any) -> "ObservationBlock | None":
    """Validate the block a TIER RECORD carries, recomposing it from its two halves.

    A tier record keeps the observation list and the instrument's self-description
    (:data:`TIMING_CAPABILITY_KEY`: what went unread, whether the buckets partition, the alias
    count) in separate fields, so the list stays a list of measurements. Validation needs both
    halves, and a reader that forgets to rejoin them sees a block with no capability statement and
    correctly -- but uselessly -- refuses the whole thing.
    """
    if not isinstance(record, Mapping):
        return None
    cap = record.get(TIMING_CAPABILITY_KEY)
    if not isinstance(cap, Mapping):
        return validate_block(record)
    merged = dict(record)
    for key in (UNMEASURED_UNITS_KEY, PARTITIONED_KEY, ALIAS_COLLISIONS_KEY):
        if key in cap:
            merged[key] = cap[key]
    return validate_block(merged)
