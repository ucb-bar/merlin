"""The MACHINE a schedule is written against: units, queues, latency, banked memories, hazards.

WHY THIS EXISTS. The kernel IR in :mod:`merlin.sched.ir` is a loop nest over instruction calls — one
instruction stream, one agent, no time. That is a faithful model of exactly one machine shape, and
reading three real accelerators shows it is not the general one:

* a **statically scheduled, non-interlocked** core, whose expert corpus is 4,590 explicit ``DELAY``
  instructions against 250 total branches and jumps — roughly 20% timing instructions and 1% control
  flow — and where the same logical matmul costs 95 cycles on one matrix unit and 34 on the other;
* a **SIMT** core with an embedded matrix mesh reached over a memory-mapped command port, whose shared
  scratchpad is banked and arbitrated lowest-index-first, so that *any* bank shared between the two
  engines starves the mesh — bank assignment is a scheduling decision with a stall consequence;
* a **hardware-interlocked**, command-driven array, where the compiler's lever is the opposite one:
  reorder to overlap, and never remove a separation, because there are none to remove.

None of that is expressible as "a loop nest over instructions", and none of it is expressible in the
existing user-schedulable languages either: the closest one's instruction node carries a C code string
and nothing else — no unit, no cost, no cycle count — and its object language has no fence, no barrier,
no asynchronous completion and no statement that can occupy time without doing work.

So this module adds the smallest thing that closes the gap, and nothing else. It is NOT a scheduling
language; the scheduling language is the primitive set in :mod:`merlin.sched.primitives`, and the
transformation-scripting layer is the ``transform`` dialect. This is the *machine* those two are
written against.

WHAT IS AND IS NOT DECLARED HERE. Everything in a :class:`Machine` is DERIVED — from a target's
compute-unit contract, its RTL facts, its address space and its declared issue-scheduling data. A
target contributes data; it never contributes a machine. Nothing in this module names a target, and the
gate that enforces that (``build_tools/scripts/check_no_target_name.py``) covers this package.

THE THREE-STATE RULE, WHICH IS LOAD-BEARING. A quantity we have not derived is ``None`` **and** carries
an :class:`Unknown` saying why. It is never 0 and never a plausible default. This is not style: the
repo has a measured instance of the cost. A unit with no top-level busy port read as permanently idle,
which inflated one kernel's idle fraction to 89.9% when the true figure was 39.2%, and moved a headline
from 76.7% to 46.2%. An unmeasured unit is UNKNOWN, never idle; an underived latency is UNKNOWN, never
free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping

__all__ = [
    "COMPLETION_KINDS",
    "DISCHARGES_DEPENDENCE",
    "SYNC_ORDERS",
    "Sync",
    "HAZARD_RESOLUTIONS",
    "MOVER_KINDS",
    "ROLE_UNKNOWN",
    "UNIT_KINDS",
    "Hierarchy",
    "Latency",
    "Level",
    "MachError",
    "Machine",
    "Memory",
    "Unit",
    "Unknown",
]


class MachError(ValueError):
    """A machine description that cannot be true of any hardware."""


#: A datapath that MOVES bytes without computing on them. ``compute_units.KINDS`` deliberately does not
#: contain this — a DMA engine is not a compute unit and declaring it as one would corrupt every
#: capability query that reads that vocabulary. But it *is* a schedulable resource: it has occupancy, it
#: has a completion event, and on one of the three targets it is eight independent channels whose
#: overlap is the whole prologue. So ``mach`` adds the kind rather than widening the other vocabulary.
MOVER_KINDS: frozenset[str] = frozenset({"dma"})

#: A unit the facts EVIDENCE but whose role they do not decide.
#:
#: This is the three-state rule applied to a unit's kind, and it is load-bearing rather than tidy. The
#: units a schedule most needs on a command-driven array are its decoupled load / execute / store
#: controllers — they are the entire overlap lever there — and the RTL derivation that finds them
#: refuses to name their role, because deciding that from a module's spelling is exactly the assumption
#: the derivation exists to avoid. Without this token the derivation has two choices, and both are
#: worse than saying so: drop the controllers, which silently deletes the lever, or label them, which
#: asserts a role nothing derived.
#:
#: A unit carrying it obliges the machine to carry a matching ``Unknown("unit.kind", where=<name>)``,
#: and :meth:`Machine.units_of` refuses to be asked for it — "give me the units whose kind we could not
#: determine" is never the question a scheduling decision should be answering.
ROLE_UNKNOWN: str = "unknown"

#: The kinds a unit may COMPUTE in. Declared here rather than imported from
#: ``targetgen.compute_units.KINDS``, and held equal to it by a test instead.
#:
#: The import was there so the two could not drift, which it achieved at a cost that only became
#: visible later: it was this module's single reach outside ``merlin.sched``, and it is what kept the
#: machine model out of a minted package. Such a package may not import this tree at all -- the
#: integrity scan rejects it outright -- so a model that cannot travel cannot be carried alongside the
#: vocabulary that consumes it.
#:
#: A checked agreement is not a weaker guarantee here, only a louder one. A kind added upstream used to
#: arrive silently and widen what a ``Machine`` accepts; now it fails a test that says to declare it
#: here too. Widening what this model admits is a decision about the model, and it should read like one.
COMPUTE_KINDS: frozenset[str] = frozenset({"systolic", "simt", "vector", "scalar", "spatial"})

#: Every kind a schedulable unit may have.
UNIT_KINDS: frozenset[str] = COMPUTE_KINDS | MOVER_KINDS | {ROLE_UNKNOWN}

#: How the hardware resolves a hazard between a producer and a consumer.
#:
#: ``interlocked`` — the hardware tracks the dependency and stalls. A reordering cannot change the
#: answer, so a bit-exactness check can never refute a schedule, and the falsifier must be occupancy.
#: ``explicit``    — the compiler must separate the two, by cycles or by a wait. A reordering CAN change
#: the answer, so bit-exactness is a sharp falsifier and the dominant lever is removing separation that
#: is more conservative than the hardware requires.
#:
#: This distinction is not cosmetic: it inverts the performance lever and it decides which gate is
#: capable of failing. It is the one field of this model that must never be guessed.
HAZARD_RESOLUTIONS: tuple[str, ...] = ("interlocked", "explicit")

#: How a consumer learns that a producer has finished.
#:
#: ``immediate`` — the result is visible to the next instruction; nothing to wait on.
#: ``counted``   — discharged by elapsed cycles. The compiler must emit the wait itself (a delay
#:                 instruction), and the count comes from :class:`Latency`.
#: ``polled``    — discharged by a wait instruction or a spin on a status word. The duration is NOT
#:                 statically known; a counted wait in its place is a bug, not a conservatism. Bulk
#:                 movement is the canonical case: its latency is data-dependent.
COMPLETION_KINDS: tuple[str, ...] = ("immediate", "counted", "polled")

#: What a synchronisation instruction actually ORDERS. Not one concept, because one target distinguishes
#: five primitives and they are not interchangeable:
#:
#: ``completion``  the named work has FINISHED. This is the only kind that discharges a data dependence.
#: ``issue``       the producer was ACCEPTED, and nothing more. Backpressure. A target ships a primitive
#:                 that reads like a fence and means exactly this; treating it as completion is how a
#:                 consumer runs against a result that does not exist yet, at full speed, silently.
#: ``visibility``  writes in one memory scope are visible to readers in it. Says nothing about whether a
#:                 UNIT's work has finished, so it does not discharge a unit dependence either.
#: ``arrival``     every party of a named n-party barrier has arrived. Orders the parties against each
#:                 other, not any one party's outstanding work.
#:
#: The distinction is load-bearing rather than descriptive: only ``completion`` may discharge a data
#: dependence, and that is a checkable property. With one ``fence`` concept it is not even statable.
SYNC_ORDERS: tuple[str, ...] = ("completion", "issue", "visibility", "arrival")

#: The orders that discharge a dependence on a producer's RESULT. Deliberately a set of ONE: widening it
#: is the change that would make the check above vacuous, so it is written where that is obvious.
DISCHARGES_DEPENDENCE: frozenset[str] = frozenset({"completion"})


@dataclass(frozen=True)
class Unknown:
    """A quantity this machine does not have, with the reason it does not have it.

    Deliberately the same shape as ``targetgen.address_space.Unknown``, generalised from "which store"
    to "which element of the machine", because the discipline is the same one and a second vocabulary
    for it would be a second thing to keep in agreement. :func:`merlin.sched.mach.derive` converts
    address-space unknowns into these.
    """

    quantity: str
    reason: str
    where: str | None = None


@dataclass(frozen=True)
class Unit:
    """One schedulable datapath: the stream work is ISSUED through, and the resource it EXECUTES on.

    These are two different things and conflating them gets the answer wrong on every one of the three
    machines. A real non-interlocked core states it plainly: long-latency tensor operations *"execute
    concurrently with each other and with scalar instructions, but only one new instruction issues per
    cycle"*. Its two matrix units share one frontend and still overlap — that overlap is the whole
    performance lever. The same holds for a command-driven array whose decoupled load / execute / store
    controllers sit behind one in-order command port.

    So: ``queue`` serialises ISSUE and fixes program order; ``executes_on`` decides what may be in
    flight at once. Asking "can these overlap?" is a question about the second.
    """

    name: str
    kind: str
    #: Name of the ordered stream work is issued through. Units sharing a queue issue serially and in
    #: program order; it does NOT stop them running at the same time. A target with one instruction
    #: stream gives every unit the same queue, which is the common case, not a special one.
    queue: str
    #: How SOFTWARE drives it — a ``targetgen.families.ENDPOINT_KINDS`` token where one is derivable.
    #: ``None`` with an :class:`Unknown` where it is not. One real machine drives its mesh by storing
    #: command words to a control aperture rather than by issuing an instruction, and a model that
    #: assumes "instruction" cannot describe it.
    exposure: str | None = None
    #: Datapath width in lanes, where that is a derived fact.
    lanes: int | None = None
    #: How many operations of this unit may be in flight at once. ``1`` for a channel that refuses a
    #: second transfer; ``None`` when underived.
    in_flight: int | None = None
    #: The physical resource this unit's work occupies. Two units naming the same resource cannot run
    #: at the same time; two units naming different ones can. Defaults to the unit's own name, which is
    #: the honest default — a separately named datapath is a separate datapath until a target says
    #: otherwise. Set it only to record a resource that two declared units genuinely share.
    executes_on: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise MachError("Unit.name must be non-empty")
        if self.kind not in UNIT_KINDS:
            raise MachError(f"Unit {self.name!r}: kind {self.kind!r} not in {sorted(UNIT_KINDS)}")
        if not self.queue:
            raise MachError(f"Unit {self.name!r}: queue must be non-empty")
        for attr in ("lanes", "in_flight"):
            value = getattr(self, attr)
            if value is not None and value <= 0:
                raise MachError(f"Unit {self.name!r}: {attr} must be positive, got {value}")


@dataclass(frozen=True)
class Latency:
    """What one instruction costs on one unit.

    Keyed by the PAIR, not by the instruction. On a real machine the same logical matmul is a 95-cycle
    operation on a systolic array and a 34-cycle operation on an inner-product tree in the same chip; a
    single latency attribute on the instruction cannot say that, and a cost model built on one would
    misplace every schedule that chooses between them.
    """

    instr: str
    unit: str
    #: Cycles the issue stage is occupied. Distinct from ``result`` because a unit can accept the next
    #: operation long before the previous one's result is readable.
    issue: int | None
    #: Cycles until the result is visible to a consumer.
    result: int | None
    completion: str = "immediate"
    #: True when ``result`` is a floor measured with every other unit idle, so a schedule that
    #: deliberately overlaps traffic must treat it as a lower bound rather than a value. Real latency
    #: tables carry this caveat in prose; carrying it as a field is how a verifier can act on it.
    contended: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        if self.completion not in COMPLETION_KINDS:
            raise MachError(
                f"Latency {self.instr!r}@{self.unit!r}: completion {self.completion!r} not in {list(COMPLETION_KINDS)}"
            )
        if self.completion == "counted" and self.result is None:
            raise MachError(
                f"Latency {self.instr!r}@{self.unit!r}: a counted completion needs a result latency; "
                "a wait that the compiler must emit cannot be emitted from an unknown count"
            )
        for attr in ("issue", "result"):
            value = getattr(self, attr)
            if value is not None and value < 0:
                raise MachError(f"Latency {self.instr!r}@{self.unit!r}: {attr} must not be negative")


@dataclass(frozen=True)
class Memory:
    """An on-chip store, with the structure that makes placement a scheduling decision.

    ``banks`` and ``shared_by`` are the fields the loop-nest model lacked. A store modelled as a flat
    byte range cannot express that two engines placed in one bank serialise against each other, and
    cannot express that an operand tile and an output tile were allocated on top of one another — a
    real shipped kernel produced 0 of 16,384 correct elements that way, silently, because its
    destination address was a constant chosen for a smaller tile.
    """

    name: str
    rows: int | None
    row_bytes: int | None
    banks: int | None
    read_ports: int | None = None
    write_ports: int | None = None
    #: How the store resolves simultaneous access, when that is a derived fact. The value is a token
    #: from the target's data, not an enum here, because an arbitration policy nobody has modelled must
    #: be reportable rather than coerced into the closest one we happen to know.
    arbiter: str | None = None
    #: Names of the units that contend for this store. A placement transformation reads this to decide
    #: whether two agents can share a bank.
    shared_by: tuple[str, ...] = ()
    #: True when this store accumulates rather than merely holding operands.
    accumulates: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise MachError("Memory.name must be non-empty")
        for attr in ("rows", "row_bytes", "banks", "read_ports", "write_ports"):
            value = getattr(self, attr)
            if value is not None and value <= 0:
                raise MachError(f"Memory {self.name!r}: {attr} must be positive, got {value}")

    @property
    def rows_per_bank(self) -> int | None:
        if self.rows is None or self.banks is None:
            return None
        return self.rows // self.banks

    @property
    def nbytes(self) -> int | None:
        if self.rows is None or self.row_bytes is None:
            return None
        return self.rows * self.row_bytes


@dataclass(frozen=True)
class Level:
    """One level of the parallel iteration hierarchy."""

    name: str
    extent: int | None

    def __post_init__(self) -> None:
        if not self.name:
            raise MachError("Level.name must be non-empty")
        if self.extent is not None and self.extent <= 0:
            raise MachError(f"Level {self.name!r}: extent must be positive, got {self.extent}")


@dataclass(frozen=True)
class Hierarchy:
    """The parallel iteration hierarchy, outermost first.

    A machine with one instruction stream has a hierarchy of one level of extent 1. That is the
    DEGENERATE case and it is written the same way as every other, on purpose: a design that treats
    "one stream" as the absence of a hierarchy bakes one stream into the abstraction, and then an
    obligation to map work across threads can never be discharged by anything.
    """

    levels: tuple[Level, ...] = ()

    @property
    def degenerate(self) -> bool:
        return all(level.extent == 1 for level in self.levels)

    @property
    def width(self) -> int | None:
        total = 1
        for level in self.levels:
            if level.extent is None:
                return None
            total *= level.extent
        return total

    def level(self, name: str) -> Level:
        for level in self.levels:
            if level.name == name:
                return level
        raise MachError(f"no hierarchy level named {name!r}; have {[l.name for l in self.levels]}")


@dataclass(frozen=True)
class Sync:
    """One of the target's synchronisation instructions, and what it orders.

    A target declares these; nothing here invents one. The reason they are declared per instruction
    rather than collapsed into a single fence is the measured one: an IR with one fence concept cannot
    state that a particular primitive signals issue backpressure and NOT completion, and therefore cannot
    check a schedule that relies on it.
    """

    instr: str
    #: One of :data:`SYNC_ORDERS`.
    orders: str
    #: What it covers -- a unit name, a memory name, or a barrier id. ``None`` means "everything the
    #: target has", which is what a full drain means and is NOT the same as "unspecified".
    scope: str | None = None
    #: For a PARTIAL drain: the outstanding depth it drains to. ``None`` on a primitive that is not one.
    depth: int | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.instr:
            raise MachError("Sync.instr must be non-empty")
        if self.orders not in SYNC_ORDERS:
            raise MachError(f"Sync {self.instr!r}: orders {self.orders!r} not in {list(SYNC_ORDERS)}")
        if self.depth is not None and self.depth < 0:
            raise MachError(f"Sync {self.instr!r}: depth must be non-negative, got {self.depth}")

    @property
    def discharges_dependence(self) -> bool:
        """Whether waiting on this establishes that a producer's result exists."""
        return self.orders in DISCHARGES_DEPENDENCE


@dataclass(frozen=True)
class Machine:
    """Everything a schedule must know about a target that is not an instruction's semantics."""

    target: str
    #: One of :data:`HAZARD_RESOLUTIONS`, or ``None`` WITH an :class:`Unknown`. Never guessed: it
    #: inverts the performance lever and it decides which gate is able to fail at all.
    hazard_resolution: str | None
    units: tuple[Unit, ...] = ()
    memories: tuple[Memory, ...] = ()
    hierarchy: Hierarchy = field(default_factory=Hierarchy)
    latencies: tuple[Latency, ...] = ()
    #: The target's synchronisation instructions and what each orders. Empty on a machine that declares
    #: none -- which is a statement about the DECLARATION, not about the hardware, so a consumer that
    #: needs one asks and gets an honest absence rather than a default fence.
    syncs: tuple[Sync, ...] = ()
    #: The instruction that occupies time without doing work, where the target has one. Required when
    #: any latency completes ``counted``: that completion says the compiler emits the wait, and this
    #: names what it emits.
    delay_instruction: str | None = None
    #: The square edge of the compute array, where the target has one and the facts resolve it — the
    #: granularity a block-scheduling pass addresses operand rows in.
    #:
    #: Typed rather than left to ``provenance`` because it is the most load-bearing number a projection
    #: of this machine carries: it decides every address a block schedule emits, and an untyped string
    #: map would keep it out of :meth:`digest` and out of every validator.
    #:
    #: ``None`` is NOT required to carry an :class:`Unknown` here, unlike
    #: :attr:`hazard_resolution`, and the asymmetry is deliberate. Every machine has a hazard model, so
    #: an absent one is always ignorance. A SIMT or vector machine has no compute array at all, so an
    #: absent edge is usually a fact about the device. Telling those two apart is the DERIVATION's job
    #: (:func:`merlin.sched.mach.derive`), which emits the ``Unknown`` only when the facts were
    #: unreadable rather than empty — the same ABSENT-versus-UNKNOWN distinction
    #: ``targetgen.address_space`` keeps for its store list.
    array_edge: int | None = None
    unknowns: tuple[Unknown, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target:
            raise MachError("Machine.target must be non-empty")
        if self.hazard_resolution is not None and self.hazard_resolution not in HAZARD_RESOLUTIONS:
            raise MachError(
                f"Machine {self.target!r}: hazard_resolution {self.hazard_resolution!r} "
                f"not in {list(HAZARD_RESOLUTIONS)}"
            )
        if self.hazard_resolution is None and not self._has_unknown("hazard_resolution"):
            raise MachError(
                f"Machine {self.target!r}: hazard_resolution is None with no Unknown explaining why. "
                "An underived hazard model must be reported, never defaulted: assuming 'interlocked' "
                "makes every illegal schedule look legal, and assuming 'explicit' makes a correctness "
                "gate that cannot fire look like one that can."
            )
        if self.array_edge is not None and self.array_edge <= 0:
            raise MachError(f"Machine {self.target!r}: array_edge must be positive, got {self.array_edge}")
        names = [u.name for u in self.units]
        if len(names) != len(set(names)):
            raise MachError(f"Machine {self.target!r}: duplicate unit names in {names}")
        for u in self.units:
            if u.kind == ROLE_UNKNOWN and not self._has_unknown("unit.kind", where=u.name):
                raise MachError(
                    f"Machine {self.target!r}: unit {u.name!r} has an undetermined kind with no "
                    f"Unknown('unit.kind', where={u.name!r}) explaining why. A unit the evidence "
                    "found but did not classify must say which one it is."
                )
        mem_names = [m.name for m in self.memories]
        if len(mem_names) != len(set(mem_names)):
            raise MachError(f"Machine {self.target!r}: duplicate memory names in {mem_names}")
        known = set(names)
        for memory in self.memories:
            for unit in memory.shared_by:
                if unit not in known:
                    raise MachError(
                        f"Machine {self.target!r}: memory {memory.name!r} is shared_by {unit!r}, "
                        f"which is not a unit of this machine"
                    )
        for lat in self.latencies:
            if lat.unit not in known:
                raise MachError(
                    f"Machine {self.target!r}: latency for {lat.instr!r} names unit {lat.unit!r}, "
                    f"which is not a unit of this machine"
                )
            if lat.completion == "counted" and self.delay_instruction is None:
                raise MachError(
                    f"Machine {self.target!r}: {lat.instr!r}@{lat.unit!r} completes by elapsed cycles, "
                    "but the machine declares no delay instruction to emit them with"
                )
        seen: set[tuple[str, str]] = set()
        for lat in self.latencies:
            key = (lat.instr, lat.unit)
            if key in seen:
                raise MachError(f"Machine {self.target!r}: duplicate latency for {key}")
            seen.add(key)

    def _has_unknown(self, quantity: str, *, where: str | None = None) -> bool:
        return any(u.quantity == quantity and (where is None or u.where == where) for u in self.unknowns)

    # -- lookups ---------------------------------------------------------------------------------

    def unit(self, name: str) -> Unit:
        for u in self.units:
            if u.name == name:
                return u
        raise MachError(f"{self.target}: no unit named {name!r}; have {[u.name for u in self.units]}")

    def memory(self, name: str) -> Memory:
        for m in self.memories:
            if m.name == name:
                return m
        raise MachError(f"{self.target}: no memory named {name!r}; have {[m.name for m in self.memories]}")

    def latency(self, instr: str, unit: str) -> Latency | None:
        """The cost of ``instr`` on ``unit``, or ``None`` when it was never derived.

        Returning ``None`` rather than a default is the point: a caller that needs a number must decide
        what to do about not having one, and cannot be handed a zero that reads as "free".
        """
        for lat in self.latencies:
            if lat.instr == instr and lat.unit == unit:
                return lat
        return None

    def units_of(self, kind: str) -> tuple[Unit, ...]:
        """Every unit of one kind. Refuses :data:`ROLE_UNKNOWN`.

        "Give me the units whose role we could not determine" is never the question a scheduling
        decision should be answering: a caller that got a list back would place work on them.
        Read :attr:`unknowns` to find out which units are in that state, and why.
        """
        if kind == ROLE_UNKNOWN:
            raise MachError(
                f"{self.target}: units_of({ROLE_UNKNOWN!r}) is not a question a schedule may ask; "
                "read Machine.unknowns to see which units were evidenced but not classified"
            )
        return tuple(u for u in self.units if u.kind == kind)

    def queues(self) -> tuple[str, ...]:
        out: list[str] = []
        for u in self.units:
            if u.queue not in out:
                out.append(u.queue)
        return tuple(out)

    def resource_of(self, name: str) -> str:
        """The physical resource a unit's work occupies (its own name unless it declares a shared one)."""
        unit = self.unit(name)
        return unit.executes_on or unit.name

    def can_overlap(self, a: str, b: str) -> bool:
        """Whether two units may have work IN FLIGHT at once.

        A question about execution, not about issue. Two units behind one in-order command port still
        overlap — that is what a decoupled controller is for, and on one target it is the entire
        performance lever. Use :meth:`shares_issue` for the ordering question.
        """
        return self.resource_of(a) != self.resource_of(b)

    def shares_issue(self, a: str, b: str) -> bool:
        """Whether two units' work is issued through one stream, hence serially and in program order.

        True does not mean they cannot overlap; it means their *starts* are ordered and each start
        costs the other one its issue slot.
        """
        return self.unit(a).queue == self.unit(b).queue

    # -- identity --------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "hazard_resolution": self.hazard_resolution,
            "units": [asdict(u) for u in self.units],
            "memories": [asdict(m) for m in self.memories],
            "hierarchy": [asdict(l) for l in self.hierarchy.levels],
            "latencies": [asdict(l) for l in self.latencies],
            "syncs": [asdict(s) for s in self.syncs],
            "delay_instruction": self.delay_instruction,
            "array_edge": self.array_edge,
            "unknowns": [asdict(u) for u in self.unknowns],
            "provenance": dict(self.provenance),
        }

    def digest(self) -> str:
        """sha256 over everything that could change a schedule's legality or cost.

        Provenance is included: two machines derived from different RTL are different machines even
        when every number happens to match, and a measurement keyed on one must not be reused for the
        other.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def text(self) -> str:
        """A readable form, for receipts and for diffing two derivations of one target."""
        lines = [f"machine {self.target} hazards={self.hazard_resolution or 'UNKNOWN'}"]
        for u in self.units:
            bits = [f"kind={u.kind}", f"queue={u.queue}"]
            if u.lanes is not None:
                bits.append(f"lanes={u.lanes}")
            if u.in_flight is not None:
                bits.append(f"in_flight={u.in_flight}")
            if u.exposure:
                bits.append(f"driven_by={u.exposure}")
            lines.append(f"  unit {u.name} " + " ".join(bits))
        for m in self.memories:
            bits = [f"rows={m.rows}", f"row_bytes={m.row_bytes}", f"banks={m.banks}"]
            if m.arbiter:
                bits.append(f"arbiter={m.arbiter}")
            if m.shared_by:
                bits.append("shared_by=" + ",".join(m.shared_by))
            if m.accumulates:
                bits.append("accumulates")
            lines.append(f"  memory {m.name} " + " ".join(bits))
        if self.hierarchy.levels:
            lines.append("  hierarchy " + " / ".join(f"{l.name}={l.extent}" for l in self.hierarchy.levels))
        if self.delay_instruction:
            lines.append(f"  delay_instruction {self.delay_instruction}")
        for lat in self.latencies:
            floor = " (floor, assumes no contention)" if lat.contended else ""
            lines.append(
                f"  latency {lat.instr}@{lat.unit} issue={lat.issue} result={lat.result} "
                f"completion={lat.completion}{floor}"
            )
        for unk in self.unknowns:
            where = f" [{unk.where}]" if unk.where else ""
            lines.append(f"  UNKNOWN {unk.quantity}{where}: {unk.reason}")
        return "\n".join(lines) + "\n"
