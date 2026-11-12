"""The machine a schedule is written against: units, queues, latency and memories.

Built from what a target already declares (its contract and RTL-derived facts) and cross-checked
against the declared on-chip geometry; never assumed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from merlin.common import facts_view

# ---- model -------------------------------------------------------------------------------------


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
#: visible later: it was this module's single reach outside ``merlin.compiler.kernel_ir``, and it is what kept the
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
    for it would be a second thing to keep in agreement. :func:`merlin.compiler.kernel_ir.machine` converts
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
    #: (:func:`merlin.compiler.kernel_ir.machine`), which emits the ``Unknown`` only when the facts were
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


# ---- derive ------------------------------------------------------------------------------------
# Build a :class:`~merlin.compiler.kernel_ir.machine.Machine` from what a target already declares and already extracts.
#
# A target contributes DATA -- its compute-unit contract, its RTL facts, its address space, its declared
# issue-scheduling rules. It never contributes a machine. This module is the one place those four are read
# together, and nothing in it names a target.
#
# WHAT IT REFUSES TO DO, AND WHY EACH REFUSAL EARNED ITS PLACE.
#
# *A store the facts describe by its PORT geometry is not a memory.* One target's only extracted store is
# a write-port byte-enable geometry -- the facts say so themselves, in the store's own provenance -- and it
# is neither of that device's two real on-chip stores, over-reporting capacity by more than an order of
# magnitude. A derivation that passed it through would hand every capacity and bank decision a plausible
# wrong number, which is worse than handing it none. So it becomes an :class:`Unknown` naming what was
# found instead.
#
# *A unit the evidence finds but does not classify keeps ``ROLE_UNKNOWN``.* The RTL derivation that finds
# a command-driven array's decoupled controllers deliberately refuses to name their role, because deciding
# that from a module's spelling is the assumption it exists to avoid. Dropping them would silently delete
# the overlap lever they are; labelling them would assert a role nothing derived.
#
# *An underived quantity is ``None`` with a reason.* Never 0, never a plausible default. The measured cost
# of the alternative is on the record: a unit with no top-level busy port read as permanently idle, which
# moved one headline from 76.7% to 46.2%.
#
# WHAT IS DECLARED RATHER THAN DERIVED. Two things, and both are declared because they are not properties
# the RTL alone settles. ``hazard_resolution`` is EVIDENCED by an ISA (a delay instruction is strong
# evidence for ``explicit``) but not decided by it -- a machine may interlock *and* expose a delay -- so it
# is declared in the target's contract and cross-checked against the ISA here, where a contradiction
# raises. Completion kind per instruction class is declared in the target's issue-scheduling data for the
# same reason: whether a wait is discharged by counting cycles or by polling is a property of the unit's
# protocol, not of any width in the RTL.

#: Provenance token the extractor stamps on a store whose geometry came from PORT widths rather than
#: from a declared storage array. Read here for the same reason ``address_space`` reads it: a port
#: declares a write granularity, not a capacity.
_PORT_GEOMETRY = "firrtl_port_geometry"

#: Facts interface blocks that describe an ENCODING or a feature set rather than a command stream, and
#: therefore name no queue. These are merlin's own facts-schema block names, not target names.
_NOT_A_COMMAND_STREAM = frozenset(
    {"funct_decode_table", "elaborated_rtl_features", "register_bundle_layouts", "dma_tlb"}
)


def _array_edge(space: Any) -> tuple[int | None, Unknown | None]:
    """The square array edge, or ``None`` -- distinguishing "no array" from "unreadable array"."""
    rows, cols = getattr(space, "array_rows", None), getattr(space, "array_cols", None)
    if rows is None:
        # ABSENT, not UNKNOWN. A SIMT or vector device has no compute array, and saying "we could not
        # read one" about a device that has none is a false gap. A lone column count is such a device's
        # LANE width, not half an array -- two dimensions are what makes an array, and one of them
        # standing alone means the facts described something else.
        return None, None
    if cols is None or rows != cols:
        return None, Unknown(
            "array_edge",
            f"the facts describe an array of {rows}x{cols}, which is not a square edge a block "
            "schedule can address operand rows in",
            getattr(space, "array_name", None),
        )
    return int(rows), None


def _memories(space: Any) -> tuple[tuple[Memory, ...], tuple[Unknown, ...]]:
    """On-chip stores, minus any whose geometry is a port's rather than a storage array's."""
    mems: list[Memory] = []
    unknowns: list[Unknown] = []
    accumulator = _accumulator_name(space)
    for store in getattr(space, "stores", ()) or ():
        sources = dict(getattr(store, "sources", {}) or {})
        if sources.get("bytes_depth") == _PORT_GEOMETRY:
            unknowns.append(
                Unknown(
                    "memory",
                    f"the only geometry the facts give for {store.name!r} is a write-port byte-enable "
                    "geometry, which the facts themselves decline to read as a datapath width; it is a "
                    "write granularity, not a capacity, so it is reported rather than used",
                    store.name,
                )
            )
            continue
        mems.append(
            Memory(
                name=store.name,
                rows=getattr(store, "total_rows", None),
                row_bytes=getattr(store, "row_bytes", None),
                banks=getattr(store, "banks", None),
                accumulates=(store.name == accumulator),
                source="; ".join(f"{k}={v}" for k, v in sorted(sources.items())) or "",
            )
        )
    for quantity in ("read_ports", "write_ports", "arbiter"):
        unknowns.append(Unknown(quantity, "no extraction reports it for any store; bank contention stays undecidable"))
    return tuple(mems), tuple(unknowns)


def _accumulator_name(space: Any) -> str | None:
    """The store the accumulate datapath fills, when one is addressable; else ``None``."""
    try:
        from merlin.development.targetgen.address_space import ADDRESSABLE, accumulator_kind
    except ImportError:  # pragma: no cover - the address space is always importable in-tree
        return None
    kind = accumulator_kind(space)
    store = getattr(kind, "store", None)
    return getattr(store, "name", None) if getattr(kind, "kind", None) == ADDRESSABLE else None


def _queue(contract: Mapping[str, Any], body: Mapping[str, Any] | None) -> tuple[str | None, str]:
    """The ordered stream work is issued through, and how it was decided.

    Derived from the target's declared ENDPOINT KIND -- how software drives the device -- rather than
    from any interface's spelling. A target with one instruction stream gives every unit the same queue,
    which is the common case and not a special one.
    """
    declared = contract.get("endpoint_kind")
    if isinstance(declared, str) and declared:
        return declared, "target_contract.endpoint_kind"
    named = [
        block.get("name")
        for block in facts_view.interfaces(body)
        if isinstance(block.get("name"), str) and block.get("name") not in _NOT_A_COMMAND_STREAM
    ]
    if len(named) == 1:
        return named[0], "the single facts interface that describes a command stream"
    return None, f"endpoint_kind undeclared and {len(named)} interfaces could be a command stream"


def _units(
    contract: Mapping[str, Any], body: Mapping[str, Any] | None, space: Any
) -> tuple[tuple[Unit, ...], tuple[Unknown, ...]]:
    queue, basis = _queue(contract, body)
    if queue is None:
        return (), (Unknown("units", f"no queue could be derived: {basis}"),)
    units: list[Unit] = []
    unknowns: list[Unknown] = []
    lanes = getattr(space, "array_cols", None)
    for declared in contract.get("compute_units") or ():
        if not isinstance(declared, Mapping) or not declared.get("name"):
            continue
        kind = declared.get("kind")
        if kind == ROLE_UNKNOWN or not isinstance(kind, str):
            unknowns.append(Unknown("unit.kind", "the contract declares the unit but not its kind", declared["name"]))
            kind = ROLE_UNKNOWN
        units.append(
            Unit(
                name=str(declared["name"]),
                kind=kind,
                queue=queue,
                exposure=declared.get("exposure") or contract.get("endpoint_kind"),
                lanes=int(lanes) if kind == "systolic" and isinstance(lanes, int) and lanes > 0 else None,
            )
        )
    unknowns.append(Unknown("unit.in_flight", "no extraction reports queue depth or outstanding-transfer capacity"))
    return tuple(units), tuple(unknowns)


def _hierarchy(body: Mapping[str, Any] | None) -> Hierarchy:
    """The parallel iteration hierarchy, degenerate where a target has one instruction stream.

    A degenerate hierarchy is written the same way as every other -- one level of extent 1 -- so that a
    consumer never has to special-case its absence, and so that "this machine has no warps" and "we
    never looked" cannot read alike.
    """
    simt = facts_view.simt(body)
    if not simt:
        return Hierarchy((Level("stream", 1),))
    levels = [
        Level(name, int(simt[key]))
        for name, key in (("core", "cores"), ("warp", "warps_per_core"), ("lane", "lanes_per_warp"))
        if isinstance(simt.get(key), int) and simt[key] > 0
    ]
    return Hierarchy(tuple(levels)) if levels else Hierarchy((Level("stream", 1),))


def _latencies(
    schedule_contract: Mapping[str, Any] | None,
) -> tuple[tuple[Latency, ...], str | None, tuple[Unknown, ...]]:
    """Declared issue gaps and result-visibility gaps, as ``(instr, unit)``-keyed costs.

    An issue gap is stated as an issue DISTANCE in instruction slots: the target's own rationale records
    that a 34-cycle resource occupancy yields a 35-cycle issue distance "because the producer and delay
    instruction each consume an issue cycle". So the occupancy is the distance minus the producer's own
    slot, and writing that conversion down here is what keeps the two numbers from being reconciled by
    deleting one of them.
    """
    if not schedule_contract:
        # Not "this machine has no latencies" -- that would price every instruction at nothing and make
        # a schedule that overlaps nothing look identical to one that overlaps everything. It is "no
        # caller supplied a schedule contract", and the census has to say so, because a reader who sees
        # latency missing from the list of gaps will read the gaps as complete.
        return (
            (),
            None,
            (
                Unknown(
                    "latency",
                    "no schedule contract was supplied, so no per-instruction issue or result cost is "
                    "derived for any unit. The only schedule contract in this tree lives under "
                    "experiments/studies/, which library code may not read, so a caller that wants costs "
                    "must pass them",
                ),
            ),
        )
    delay = (schedule_contract.get("delay_instruction") or {}).get("mnemonic")
    unknowns: list[Unknown] = []
    costs: dict[tuple[str, str], Latency] = {}
    for rule in schedule_contract.get("minimum_issue_gap") or ():
        _rule_into(rule, costs, unknowns, field="issue", delay=delay)
    for rule in schedule_contract.get("register_dependency_gap") or ():
        _rule_into(rule, costs, unknowns, field="result", delay=delay)
    return tuple(costs.values()), delay, tuple(unknowns)


def _rule_into(
    rule: Any, costs: dict[tuple[str, str], Latency], unknowns: list[Unknown], *, field: str, delay: str | None
) -> None:
    if not isinstance(rule, Mapping):
        return
    name = str(rule.get("name") or "")
    unit = rule.get("unit")
    if not isinstance(unit, str) or not unit:
        unknowns.append(
            Unknown(
                "latency.unit",
                "the rule declares no unit, and recovering one from its name or its mnemonics' spelling "
                "would be a guess in the middle of the one quantity that must not be guessed",
                name,
            )
        )
        return
    cycles = rule.get("cycles")
    if not isinstance(cycles, int) or cycles <= 0:
        unknowns.append(Unknown(f"latency.{field}", "the rule declares no positive cycle count", name))
        return
    declared_completion = rule.get("completion")
    if declared_completion is None:
        # An issue gap says when the NEXT instruction may issue; it does not say how a consumer learns
        # this one finished. Reading "the target has a delay instruction" as "therefore every cost is
        # discharged by counting cycles" would put a counted completion on a resource constraint that
        # has no completion at all, and oblige a compiler to emit a wait for a count nothing declared.
        declared_completion = "immediate"
    elif declared_completion not in COMPLETION_KINDS:
        # DECLARED, but not a kind this model knows. Reading it as immediate would turn a misspelled
        # polled or counted completion into no wait at all -- the one wrong answer nothing downstream
        # can notice -- so the rule is reported and not costed.
        unknowns.append(
            Unknown(
                "latency.completion",
                f"the rule declares completion {declared_completion!r}, which is none of {COMPLETION_KINDS}",
                name,
            )
        )
        return
    producers: Sequence[Any] = rule.get("producers") or ()
    for instr in producers:
        if not isinstance(instr, str) or not instr:
            continue
        key = (instr, unit)
        prior = costs.get(key)
        issue = (cycles - 1) if field == "issue" else (prior.issue if prior else None)
        result = cycles if field == "result" else (prior.result if prior else None)
        completion = prior.completion if prior else declared_completion
        if completion == "counted" and result is None:
            unknowns.append(
                Unknown(
                    "latency.completion",
                    "the rule declares a counted completion but no result latency to count; a wait the "
                    "compiler must emit cannot be emitted from an unknown count, so the completion is "
                    "reported rather than assumed",
                    name,
                )
            )
            completion = "immediate"
        costs[key] = Latency(
            instr=instr,
            unit=unit,
            issue=issue,
            result=result,
            completion=completion,
            contended=True,
            source=f"schedule_contract.{name}" if prior is None else f"{prior.source}+{name}",
        )


def _hazard(contract: Mapping[str, Any], delay: str | None) -> tuple[str | None, tuple[Unknown, ...]]:
    """The declared hazard model, cross-checked against the ISA that evidences it."""
    declared = (contract.get("memory_model") or {}).get("hazard_resolution")
    if declared is None:
        return None, (
            Unknown(
                "hazard_resolution",
                "the target's contract declares no memory_model.hazard_resolution, and no extraction "
                "decides it: a delay instruction is evidence for 'explicit' but its absence is not "
                "evidence for 'interlocked', since a machine may interlock and still expose a delay",
            ),
        )
    if declared not in HAZARD_RESOLUTIONS:
        raise MachError(f"hazard_resolution {declared!r} is not one of {list(HAZARD_RESOLUTIONS)}")
    if declared == "interlocked" and delay:
        raise MachError(
            f"the contract declares hazards 'interlocked' while the target's schedule contract declares "
            f"a delay instruction {delay!r}. A machine whose hardware resolves hazards has no separation "
            "for a compiler to emit; one of the two declarations is wrong, and this derivation will not "
            "pick which."
        )
    return declared, ()


def machine_from_address_space(space: Any, *, target: str | None = None) -> Machine:
    """The MEMORY and ARRAY half of a machine, for a caller that already holds an address space.

    Carries an ``Unknown`` for ``hazard_resolution`` because an address space says nothing about it; a
    caller wanting the whole machine calls :func:`derive`.
    """
    name = target or getattr(space, "target", None) or ""
    mems, mem_unknowns = _memories(space)
    edge, edge_unknown = _array_edge(space)
    unknowns = [
        Unknown("hazard_resolution", "derived from an address space alone, which does not describe issue"),
        *mem_unknowns,
        *([edge_unknown] if edge_unknown else []),
        *(Unknown(u.quantity, u.reason, getattr(u, "where", None)) for u in getattr(space, "unknowns", ()) or ()),
    ]
    return Machine(
        target=name,
        hazard_resolution=None,
        memories=mems,
        array_edge=edge,
        unknowns=tuple(unknowns),
        provenance={"address_space": str(getattr(space, "sources", {}) or {})},
    )


def derive(
    target: str,
    *,
    facts: dict[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
    space: Any | None = None,
    schedule_contract: Mapping[str, Any] | None = None,
) -> Machine:
    """The machine ``target`` declares and extracts, with every gap named.

    Every input is injectable so the derivation is testable without an RTL toolchain -- which matters,
    because an absent extractor must read as an absent MEASUREMENT, never as an absent unit.
    """
    from merlin.development.targetgen.address_space import derive_address_space
    from merlin.development.targetgen.rtl import facts as rtl_facts

    if space is None:
        space = derive_address_space(target, facts=facts)
    body = facts.get("facts") if isinstance(facts, Mapping) and "facts" in facts else facts
    if body is None:
        body = rtl_facts.body_if_present(target)
    if contract is None:
        contract = _contract_for(target)

    # The hazard cross-check runs FIRST, on the delay instruction alone. A contradiction between the two
    # declarations must surface as itself; run after the costs are built, it is masked by whatever the
    # first malformed rule raises, and the reader is told about the wrong thing.
    hazard, hazard_unknowns = _hazard(contract, (schedule_contract or {}).get("delay_instruction", {}).get("mnemonic"))
    units, unit_unknowns = _units(contract, body, space)
    mems, mem_unknowns = _memories(space)
    edge, edge_unknown = _array_edge(space)
    costs, delay, latency_unknowns = _latencies(schedule_contract)

    # A declared cost naming a unit this machine does not have is dropped -- but saying so is the whole
    # point. Dropped silently it reads as "that instruction is free", and the likeliest cause is the one
    # that matters most: the contract and the unit list disagree about what the units are called, so
    # EVERY cost vanishes and the machine prices a fully serial schedule at zero.
    known = {u.name for u in units}
    orphans = tuple(
        Unknown(
            "latency.unit",
            f"a declared cost names the unit {c.unit!r}, which this machine does not have "
            f"(it has {sorted(known)}), so the cost is dropped rather than attached to a guess",
            c.instr,
        )
        for c in costs
        if c.unit not in known
    )
    costs = tuple(c for c in costs if c.unit in known)
    unknowns = (
        *hazard_unknowns,
        *unit_unknowns,
        *mem_unknowns,
        *([edge_unknown] if edge_unknown else []),
        *latency_unknowns,
        *orphans,
        *(Unknown(u.quantity, u.reason, getattr(u, "where", None)) for u in getattr(space, "unknowns", ()) or ()),
    )
    source = facts_view.source(body) or {}
    return Machine(
        target=target,
        hazard_resolution=hazard,
        units=units,
        memories=mems,
        hierarchy=_hierarchy(body),
        latencies=costs,
        delay_instruction=delay,
        array_edge=edge,
        unknowns=unknowns,
        provenance={
            "facts_source": "; ".join(f"{k}={v}" for k, v in sorted(source.items()))[:400],
            "address_space": str(getattr(space, "sources", {}) or {})[:400],
        },
    )


def _contract_for(target: str) -> Mapping[str, Any]:
    """The target's RESOLVED contract.

    Through ``target_registry.resolve`` first, which is the seam that answers "which package is this
    target" for a curated package and a discovered one alike, and which reads the contract that package
    actually ships. Falling back to the residual deriver for a target that has a residual but no
    materialised package. A target with neither gets an empty mapping, and every quantity a contract
    would have supplied becomes an ``Unknown`` -- never a family default wearing a contract's clothes.
    """
    from merlin.development.targetgen import capability_manifests, target_registry

    try:
        selected = target_registry.resolve(target)
        if selected.capability_contract_path.is_file():
            doc = selected.load_contract()
            if isinstance(doc, Mapping):
                return doc
    except Exception:  # noqa: S110 - not resolvable as a package; try the residual deriver below
        pass
    try:
        return capability_manifests.manifest_for(target)
    except Exception as exc:  # the target declares no manifest; the caller gets Unknowns, not defaults
        return {"_unavailable": f"{type(exc).__name__}: {exc}"}


def unknown_census(machine: Machine) -> dict[str, int]:
    """``{quantity: count}`` -- the countable form of what a machine could not answer.

    An exit criterion states this set, not a total: a NEW quantity appearing is a regression even when
    the count falls, and a quantity disappearing is progress that must be re-asserted.
    """
    census: dict[str, int] = {}
    for unknown in machine.unknowns:
        census[unknown.quantity] = census.get(unknown.quantity, 0) + 1
    return dict(sorted(census.items()))


# ---- crosscheck --------------------------------------------------------------------------------
# Hold a target's DECLARED on-chip geometry to the machine DERIVED from its RTL facts.
#
# Every target that schedules against on-chip storage learns its geometry twice. Once from something it
# DECLARES -- a C header its programs compile against, a params file, a contract field -- and once from
# what an extractor DERIVES from the RTL. The two describe the same silicon by different routes, and
# nothing compares them, so a declaration regenerated against a different elaboration schedules against a
# machine that is not the one being measured. Every cycle count still looks plausible, because the program
# still runs.
#
# WHY THIS LIVES IN THE CORE AND NOT IN A TARGET'S BACKEND. The comparison names no target and needs
# none: it takes a mapping of declared numbers, derives the machine for whichever target it was given, and
# reports. Written inside one target's binding it would be correct and then copied, and the second copy is
# where the two drift -- which is the shape of overfit the repo's cardinal rule exists to prevent. Here it
# is covered by ``check_no_target_name.py`` for free, so this file can never grow a target literal without
# a gate saying so.
#
# A DISAGREEMENT RAISES, and names both numbers rather than preferring one. Which of the two is stale is
# not a question this function can answer, and picking silently is how the wrong number becomes the agreed
# number. An UNDERIVABLE machine does NOT raise: inside a sandbox the RTL facts are not mounted, and
# refusing there would make "we could not look" indistinguishable from "the two disagree". It is recorded
# instead, so a reader can tell agreement from silence.

#: Declared-geometry axis -> the attribute of the projected geometry it must equal. The declaring side
#: supplies its OWN spelling for each; this is the shared vocabulary the comparison is made in.
AXES: tuple[str, ...] = ("block", "operand_rows", "accumulator_rows")


class GeometryDisagreement(ValueError):
    """A target's declared geometry and its derived machine describe different silicon."""


def crosscheck_declared_geometry(declared: Mapping[str, int], target: str | None) -> str:
    """Compare ``declared`` against the machine derived for ``target``; return what was established.

    ``declared`` maps :data:`AXES` names to the numbers the target's own declaration states. A caller
    whose declaration spells them differently translates at the call site -- the translation is the one
    genuinely target-specific part, and it stays with the target.

    Returns a sentence a receipt can carry: which machine agreed, or why none could be consulted.
    Raises :class:`GeometryDisagreement` when both spoke and they differ.
    """
    if not target:
        return "not checked: no target named"
    unknown = sorted(set(declared) - set(AXES))
    if unknown:
        raise GeometryDisagreement(
            f"{target}: declared geometry names {unknown}, which is not one of {list(AXES)}. The "
            "comparison is made in a shared vocabulary; a caller translates its own spelling at the "
            "call site rather than widening this one."
        )
    try:
        from merlin.compiler.schedule.derive import geometry_from_machine

        geometry = geometry_from_machine(derive(target))
    except Exception as exc:  # noqa: BLE001 - an unavailable derivation is recorded, never fatal
        return f"unavailable: {type(exc).__name__}: {exc}"

    disagree = {
        axis: (value, getattr(geometry, axis)) for axis, value in declared.items() if value != getattr(geometry, axis)
    }
    if disagree:
        raise GeometryDisagreement(
            f"{target}: what this target DECLARES and the machine derived from its RTL facts describe "
            "different silicon: "
            + "; ".join(f"{k} declared={d} derived={v}" for k, (d, v) in sorted(disagree.items()))
            + ". One of the two is stale. This refuses rather than picking, because a schedule built on "
            "the wrong one still runs and still reports a number."
        )
    return f"agrees with merlin.compiler.kernel_ir.machine ({geometry.sources.get('machine', 'no digest')})"


__all__ = [
    "AXES",
    "COMPLETION_KINDS",
    "DISCHARGES_DEPENDENCE",
    "GeometryDisagreement",
    "HAZARD_RESOLUTIONS",
    "Hierarchy",
    "Latency",
    "Level",
    "MOVER_KINDS",
    "MachError",
    "Machine",
    "Memory",
    "ROLE_UNKNOWN",
    "SYNC_ORDERS",
    "Sync",
    "UNIT_KINDS",
    "Unit",
    "Unknown",
    "crosscheck_declared_geometry",
    "derive",
    "machine_from_address_space",
    "unknown_census",
]
