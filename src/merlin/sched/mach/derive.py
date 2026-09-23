"""Build a :class:`~merlin.sched.mach.Machine` from what a target already declares and already extracts.

A target contributes DATA -- its compute-unit contract, its RTL facts, its address space, its declared
issue-scheduling rules. It never contributes a machine. This module is the one place those four are read
together, and nothing in it names a target.

WHAT IT REFUSES TO DO, AND WHY EACH REFUSAL EARNED ITS PLACE.

*A store the facts describe by its PORT geometry is not a memory.* One target's only extracted store is
a write-port byte-enable geometry -- the facts say so themselves, in the store's own provenance -- and it
is neither of that device's two real on-chip stores, over-reporting capacity by more than an order of
magnitude. A derivation that passed it through would hand every capacity and bank decision a plausible
wrong number, which is worse than handing it none. So it becomes an :class:`Unknown` naming what was
found instead.

*A unit the evidence finds but does not classify keeps ``ROLE_UNKNOWN``.* The RTL derivation that finds
a command-driven array's decoupled controllers deliberately refuses to name their role, because deciding
that from a module's spelling is the assumption it exists to avoid. Dropping them would silently delete
the overlap lever they are; labelling them would assert a role nothing derived.

*An underived quantity is ``None`` with a reason.* Never 0, never a plausible default. The measured cost
of the alternative is on the record: a unit with no top-level busy port read as permanently idle, which
moved one headline from 76.7% to 46.2%.

WHAT IS DECLARED RATHER THAN DERIVED. Two things, and both are declared because they are not properties
the RTL alone settles. ``hazard_resolution`` is EVIDENCED by an ISA (a delay instruction is strong
evidence for ``explicit``) but not decided by it -- a machine may interlock *and* expose a delay -- so it
is declared in the target's contract and cross-checked against the ISA here, where a contradiction
raises. Completion kind per instruction class is declared in the target's issue-scheduling data for the
same reason: whether a wait is discharged by counting cycles or by polling is a property of the unit's
protocol, not of any width in the RTL.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.common import facts_view
from merlin.sched.mach.model import (
    COMPLETION_KINDS,
    HAZARD_RESOLUTIONS,
    ROLE_UNKNOWN,
    Hierarchy,
    Latency,
    Level,
    MachError,
    Machine,
    Memory,
    Unit,
    Unknown,
)

__all__ = ["derive", "machine_from_address_space", "unknown_census"]

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
        from merlin.targetgen.address_space import ADDRESSABLE, accumulator_kind
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
                    "merlin/experiments/, which library code may not read, so a caller that wants costs "
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
    if declared_completion not in COMPLETION_KINDS:
        # An issue gap says when the NEXT instruction may issue; it does not say how a consumer learns
        # this one finished. Reading "the target has a delay instruction" as "therefore every cost is
        # discharged by counting cycles" would put a counted completion on a resource constraint that
        # has no completion at all, and oblige a compiler to emit a wait for a count nothing declared.
        declared_completion = "immediate"
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
    from merlin.targetgen.address_space import derive_address_space
    from merlin.targetgen.rtl import facts as rtl_facts

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
    import yaml

    from merlin.targetgen import capability_manifests, target_registry

    try:
        path = target_registry.resolve(target).contract_path
        if path and path.is_file():
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
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
