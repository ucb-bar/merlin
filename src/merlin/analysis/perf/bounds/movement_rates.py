"""Measure a systolic array's fill/drain depth from its circuit, instead of assuming a law.

A fill/drain depth is an intercept, not a rate: it is paid once per weight reload and it dominates
small tiles. It is also the term most easily assumed. The obvious closed form for a square
weight-stationary array is ``2*DIM`` -- rows plus columns -- and a slightly better one is ``2*DIM-2``.
Both are guesses about a pipeline whose actual length is a property of the emitted circuit, and the
gap between them is two cycles on one array and much larger on another microarchitecture.

So this reads the depth out of the circuit: the length of the output-valid delay-line register chain,
counted directly in the IR. Nothing is fitted and no law is applied.

WHY A LAW IS STILL WORTH KEEPING, and why this module reports BOTH. A measured depth is the truth for
the design that was elaborated; a law is what lets the model answer for a design that has not been
elaborated yet -- a different mesh dimension in a design-space sweep, where there is no circuit to
read. Keeping the two apart, and reporting when they disagree, is what tells you whether the law may
be extrapolated at all. A law that matches the circuit on the one point it was checked against is
evidence, not proof; a law that does not match is refuted for this family and must not be swept with.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from merlin.analysis.perf.bounds.decomposition import UNKNOWN, Trait, Unavailable, _Unknown

# ---- handshake ---------------------------------------------------------------------------------


class HandshakeUnavailable(RuntimeError):
    """The circuit could not be read, so the fill/drain depth is UNKNOWN -- never a default."""


@dataclass(frozen=True)
class FillDepth:
    """A fill/drain depth, its source, and how it compares with the law it would otherwise be given."""

    dim: int
    measured_cycles: int
    #: What the named law predicts for this dimension, or None when no law was offered.
    law_cycles: int | None
    law: str | None
    #: Double-buffering the circuit carries. A second slot lets the next reload overlap the current
    #: compute, which is why it belongs beside the depth rather than in a separate report.
    weight_buffer_slots: int
    accumulator_banks: int
    source: str

    @property
    def law_agrees(self) -> bool | None:
        """True/False when a law was offered, None when none was."""
        return None if self.law_cycles is None else self.law_cycles == self.measured_cycles

    def claim(self) -> str:
        if self.law_cycles is None:
            return f"fill/drain {self.measured_cycles} cycles, measured from {self.source}"
        if self.law_agrees:
            return (
                f"fill/drain {self.measured_cycles} cycles, measured; the law {self.law!r} agrees "
                f"at DIM={self.dim} -- evidence it may extrapolate, not proof"
            )
        return (
            f"fill/drain {self.measured_cycles} cycles, measured; the law {self.law!r} predicts "
            f"{self.law_cycles} and is REFUTED for this design -- do not sweep with it"
        )


def _instance_ports(line: str) -> tuple[str, dict[str, str]]:
    """``(instance name, {port: driving value})`` for one ``hw.instance`` line, else ``("", {})``.

    Parsed structurally -- membership and ``partition`` only, no pattern matching -- because the
    result list on the left of the ``=`` is a comma-separated series of the instance's own results
    and must not be mistaken for its name. The name is the quoted string after ``hw.instance``.
    """
    if "hw.instance" not in line:
        return "", {}
    _, _, after = line.strip().partition("hw.instance")
    after = after.strip()
    if after.count('"') < 2:
        return "", {}
    _, _, rest = after.partition('"')
    name, _, rest = rest.partition('"')
    _, _, args = rest.partition("(")
    ports: dict[str, str] = {}
    for part in args.split(","):
        port, sep, value = part.partition(":")
        if not sep:
            continue
        value = value.strip()
        if value.startswith("%"):
            ports[port.strip()] = value.split(":")[0].strip().lstrip("%")
    return name, ports


def _valid_path_depth(bodies: "dict[str, list[str]]", array_module: str) -> tuple[str, int]:
    """The longest register-stage path a valid signal takes through ``array_module``.

    The upstream pass reads the depth off a register chain whose OWN NAME contains "valid". That is
    a naming convention, not a structural fact, and a conformant design loses it: firtool named this
    array's chain ``%r_256_0 ... %r_1115_0`` and put the word "valid" only on the SIGNAL each stage
    samples. The pass then reports "no output-valid delay-line found", which reads as an unreadable
    circuit when the delay line is plainly there -- 257 registers of it.

    So the path is walked instead of name-matched: a stage is a register, and the valid signal
    crosses a submodule through the ports whose names carry the handshake's own "valid" (the port
    names are the DESIGN's vocabulary, unlike the register names, which the emitter invents). The
    answer is the deepest such path reaching a module output, which is the same physical quantity --
    cycles from an input valid to the output valid -- that the named chain measures where one exists.
    """
    body = bodies[array_module]
    regs: dict[str, str] = {}
    instances: dict[str, dict[str, str]] = {}
    outputs: list[str] = []
    for line in body:
        stripped = line.strip()
        if stripped.startswith("%") and "seq.firreg" in stripped:
            lhs, _, rhs = stripped.partition("=")
            _, _, driver = rhs.partition("seq.firreg")
            head = driver.split()
            if head:
                regs[lhs.strip().lstrip("%")] = head[0].lstrip("%")
        elif "hw.instance" in stripped:
            name, ports = _instance_ports(stripped)
            if name:
                instances[name] = ports
        elif stripped.startswith("hw.output"):
            _, _, values = stripped.partition("hw.output")
            outputs = [v.strip().lstrip("%") for v in values.split(",") if v.strip()]

    depth_of: dict[str, int] = {}

    def depth(token: str) -> int:
        seen = depth_of.get(token)
        if seen is not None:
            return seen
        depth_of[token] = 0  # break combinational cycles rather than recursing forever
        if token in regs:
            result = 1 + depth(regs[token])
        else:
            owner, sep, _ = token.partition(".")
            ports = instances.get(owner) if sep else None
            result = max((depth(v) for port, v in (ports or {}).items() if "valid" in port.lower()), default=0)
        depth_of[token] = result
        return result

    best, where = 0, ""
    for token in outputs:
        found = depth(token)
        if found > best:
            best, where = found, token
    return where, best


def measure_fill_depth(target: str, *, law: str | None = "systolic_2d", hw_mlir: Any = None) -> FillDepth:
    """Read the fill/drain depth from the target's elaborated circuit, and check any offered law.

    ``law`` names a fill law from :mod:`merlin.analysis.perf.contract.record` to cross-check against; pass None to skip
    the comparison. Raises rather than returning a default when the circuit is unreachable: an
    unmeasurable intercept is UNKNOWN, and a pipeline silently given a depth of zero reads as an array
    that fills instantly."""
    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_1
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

    path = hw_mlir if hw_mlir is not None else mlc_bridge.core_hw_mlir(target)
    if path is None:
        raise HandshakeUnavailable(
            f"no elaborated circuit is resolvable for {target!r}; the fill/drain depth is UNKNOWN"
        )
    # MODULE NAMES ARE DERIVED, NOT DEFAULTED. `infer_handshake_depth` takes the array/mesh/PE module
    # names and defaults them to one family's spelling; calling it without them asks every target's
    # circuit for another target's module. That produced "array module @SystolicArray not found" on a
    # design whose array container the facts name `Mesh` -- a refusal about the wrong thing, which
    # reads as "this circuit is unreadable" rather than "we looked for the wrong module". The names
    # come from the target's own discovered array fact; absent one, the defaults still apply.
    # The pass distinguishes the outer array WRAPPER from the inner MESH, and the discovered array
    # fact names the mesh (its `container`) -- so the derived name goes to `mesh_module`, never to
    # `array_module`. Mapping it to the wrapper broke a target whose circuit reads correctly under
    # the defaults (atlas: dim=32, fill_drain_depth=62), which is why the wrapper name is only
    # retried, not replaced: a design that carries no separate wrapper is then asked for its mesh.
    names: dict[str, str] = {}
    try:
        from merlin.target.rtl import facts as _facts

        arrays = (_facts.body_if_present(target)).get("arrays") or ()
        discovered = arrays[0] if arrays else {}
        if discovered.get("container"):
            names["mesh_module"] = str(discovered["container"])
        if discovered.get("element"):
            names["pe_module"] = str(discovered["element"])
    except Exception:  # noqa: BLE001 - no facts bundle just means the defaults stand
        names = {}
    walked: tuple[str, int] | None = None
    try:
        with _merlin_mlc_discovery_1._mlc_cwd():
            import importlib

            module = importlib.import_module("mlc.passes.infer_handshake_depth")
            infer_handshake_depth = module.infer_handshake_depth

            def _attempt(**kw: Any) -> Any:
                return infer_handshake_depth(str(path), **kw)

            try:
                facts = _attempt(**names)
            except ValueError as missing:
                # Only a MISSING WRAPPER is retried with the derived mesh name; any other refusal is
                # about this design and must surface unchanged rather than be masked by a retry.
                if "array module" not in str(missing) or "mesh_module" not in names:
                    raise
                names = {**names, "array_module": names["mesh_module"]}
                facts = _attempt(**names)
    except ValueError as unnamed:
        # A design whose delay line the emitter did not NAME "valid" is not an unreadable circuit.
        # Walk the valid path instead, and let every other refusal fall through unchanged.
        if "output-valid delay-line" not in str(unnamed):
            raise HandshakeUnavailable(
                f"the circuit for {target!r} could not be read for a fill/drain depth: "
                f"{type(unnamed).__name__}: {unnamed}"
            ) from unnamed
        try:
            with _merlin_mlc_discovery_1._mlc_cwd():
                import importlib

                module = importlib.import_module("mlc.passes.infer_handshake_depth")
                lines = Path(path).read_text().split("\n")
                bodies = {name: lines[start:end] for name, start, end in module._iter_modules(lines)}
                array_module = names.get("array_module") or names.get("mesh_module") or ""
                if array_module not in bodies:
                    raise HandshakeUnavailable(
                        f"the array module for {target!r} is not in its own circuit: {array_module!r}"
                    )
                where, depth = _valid_path_depth(bodies, array_module)
                if depth <= 0:
                    raise HandshakeUnavailable(
                        f"{target!r} has no valid path through @{array_module}, named or walked, "
                        f"so its fill/drain depth is UNKNOWN"
                    )
                mesh_module = names.get("mesh_module", array_module)
                pe_module = names.get("pe_module", "PE")
                pe_instances = module._count_pe_instances(bodies.get(mesh_module, bodies[array_module]), pe_module)
                if pe_instances <= 0:
                    raise HandshakeUnavailable(
                        f"no @{pe_module} instances in @{mesh_module}, so {target!r} has no measurable array dimension"
                    )
                from math import isqrt

                dim = isqrt(pe_instances)
                if dim * dim != pe_instances:
                    raise HandshakeUnavailable(
                        f"@{pe_module} count {pe_instances} in @{mesh_module} is not a square mesh"
                    )
                walked = (f"@{array_module}.%{where} valid path ({pe_instances} @{pe_module} instances)", depth)
                facts = SimpleNamespace(
                    dim=dim,
                    fill_drain_depth=depth,
                    weight_buffer_slots=module._weight_slot_count(bodies[array_module]),
                    accumulator_banks=module._accsel_banks(bodies[array_module]),
                    array_module=array_module,
                    valid_register_family=where,
                    pe_instances=pe_instances,
                )
        except HandshakeUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - an unwalkable circuit is UNKNOWN, not a default
            raise HandshakeUnavailable(
                f"the circuit for {target!r} could not be walked for a fill/drain depth: {type(exc).__name__}: {exc}"
            ) from exc
    except Exception as exc:  # noqa: BLE001 - a circuit we cannot read is UNKNOWN, not a default
        raise HandshakeUnavailable(
            f"the circuit for {target!r} could not be read for a fill/drain depth: {type(exc).__name__}: {exc}"
        ) from exc

    law_cycles = None
    if law is not None:
        from merlin.analysis.perf.contract.record import fill_cycles

        law_cycles = int(fill_cycles(law, facts.dim))
    source = (
        walked[0]
        if walked is not None
        else f"{facts.array_module}.{facts.valid_register_family} ({facts.pe_instances} MAC instances)"
    )
    return FillDepth(
        dim=int(facts.dim),
        measured_cycles=int(facts.fill_drain_depth),
        law_cycles=law_cycles,
        law=law,
        weight_buffer_slots=int(facts.weight_buffer_slots),
        accumulator_banks=int(facts.accumulator_banks),
        source=source,
    )


# ---- movement_peak -----------------------------------------------------------------------------
# The memory-movement peak of an accelerator, read from its elaborated timing structure.
#
# The roofline's movement term needs a rate. :mod:`merlin.analysis.perf.bounds.per_shape_ceiling` accepts a transfer
# width from a datapath or interface block, and no target's facts published one, so the term stayed
# UNKNOWN on every target. The facts extractor's ``timing_structure.memory_clients`` section does
# publish the movement engine's own ports (``merlin.target.rtl.hw.timing``): per memory
# client, the beat width, the channel count, the in-flight transaction slots selected by the response
# ID, and the largest request the size field can carry. This module turns those into:
#
# * a **beat peak** per client: ``beat_bytes x beats_per_cycle``. A synchronous ready/valid channel
#   retires at most one beat per cycle, so ``beats_per_cycle`` is the client's channel count;
# * an **in-flight bound**: ``in_flight x max_request_bytes`` bytes may be outstanding, so by Little's
#   law the sustained rate is at most that volume divided by the memory round trip. The round trip is
#   NOT an RTL fact (the extractor lists it under ``fit_from_measurement``), so without a calibrated
#   latency the in-flight limit is reported as the latency above which it binds, never as a rate.
#
# Direction is read structurally, never from a client's name: a client whose RESPONSE carries data
# reads memory, and one whose REQUEST carries data with a data-less response writes it.
#
# Everything absent is UNKNOWN with a reason. A peak is an upper bound on the movement rate; it never
# describes contention, translation misses or arbitration, which remain measured quantities.

SCHEMA = "merlin.analysis.perf.bounds.movement_rates.v1"
READ = "read"
WRITE = "write"


def _positive(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _body(document: Mapping[str, Any]) -> Mapping[str, Any]:
    body = document.get("facts") if isinstance(document, Mapping) and "facts" in document else document
    return body if isinstance(body, Mapping) else {}


def _direction(client: Mapping[str, Any]) -> str | None:
    request = _positive(client.get("request_data_bits"))
    response = _positive(client.get("response_data_bits"))
    if response is not None:
        return READ
    if request is not None and client.get("response_data_bits") is None:
        return WRITE
    return None


def _client(name: str, client: Mapping[str, Any], latency: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "direction": _direction(client), "unresolved": {}}
    beat = _positive(client.get("beat_bytes"))
    channels = _positive(client.get("request_channels"))
    in_flight_block = client.get("in_flight")
    in_flight = (
        _positive(in_flight_block.get("value"))
        if isinstance(in_flight_block, Mapping) and in_flight_block.get("status") == "derived"
        else None
    )
    largest = _positive(client.get("max_request_bytes"))
    if row["direction"] is None:
        row["unresolved"]["direction"] = "neither the request nor the response states a data width"
    if beat is None or channels is None:
        row["unresolved"]["peak"] = "the client states no beat width or channel count"
        row["peak_bytes_per_cycle"] = None
    else:
        row["beat_bytes"] = beat
        row["beats_per_cycle"] = channels
        row["peak_bytes_per_cycle"] = beat * channels
    row["in_flight"] = in_flight
    row["max_request_bytes"] = largest
    if in_flight is None or largest is None:
        row["unresolved"]["in_flight"] = "the in-flight slots or the largest request size are not derived"
        row["in_flight_bytes"] = None
        row["binding_latency_cycles"] = None
    else:
        row["in_flight_bytes"] = in_flight * largest
        peak = row["peak_bytes_per_cycle"]
        # Above this round trip the outstanding-request window, not the beat width, limits the rate.
        row["binding_latency_cycles"] = (row["in_flight_bytes"] / peak) if peak else None
    peak = row["peak_bytes_per_cycle"]
    if peak is None:
        row["sustained_bytes_per_cycle"] = None
        row["sustained_basis"] = "no beat peak"
    elif latency is None:
        row["sustained_bytes_per_cycle"] = float(peak)
        row["sustained_basis"] = (
            "beat peak only: the memory round trip is a calibration fit, so the in-flight window is "
            "reported as the latency above which it binds, not applied"
        )
    elif row["in_flight_bytes"] is None:
        row["sustained_bytes_per_cycle"] = float(peak)
        row["sustained_basis"] = "beat peak; the in-flight window is not derived"
    else:
        windowed = row["in_flight_bytes"] / float(latency)
        row["sustained_bytes_per_cycle"] = min(float(peak), windowed)
        row["sustained_basis"] = (
            f"min(beat peak {peak}, in-flight {row['in_flight_bytes']} bytes / round trip {latency:g} cycles)"
        )
    return row


def movement_peak(document: Mapping[str, Any], *, round_trip_latency_cycles: float | None = None) -> dict[str, Any]:
    """Per-direction movement peaks from ``timing_structure.memory_clients``, or UNKNOWN with why.

    ``round_trip_latency_cycles`` is a fitted memory round trip (a calibration product). Supplied, the
    in-flight window caps each client's sustained rate; omitted, only the beat peak applies.
    """
    if round_trip_latency_cycles is not None and (
        isinstance(round_trip_latency_cycles, bool)
        or not isinstance(round_trip_latency_cycles, (int, float))
        or round_trip_latency_cycles <= 0
    ):
        raise ValueError("a round-trip latency must be a positive number of cycles")
    structure = _body(document).get("timing_structure")
    clients_block = structure.get("memory_clients") if isinstance(structure, Mapping) else None
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "unknown",
        "clients": {},
        "read_bytes_per_cycle": None,
        "write_bytes_per_cycle": None,
        "total_bytes_per_cycle": None,
        "round_trip_latency_cycles": round_trip_latency_cycles,
        "unresolved": {},
        "fit_from_measurement": list((structure or {}).get("fit_from_measurement") or [])
        if isinstance(structure, Mapping)
        else [],
    }
    if not isinstance(clients_block, Mapping) or clients_block.get("status") != "derived":
        out["unresolved"]["memory_clients"] = (
            "the facts carry no derived timing_structure.memory_clients section, so no movement port is known"
        )
        return out
    clients = clients_block.get("clients")
    if not isinstance(clients, Mapping) or not clients:
        out["unresolved"]["memory_clients"] = "the derived memory-client section names no client"
        return out
    totals: dict[str, float] = {}
    for name in sorted(clients):
        client = clients[name]
        if not isinstance(client, Mapping):
            out["unresolved"][str(name)] = "the client record is not a mapping"
            continue
        row = _client(str(name), client, round_trip_latency_cycles)
        out["clients"][str(name)] = row
        if row["direction"] is not None and row["sustained_bytes_per_cycle"] is not None:
            totals[row["direction"]] = totals.get(row["direction"], 0.0) + row["sustained_bytes_per_cycle"]
        else:
            out["unresolved"][str(name)] = "; ".join(f"{k}: {v}" for k, v in row["unresolved"].items())
    out["read_bytes_per_cycle"] = totals.get(READ)
    out["write_bytes_per_cycle"] = totals.get(WRITE)
    if READ in totals and WRITE in totals:
        # Separate read and write clients may run concurrently, so total traffic over their summed
        # rates is a lower bound on movement time: an upper bound on the achievable rate.
        out["total_bytes_per_cycle"] = totals[READ] + totals[WRITE]
        out["status"] = "derived"
    else:
        missing = sorted({READ, WRITE} - set(totals))
        out["unresolved"]["direction"] = f"no resolved client moves data in direction(s) {missing}"
    return out


# ---- amplification -----------------------------------------------------------------------------
# Data-movement amplification: bytes actually moved vs bytes the computation needs.
#
# A program that needs 4 KiB of operands but pulls 64 KiB across the bus is paying 16x for its data.
# On a target where the movement engine is busy for most of the run that ratio, not overlap, is the
# lever worth pulling -- and a performance bound built on *algorithmic* bytes rather than *moved* bytes
# is optimistic by exactly this factor, which is the flattering direction.
#
# **The ratio alone is a trap.** A fixed per-transfer granule -- a DMA command that always moves a
# whole block whatever you asked for -- inflates the ratio on small tiles and amortizes away on large
# ones. Reporting 16x without saying how much of it is that artifact invites someone to go chase a win
# that proper tiling would have taken anyway. So every result here splits the ratio into two
# multiplicative factors:
#
# ``granularity_factor``
#     ``(transfers_min * block_bytes) / useful_bytes`` -- the fixed-block padding, where
#     ``transfers_min = ceil(useful_bytes / block_bytes)`` is the fewest transfers that could carry the
#     useful bytes. **This is the small-tile artifact.** It falls towards 1 as the tile grows past the
#     block, so it is not a win that survives tiling.
#
# ``redundancy_factor``
#     ``transfers / transfers_min`` -- bytes moved more than once: refetch, replication, re-reading an
#     operand the program could have kept resident. **This is the part that survives amortization**,
#     and it is the real target.
#
# ``ratio == granularity_factor * redundancy_factor`` exactly, so the split is a decomposition and not
# an estimate.
#
# Gated on the derived trait "explicit DMA / managed scratchpad": on a target whose data movement is
# implicit (a hardware cache) there is no program-visible transfer to count and no useful/moved
# distinction the compiler can act on, so the honest answer is
# :class:`~merlin.analysis.perf.bounds.decomposition.Unavailable`. The trait holding is not enough on its own -- the byte
# evidence has to exist too, and a target with the trait but no measurement gets an ``Unavailable``
# that names *evidence*, not *trait*, so the two situations stay distinguishable.


@dataclass(frozen=True)
class TensorOperand:
    """One tensor a workload reads or writes, sized from its shape and dtype.

    ``broadcast`` marks an operand that materializes a single repeated value (a splat constant, an
    epsilon, a reciprocal-of-N). Its footprint is a lowering artifact rather than data the
    computation needs -- the program could synthesize it -- so it is excluded from the useful bytes
    and reported separately in ``broadcast_bytes``. Counting it would understate the amplification.
    """

    name: str
    elements: int
    element_bytes: float
    #: ``True`` for an operand the workload writes, ``False`` for one it reads.
    is_output: bool = False
    broadcast: bool = False

    @property
    def bytes(self) -> float:
        return self.elements * self.element_bytes


def useful_bytes(operands: Sequence[TensorOperand]) -> tuple[int, int]:
    """``(useful_bytes, broadcast_bytes)`` for a workload's declared operands.

    Useful bytes are the operands and results a computation genuinely consumes and produces, sized
    from shape and dtype. Broadcast splats are separated out rather than dropped silently.
    """
    useful = sum(o.bytes for o in operands if not o.broadcast)
    splat = sum(o.bytes for o in operands if o.broadcast)
    return int(round(useful)), int(round(splat))


def moved_bytes_from_beats(beats: int, beat_bytes: int) -> int:
    """Bytes on the bus for ``beats`` transfers of a ``beat_bytes``-wide data port.

    ``beat_bytes`` is an RTL fact about the target's data port and must be supplied by the caller
    from that target's derived facts. There is no default: a guessed bus width silently rescales
    every ratio in this module.
    """
    if beat_bytes <= 0:
        raise ValueError(f"beat_bytes must be positive, got {beat_bytes}")
    return int(beats) * int(beat_bytes)


@dataclass(frozen=True)
class MovementObservation:
    """What one workload actually moved, and what it needed."""

    workload: str
    #: Bytes that crossed the bus (from measured beats x the port width -- not from the shape).
    moved_bytes: int
    #: Bytes the computation needs: operands + results from shape and dtype.
    useful_bytes: int
    #: Number of movement commands the program issued. Needed to separate the fixed per-transfer
    #: granule from genuine refetch; ``None`` leaves the sensitivity UNKNOWN rather than guessed.
    transfers: int | None = None
    #: Per-command byte counts when they are individually known (heterogeneous descriptors). When
    #: given, the block is the largest command rather than the mean.
    transfer_bytes: tuple[int, ...] = ()
    broadcast_bytes: int = 0
    provenance: str = ""


@dataclass(frozen=True)
class WorkloadAmplification:
    """One workload's amplification, split into the artifact and the part that survives tiling."""

    workload: str
    moved_bytes: int
    useful_bytes: int
    ratio: float
    #: Bytes one movement command carries (the fixed granule), derived from the observation.
    block_bytes: float | _Unknown
    #: Fewest commands that could carry the useful bytes at that granule.
    transfers_min: int | _Unknown
    #: The small-tile fixed-block artifact. Falls to 1 as the tile grows past the block.
    granularity_factor: float | _Unknown
    #: Bytes moved more than once. Survives amortization; this is the real lever.
    redundancy_factor: float | _Unknown
    #: How full an average command is. Low fill == the ratio is mostly the artifact.
    fill_fraction: float | _Unknown
    provenance: str = ""

    @property
    def artifact_share(self) -> float | _Unknown:
        """Fraction of the ratio's magnitude attributable to the fixed granule, in log terms.

        ``log(granularity) / log(ratio)`` -- the two factors multiply, so their contributions add in
        logs. 1.0 means the whole ratio is the small-tile artifact; 0.0 means none of it is.
        """
        g, r = self.granularity_factor, self.ratio
        if g is UNKNOWN or r <= 1.0:
            return UNKNOWN
        return math.log(g) / math.log(r)


def movement_trait(manifest: Mapping[str, Any] | None = None, facts: Mapping[str, Any] | None = None) -> Trait:
    """Does this target have explicit DMA or a software-managed scratchpad?

    Derived from the target's own description, never from its name:

    * a **declared movement capability** on any compute unit (the target says a unit moves data
      under program control), and/or
    * a **software-managed on-chip memory**: a resident/scratchpad memory model, or discovered
      on-chip memories in the RTL facts.

    Either establishes the trait, because either gives the program a transfer it chose to issue.
    A hardware cache gives neither, and on such a target this analysis has nothing to count.
    """
    evidence: list[str] = []
    body = (facts or {}).get("facts", facts or {})

    units = list((manifest or {}).get("compute_units") or ())
    units += list((manifest or {}).get("derived_compute_units") or ())
    for u in units:
        if not isinstance(u, Mapping):
            continue
        for cap in u.get("semantic_capabilities") or ():
            if cap.get("family") == "movement":
                evidence.append(f"compute unit {u.get('name')!r} declares a movement capability")
                break

    mem = (manifest or {}).get("memory_model") or {}
    managed = [k for k, v in mem.items() if v is True]
    if managed:
        evidence.append(f"memory_model declares {sorted(managed)} (software-managed on-chip state)")

    memories = [m for m in (body.get("memories") or ()) if isinstance(m, Mapping)]
    if memories:
        evidence.append(f"RTL facts discover on-chip memories {sorted(m.get('name', '?') for m in memories)}")

    if evidence:
        return Trait("explicit_movement", True, evidence="; ".join(evidence))
    return Trait(
        "explicit_movement",
        None,
        evidence="no declared movement capability, managed memory model or discovered on-chip memory",
        missing=("explicit DMA or a software-managed scratchpad",),
    )


def _block_and_min(obs: MovementObservation) -> tuple[float | _Unknown, int | _Unknown]:
    """Derive the fixed per-command granule and the fewest commands that could carry the payload.

    Refuses on fewer than two observed commands: the granule is a fitted parameter and one transfer
    is one point. A block derived from a single command cannot be told apart from that command's
    payload, which would report every workload as 100% redundancy and 0% artifact.
    """
    if obs.transfer_bytes:
        if len(obs.transfer_bytes) < 2:
            return UNKNOWN, UNKNOWN
        block: float = float(max(obs.transfer_bytes))
    else:
        if obs.transfers is None or obs.transfers < 2:
            return UNKNOWN, UNKNOWN
        block = obs.moved_bytes / obs.transfers
    if block <= 0 or obs.useful_bytes <= 0:
        return UNKNOWN, UNKNOWN
    return block, max(1, math.ceil(obs.useful_bytes / block))


def amplification(
    obs: MovementObservation,
    *,
    trait: Trait | None = None,
    manifest: Mapping[str, Any] | None = None,
    facts: Mapping[str, Any] | None = None,
) -> WorkloadAmplification | Unavailable:
    """Amplification for one workload, with the fixed-granule sensitivity attached.

    Returns :class:`Unavailable` when the movement trait is not established (the target has no
    program-visible transfers) or when the byte evidence is missing (it does, but nobody measured
    it). Those are different failures and the returned ``missing`` says which.
    """
    tr = trait if trait is not None else movement_trait(manifest, facts)
    if tr.satisfied is not True:
        return Unavailable("data-movement amplification", tr.missing or ("explicit data movement",), tr.evidence)
    if obs.useful_bytes <= 0:
        return Unavailable(
            "data-movement amplification",
            ("the bytes the computation needs (operand shapes + dtypes)",),
            f"{obs.workload}: useful_bytes={obs.useful_bytes}",
        )
    if obs.moved_bytes <= 0:
        return Unavailable(
            "data-movement amplification",
            ("measured moved bytes (beats x the port width)",),
            f"{obs.workload}: moved_bytes={obs.moved_bytes}",
        )

    ratio = obs.moved_bytes / obs.useful_bytes
    block, tmin = _block_and_min(obs)
    if block is UNKNOWN:
        return WorkloadAmplification(
            workload=obs.workload,
            moved_bytes=obs.moved_bytes,
            useful_bytes=obs.useful_bytes,
            ratio=ratio,
            block_bytes=UNKNOWN,
            transfers_min=UNKNOWN,
            granularity_factor=UNKNOWN,
            redundancy_factor=UNKNOWN,
            fill_fraction=UNKNOWN,
            provenance=obs.provenance,
        )

    n_transfers = len(obs.transfer_bytes) if obs.transfer_bytes else int(obs.transfers)
    granularity = (tmin * block) / obs.useful_bytes
    redundancy = n_transfers / tmin
    return WorkloadAmplification(
        workload=obs.workload,
        moved_bytes=obs.moved_bytes,
        useful_bytes=obs.useful_bytes,
        ratio=ratio,
        block_bytes=block,
        transfers_min=tmin,
        granularity_factor=granularity,
        redundancy_factor=redundancy,
        fill_fraction=(obs.useful_bytes / n_transfers) / block,
        provenance=obs.provenance,
    )


@dataclass(frozen=True)
class AmplificationSensitivity:
    """How much of a corpus's amplification is the small-tile artifact, and how much survives.

    ``block_bytes_by_workload`` is kept per workload because a per-command granule is a *descriptor*
    choice: two workloads carrying the same tile can enqueue different transfer sizes, so a single
    corpus-wide block would be a fit to a bimodal population. When the blocks disagree the report
    says so instead of averaging them into a number nobody can act on.
    """

    n_points: int
    block_bytes_by_workload: dict[str, float]
    block_bytes_consistent: bool | None
    #: Geometric mean of the ratio that survives amortization, over the workloads that resolved.
    amortized_ratio: float | _Unknown
    #: Geometric mean of the small-tile fixed-block factor.
    granularity_factor: float | _Unknown
    #: Mean fraction (in logs) of the observed ratio explained by the fixed granule.
    artifact_share: float | _Unknown
    note: str = ""


def _geomean(values: Sequence[float]) -> float | _Unknown:
    vals = [v for v in values if v > 0]
    if not vals:
        return UNKNOWN
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


@dataclass(frozen=True)
class CorpusAmplification:
    """Amplification over a corpus, plus its sensitivity to the fixed per-transfer granule."""

    workloads: dict[str, WorkloadAmplification] = field(default_factory=dict)
    unavailable: dict[str, Unavailable] = field(default_factory=dict)
    sensitivity: AmplificationSensitivity | Unavailable | None = None

    @property
    def ratios(self) -> dict[str, float]:
        return {n: w.ratio for n, w in self.workloads.items()}


def corpus_amplification(
    observations: Iterable[MovementObservation],
    *,
    trait: Trait | None = None,
    manifest: Mapping[str, Any] | None = None,
    facts: Mapping[str, Any] | None = None,
) -> CorpusAmplification:
    """:func:`amplification` over a corpus, with the corpus-level sensitivity.

    The sensitivity needs **at least two workloads that resolved a granule** -- one point cannot
    separate a fixed per-transfer cost from a scale-invariant one, and pretending otherwise is how a
    single measurement becomes a fitted law.
    """
    tr = trait if trait is not None else movement_trait(manifest, facts)
    ok: dict[str, WorkloadAmplification] = {}
    bad: dict[str, Unavailable] = {}
    for obs in observations:
        r = amplification(obs, trait=tr)
        if isinstance(r, Unavailable):
            bad[obs.workload] = r
        else:
            ok[obs.workload] = r

    resolved = {n: w for n, w in ok.items() if w.block_bytes is not UNKNOWN}
    if len(resolved) < 2:
        sens: AmplificationSensitivity | Unavailable = Unavailable(
            "amplification sensitivity",
            ("at least two workloads with >=2 movement commands each",),
            f"{len(resolved)} workload(s) resolved a per-transfer granule; a fixed per-transfer "
            f"cost cannot be separated from a scale-invariant one on one point",
        )
        return CorpusAmplification(workloads=ok, unavailable=bad, sensitivity=sens)

    blocks = {n: float(w.block_bytes) for n, w in resolved.items()}
    consistent = max(blocks.values()) - min(blocks.values()) < 1e-9
    shares = [w.artifact_share for w in resolved.values() if w.artifact_share is not UNKNOWN]
    sens = AmplificationSensitivity(
        n_points=len(resolved),
        block_bytes_by_workload=blocks,
        block_bytes_consistent=consistent,
        amortized_ratio=_geomean([float(w.redundancy_factor) for w in resolved.values()]),
        granularity_factor=_geomean([float(w.granularity_factor) for w in resolved.values()]),
        artifact_share=(sum(shares) / len(shares)) if shares else UNKNOWN,
        note=(
            "the per-command granule is uniform across the corpus"
            if consistent
            else "the per-command granule VARIES across the corpus -- it is a per-workload descriptor "
            "choice, so a single corpus-wide block would be a fit to a mixed population"
        ),
    )
    return CorpusAmplification(workloads=ok, unavailable=bad, sensitivity=sens)


__all__ = [
    "AmplificationSensitivity",
    "CorpusAmplification",
    "MovementObservation",
    "READ",
    "SCHEMA",
    "TensorOperand",
    "WRITE",
    "WorkloadAmplification",
    "amplification",
    "corpus_amplification",
    "moved_bytes_from_beats",
    "movement_peak",
    "movement_trait",
    "useful_bytes",
]
