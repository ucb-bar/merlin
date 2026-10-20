"""Bind observed counter readings to their independently declared RTL source."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def counter_observations(
    console: str, *, target: str, simulator: str, cycles: int | None, oracle: Any
) -> tuple[list[dict] | None, dict | None]:
    """``(timing_observations, timing_capability)`` a bracketed run earned, or ``(None, None)``.

    THE HOP THAT WAS MISSING. The bracket emitter, the console parser, the wire contract and every
    consumer of a per-unit activity vector already existed; nothing turned the readings into the block
    a tier record carries, so a target that counts overlap in HARDWARE still reported
    ``missing: ['at least one activity source']`` and no composition operator, headroom or eta could
    resolve from it.

    ONLY AN RTL ORACLE MAY CARRY ONE. A functional model runs the program correctly without modelling
    the engines, so its counter CSRs describe nothing: measured on one, a 52-cycle window returned
    per-engine busy totals in the THOUSANDS. Those are not imprecise numbers, they are numbers about a
    different machine, and a composition operator derived from them would be a fabrication carrying a
    measurement's provenance. An oracle that states nothing fails closed for the same reason.

    NOTHING IS SWALLOWED. The first version of this guard put an unimported name inside a broad
    ``except``, so EVERY call returned "no capability": the negative cases passed for the wrong reason
    and the positive case silently never fired. Each refusal below is a specific, reachable condition.
    """
    if not isinstance(oracle, Mapping) or oracle.get("derived_from_rtl") is not True:
        return None, None  # a model's counters describe a different machine
    from merlin.perf import hw_counters
    from merlin.perf import observations as _observations

    readings = hw_counters.parse_counter_output(console)
    if not readings:
        return None, None  # unbracketed: byte-identical to before
    discovery = hw_counters.counter_source_for_target(target)
    if discovery.get("status") != "derived":
        return None, None  # no counter set derived from this target's own header
    measured_schema = hw_counters.parse_counter_schema(console)
    if measured_schema is not None and measured_schema != discovery.get("header_sha256"):
        return None, None  # the ELF was bracketed against a DIFFERENT schema
    # An ABSENT schema line is UNKNOWN, not a mismatch -- a real bracketed run need not emit one, and
    # refusing on its absence would refuse every such run. What actually binds the readings to this
    # header is the coverage check below: the reading set must contain every combination the header
    # derives, which a run bracketed against a different counter set cannot satisfy.
    try:
        occupancy, _codes = hw_counters.occupancy_for_discovery(discovery)
    except (OSError, ValueError):
        return None, None  # the counter source changed or vanished after discovery
    required = set(occupancy.by_combination.values())
    if not required or not required <= set(readings):
        return None, None  # a partial combination set is a lower bound, not a total
    # The KIND of each engine is the TARGET's declaration: a kind cannot be read off a counter name,
    # and a consumer refuses a unit that lacks one. Absent when the backend declares none.
    from merlin.runtime.backends import base as _backends

    _kinds_reader = getattr(_backends.get_backend(target), "counter_engine_kinds", None)
    kinds = _kinds_reader() if callable(_kinds_reader) else hw_counters.declared_engine_kinds(target)
    block = hw_counters.observations_from_counters(
        readings,
        occupancy,
        total_cycles=cycles,
        source=f"hardware combination counters ({discovery.get('header') or 'contract counter-event table'})",
        kind_of=kinds,
    )
    validated = _observations.validate_block(block)
    if validated is None:
        return None, None
    # A refused block still travels as a capability record: "the producer emitted a block we could not
    # believe" is a fact about the instrument, and dropping it hides the instrument rather than the bug.
    return ([dict(o) for o in validated.observations] or None), validated.to_dict()
