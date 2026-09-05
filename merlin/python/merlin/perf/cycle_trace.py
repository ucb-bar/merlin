"""Reduce a per-cycle activity trace to the timing block :mod:`merlin.perf.observations` validates.

Some RTL engines compute the occupancy decomposition *inside* the simulator and hand it back on the
result (the Verilator program harness does: it accumulates the buckets as it steps and prints
``timing_observations`` beside the outputs). Others only *dump* the design's activity ports, one row
per cycle, and return nothing but ``{halted, cycles, outputs, reads, writes}``. The second shape is
not a fidelity gap -- the same wires were sampled on the same cycles -- but every consumer that reads
occupancy telemetry saw the second kind of engine as an instrument with no timing capability, so
adopting a faster engine silently cost the perf campaign its per-capsule decomposition.

This module closes that by reduction rather than by re-simulation: the trace the engine ALREADY wrote
is folded into the SAME quantities, with the same spellings, that an in-sim harness emits. Nothing is
modelled, estimated or scaled -- every number here is a count of rows in a file the engine produced.

WHAT IS DECLARED AND WHY IT CANNOT BE GUESSED
---------------------------------------------
A trace is a table of integers. Which of its columns is a *unit's busy signal* is not a property of
the table, and three plausible ways to guess it are all wrong in the flattering direction:

* **By spelling.** Picking the columns whose names end in "busy" is a name heuristic, and it reads a
  role out of an identifier -- the failure :meth:`~merlin.perf.observations.ObservationBlock.kinds`
  exists to refuse. It also silently drops a busy port spelled any other way, which makes the unit
  read as permanently idle and inflates the idle figure.
* **By value range.** A column that only ever holds 0/1 need not be an occupancy signal (a halt flag
  is not a unit), and a genuine busy port that is high on every sampled cycle looks constant.
* **By nonzero-means-busy.** Applied to a state register this is simply false: state 0 is a state,
  not idleness. :func:`merlin.perf.occupancy.calibrate_state_idle` exists precisely because which
  value means idle has to be established, not assumed.

So the column-to-unit binding and each unit's KIND are **declared by the producer** -- the party that
chose the columns and knows what they are wired to -- and this module refuses to invent one. An
engine with no declaration reports no timing capability, exactly as it did before, which is the
honest answer rather than a fabricated block.

The declaration is a small JSON document beside the engine's wrapper (:data:`DECLARATION_NAME`), so
it lives with the harness whose ports it describes rather than in library code that must not know any
target's unit names. Nothing in this module names a target, a unit, a kind or a column.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf import observations as _OBS

__all__ = [
    "DECLARATION_NAME", "DECLARATION_SCHEMA", "block_from_rows", "block_from_trace",
    "load_declaration", "read_rows",
]

#: The producer's column declaration, read from beside the engine wrapper it describes.
DECLARATION_NAME = "timing_columns.json"
#: Its schema tag. A declaration without it is refused rather than read optimistically -- an
#: unlabelled JSON file next to a wrapper is not evidence that anyone declared these columns.
DECLARATION_SCHEMA = "merlin.cycle-trace-columns.v1"

#: A busy column is boolean by construction (a unit is busy on a cycle or it is not). The trace
#: stores it as an integer, so exactly these two spellings are accepted and anything else makes the
#: column UNREADABLE -- never "nonzero means busy", which on a state register is false.
_LOW = "0"
_HIGH = "1"


def load_declaration(engine_dir: "str | Path") -> "dict[str, Any] | None":
    """The producer's column declaration in ``engine_dir``, or ``None`` when it declares none.

    ``None`` is the answer for "this engine has no timing capability", and it is deliberately the
    same answer an engine that writes no trace at all gets: a missing declaration is a missing
    instrument, not an instrument reading zero.
    """
    path = Path(engine_dir) / DECLARATION_NAME
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, Mapping):
        return None
    if str(doc.get("schema_version") or "") != DECLARATION_SCHEMA:
        return None
    units = doc.get("busy_columns")
    if not isinstance(units, Mapping) or not units:
        return None
    unmeasured = doc.get(_OBS.UNMEASURED_UNITS_KEY)
    if not isinstance(unmeasured, Sequence) or isinstance(unmeasured, (str, bytes)):
        # REQUIRED of the producer for the same reason validate_block requires it downstream: a
        # declaration that does not say which of its own columns it could not read is claiming a
        # completeness it has not earned. Refused whole rather than defaulted to complete.
        return None
    return dict(doc)


def read_rows(csv_path: "str | Path") -> "list[dict[str, str]] | None":
    """The trace as rows of strings, or ``None`` if it is absent/unreadable/headerless.

    Parsed with :mod:`csv` rather than by splitting lines: the header names the columns and a
    positional read would silently re-bind every unit the day a column is inserted.
    """
    path = Path(csv_path)
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return rows or None


def block_from_rows(rows: "Sequence[Mapping[str, str]]", declaration: Mapping) -> "dict[str, Any] | None":
    """Fold trace ``rows`` into the ``timing_observations`` block, per ``declaration``.

    Returns the producer-side mapping :func:`merlin.perf.observations.validate_block` consumes --
    the observation list plus the ``unmeasured_units`` / ``partitioned`` assertions -- or ``None``
    when the declaration binds no column the trace actually carries. Every value is a row count.

    The buckets are JOINT, not a partition: a cycle in which two units are busy is counted once for
    each, and once more in the overlap readings. ``partitioned`` is therefore asserted ``False``,
    which is what licenses an overlap reading at all -- and it is an assertion about this reduction,
    so a later change that reduced these to marginal counts would make it fail loudly instead of
    quietly reporting zero overlap.
    """
    units = declaration.get("busy_columns")
    if not isinstance(units, Mapping) or not rows:
        return None
    present = set(rows[0].keys())

    # Bind each DECLARED column to its unit and kind. A declared column the trace does not carry is
    # unmeasured -- reported, never treated as a unit that was idle all run.
    bound: list[tuple[str, str, str]] = []          # (column, unit, kind)
    missing: list[str] = []
    for column, spec in units.items():
        col = str(column)
        unit = str((spec or {}).get("unit") or col) if isinstance(spec, Mapping) else col
        kind = str((spec or {}).get("kind") or "") if isinstance(spec, Mapping) else ""
        (bound.append((col, unit, kind)) if col in present else missing.append(col))
    if not bound:
        return None

    busy = {unit: 0 for _, unit, _ in bound}
    idle = 0
    overlap_any = 0
    overlap_kinds = 0
    sampled = 0
    unreadable: set[str] = set()
    for row in rows:
        sampled += 1
        hot_units: set[str] = set()
        hot_kinds: set[str] = set()
        for col, unit, kind in bound:
            raw = str(row.get(col, "")).strip()
            if raw == _HIGH:
                hot_units.add(unit)
                if kind:
                    hot_kinds.add(kind)
            elif raw != _LOW:
                # NOT counted either way. A column whose value this reduction cannot read is a column
                # whose occupancy is unknown, and folding it into idle is the reading that flatters.
                unreadable.add(col)
        # ONCE PER UNIT PER CYCLE, off the SET -- several columns may name one unit (a unit's load and
        # store halves nest inside it, and the producer folds them by declaring one unit for all
        # three). Incrementing per COLUMN would charge such a unit two or three cycles for one, and
        # would report it overlapping with itself.
        for unit in hot_units:
            busy[unit] += 1
        if not hot_units:
            idle += 1
        if len(hot_units) >= 2:
            overlap_any += 1
        if len(hot_kinds) >= 2:
            overlap_kinds += 1

    declared_unmeasured = [str(u) for u in declaration.get(_OBS.UNMEASURED_UNITS_KEY) or ()]
    unmeasured = sorted(set(declared_unmeasured) | set(missing) | unreadable)

    obs: list[dict[str, Any]] = []
    kind_by_unit = {unit: kind for _, unit, kind in bound}
    for unit in sorted(busy):
        entry: dict[str, Any] = {
            "quantity": f"{_OBS.BUSY_PREFIX}{unit}{_OBS.IN_PROGRAM_SUFFIX}",
            "value": busy[unit], "unit": "cycles", "concurrent": True,
            "note": "cycles this unit's activity port was high while the program ran; CONTENDED "
                    "(other units may be busy in the same cycle), counted off the engine's own "
                    "per-cycle trace"}
        if kind_by_unit.get(unit):
            entry["kind"] = kind_by_unit[unit]
        obs.append(entry)
    # IDLE IS RELATIVE TO WHAT WAS READ, and saying so is the difference between a measurement and a
    # flattering one. A unit this trace does not carry contributes no busy cycles, so every cycle it
    # alone was busy in lands here -- which is the "unit with no busy port, read as permanently idle"
    # failure. Cross-checked on a design whose other engine also reports per-channel DMA units the
    # trace omits: every shared unit agreed exactly and the idle figures differed by precisely those
    # units' cycles. So idle is exact when nothing is unmeasured and an UPPER BOUND otherwise, and
    # the note carries which case this run is.
    _idle_note = "cycles in which no DECLARED unit's port was high"
    if unmeasured:
        _idle_note += (f"; an UPPER BOUND, not an exact figure -- {len(unmeasured)} unit(s) were not "
                       "read (see unmeasured_units) and any cycle in which only those were busy is "
                       "counted here. Not comparable with an instrument that reads more units")
    obs.append({"quantity": _OBS.IDLE_QUANTITY, "value": idle, "unit": "cycles", "concurrent": False,
                "note": _idle_note})
    obs.append({"quantity": _OBS.OVERLAP_OBSERVED, "value": overlap_any, "unit": "cycles",
                "concurrent": True,
                "note": "cycles with two or more declared units busy together -- a JOINT count, not "
                        "derivable from the per-unit numbers above"})
    obs.append({"quantity": _OBS.OVERLAP_ACROSS_KINDS, "value": overlap_kinds, "unit": "cycles",
                "concurrent": True,
                "note": "cycles with two or more DISTINCT declared kinds busy together; two units of "
                        "one kind running together is not cross-kind overlap"})
    obs.append({"quantity": _OBS.SAMPLED_QUANTITY, "value": sampled, "unit": "cycles",
                "concurrent": False,
                "note": "rows in the engine's per-cycle trace; the buckets reconcile against this, "
                        "which need not equal the run's reported cycle count"})

    notes = [str(declaration.get("unmeasured_note") or "").strip()]
    if missing:
        notes.append(f"declared but absent from the trace: {sorted(missing)}")
    if unreadable:
        notes.append(f"present but not boolean on every row, so unread: {sorted(unreadable)}")
    out: dict[str, Any] = {
        _OBS.TIMING_OBSERVATIONS_KEY: obs,
        _OBS.UNMEASURED_UNITS_KEY: unmeasured,
        _OBS.PARTITIONED_KEY: False,
    }
    note = " | ".join(n for n in notes if n)
    if note:
        out["unmeasured_note"] = note
    return out


def block_from_trace(csv_path: "str | Path", engine_dir: "str | Path") -> "dict[str, Any] | None":
    """The block for a trace at ``csv_path``, using the declaration in ``engine_dir``.

    ``None`` -- meaning "no timing capability", which is what the caller then reports -- whenever the
    trace is absent or the producer declared no columns. Both are answers about the instrument, and
    neither is ever turned into a block of zeros.
    """
    decl = load_declaration(engine_dir)
    if decl is None:
        return None
    rows = read_rows(csv_path)
    if rows is None:
        return None
    return block_from_rows(rows, decl)
