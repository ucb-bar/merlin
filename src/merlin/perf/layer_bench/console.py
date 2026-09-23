"""Parse what a layer program and its engine print, structurally (no regex).

Two sources, two grammars:

- The layer program prints one line per measured group::

      LB_RECORD <label> cycles=<n> [<field>=<int> ...]

  ``label`` is the caller's handle (e.g. a key prefix). Fields are decimal integers; ``cycles`` is
  required. Records are small on purpose: a slow console decides what is measurable.
- The engine prints a completion line on stderr, e.g.::

      [gsim-emu] FINISHED: cycles=601947 wall=42.71s (14095 cyc/s) done=1 exit_code=0

  Only ``key=value`` tokens after ``FINISHED:`` are read; a trailing ``s`` on ``wall`` is seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RECORD_TAG = "LB_RECORD"
FINISH_TAG = "FINISHED:"


class ConsoleError(ValueError):
    """A console line claimed to be a record but did not parse. Never silently skipped."""


@dataclass(frozen=True)
class LayerRecord:
    label: str
    cycles: int
    fields: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EngineFinish:
    cycles: int
    wall_seconds: float
    done: bool
    exit_code: int


def parse_layer_records(console: str) -> list[LayerRecord]:
    """Every ``LB_RECORD`` line, in order. A malformed record raises instead of disappearing."""
    records = []
    for line in console.splitlines():
        parts = line.split()
        if not parts or parts[0] != RECORD_TAG:
            continue
        if len(parts) < 3:
            raise ConsoleError(f"truncated record: {line!r}")
        label = parts[1]
        values: dict[str, int] = {}
        for token in parts[2:]:
            name, sep, raw = token.partition("=")
            if not sep or not name or not raw.lstrip("-").isdigit():
                raise ConsoleError(f"record field is not name=<int>: {token!r} in {line!r}")
            if name in values:
                raise ConsoleError(f"duplicate field {name!r} in {line!r}")
            values[name] = int(raw)
        if "cycles" not in values:
            raise ConsoleError(f"record without cycles: {line!r}")
        if values["cycles"] < 0:
            raise ConsoleError(f"negative cycles: {line!r}")
        cycles = values.pop("cycles")
        records.append(LayerRecord(label=label, cycles=cycles, fields=values))
    return records


def parse_engine_finish(text: str) -> EngineFinish | None:
    """The engine's completion line, or None when the engine never finished (e.g. hit its cap)."""
    found = None
    for line in text.splitlines():
        head, sep, tail = line.partition(FINISH_TAG)
        if not sep:
            continue
        values: dict[str, str] = {}
        for token in tail.split():
            name, eq, raw = token.partition("=")
            if eq:
                values[name] = raw
        try:
            found = EngineFinish(
                cycles=int(values["cycles"]),
                wall_seconds=float(values["wall"].removesuffix("s")),
                done=values["done"] == "1",
                exit_code=int(values["exit_code"]),
            )
        except (KeyError, ValueError) as why:
            raise ConsoleError(f"unparseable engine completion line {line!r}: {why}") from None
    return found
