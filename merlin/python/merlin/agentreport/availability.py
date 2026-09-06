"""Per-field provenance for everything the run report reads.

WHY THIS EXISTS. A run report draws numbers from a dozen files written by four drivers over three
months, and the files disagree about what they contain. The dangerous case is not a missing file --
it is a field that is *present and zero* because the writer could not measure it. Those read as
findings ("this run did no thinking", "this run cost nothing") and they are not.

So every field a reader produces is accompanied by a :class:`Status`. ``UNAVAILABLE`` carries the
reason VERBATIM from whichever reader refused, because those reasons are already good -- see
``merlin.agent_trace.timeline`` -- and rewriting them loses the diagnostic.

``DERIVED`` is deliberately distinct from ``MEASURED``: capsules-in-flight reconstructed from
self-check state changes is a real number, but it is not the same kind of number as a tool span read
off a stamped event, and a figure that renders them identically is making a claim the data does not
support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping

MEASURED = "measured"
DERIVED = "derived"
UNAVAILABLE = "unavailable"

#: Ordered best-to-worst, so a caller can rank runs by how much of them is real.
KINDS = (MEASURED, DERIVED, UNAVAILABLE)


@dataclass(frozen=True)
class Status:
    """How a single field came to be -- and, when it did not, why."""

    kind: str
    reason: str = ""
    #: Which reader/source produced it (e.g. "transcript_tool_use_ids", "codex_item_events").
    source: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown status kind {self.kind!r}; known: {KINDS}")
        # An UNAVAILABLE with no reason is the exact failure this module exists to prevent: it
        # renders as a blank cell that reads like "nothing to say" rather than "could not measure".
        if self.kind == UNAVAILABLE and not self.reason:
            raise ValueError("an UNAVAILABLE status must carry a reason")

    @property
    def ok(self) -> bool:
        return self.kind in (MEASURED, DERIVED)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "reason": self.reason, "source": self.source}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Status":
        return cls(kind=str(d.get("kind") or UNAVAILABLE),
                   reason=str(d.get("reason") or ""), source=str(d.get("source") or ""))


def measured(source: str = "") -> Status:
    return Status(MEASURED, source=source)


def derived(reason: str = "", source: str = "") -> Status:
    """A real number, reconstructed rather than read. ``reason`` says how it was reconstructed."""
    return Status(DERIVED, reason=reason, source=source)


def unavailable(reason: str, source: str = "") -> Status:
    return Status(UNAVAILABLE, reason=reason, source=source)


@dataclass
class Availability:
    """The per-field ledger for one run, plus the aggregate a selector can rank on."""

    fields: dict[str, Status] = field(default_factory=dict)

    def set(self, name: str, status: Status) -> None:
        self.fields[name] = status

    def get(self, name: str) -> Status:
        return self.fields.get(name, unavailable(f"field {name!r} was never read"))

    def ok(self, name: str) -> bool:
        return self.get(name).ok

    def __iter__(self) -> Iterator[tuple[str, Status]]:
        return iter(sorted(self.fields.items()))

    @property
    def score(self) -> float:
        """Fraction of read fields that carry a real number. Ranks a plottable run above a stub."""
        if not self.fields:
            return 0.0
        return sum(1 for s in self.fields.values() if s.ok) / len(self.fields)

    def reasons(self) -> dict[str, list[str]]:
        """Unavailable field names grouped by reason -- the raw material for a figure caption."""
        out: dict[str, list[str]] = {}
        for name, st in sorted(self.fields.items()):
            if st.kind == UNAVAILABLE:
                out.setdefault(st.reason, []).append(name)
        return out

    def to_dict(self) -> dict:
        return {name: st.to_dict() for name, st in sorted(self.fields.items())}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Availability":
        return cls(fields={str(k): Status.from_dict(v) for k, v in (d or {}).items()})
