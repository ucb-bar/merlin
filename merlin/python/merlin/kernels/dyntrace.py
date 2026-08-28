"""A normalized dynamic-execution trace: one vocabulary, many producers.

WHY. Static assembly cannot answer the questions that decide whole-model time. Whether a copy
overlaps the compute it feeds, how long an engine sat idle waiting for its operand, how much of a
dispatch was queue wait rather than work -- none of it is in the instruction stream, and all of it is
where a model's time goes once the inner loop is fast. Measured here: once the GEMM kernel was quick,
the matmul bucket was 1.3-6% of model time, so 94-97% had never been attributed to anything.

THE INVARIANT THAT MATTERS. A trace with no ``dma_read`` events cannot tell you that no DMA happened.
It can equally mean this producer does not record DMA. Those two readings differ by everything, and
collapsing them is a recorded failure mode in this tree ("none means unavailable hides real bugs" --
a mesh path that turned every failure into 'no result'). So a :class:`Trace` declares which event
kinds its producer CAN emit, and every aggregate over a kind outside that set returns ``None``
(UNKNOWN) rather than ``0``. Asking for DMA bytes from a producer that cannot see DMA is answered
"I don't know", never "zero".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: The closed event vocabulary. Closed on purpose: a producer that needs a kind not in this list is
#: describing something the rest of the pipeline cannot reason about, and adding it is a decision.
EVENT_KINDS: tuple[str, ...] = (
    "host_compute",              # work on the host/scalar core, outside any dispatch
    "dispatch_begin",            # a launch starts
    "dispatch_end",              # a launch retires
    "dma_read", "dma_write",     # bulk asynchronous movement, not through the compute datapath
    "local_load", "local_store", # movement into/out of the endpoint's own local memory
    "compute",                   # the endpoint doing arithmetic
    "commit", "readout",         # draining an accumulator, and making the result visible
    "sync",                      # a fence/barrier/completion wait
    "queue_wait",                # ready work not yet started, because the queue was busy
    "engine_idle",               # the engine had nothing to do
    "engine_stall",              # the engine had work and could not proceed
)

#: Kinds that consume wall/cycle time without doing the work the model asked for. Named so a report
#: cannot quietly count a stall as compute.
OVERHEAD_KINDS: frozenset[str] = frozenset({"sync", "queue_wait", "engine_idle", "engine_stall"})


@dataclass(frozen=True)
class TraceEvent:
    """One thing that happened, on one engine, over a cycle interval."""

    kind: str
    #: which engine it happened on, in the engine vocabulary (spatial/simt/vector/scalar).
    engine: str = ""
    #: the dispatch this belongs to, so an event can be joined back to a model op.
    dispatch_id: str | None = None
    #: cycles. ``end`` is exclusive. None when the producer timestamps only one edge.
    start: int | None = None
    end: int | None = None
    #: bytes moved, for movement kinds. None is UNKNOWN, never zero.
    nbytes: int | None = None
    #: the emitted-code range this corresponds to, when the producer can attribute one.
    symbol: str | None = None
    insn_range: tuple[int, int] | None = None
    #: ids of events this one waited on.
    depends_on: tuple[str, ...] = ()
    event_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> int | None:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def problems(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.kind not in EVENT_KINDS:
            out.append(f"event kind {self.kind!r} is not in the closed vocabulary")
        if self.start is not None and self.end is not None and self.end < self.start:
            out.append(f"event ends before it starts ({self.start} -> {self.end})")
        return tuple(out)


@dataclass
class Trace:
    """A sequence of events PLUS what its producer is capable of recording.

    ``records`` is the load-bearing field. Without it, every absence reads as a zero.
    """

    source: str
    #: the substrate that produced this (gsim, firesim, spike, ...). Joins to measurement authority.
    substrate: str = ""
    events: tuple[TraceEvent, ...] = ()
    #: which EVENT_KINDS this producer can emit at all. An empty set means "unknown capability",
    #: which makes every aggregate UNKNOWN -- the honest reading of a trace nobody characterized.
    records: frozenset[str] = frozenset()
    notes: tuple[str, ...] = ()

    def problems(self) -> tuple[str, ...]:
        out: list[str] = []
        for bad in sorted({k for k in self.records if k not in EVENT_KINDS}):
            out.append(f"declared recordable kind {bad!r} is not in the closed vocabulary")
        for e in self.events:
            out.extend(e.problems())
        emitted = {e.kind for e in self.events}
        for k in sorted(emitted - set(self.records)):
            out.append(f"emitted {k!r} events without declaring the kind recordable — the capability "
                       f"declaration is what makes an ABSENCE readable, so it must be complete")
        return tuple(out)

    # --- aggregates: UNKNOWN unless the producer can see the kind -----------------------------

    def can_see(self, *kinds: str) -> bool:
        return bool(self.records) and all(k in self.records for k in kinds)

    def bytes_moved(self, *, kinds: Iterable[str] = ("dma_read", "dma_write")) -> int | None:
        """Total bytes over ``kinds``; None when the producer cannot see them or did not size them."""
        kinds = tuple(kinds)
        if not self.can_see(*kinds):
            return None
        sized = [e.nbytes for e in self.events if e.kind in kinds]
        if any(b is None for b in sized):
            return None            # a partially-sized total is not a total
        return sum(b for b in sized if b is not None)

    def cycles_in(self, kind: str, *, engine: str | None = None) -> int | None:
        """Cycles spent in ``kind``; None when unrecordable or untimed."""
        if not self.can_see(kind):
            return None
        picked = [e for e in self.events
                  if e.kind == kind and (engine is None or e.engine == engine)]
        durations = [e.duration for e in picked]
        if any(d is None for d in durations):
            return None
        return sum(d for d in durations if d is not None)

    def overhead_cycles(self) -> int | None:
        """Cycles in sync/queue-wait/idle/stall. None if ANY overhead kind is invisible here.

        Deliberately all-or-nothing: summing the overhead kinds a producer happens to emit and
        calling it "overhead" understates it by exactly the kinds it cannot see, which is the
        direction that flatters the result.
        """
        if not self.can_see(*sorted(OVERHEAD_KINDS)):
            return None
        parts = [self.cycles_in(k) for k in sorted(OVERHEAD_KINDS)]
        if any(p is None for p in parts):
            return None
        return sum(p for p in parts if p is not None)

    def engines_seen(self) -> tuple[str, ...]:
        return tuple(sorted({e.engine for e in self.events if e.engine}))

    def gaps(self) -> tuple[str, ...]:
        """What this trace cannot answer, stated. An empty trace is a gap, not a quiet zero."""
        out: list[str] = []
        if not self.records:
            out.append(f"{self.source}: producer capability undeclared — every aggregate is UNKNOWN, "
                       f"which is NOT the same as zero")
            return tuple(out)
        for k in sorted(set(EVENT_KINDS) - set(self.records)):
            out.append(f"{self.source}: cannot record {k!r}; an absence of it means nothing")
        return tuple(out)
