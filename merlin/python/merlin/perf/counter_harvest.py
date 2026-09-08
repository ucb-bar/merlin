"""Harvest counter-bracketed runs into the corpus attribution map the agent reads.

WHY THIS IS THE LAST HOP. The feedback stack was complete except for its input, and the input turned
out to need no hardware at all: the graded harness already emits ``counter_bracket_c`` over
``derive_occupancy_counters`` -- the complete one-hot ``MAIN_*`` partition -- gated on one opt-in
environment variable, and :mod:`merlin.perf.counter_trust` declares the cycle-accurate FIRRTL
simulator's counters ``real`` on the same footing as the FPGA's. So a full partition, and therefore a
CLOSED accelerator-busy figure rather than a lower bound, is one local run away. This module walks
those runs and hands :func:`merlin.perf.attribution.attribute_corpus` its sources.

⚠️ **THE ENGINE IS CHECKED, NOT ASSUMED, AND THIS IS NOT THEORETICAL.** A single graded run writes one
console per oracle tier, and on the run this was built from the functional ISS reported all seven
counters at 2,609..4,744 against its own 65-cycle window -- **35x the window**, which is impossible,
because that model increments every counter with ``rand()`` on every accelerator instruction. The
cycle-accurate console for the same capsule reported 0..83 against 366. Both files sit in the same
directory and the numbers look equally plausible in isolation. So every reading is admitted only
through :func:`merlin.perf.counter_trust.verdict_for`, and a console whose engine cannot be
identified from its own path is refused rather than guessed at.

A second guard follows from the same run: the partition sum may not exceed the measured window. That
is what caught the fabricated set as arithmetically impossible rather than merely untrusted, and it
would also catch a wrapped counter or two windows mixed into one reading.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["CounterRun", "harvest_counter_runs", "corpus_map", "format_corpus_map",
           "engine_of", "CYCLE_MARKER"]

#: The engine that produced a console is read from the FILE NAME, because the readings themselves
#: carry no engine and a graded run writes one console per oracle tier into one directory. The
#: candidate names are the TARGET'S OWN declared oracle keys -- never a table of filename spellings
#: here, which would name one target's console conventions in shared code and would go stale the
#: moment a target added a tier. See :func:`engine_of`.

#: The harness's cycle line. The window the partition is charged against.
CYCLE_MARKER = "METRIC cycles"


@dataclass(frozen=True)
class CounterRun:
    """One counter-bracketed run: its engine, its window, and its readings."""

    workload: str
    engine: str
    total_cycles: int
    readings: Mapping[str, int]
    console: Path
    trusted: bool
    refusal: str = ""

    @property
    def charged(self) -> int:
        return sum(int(v) for v in self.readings.values())

    def to_dict(self) -> dict[str, Any]:
        return {"workload": self.workload, "engine": self.engine,
                "total_cycles": self.total_cycles, "charged_cycles": self.charged,
                "readings": {k: int(v) for k, v in sorted(self.readings.items())},
                "console": str(self.console), "trusted": self.trusted, "refusal": self.refusal}


@dataclass
class HarvestResult:
    runs: list[CounterRun] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)

    def trusted(self) -> list[CounterRun]:
        return [r for r in self.runs if r.trusted]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_counter_harvest_v1",
                "n_runs": len(self.runs), "n_trusted": len(self.trusted()),
                "runs": [r.to_dict() for r in self.runs], "refusals": list(self.refusals)}


def engine_of(path: Path, engines: Iterable[str]) -> str | None:
    """Which of ``engines`` this console belongs to, or None when its name names none of them.

    LONGEST match wins, so a tier whose key is a prefix or substring of another cannot capture it.
    ``engines`` is the target's own declared oracle vocabulary (its backend ``ORACLE`` keys), passed
    in rather than tabulated here: a filename table in shared code would name one target's console
    conventions and would silently mis-attribute a tier the target added later.
    """
    stem = path.name
    hits = [e for e in engines if e and e in stem]
    return max(hits, key=len) if hits else None


def harvest_counter_runs(runs_root: Any, *, engines: Iterable[str]) -> HarvestResult:
    """Every counter-bracketed console under ``runs_root``, with each refusal carrying its reason.

    ``engines`` is the target's declared oracle vocabulary -- for a backend registered with
    :mod:`merlin.runtime.backends.base`, ``sorted(get_backend(target).ORACLE)``. Required, because a
    console's engine decides whether its readings are measurements at all, and this module refuses to
    guess: the run this was built from had a fabricated set and a real set in ONE directory.
    """
    from merlin.perf import counter_trust as CT  # noqa: PLC0415
    from merlin.perf import hw_counters as HC  # noqa: PLC0415

    out = HarvestResult()
    root = Path(runs_root)

    for console in sorted(root.rglob("*console*")):
        if not console.is_file() or console.suffix not in ("", ".log", ".txt"):
            continue
        try:
            text = console.read_text(errors="replace")
        except OSError as exc:
            out.refusals.append({"console": str(console), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if HC.COUNTER_MARKER not in text:
            continue
        engine = engine_of(console, engines)
        if engine is None:
            out.refusals.append({
                "console": str(console),
                "reason": (f"the console's name names none of the declared engines "
                           f"{sorted(engines)}, and the "
                           "readings do not carry it; a graded run writes one console per tier into "
                           "one directory, so guessing would mix a fabricated set with a real one")})
            continue
        readings = HC.parse_counter_output(text)
        if not readings:
            out.refusals.append({"console": str(console), "reason": "no readings parsed"})
            continue
        cycle_lines = [line for line in text.splitlines() if CYCLE_MARKER in line]
        if not cycle_lines:
            out.refusals.append({
                "console": str(console),
                "reason": (f"no {CYCLE_MARKER!r} line, so the partition has no window to be charged "
                           f"against and the host residue cannot be computed")})
            continue
        try:
            total = int(cycle_lines[-1].split()[-1])
        except ValueError:
            out.refusals.append({"console": str(console),
                                 "reason": f"unparseable cycle line {cycle_lines[-1]!r}"})
            continue

        verdict = CT.verdict_for(engine)
        refusal = verdict.refusal() or ""
        # ARITHMETIC IMPOSSIBILITY, checked independently of the trust contract. A partition cannot
        # charge more cycles than the window it was read in; the fabricated set failed this by 35x.
        charged = sum(int(v) for v in readings.values())
        if total > 0 and charged > total:
            impossible = (f"the readings charge {charged} cycles against a {total}-cycle window "
                          f"({charged / total:.1f}x): a partition cannot exceed its own window, so "
                          f"this is fabricated, wrapped, or two windows mixed into one reading")
            refusal = (refusal + "; " if refusal else "") + impossible

        run = CounterRun(workload=console.parent.parent.name, engine=engine, total_cycles=total,
                         readings=readings, console=console,
                         trusted=not refusal, refusal=refusal)
        out.runs.append(run)
        if refusal:
            out.refusals.append({"console": str(console), "engine": engine, "reason": refusal})
    return out


def corpus_map(runs: Iterable[CounterRun], *, header_text: str,
               kind_of: Mapping[str, str]) -> tuple[Any, list[dict[str, Any]]]:
    """``(CorpusAttribution, refusals)`` over the TRUSTED runs, or a refusal per run that cannot map.

    Each run is converted through :func:`merlin.perf.attribution.activity_from_counter_readings`,
    which refuses a partial partition -- so a corpus map is built only from runs whose
    accelerator-busy figure is closed.
    """
    from merlin.perf import attribution as A  # noqa: PLC0415

    sources = []
    refusals: list[dict[str, Any]] = []
    for run in runs:
        if not run.trusted:
            refusals.append({"workload": run.workload, "reason": run.refusal})
            continue
        try:
            sources.append(A.activity_from_counter_readings(
                run.readings, workload=run.workload, total_cycles=run.total_cycles,
                header_text=header_text, kind_of=kind_of,
                provenance=f"{run.engine} + full partition ({run.console.name})"))
        except ValueError as exc:
            refusals.append({"workload": run.workload, "reason": str(exc)})
    if not sources:
        return None, refusals
    buckets = A.buckets_from_kinds({r.name: r.kind for r in sources[0].resources},
                                   fixed_bucket="host")
    return A.attribute_corpus(sources, buckets=buckets), refusals


def format_corpus_map(corpus: Any, *, refusals: Sequence[Mapping[str, Any]] = ()) -> str:
    """The table the agent reads to choose where to work. ``NONE`` means nothing to win here."""
    if corpus is None:
        rows = ["== no workload produced a closed partition"]
        rows += [f"   refused {r.get('workload', '?')}: {str(r.get('reason'))[:110]}"
                 for r in refusals]
        return "\n".join(rows)
    rows = [f"{'workload':26} {'bucket':9} {'cycles':>10} {'%window':>8}  family"]
    for workload, att in sorted(corpus.workloads.items()):
        total = att.total_cycles or 1
        for component in att.components:
            family = (component.family.value if hasattr(component.family, "value")
                      else str(component.family))
            rows.append(f"{workload[:26]:26} {component.bucket:9} "
                        f"{component.measured_cycles:>10,} "
                        f"{100.0 * component.measured_cycles / total:>7.2f}%  {family}")
    for r in refusals:
        rows.append(f"   refused {r.get('workload', '?')}: {str(r.get('reason'))[:110]}")
    return "\n".join(rows)
