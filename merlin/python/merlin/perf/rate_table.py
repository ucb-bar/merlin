"""Per-compute-class throughput, derived from the certification runs phase 1 already paid for.

THE PROBLEM THIS SOLVES. A performance corpus wants members big enough to be worth optimising, and a
cycle-accurate oracle charges by output size, so the members that matter are exactly the ones nobody
can afford to measure per candidate. The escape is not a cheaper oracle and not a weaker tier: it is
that the functional corpus is ALREADY certified on the cycle-accurate tier, and every one of those
runs emitted a cycle count next to an emitted program. Phase 1 is the instrument. This module reads
it, and phase 2 becomes a CONSUMER of that instrument rather than a second one.

Two consumers are blocked on precisely this table and neither invents it:

* :func:`merlin.perf.compose_estimate._empirical_ceiling` divides priced MAC demand by "the slowest
  rate anything on this machine has been MEASURED at", and says of it: *the rate is a parameter,
  never derived here, and it must come from measured baselines*;
* :class:`merlin.targetgen.routing.MeasuredCost` DECLINES any unit absent from its measurement table,
  because *an unmeasured unit that scored well would win routing decisions on the strength of having
  no data*.

WHAT A RATE HERE IS, AND IS NOT
-------------------------------
A rate is ``priced MAC demand / measured cycles`` for one program, and the table keeps the SLOWEST per
compute class. That makes it usable as an empirical ceiling and nothing else. It is not a prediction,
not a model of the machine, and not promotable: every number is contended in exactly the sense
:mod:`merlin.perf.harvest` defines -- other engines were live inside the same window -- so the honest
reading is "no program of this class has been observed slower than this", which is a bound.

Split by class because one global rate is sound and useless: the slowest and fastest classes on the
target measured here differ by 35x (conv at 2.67 MACs/cycle against resident matmul at 94.1), and a
single rate covering both produced bands 95.7x wide against 18.9x per class. The class is read off the
emitted program, never fitted.

FIVE REFUSALS, EACH OF WHICH WAS A REAL DEFECT SOMEWHERE
--------------------------------------------------------
1. **Only a citable tier contributes.** The target's :class:`MeasurementAuthority` decides, not this
   module. On the corpus here that drops 872 functional-tier observations and keeps the RTL ones, and
   the functional tier is the one that overcounts.
2. **Only a PASSING stage contributes.** A cycle count from a stage that failed is timing evidence
   about a program that did not compute the declared operation. One workload carries 2 cycles from a
   failing stage beside 6349 and 19928 from passing ones; pooling them reports a 9964x "spread".
3. **A program is identified by its CONTENT.** Hashing a path counts one program measured twenty
   times as twenty programs and inflates every count downstream.
4. **A buffer measured twice is more evidence, not spoiled evidence -- but never an average.** A
   command buffer is the INPUT to code generation, so two submissions can compile one buffer into
   different programs and legitimately measure 311, 316 and 317 cycles. A containment test must
   discard that (there is no single measurement to test against); a SLOWEST-rate ceiling must not,
   because the slowest of the three is a real observation of how slow this class runs. So the
   conservative measurement is used, the spread travels on the rate, and nothing is averaged --
   a mean of 311/316/317 describes no run that happened. Measured here: discarding instead of
   using the slowest threw away 42 of 84 programs, half the evidence on disk.
5. **A partial price cannot make a ceiling.** If any command in the program has no work-counting rule
   the priced MACs are a lower bound, and dividing a lower bound by cycles understates the rate --
   which would make the ceiling built from it too tight, i.e. not a ceiling. Those programs are
   refused rather than included pessimistically.

A class with no qualifying program gets NO rate and a reason. There is no default, because a default
rate is exactly the "scored well on the strength of having no data" failure the routing cost model
already refuses.

WHAT PHASE 1 CAN AND CANNOT PRICE, MEASURED
-------------------------------------------
The table's coverage is bounded by the OPCODES the certified corpus actually emits, and that is much
narrower than the capsule names suggest. Over every citable RTL observation on this repo's own disk --
84 programs across 30 workloads -- the emitted opcodes are exactly ``MATMUL_RESIDENT``, ``COMMIT``,
``RES_PACK``, ``EVICT`` and ``VECTOR_MAP``. There is no ``CONV2D``, no ``ATTENTION_QK``, no
``ATTENTION_PV``, no ``BATCHED_MATMUL`` and not one plain ``MATMUL``. So **78 of 84 programs are one
compute class and the other six have none**, and the per-class split that took bands from 95.7x to
18.9x has a single class to work with.

That is not a classifier defect. A capsule named for a convolution lowers THROUGH im2col: the buffer
for ``GH1_conv2d_i8_hidden`` carries ``params.im2col_recipes`` and emits
``RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT``, and attention projections lower to matmuls the same
way. At the command-buffer level this target's whole certified corpus is resident matmul.

The consequence is the load-bearing one, and it is why :attr:`RateTable.unpriced_classes` is reported
rather than left as an absence: **phase 2 can only price the classes phase 1 emits.** Certifying more
LARGE capsules does not by itself widen the table -- certifying capsules that emit a DIFFERENT
compute class does. A corpus whose every member lowers to one opcode gives an instrument with one
reading, however many members it has.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ClassRate", "Program", "RateTable", "observed_programs", "rates_for", "program_digest"]

#: The emitted program, relative to the run directory that certified it.
_BUFFER_RELATIVE = ("generated", "command_buffer.json")
#: The per-capsule result the measurement authority is applied to.
_RESULT_NAME = "capsule_result.json"
#: The quantity a rate is built from -- the WHOLE-PROGRAM cycle count, spelled as
#: :mod:`merlin.perf.harvest` spells it when it reads a tier's own count. Other harvested quantities
#: share the unit ``cycles`` while measuring something else entirely (a per-unit busy count, a
#: ``scheduled_delay.<mnemonic>`` gap), so filtering on the UNIT admits them and silently prices a
#: program by one engine's occupancy. Matched on the quantity, never on the unit.
_TOTAL_CYCLES = "total_cycles"
_CYCLES_UNIT = "cycles"
_PASS = "pass"


#: Keys whose values are a filesystem path or a workload identity. WITHHELD by default when a table
#: is serialized, and the reason is a measured leak rather than caution: a sibling cost fit emitted the
#: run paths its samples came from, those runs include the grading passes over the HELD-OUT capsules,
#: and a writer that embedded the dict verbatim published ten holdout capsule names and 238 local
#: absolute paths into the tree every graded arm can read. This table is harvested from exactly the
#: same runs, so it carries exactly the same hazard.
#:
#: Two separate rules are being obeyed. An answer key must never reach a graded agent -- a run
#: directory named `..._hidden` or `_holdout_...` names a holdout capsule in its path. And a public
#: artifact carries no local absolute paths. Withholding satisfies both, and a caller doing local
#: diagnosis can still ask for them.
_PROVENANCE_KEYS = ("where", "source", "sources", "workload", "slowest_from",
                    "submissions", "submission", "slowest_from_workload")


def _redact_rows(rows: "Sequence[Mapping[str, Any]]",
                 include_provenance: bool) -> list[dict[str, Any]]:
    """Rows with every path- or identity-bearing field dropped unless explicitly requested.

    The REASON survives redaction. A refusal that keeps its reason and loses its location still tells
    a reader what could not contribute and why, which is the part that is actionable; the path is only
    useful to someone standing on this filesystem, and that someone can pass the flag.
    """
    if include_provenance:
        return [dict(r) for r in rows]
    return [{k: v for k, v in r.items() if k not in _PROVENANCE_KEYS} for r in rows]


def program_digest(buffer: Mapping[str, Any]) -> str:
    """Identity of the emitted PROGRAM, independent of where it was written.

    Commands and tensors only: a run directory differs per run, so hashing the file would count one
    program measured twenty times as twenty programs.
    """
    body = [[row.get("opcode"), row.get("operands"), row.get("attributes")]
            for row in (buffer.get("commands") or []) if isinstance(row, Mapping)]
    payload = {"commands": body, "tensors": buffer.get("tensors")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class Program:
    """One emitted program, its measured cycle counts, and where they came from."""

    digest: str
    buffer: Mapping[str, Any]
    workload: str
    #: Every distinct cycle count observed for this program. More than one is a DISAGREEMENT.
    measured: set[float] = field(default_factory=set)
    submissions: set[str] = field(default_factory=set)
    tiers: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)

    @property
    def agrees(self) -> bool:
        """Did every run of this buffer measure the same? Informational -- not an admission test."""
        return len(self.measured) == 1

    @property
    def cycles(self) -> float | None:
        """The SLOWEST cycle count observed for this buffer, which is the conservative one.

        A rate built from it is the smallest rate the evidence supports, and a ceiling built from the
        smallest rate is the loosest -- erring in the only direction a ceiling may err.
        """
        return max(self.measured) if self.measured else None


@dataclass(frozen=True)
class ClassRate:
    """The slowest measured rate for one compute class, with everything needed to distrust it."""

    compute_class: str
    slowest_macs_per_cycle: float
    slowest_from: str
    fastest_macs_per_cycle: float
    n_programs: int
    cycles_min: float
    cycles_max: float

    def to_dict(self, *, include_provenance: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"compute_class": self.compute_class,
                "slowest_macs_per_cycle": self.slowest_macs_per_cycle,
                "fastest_macs_per_cycle": self.fastest_macs_per_cycle,
                "n_programs": self.n_programs,
                "cycles_min": self.cycles_min, "cycles_max": self.cycles_max,
                "licence": ("an EMPIRICAL bound over the cycle domain stated here; a program outside "
                            "that domain is not covered by this rate and must not be priced with it")}
        if include_provenance:
            out["slowest_from"] = self.slowest_from
        return out


@dataclass
class RateTable:
    """Per-class rates for one target, plus every source that could have contributed and did not."""

    target: str
    peak_macs_per_cycle: float
    rates: dict[str, ClassRate] = field(default_factory=dict)
    refusals: list[dict[str, Any]] = field(default_factory=list)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    n_programs_seen: int = 0

    def rate_for(self, compute_class: str | None) -> float | None:
        """The slowest measured rate for ``compute_class``, or ``None`` -- never a default."""
        entry = self.rates.get(str(compute_class or ""))
        return entry.slowest_macs_per_cycle if entry else None

    @property
    def unpriced_classes(self) -> tuple[str, ...]:
        """Compute classes this target's certified runs never emitted, so nothing can price them.

        Reported rather than left as an absence: a caller reading only ``rates`` sees a table that
        looks complete for whatever it happens to contain, and the difference between "this class is
        fast" and "no certified program of this class exists" is the whole question.
        """
        from merlin.perf import compose_estimate as CE

        return tuple(k for k in CE.COMPUTE_CLASSES if k not in self.rates)

    def to_dict(self, *, include_provenance: bool = False) -> dict[str, Any]:
        return {"target": self.target, "peak_macs_per_cycle": self.peak_macs_per_cycle,
                "rates": {k: v.to_dict(include_provenance=include_provenance)
                          for k, v in sorted(self.rates.items())},
                "n_programs_seen": self.n_programs_seen,
                "n_classes_rated": len(self.rates),
                "unpriced_classes": list(self.unpriced_classes),
                "disagreements": _redact_rows(self.disagreements, include_provenance),
                "refusals": _redact_rows(self.refusals, include_provenance),
                "provenance": {
                    "derived_from": "phase-1 certification runs, via merlin.perf.harvest",
                    "kind": "trace_derived",
                    "note": ("contended: other engines were live in the same window, so each rate is "
                             "an upper bound on how slow this class has been seen to run, never an "
                             "isolated cost and never promotable to a constant")}}


def _run_dirs(roots: Sequence[Path]):
    for root in roots:
        if root.is_dir():
            yield from sorted(root.rglob(_RESULT_NAME))


def observed_programs(target: str, *, authority: Any = None,
                      roots: Sequence[Path] | None = None) -> tuple[dict[str, Program], list[dict]]:
    """``({digest: Program}, refusals)`` over every certification run this target owns.

    The tier gate is :mod:`merlin.perf.harvest`'s, applied through the target's declared authority, so
    citability is decided in exactly one place. A run whose result cannot be paired with the program
    it measured contributes nothing and says so.
    """
    from merlin.kernels.measurement import authority_for
    from merlin.perf import harvest as HV

    auth = authority if authority is not None else authority_for(target)
    if roots is None:
        roots = HV.discover_roots(target)
    roots = [Path(r) for r in roots]

    found: dict[str, Program] = {}
    refusals: list[dict[str, Any]] = []
    for result_path in _run_dirs(roots):
        run_dir = result_path.parent
        buffer_path = run_dir.joinpath(*_BUFFER_RELATIVE)
        if not buffer_path.is_file():
            # NOT an error and not silent: a run that certified a capsule without keeping its emitted
            # program is a real gap in the record, and it is the difference between "this program has
            # no measurement" and "this measurement has no program".
            refusals.append({"what": "emitted program", "where": str(run_dir),
                             "reason": "the run kept no command buffer, so its cycles cannot be "
                                       "attributed to a program"})
            continue
        try:
            buffer = json.loads(buffer_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            refusals.append({"what": "emitted program", "where": str(buffer_path),
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        observations, refused = HV.harvest_capsule_result(result_path, authority=auth)
        refusals.extend(r.to_dict() for r in refused)
        digest = program_digest(buffer)
        for obs in observations:
            if obs.quantity != _TOTAL_CYCLES or obs.unit != _CYCLES_UNIT:
                continue
            if obs.status and obs.status != _PASS:
                refusals.append({"what": "cycle count", "where": str(result_path),
                                 "reason": f"the stage reached status {obs.status!r}; a cycle count "
                                           f"from a stage that did not compute the declared "
                                           f"operation is not poolable with one that did"})
                continue
            program = found.setdefault(digest, Program(digest=digest, buffer=buffer,
                                                       workload=obs.workload or run_dir.name))
            program.measured.add(float(obs.value))
            program.submissions.add(obs.submission)
            program.tiers.add(obs.tier)
            program.sources.append(str(result_path))
    return found, refusals


def rates_for(target: str, *, peak_macs_per_cycle: float, authority: Any = None,
              roots: Sequence[Path] | None = None,
              programs: Mapping[str, Program] | None = None) -> RateTable:
    """The slowest measured MACs-per-cycle per compute class, over this target's certified runs.

    ``peak_macs_per_cycle`` is the target's DERIVED structural peak and is required: it is what turns
    a command buffer into a priced MAC demand, and inventing one here would make every rate a function
    of a number nobody measured.
    """
    from merlin.perf import compose_estimate as CE

    if not peak_macs_per_cycle or peak_macs_per_cycle <= 0:
        raise ValueError("rates_for needs the target's derived structural peak; a rate table built "
                         "on an assumed peak describes a machine nobody has")

    refusals: list[dict[str, Any]] = []
    if programs is None:
        programs, refusals = observed_programs(target, authority=authority, roots=roots)

    table = RateTable(target=str(target), peak_macs_per_cycle=float(peak_macs_per_cycle),
                      refusals=refusals, n_programs_seen=len(programs))
    per_class: dict[str, list[tuple[float, str, float]]] = {}
    for program in programs.values():
        if not program.measured:
            # NOT a disagreement -- absent evidence. Collapsing the two would report a program
            # nobody measured as one whose measurements conflict.
            continue
        if not program.agrees:
            # RECORDED AND USED, NEVER AVERAGED. One buffer compiled by two submissions is two
            # programs; the slowest is the conservative evidence and the spread is what a reader
            # needs to judge it.
            table.disagreements.append({
                "workload": program.workload, "digest": program.digest[:12],
                "measured": sorted(program.measured),
                "submissions": sorted(program.submissions),
                "used": max(program.measured),
                "reason": ("one buffer measured several cycle counts -- the buffer is the input to "
                           "code generation, so these are different emitted programs; the slowest "
                           "is used and none is averaged")})
        cycles = program.cycles
        if not cycles or cycles <= 0:
            continue
        klass = CE.compute_class(program.buffer)
        if klass is None:
            refusals.append({"what": "compute class", "where": program.workload,
                             "reason": "the program declares no opcode from the priced vocabulary"})
            continue
        floor = CE._structural_floor(program.buffer, peak_macs_per_cycle)  # noqa: SLF001 -- one pricer
        if floor.get("status") != CE.DERIVED:
            refusals.append({"what": "priced work", "where": program.workload,
                             "reason": str(floor.get("reason") or "the buffer prices no work")})
            continue
        if not floor.get("counts_every_command"):
            # A LOWER-BOUND PRICE MAKES THE RATE LOOK SLOWER THAN IT IS, and a ceiling built from a
            # too-slow rate is too tight -- which is to say, not a ceiling. The asymmetry is why this
            # refuses where the structural floor happily accepts the same program.
            refusals.append({"what": "priced work", "where": program.workload,
                             "reason": "some commands have no work-counting rule, so the priced MACs "
                                       "are a lower bound and the rate derived from them would "
                                       "understate how fast this class runs"})
            continue
        per_class.setdefault(klass, []).append((floor["macs"] / cycles, program.workload, cycles))

    for klass, observed in per_class.items():
        rate, workload, _ = min(observed)
        table.rates[klass] = ClassRate(
            compute_class=klass, slowest_macs_per_cycle=rate, slowest_from=workload,
            fastest_macs_per_cycle=max(o[0] for o in observed), n_programs=len(observed),
            cycles_min=min(o[2] for o in observed), cycles_max=max(o[2] for o in observed))
    return table

def holdout_containment(target: str, *, peak_macs_per_cycle: float, authority: Any = None,
                        roots: Sequence[Path] | None = None,
                        programs: Mapping[str, Program] | None = None,
                        include_provenance: bool = False) -> dict[str, Any]:
    """Do rates derived from HALF the programs produce bands containing the other half?

    This is the acceptance gate on using a rate table to price anything, and it is the same reasoning
    that condemned every cheap ordering signal in this tree: a signal is exposed on measured
    agreement, never on the plausibility of its derivation. Deriving and testing on one set would
    report how well a bound covers the very data that set it.

    The two miss directions are reported separately and never averaged. A measurement BELOW the floor
    means the floor is not a floor -- a mis-derived peak, or work the counter did not see. One ABOVE
    the ceiling means the slowest observed rate was not slow enough. They are different defects and
    one combined rate would hide both.

    The split is deterministic on the program digest, so the same corpus always yields the same
    verdict; a random split would let a rerun launder a failure into a pass.
    """
    from merlin.perf import compose_estimate as CE

    if programs is None:
        programs, _ = observed_programs(target, authority=authority, roots=roots)
    usable = {d: p for d, p in programs.items() if p.measured}
    # Deterministic halves: the digest is already a hash, so its low bit is a fair coin that does not
    # move between runs.
    train = {d: p for d, p in usable.items() if int(d[-1], 16) % 2 == 0}
    test = {d: p for d, p in usable.items() if int(d[-1], 16) % 2 == 1}

    table = rates_for(target, peak_macs_per_cycle=peak_macs_per_cycle, programs=train)
    below = above = inside = 0
    widths: list[float] = []
    undecided: list[dict[str, Any]] = []
    for program in test.values():
        klass = CE.compute_class(program.buffer)
        band = CE.band(program.buffer, target=target, peak_macs_per_cycle=peak_macs_per_cycle,
                       slowest_macs_per_cycle=table.rate_for(klass))
        if band.get("status") != CE.DERIVED:
            undecided.append({"workload": program.workload, "compute_class": klass,
                              "reason": str(band.get("reason") or "band not derived")})
            continue
        if band.get("lower"):
            widths.append(float(band["upper"]) / float(band["lower"]))
        measured = program.cycles
        if measured < band["lower"]:
            below += 1
        elif measured > band["upper"]:
            above += 1
        else:
            inside += 1
    decided = below + above + inside
    widths.sort()
    median_width = (widths[len(widths) // 2] if len(widths) % 2
                    else (widths[len(widths) // 2 - 1] + widths[len(widths) // 2]) / 2) if widths else None
    return {"target": target, "peak_macs_per_cycle": peak_macs_per_cycle,
            "median_band_width": median_width,
            "width_note": ("CONTAINMENT ALONE IS CHEAP: a band wide enough contains everything, so the "
                           "width is half the result. A band must be about an order of magnitude to "
                           "separate anything, and at this width a difference smaller than it is "
                           "invisible -- which is what the band may not be used to deny"),
            "n_train": len(train), "n_test": len(test), "n_decided": decided,
            "contained": inside, "below_floor": below, "above_ceiling": above,
            "containment_rate": (inside / decided) if decided else None,
            "n_undecided": len(undecided),
            # An undecided row names the WORKLOAD it could not decide, and a held-out capsule's name
            # is an answer key. Redacted on the same terms as the table's own rows.
            "undecided": _redact_rows(undecided[:20], include_provenance),
            "rates_used": {k: v.to_dict(include_provenance=include_provenance)
                           for k, v in sorted(table.rates.items())},
            "unpriced_classes": list(table.unpriced_classes),
            "licence": ("a containment rate over programs the rates were NOT derived from; a band "
                        "may eliminate a candidate and may never certify one")}
