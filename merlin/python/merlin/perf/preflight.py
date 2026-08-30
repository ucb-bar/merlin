"""What a workload will cost, and whether it may be scheduled at all -- decided before any oracle runs.

:mod:`merlin.targetgen.oracle_schedule` decides *which* oracle queries buy the most information per
second. This is its performance analogue and it answers the question one step earlier: given only a
workload's DECLARED shapes and dtypes, how much work is it, how long will each tier take to answer, and
**is it even legal to ask**. Like that module it is deliberately pure -- no I/O, no oracle, no
simulator, no target lookup -- so the whole policy is testable and costs nothing to run.

**Why the refusals are the point.** Two hazards on a staged-program accelerator do not raise, do not
warn, and do not fail: they return a *number*, and the number is wrong.

* **The finite DRAM window.** The program runner reduces every address modulo a power-of-two window.
  A tensor larger than the window aliases onto itself; two tensors whose reduced spans overlap corrupt
  each other. The run completes and reports cycles for a computation nobody asked for.
  :func:`~merlin.perf.workload_gen.alias_report` has computed exactly this since it was written and
  **is not called from the grading path**, so every layer-scale grade to date has been unchecked.
* **Instruction memory.** A program longer than the instruction memory is not rejected -- its tail is
  never loaded. The device executes the prefix and halts, and the cycle count describes the prefix.

Both become pre-flight refusals here, at zero cost, instead of results that have to be disbelieved
later. A caller that gets :attr:`Preflight.ok` False must not schedule the workload.

**Everything numeric carries how much it is worth.** Three separate honesty rules, each of which has
already cost this repo a wrong number:

1. **The cycle rate is an extrapolation until at least two points fit it.** The rate available today
   comes from ONE datapoint (a layer that ran 3,300,328 cycles over 4,394 tile passes = 751.1
   cyc/pass), and one point cannot separate a rate from a fixed fill/drain intercept -- the repo's
   at-least-two-points-per-fitted-parameter rule. :class:`Rate` therefore records its basis and
   :attr:`Rate.is_extrapolation` is True for the single-point form, so a projection built on it can be
   labelled rather than quoted. It is also not a decomposition: of those 751 cycles the contraction
   unit is busy for only ~158, and the movement engine's share varies from 60% to 93.7% across the
   reference corpus, so the rate is a summary of a mix and not a law about the machine.
2. **Wall time needs the word term, not just the cycle term.** ``seconds = a + b*cycles + c*words``
   (:mod:`merlin.perf.oracle_cost`). A cycles-only fit has nowhere to put the program-load cost and
   charges it to the cycles: measured, that overstated one tier's marginal per-cycle rate by **1.77x**
   with an r2 of 0.97, so the bad fit is not detectable from fit quality. Projections here go through
   :meth:`~merlin.perf.oracle_cost.CostLaw.estimate`, which keeps the terms separate and reports how
   far past the measured domain the projection reaches.
3. **A check that could not run is not a pass.** An unknown DRAM window or an unknown instruction-memory
   capacity produces a refusal whose code says UNCHECKED, exactly as
   :class:`~merlin.perf.workload_gen.AliasReport` already does. Silence would be indistinguishable from
   a clean footprint.

**No fact about a machine is written down here.** Tile geometry, element widths, the DRAM window, the
instruction-memory capacity and the emitter's own section lengths all arrive as
:class:`MachineBudget` / :class:`EmitterShape`, built by the caller from that target's derived facts
(:func:`merlin.perf.workload_gen.machine_facts` supplies every one of them). This module never asks
which target it is looking at.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .amplification import TensorOperand, useful_bytes
from .oracle_cost import CostEstimate, CostLaw
from .workload_gen import AliasReport, Placement, alias_report

__all__ = [
    "RateBasis", "Rate", "rate_from_observations",
    "MachineBudget", "EmitterShape", "TensorSpec",
    "Refusal", "Preflight",
    "operands_from_declaration", "place_operands", "tile_passes_for",
    "preflight_operands", "preflight_matmul", "render",
]


# --- the cycle rate, with what it is worth ----------------------------------------------------------
class RateBasis(str, Enum):
    """How much a cycles-per-tile-pass number is worth."""

    #: fitted from two or more (tile_passes, cycles) points, so the rate and the fixed term are
    #: separately identified.
    FITTED = "FITTED"
    #: a single point divided through. It contains whatever fixed fill/drain that one workload paid and
    #: cannot say how much -- usable as a projection, never as a law.
    SINGLE_POINT_EXTRAPOLATION = "SINGLE_POINT_EXTRAPOLATION"
    #: no observation at all. NOT zero: a caller must handle its absence.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Rate:
    """``cycles = fixed + per_tile_pass * passes`` for one emitter on one machine.

    ``fixed`` is ``None`` for the single-point form, because one observation cannot separate an
    intercept from a slope. That is not a formality: pipeline fill and drain are intercepts and a
    rate-only model mispredicts every small workload, which is most of a capsule corpus.
    """

    per_tile_pass: float | None
    fixed: float | None
    basis: RateBasis
    n_points: int
    domain: tuple[int, int] | None = None      # (min, max) tile passes actually observed
    note: str = ""

    @property
    def known(self) -> bool:
        return self.basis is not RateBasis.UNKNOWN and self.per_tile_pass is not None

    @property
    def is_extrapolation(self) -> bool:
        return self.basis is RateBasis.SINGLE_POINT_EXTRAPOLATION

    def cycles(self, tile_passes: int) -> int | None:
        """Projected cycles for ``tile_passes``, or ``None`` when the rate is UNKNOWN."""
        if not self.known:
            return None
        return int(round((self.fixed or 0.0) + self.per_tile_pass * int(tile_passes)))

    def reach(self, tile_passes: int) -> float | None:
        """How far past the largest observed workload this projection reaches. ``> 1`` is beyond the
        evidence; ``None`` when nothing was observed."""
        if not self.domain or self.domain[1] <= 0:
            return None
        return float(tile_passes) / float(self.domain[1])

    def as_dict(self) -> dict:
        return {"per_tile_pass": self.per_tile_pass, "fixed": self.fixed, "basis": self.basis.value,
                "n_points": self.n_points, "is_extrapolation": self.is_extrapolation,
                "domain": list(self.domain) if self.domain else None, "note": self.note}


def rate_from_observations(observations: Sequence[tuple[int, int]], *, note: str = "") -> Rate:
    """Derive a :class:`Rate` from ``(tile_passes, cycles)`` observations.

    Two or more DISTINCT tile-pass counts fit a slope and an intercept and the result is ``FITTED``.
    Exactly one point (or several at the same count) can only be divided through: the result carries
    ``SINGLE_POINT_EXTRAPOLATION`` and a ``None`` intercept, so nothing downstream can read a fill term
    that was never measured. No points is ``UNKNOWN`` -- never a default rate.
    """
    pts = [(int(p), int(c)) for p, c in observations if int(p) > 0]
    if not pts:
        return Rate(None, None, RateBasis.UNKNOWN, 0, None,
                    note or "no (tile_passes, cycles) observation was supplied")
    passes = [p for p, _ in pts]
    domain = (min(passes), max(passes))
    if len(set(passes)) < 2:
        mean_cycles = sum(c for _, c in pts) / len(pts)
        per = mean_cycles / passes[0]
        return Rate(per, None, RateBasis.SINGLE_POINT_EXTRAPOLATION, len(pts), domain,
                    note or (f"{len(pts)} observation(s) at a single tile-pass count ({passes[0]}); the "
                             f"fixed fill/drain term is folded into the rate and cannot be separated"))
    n = len(pts)
    mx = sum(passes) / n
    my = sum(c for _, c in pts) / n
    sxx = sum((p - mx) ** 2 for p in passes)
    slope = sum((p - mx) * (c - my) for p, c in pts) / sxx
    intercept = my - slope * mx
    return Rate(slope, intercept, RateBasis.FITTED, n, domain,
                note or f"least squares over {n} points at {len(set(passes))} distinct tile-pass counts")


# --- the machine's capacities, all supplied by the caller -------------------------------------------
@dataclass(frozen=True)
class MachineBudget:
    """The capacities a preflight checks a workload against.

    Every field is a DERIVED fact about one target and none has a default. ``dram_window`` and
    ``imem_words`` are ``None`` when the target's own sources do not publish them, which produces an
    UNCHECKED refusal rather than a silent pass.
    """

    tile_rows: int
    tile_cols: int
    operand_bytes: int
    accum_bytes: int
    dram_window: int | None
    imem_words: int | None
    dram_base: int = 0
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def tile_register_bytes(self) -> int:
        return self.tile_rows * self.tile_cols

    @classmethod
    def from_machine_facts(cls, facts: Any) -> "MachineBudget":
        """Build from a :class:`merlin.perf.workload_gen.MachineFacts`, whose every field is derived
        from the target's own RTL / manifest / shipped ISA reference."""
        return cls(tile_rows=facts.tile.rows, tile_cols=facts.tile.cols,
                   operand_bytes=facts.operand_bytes, accum_bytes=facts.accum_bytes,
                   dram_window=facts.dram_window, imem_words=facts.imem_words,
                   dram_base=int(getattr(facts, "dram_base", 0) or 0),
                   provenance=dict(getattr(facts, "provenance", {}) or {}))

    def as_dict(self) -> dict:
        return {"tile_rows": self.tile_rows, "tile_cols": self.tile_cols,
                "operand_bytes": self.operand_bytes, "accum_bytes": self.accum_bytes,
                "dram_window": self.dram_window, "imem_words": self.imem_words,
                "dram_base": self.dram_base, "provenance": dict(self.provenance)}


@dataclass(frozen=True)
class EmitterShape:
    """How an emitter's PROGRAM LENGTH responds to the shape, measured from a program it already emitted.

    ``looped_words`` is what the emitter actually produces -- constant in the shape, because the axes are
    real backward branches. ``unrolled_words`` is its control-flow-free twin, and the gap between them is
    the entire argument for the loops: on one measured layer the same schedule was 96 words looped and
    108,011 unrolled, which overflows a 32,768-word instruction memory 3.3x.

    Supply this from :attr:`merlin.perf.workload_gen.MatmulPlan.section_words` (its keys are exactly the
    three section fields below) so the projection is counted from an emitted program rather than guessed.
    """

    prologue_words: int
    k_step_words: int
    tile_epilogue_words: int
    looped_words: int
    provenance: str = ""

    @classmethod
    def from_section_words(cls, section_words: Mapping[str, int], *, provenance: str = "") -> "EmitterShape":
        return cls(prologue_words=int(section_words.get("prologue", 0)),
                   k_step_words=int(section_words.get("k_step", 0)),
                   tile_epilogue_words=int(section_words.get("tile_epilogue", 0)),
                   looped_words=int(section_words.get("total", 0)),
                   provenance=provenance or "measured section lengths of an emitted program")

    def unrolled_words(self, m_tiles: int, k_tiles: int, n_tiles: int) -> int:
        """The same count :meth:`merlin.perf.workload_gen.MatmulPlan.unrolled_word_estimate` produces."""
        return (self.prologue_words
                + m_tiles * k_tiles * n_tiles * self.k_step_words
                + m_tiles * n_tiles * self.tile_epilogue_words)

    def as_dict(self) -> dict:
        return {"prologue_words": self.prologue_words, "k_step_words": self.k_step_words,
                "tile_epilogue_words": self.tile_epilogue_words, "looped_words": self.looped_words,
                "provenance": self.provenance}


@dataclass(frozen=True)
class TensorSpec:
    """One DECLARED tensor: the shape and dtype a capsule states, before anything is scheduled.

    This is the input that closes the half-hand-entered hole in :mod:`merlin.perf.amplification`: the
    measurement artifact carries moved bytes but no shape, so ``useful_bytes`` had to be typed in by
    hand and the amplification ratios could not be reproduced from the artifact. A capsule DOES declare
    ``inputs[].shape`` and ``inputs[].dtype``, so the useful side becomes derivable from the corpus.
    """

    name: str
    shape: tuple[int, ...]
    element_bytes: float
    role: str = "input"
    broadcast: bool = False

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self) -> int:
        return int(round(self.elements * self.element_bytes))

    def operand(self) -> TensorOperand:
        return TensorOperand(name=self.name, elements=self.elements,
                             element_bytes=self.element_bytes,
                             is_output=(self.role == "output"), broadcast=self.broadcast)


def operands_from_declaration(inputs: Sequence[Mapping[str, Any]], *,
                              element_bytes_of) -> tuple[TensorSpec, ...]:
    """Declared tensors (a capsule's ``inputs`` list) as :class:`TensorSpec`.

    ``element_bytes_of(dtype) -> float`` is the caller's, because the numeric-format vocabulary is DATA
    (the quant-format registry), not a table this module may hold. A dtype it cannot size raises there
    rather than being silently sized as one byte.
    """
    out: list[TensorSpec] = []
    for item in inputs:
        name = str(item.get("name") or "")
        shape = tuple(int(d) for d in (item.get("shape") or ()))
        if not name or not shape:
            raise ValueError(f"declared tensor {item!r} has no name or no shape; it cannot be sized")
        dtype = str(item.get("dtype") or "")
        out.append(TensorSpec(name=name, shape=shape, element_bytes=float(element_bytes_of(dtype)),
                              role=str(item.get("role") or "input")))
    return tuple(out)


def place_operands(specs: Sequence[TensorSpec], *, origin: int, align: int) -> tuple[Placement, ...]:
    """Lay the declared tensors out end to end from ``origin``, each aligned to ``align``.

    This is the layout rule :func:`merlin.perf.workload_gen.plan_matmul` uses (sequential, aligned,
    absolute addresses computed once), reproduced over declared shapes so the footprint can be checked
    BEFORE a plan exists. It is a placement HYPOTHESIS: a caller that already has a plan should pass
    ``plan.placements`` straight to :func:`preflight_operands` instead.
    """
    cur = int(origin)
    out: list[Placement] = []
    for s in specs:
        cur = (cur + align - 1) // align * align
        out.append(Placement(name=s.name, role=s.role, shape=list(s.shape), dtype="", nbytes=s.nbytes,
                             base=cur))
        cur += s.nbytes
    return tuple(out)


def tile_passes_for(m: int, k: int, n: int, budget: MachineBudget) -> tuple[int, int, int]:
    """``(m_tiles, k_tiles, n_tiles)`` for a contraction on this budget's array.

    Raises when an extent is not a whole number of tiles -- the same refusal
    :func:`merlin.perf.workload_gen.plan_matmul` makes, brought forward to cost nothing.
    """
    for label, extent, edge in (("M", m, budget.tile_rows), ("K", k, budget.tile_cols),
                                ("N", n, budget.tile_cols)):
        if extent <= 0 or extent % edge:
            raise ValueError(f"{label}={extent} is not a whole number of {edge}-wide tiles")
    return m // budget.tile_rows, k // budget.tile_cols, n // budget.tile_cols


# --- the refusals -----------------------------------------------------------------------------------
#: A footprint that provably aliases inside the runner's window: the run WOULD return a number, for a
#: computation nobody asked for.
DRAM_ALIAS = "dram_alias"
#: The window size is not published by the runner, so the footprint COULD NOT BE CHECKED.
DRAM_WINDOW_UNKNOWN = "dram_window_unchecked"
#: The program is longer than instruction memory. Its tail is never loaded and the device runs the prefix.
IMEM_OVERFLOW = "imem_overflow"
#: The instruction-memory capacity is not derivable, or no program length was supplied to check.
IMEM_UNCHECKED = "imem_unchecked"
#: An extent is not a whole number of tiles, so the contraction cannot be emitted faithfully.
PARTIAL_TILE = "partial_tile"
#: No cycle rate was supplied, so no projection exists to budget against.
RATE_UNKNOWN = "rate_unknown"

#: Codes that mean "a hazard was PROVEN", as opposed to "a check could not run".
PROVEN_CODES = frozenset({DRAM_ALIAS, IMEM_OVERFLOW, PARTIAL_TILE})


@dataclass(frozen=True)
class Refusal:
    """One reason this workload must not be scheduled."""

    code: str
    detail: str

    @property
    def proven(self) -> bool:
        """True for a hazard that was demonstrated; False for a check that could not run.

        Both refuse. They are distinguished because they call for different actions -- shrink the
        workload versus make the fact derivable -- and collapsing them reads as if every unchecked
        workload were broken."""
        return self.code in PROVEN_CODES

    def as_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail, "proven": self.proven}


@dataclass(frozen=True)
class Preflight:
    """What the workload costs and whether it may be scheduled -- decided from declarations alone."""

    workload: str
    useful_bytes: int
    broadcast_bytes: int
    footprint_bytes: int
    alias: AliasReport | None
    tiles: tuple[int, int, int] | None
    tile_passes: int | None
    projected_cycles: int | None
    rate: Rate
    program_words: int | None
    unrolled_words: int | None
    imem_words: int | None
    refusals: tuple[Refusal, ...]
    #: tier -> projected wall time, each carrying its own concurrency, term split and extrapolation.
    wall: Mapping[str, CostEstimate] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True only when every check RAN and every one passed. An unchecked hazard is not a pass."""
        return not self.refusals

    @property
    def proven_hazards(self) -> tuple[Refusal, ...]:
        return tuple(r for r in self.refusals if r.proven)

    @property
    def unchecked(self) -> tuple[Refusal, ...]:
        return tuple(r for r in self.refusals if not r.proven)

    @property
    def cheapest_tier(self) -> str | None:
        priced = {t: e.seconds for t, e in self.wall.items() if not e.is_lower_bound}
        return min(priced, key=priced.get) if priced else None

    def as_dict(self) -> dict:
        return {
            "workload": self.workload, "ok": self.ok,
            "useful_bytes": self.useful_bytes, "broadcast_bytes": self.broadcast_bytes,
            "footprint_bytes": self.footprint_bytes,
            "alias": {"ok": self.alias.ok, "window": self.alias.window,
                      "wrapped": list(self.alias.wrapped),
                      "collisions": [list(c) for c in self.alias.collisions],
                      "reason": self.alias.reason} if self.alias else None,
            "tiles": list(self.tiles) if self.tiles else None,
            "tile_passes": self.tile_passes,
            "projected_cycles": self.projected_cycles,
            "cycles_are_an_extrapolation": self.rate.is_extrapolation,
            "rate": self.rate.as_dict(),
            "program_words": self.program_words, "unrolled_words": self.unrolled_words,
            "imem_words": self.imem_words,
            "refusals": [r.as_dict() for r in self.refusals],
            "wall": {t: e.as_dict() for t, e in self.wall.items()},
        }


# --- the decision -----------------------------------------------------------------------------------
def preflight_operands(name: str, specs: Sequence[TensorSpec], *, budget: MachineBudget,
                       rate: Rate | None = None, tile_passes: int | None = None,
                       tiles: tuple[int, int, int] | None = None,
                       placements: Sequence[Placement] | None = None,
                       program_words: int | None = None, unrolled_words: int | None = None,
                       laws: Mapping[str, CostLaw] | None = None,
                       origin: int | None = None, align: int = 64) -> Preflight:
    """Preflight a workload from its declared tensors.

    ``placements`` overrides the layout hypothesis when the caller already has a plan (then the alias
    check is about the addresses that will really be used). ``program_words`` is the length of the
    program that will actually run; without it the instruction-memory check cannot run and says so.
    ``laws`` maps a tier name to its fitted :class:`~merlin.perf.oracle_cost.CostLaw`, and the wall
    projection needs BOTH the cycle projection and a word count -- a law's word term is the one a
    cycles-only fit silently charges to the cycles.
    """
    rate = rate or Rate(None, None, RateBasis.UNKNOWN, 0, None, "no rate supplied")
    refusals: list[Refusal] = []

    ops = tuple(s.operand() for s in specs)
    useful, splat = useful_bytes(ops)

    placed = tuple(placements) if placements is not None else place_operands(
        specs, origin=budget.dram_base + align if origin is None else origin, align=align)
    report = alias_report(placed, budget.dram_window)
    if not report.ok:
        code = DRAM_ALIAS if report.window is not None else DRAM_WINDOW_UNKNOWN
        refusals.append(Refusal(code, report.reason))

    projected = rate.cycles(tile_passes) if tile_passes else None
    if tile_passes and not rate.known:
        refusals.append(Refusal(RATE_UNKNOWN,
                                "no cycles-per-tile-pass observation was supplied, so this workload "
                                "has no projected cost to budget against; " + rate.note))

    cap = budget.imem_words
    if program_words is None:
        refusals.append(Refusal(IMEM_UNCHECKED,
                                "no program length was supplied, so the instruction-memory fit COULD "
                                "NOT BE CHECKED; a program longer than IMEM is not rejected -- its "
                                "tail is never loaded and the device runs the prefix"))
    elif cap is None:
        refusals.append(Refusal(IMEM_UNCHECKED,
                                f"the target publishes no instruction-memory capacity, so a "
                                f"{program_words}-word program COULD NOT BE CHECKED; this is not a pass"))
    elif program_words > cap:
        refusals.append(Refusal(IMEM_OVERFLOW,
                                f"the program is {program_words} words and instruction memory holds "
                                f"{cap}; the tail would never be loaded and the cycle count would "
                                f"describe the prefix"))

    wall: dict[str, CostEstimate] = {}
    if laws and projected is not None and program_words is not None:
        for tier, law in laws.items():
            wall[tier] = law.estimate(projected, program_words)

    return Preflight(workload=name, useful_bytes=useful, broadcast_bytes=splat,
                     footprint_bytes=report.footprint_bytes, alias=report,
                     tiles=tiles, tile_passes=tile_passes, projected_cycles=projected, rate=rate,
                     program_words=program_words, unrolled_words=unrolled_words, imem_words=cap,
                     refusals=tuple(refusals), wall=wall)


def preflight_matmul(name: str, *, m: int, k: int, n: int, budget: MachineBudget,
                     rate: Rate | None = None, emitter: EmitterShape | None = None,
                     loops: bool = True, laws: Mapping[str, CostLaw] | None = None,
                     origin: int | None = None, align: int = 64) -> Preflight:
    """Preflight an ``[m, k] x [k, n]`` contraction from its shape alone.

    The three tensors are sized from the budget's own element widths, laid out the way the emitter lays
    them out, and checked against the window. ``emitter`` supplies the program length: ``loops`` picks
    which of the two lengths is the one that will run, and BOTH are reported, because the unrolled twin
    is the number that decides whether a shape is reachable without control flow at all.
    """
    try:
        tiles = tile_passes_for(m, k, n, budget)
    except ValueError as e:
        empty = Rate(None, None, RateBasis.UNKNOWN, 0, None, "shape refused before any rate applies")
        return Preflight(workload=name, useful_bytes=0, broadcast_bytes=0, footprint_bytes=0,
                         alias=None, tiles=None, tile_passes=None, projected_cycles=None,
                         rate=rate or empty, program_words=None, unrolled_words=None,
                         imem_words=budget.imem_words,
                         refusals=(Refusal(PARTIAL_TILE, str(e)),))
    mt, kt, nt = tiles
    specs = (
        TensorSpec("A", (m, k), float(budget.operand_bytes), role="input"),
        TensorSpec("W", (k, n), float(budget.operand_bytes), role="weight"),
        TensorSpec("C", (m, n), float(budget.accum_bytes), role="output"),
    )
    words = unrolled = None
    if emitter is not None:
        unrolled = emitter.unrolled_words(mt, kt, nt)
        words = emitter.looped_words if loops else unrolled
    return preflight_operands(name, specs, budget=budget, rate=rate, tile_passes=mt * kt * nt,
                              tiles=tiles, program_words=words, unrolled_words=unrolled,
                              laws=laws, origin=origin, align=align)


def render(pf: Preflight) -> str:
    """One human-readable block. A refused workload leads with the refusal, and a projection built on a
    single-point rate says so on the same line as the number."""
    head = "MAY SCHEDULE" if pf.ok else "REFUSED"
    lines = [f"{pf.workload}: {head}"]
    for r in pf.refusals:
        lines.append(f"  [{'hazard' if r.proven else 'unchecked'}] {r.code}: {r.detail}")
    lines.append(f"  useful_bytes={pf.useful_bytes} footprint_bytes={pf.footprint_bytes}"
                 + (f" broadcast_bytes={pf.broadcast_bytes}" if pf.broadcast_bytes else ""))
    if pf.tile_passes is not None:
        tail = ""
        if pf.rate.is_extrapolation:
            tail = " [EXTRAPOLATION from one datapoint — not a law]"
        reach = pf.rate.reach(pf.tile_passes)
        if reach and reach > 1.0:
            tail += f" [x{reach:.3g} beyond the largest observed workload]"
        lines.append(f"  tile_passes={pf.tile_passes} projected_cycles={pf.projected_cycles}{tail}")
    if pf.program_words is not None or pf.unrolled_words is not None:
        lines.append(f"  program_words={pf.program_words} unrolled_words={pf.unrolled_words} "
                     f"imem_words={pf.imem_words}")
    for tier, est in sorted(pf.wall.items()):
        lines.append(f"  {tier}: {est}")
    return "\n".join(lines)
