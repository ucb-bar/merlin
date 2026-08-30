"""What one oracle query COSTS, as a law with its terms separated by construction.

The question this answers is "how long will it take to ask the simulator about a program I have not
run yet", and the answer is a fitted law rather than a single rate, because a query's cost is not one
rate. A simulator charges for *simulated time* (the cycles the program takes to halt) and separately
for *program size* (every word of the program is loaded into the device's instruction memory before
the first one retires, one bus transaction at a time). Two terms, one per axis::

    seconds = a + b * cycles + c * words

**Why a one-term fit is wrong, and wrong in the flattering direction for the wrong tier.** On a real
corpus the two axes are correlated -- a bigger kernel is a longer-running kernel -- so a fit against
cycles alone has nowhere to put the load cost and silently charges it to the cycles. Directly
measured on one substrate: the corpus-only per-cycle slope came out **1.77x** the true marginal rate.
Every projection built on it overstates the cycle cost of a long program and understates the size
cost of a large one, and it is not detectable from the fit quality -- the bad fit's r2 was 0.97.

**How the terms are separated: by construction, not by regression.** The generic move, and it is
target-independent because every stored-program device does this:

    run a program whose FIRST word is the halt.

All W words are still loaded; roughly one instruction retires. Whatever that costs is the load term,
alone, with the cycle term pinned to its floor -- so ``c`` is obtained by measurement rather than
inferred from data in which it is confounded. The mirror construction pins the words and sweeps the
cycles (one program, a loop, a trip count) so ``b`` is isolated the same way. See :class:`ProbeKind`.

**Concurrency travels with every number.** A per-query cost measured while N queries run at once is a
throughput figure, not a latency figure, and the two differ by a lot: the same query on one substrate
here measured 3.7 s serial and 23.4 s under a 16-worker grade -- **6.3x**. A cost quoted without its
concurrency is unfalsifiable, so :class:`CostSample`, :class:`CostLaw` and :class:`CostEstimate` all
require it (no default), fitting refuses to mix concurrencies, and every rendered line prints it.

**Nothing is silently assumed.** Each term carries its own :class:`Provenance`, the construction that
produced it, and the domain it was measured over. A term no construction isolated is ``UNKNOWN`` --
never ``0.0`` -- and an estimate that had to leave it out says so in ``excluded``. Every projection
reports its :attr:`CostEstimate.extrapolation`: how far past the largest thing actually measured it
reaches, per axis.

The module is a modelling tool: it fits and reports. Measurement is the caller's, through the tiny
:class:`Substrate` protocol, so anything that can run a program -- an RTL simulator, an ISA model, a
board -- can be priced by the same code.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

__all__ = [
    "ProbeKind", "Provenance", "CostSample", "Term", "CostLaw", "CostEstimate",
    "Validation", "ConcurrencyReport", "MixedConcurrency", "NotEnoughEvidence",
    "Substrate", "RunOutcome", "Probe",
    "fit_cost_law", "cycles_only_slope", "average_replicates", "concurrency_inflation",
    "halt_first_probes", "measure", "render_law",
]


class MixedConcurrency(ValueError):
    """Samples taken at different concurrencies were pooled into one fit."""


class NotEnoughEvidence(ValueError):
    """A fit was asked for from data that cannot support it (fewer than two distinct points)."""


class ProbeKind(str, Enum):
    """What a sample was constructed to isolate.

    ``LOAD`` and ``CYCLE`` are the two isolating constructions; the whole point of the module is that
    the law is fitted from these rather than from ``CORPUS``, in which the axes are confounded.
    """

    #: words vary, cycles pinned at the floor -- the halt-first program. Isolates the per-word term.
    LOAD = "load"
    #: cycles vary, words pinned -- one program with a swept trip count. Isolates the per-cycle term.
    CYCLE = "cycle"
    #: the smallest possible query: minimum words AND minimum cycles. Measures the fixed term.
    FLOOR = "floor"
    #: a real query. Both axes vary AND ARE CORRELATED, so this is held out for validation.
    CORPUS = "corpus"


class Provenance(str, Enum):
    """How much a term's value is worth."""

    #: fitted from data constructed to isolate this term (or read off a floor probe directly).
    MEASURED = "MEASURED"
    #: fitted after another term was subtracted out -- correct only if that term is correct.
    DERIVED = "DERIVED"
    #: no construction isolated it. NOT zero: a caller must handle its absence explicitly.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CostSample:
    """One timed query. ``concurrency`` has no default: a cost without it is not interpretable."""

    seconds: float
    cycles: int
    words: int
    concurrency: int
    kind: ProbeKind = ProbeKind.CORPUS
    label: str | None = None

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency!r}")
        if self.seconds < 0:
            raise ValueError(f"seconds must be >= 0, got {self.seconds!r}")


@dataclass(frozen=True)
class Term:
    """One coefficient of the law, with what it cost to believe it."""

    name: str
    value: float | None
    unit: str
    provenance: Provenance
    construction: str
    n: int = 0
    r2: float | None = None
    #: the (min, max) of this term's own axis over the samples that fitted it.
    domain: tuple[float, float] | None = None
    note: str | None = None

    @property
    def known(self) -> bool:
        return self.provenance is not Provenance.UNKNOWN and self.value is not None

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "unit": self.unit,
                "provenance": self.provenance.value, "construction": self.construction,
                "n": self.n, "r2": self.r2, "domain": list(self.domain) if self.domain else None,
                "note": self.note}


@dataclass(frozen=True)
class CostEstimate:
    """A projected cost. Carries its concurrency, its term breakdown and its extrapolation."""

    substrate: str
    concurrency: int
    cycles: int
    words: int
    seconds: float
    #: seconds contributed per term; a term that could not be included is absent here and named in
    #: :attr:`excluded`.
    by_term: dict[str, float]
    measured: tuple[str, ...]
    assumed: tuple[str, ...]
    #: terms the law could not supply, so ``seconds`` is a LOWER BOUND missing them.
    excluded: tuple[str, ...]
    #: axis -> requested / largest actually measured on that axis. > 1 means beyond the evidence.
    extrapolation: dict[str, float]

    @property
    def within_measured_domain(self) -> bool:
        return all(f <= 1.0 for f in self.extrapolation.values())

    @property
    def is_lower_bound(self) -> bool:
        return bool(self.excluded)

    def as_dict(self) -> dict:
        return {"substrate": self.substrate, "concurrency": self.concurrency,
                "cycles": self.cycles, "words": self.words, "seconds": self.seconds,
                "by_term": dict(self.by_term), "measured": list(self.measured),
                "assumed": list(self.assumed), "excluded": list(self.excluded),
                "extrapolation": dict(self.extrapolation),
                "within_measured_domain": self.within_measured_domain,
                "is_lower_bound": self.is_lower_bound}

    def __str__(self) -> str:  # concurrency is never optional in the human-readable form either
        tail = ""
        if self.excluded:
            tail += f" [LOWER BOUND — excludes {'+'.join(self.excluded)}]"
        beyond = {a: f for a, f in self.extrapolation.items() if f > 1.0}
        if beyond:
            tail += " [EXTRAPOLATED " + ", ".join(f"{a} x{f:.3g}" for a, f in beyond.items()) + "]"
        return (f"{self.substrate}: {self.seconds:.4g} s for {self.cycles} cycles / "
                f"{self.words} words at concurrency={self.concurrency}{tail}")


@dataclass(frozen=True)
class Validation:
    """How the law does on queries it was not fitted on."""

    n: int
    median_abs_rel_err: float
    max_abs_rel_err: float
    rows: tuple[dict, ...] = ()

    def as_dict(self) -> dict:
        return {"n": self.n, "median_abs_rel_err": self.median_abs_rel_err,
                "max_abs_rel_err": self.max_abs_rel_err, "rows": [dict(r) for r in self.rows]}


@dataclass(frozen=True)
class ConcurrencyReport:
    """What running N at once did to the per-query number."""

    workers: int
    serial_seconds: float
    observed_seconds: float
    inflation_x: float
    note: str

    def as_dict(self) -> dict:
        return {"workers": self.workers, "serial_seconds": self.serial_seconds,
                "observed_seconds": self.observed_seconds, "inflation_x": self.inflation_x,
                "note": self.note}


@dataclass(frozen=True)
class CostLaw:
    """``seconds = fixed + per_cycle*cycles + per_word*words`` for one substrate at one concurrency."""

    substrate: str
    concurrency: int
    fixed: Term
    per_cycle: Term
    per_word: Term
    #: the naive one-term fit, kept so the mistake it embodies is reportable rather than hypothetical.
    cycles_only_per_cycle: float | None = None
    n_samples: int = 0
    notes: tuple[str, ...] = ()

    # -- rates -------------------------------------------------------------------------------------
    @property
    def cycles_per_second(self) -> float | None:
        v = self.per_cycle.value
        return None if not v else 1.0 / v

    @property
    def words_per_second(self) -> float | None:
        v = self.per_word.value
        return None if not v else 1.0 / v

    @property
    def cycles_only_overstatement(self) -> float | None:
        """How much a cycles-only fit overstates the true marginal per-cycle rate."""
        if self.cycles_only_per_cycle is None or not self.per_cycle.value:
            return None
        return self.cycles_only_per_cycle / self.per_cycle.value

    @property
    def measured_domain(self) -> dict[str, float]:
        """Largest value actually measured on each axis -- the edge every projection is judged against."""
        out: dict[str, float] = {}
        if self.per_cycle.domain:
            out["cycles"] = self.per_cycle.domain[1]
        if self.per_word.domain:
            out["words"] = self.per_word.domain[1]
        return out

    # -- projection --------------------------------------------------------------------------------
    def estimate(self, cycles: int, words: int) -> CostEstimate:
        """Project one query, with the measured/assumed split and the extrapolation factors."""
        by_term: dict[str, float] = {}
        measured: list[str] = []
        assumed: list[str] = []
        excluded: list[str] = []
        for term, x in ((self.fixed, 1.0), (self.per_cycle, float(cycles)), (self.per_word, float(words))):
            if not term.known:
                excluded.append(term.name)
                continue
            by_term[term.name] = term.value * x
            (measured if term.provenance is Provenance.MEASURED else assumed).append(term.name)
        extrapolation: dict[str, float] = {}
        domain = self.measured_domain
        for axis, requested in (("cycles", float(cycles)), ("words", float(words))):
            edge = domain.get(axis)
            if edge:
                extrapolation[axis] = requested / edge
            else:
                extrapolation[axis] = float("inf")
                assumed.append(f"{axis}_domain")
        return CostEstimate(
            substrate=self.substrate, concurrency=self.concurrency, cycles=cycles, words=words,
            seconds=sum(by_term.values()), by_term=by_term,
            measured=tuple(measured), assumed=tuple(assumed), excluded=tuple(excluded),
            extrapolation=extrapolation)

    def validate(self, samples: Iterable[CostSample], *, average_reps: bool = True) -> Validation:
        """Held-out error. Samples must share the law's concurrency, or the comparison is meaningless."""
        rows = list(average_replicates(samples)) if average_reps else list(samples)
        errs: list[float] = []
        out: list[dict] = []
        for s in rows:
            if s.concurrency != self.concurrency:
                raise MixedConcurrency(
                    f"validating a law fitted at concurrency={self.concurrency} against a sample taken "
                    f"at concurrency={s.concurrency}: a throughput number cannot check a latency law")
            est = self.estimate(s.cycles, s.words)
            rel = (est.seconds - s.seconds) / s.seconds if s.seconds else float("inf")
            errs.append(abs(rel))
            out.append({"label": s.label, "cycles": s.cycles, "words": s.words,
                        "predicted_seconds": est.seconds, "measured_seconds": s.seconds,
                        "rel_err": rel})
        if not errs:
            raise NotEnoughEvidence("validate() was given no samples")
        return Validation(n=len(errs), median_abs_rel_err=statistics.median(errs),
                          max_abs_rel_err=max(errs), rows=tuple(out))

    def as_dict(self) -> dict:
        return {"substrate": self.substrate, "concurrency": self.concurrency,
                "form": "seconds = fixed + per_cycle*cycles + per_word*words",
                "fixed": self.fixed.as_dict(), "per_cycle": self.per_cycle.as_dict(),
                "per_word": self.per_word.as_dict(),
                "cycles_per_second": self.cycles_per_second,
                "words_per_second": self.words_per_second,
                "cycles_only_per_cycle": self.cycles_only_per_cycle,
                "cycles_only_overstatement": self.cycles_only_overstatement,
                "measured_domain": self.measured_domain,
                "n_samples": self.n_samples, "notes": list(self.notes)}


# --- the fit ----------------------------------------------------------------------------------------

def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """``(intercept, slope, r2)`` of a least-squares line. Two distinct x values minimum."""
    n = len(xs)
    if n < 2:
        raise NotEnoughEvidence(f"a line needs at least 2 points, got {n}")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        raise NotEnoughEvidence("all sample x values are identical — the slope is not identifiable")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r2


def _one_concurrency(samples: Sequence[CostSample], allow_mixed: bool) -> int:
    found = sorted({s.concurrency for s in samples})
    if not found:
        raise NotEnoughEvidence("no samples")
    if len(found) > 1 and not allow_mixed:
        raise MixedConcurrency(
            f"samples span concurrencies {found}: a per-query time measured under parallelism is a "
            f"throughput figure and pooling it with serial latencies fits neither. Split the fit, or "
            f"pass allow_mixed_concurrency=True and own the consequence.")
    return found[0]


def average_replicates(samples: Iterable[CostSample]) -> list[CostSample]:
    """Collapse repeated timings of the SAME query (same label/kind/cycles/words) to their mean.

    Repetitions measure host noise, not a new point; leaving them in weights that query twice.
    """
    buckets: dict[tuple, list[CostSample]] = {}
    for s in samples:
        buckets.setdefault((s.kind, s.label, s.cycles, s.words, s.concurrency), []).append(s)
    out = []
    for key, group in buckets.items():
        first = group[0]
        mean = sum(g.seconds for g in group) / len(group)
        out.append(CostSample(seconds=mean, cycles=first.cycles, words=first.words,
                              concurrency=first.concurrency, kind=first.kind, label=first.label))
    return out


def cycles_only_slope(samples: Iterable[CostSample]) -> float | None:
    """The naive one-term fit: seconds ~ cycles, ignoring program size. Reported so it can be BEATEN.

    This is the number a cycles-only characterization would publish. Compare it to
    :attr:`CostLaw.per_cycle`; the ratio is how much the load term was charged to the cycles.
    """
    rows = [s for s in samples]
    if len(rows) < 2:
        return None
    try:
        _, slope, _ = _ols([float(s.cycles) for s in rows], [s.seconds for s in rows])
    except NotEnoughEvidence:
        return None
    return slope


def fit_cost_law(
    samples: Iterable[CostSample],
    *,
    substrate: str,
    cycle_fit_min_cycles: int | None = None,
    allow_mixed_concurrency: bool = False,
    origin_reach: float = 0.05,
) -> CostLaw:
    """Fit ``seconds = a + b*cycles + c*words``, separating the terms by construction.

    The order matters and is the whole method:

    1. ``c`` from the :attr:`ProbeKind.LOAD` probes -- the halt-first program, where the words vary
       and the cycles are pinned at the floor. Nothing else can contaminate this slope.
    2. every other sample has ``c*words`` SUBTRACTED before the cycle fit, so the cycle slope is not
       paid for the program load.
    3. ``b`` from the :attr:`ProbeKind.CYCLE` probes (words pinned, so the subtraction is a constant
       and ``b`` is isolated by construction). With no cycle ladder the corpus residual is used
       instead and ``b`` is marked :attr:`Provenance.DERIVED` -- it is then only as right as ``c``.
    4. ``a`` from a :attr:`ProbeKind.FLOOR` probe if one exists (measured directly), else from the
       cycle fit's intercept -- but only when that fit's domain actually reaches toward the origin
       (``min(x) <= origin_reach * max(x)``). An intercept extrapolated back across the whole domain
       is not a measurement of it. A fitted fixed term that comes out NEGATIVE is not a cost either;
       both cases record ``UNKNOWN``, and :meth:`CostLaw.estimate` then reports a lower bound that
       names the missing term rather than quietly substituting zero.

    ``cycle_fit_min_cycles`` restricts the cycle fit to the large end, which is how you check that the
    law has not bent: refit above a threshold and compare the slope to the whole-range slope.
    """
    rows = [s for s in samples]
    if not rows:
        raise NotEnoughEvidence("fit_cost_law() was given no samples")
    concurrency = _one_concurrency(rows, allow_mixed_concurrency)
    notes: list[str] = []

    load = [s for s in rows if s.kind is ProbeKind.LOAD]
    cycle = [s for s in rows if s.kind is ProbeKind.CYCLE]
    floors = [s for s in rows if s.kind is ProbeKind.FLOOR]
    corpus = [s for s in rows if s.kind is ProbeKind.CORPUS]

    # 1. the WORD term, from the halt-first load probe.
    if len({s.words for s in load}) >= 2:
        w_intercept, c, w_r2 = _ols([float(s.words) for s in load], [s.seconds for s in load])
        span = (min(s.cycles for s in load), max(s.cycles for s in load))
        pinned = span[1] <= max(2 * span[0], span[0] + 4)
        if not pinned:
            notes.append(
                f"the load probe's cycles are NOT pinned ({span[0]}..{span[1]}): its slope carries "
                f"some of the cycle term, so the word term is DERIVED rather than isolated")
        per_word = Term(
            name="per_word", value=c, unit="s/word",
            provenance=Provenance.MEASURED if pinned else Provenance.DERIVED,
            construction="halt-first program: all W words load, ~1 instruction retires",
            n=len(load), r2=w_r2,
            domain=(float(min(s.words for s in load)), float(max(s.words for s in load))),
            note=None if pinned else f"load-probe cycles spanned {span[0]}..{span[1]}")
    else:
        c = None
        per_word = Term(
            name="per_word", value=None, unit="s/word", provenance=Provenance.UNKNOWN,
            construction="none — no halt-first load probe was supplied",
            note="without a load probe the program-size cost cannot be separated from the cycle cost; "
                 "any cycle slope below has absorbed it")
        notes.append("NO LOAD PROBE: the per-word term is UNKNOWN and the per-cycle term is "
                     "contaminated by it. Run a halt-first ladder to separate them.")

    # 2. remove the word term before fitting cycles.
    def residual(s: CostSample) -> float:
        return s.seconds - (c * s.words if c is not None else 0.0)

    # 3. the CYCLE term.
    cycle_pool = cycle or corpus
    from_ladder = bool(cycle)
    if cycle_fit_min_cycles is not None:
        cycle_pool = [s for s in cycle_pool if s.cycles >= cycle_fit_min_cycles]
    if len({s.cycles for s in cycle_pool}) >= 2:
        c_intercept, b, c_r2 = _ols([float(s.cycles) for s in cycle_pool],
                                    [residual(s) for s in cycle_pool])
        words_span = (min(s.words for s in cycle_pool), max(s.words for s in cycle_pool))
        words_pinned = from_ladder and words_span[1] <= max(2 * words_span[0], words_span[0] + 8)
        if from_ladder and words_pinned and per_word.known:
            prov, how = Provenance.MEASURED, ("cycle ladder at pinned program size, word term removed")
        elif per_word.known:
            prov, how = Provenance.DERIVED, ("correlated samples with the measured word term removed")
        else:
            prov, how = Provenance.DERIVED, ("cycles-only fit — NO word term was available to remove")
        per_cycle = Term(
            name="per_cycle", value=b, unit="s/cycle", provenance=prov, construction=how,
            n=len(cycle_pool), r2=c_r2,
            domain=(float(min(s.cycles for s in cycle_pool)), float(max(s.cycles for s in cycle_pool))),
            note=None if prov is Provenance.MEASURED else
                 "not isolated by construction; only as correct as the term subtracted from it")
    else:
        c_intercept = None
        b = None
        per_cycle = Term(name="per_cycle", value=None, unit="s/cycle", provenance=Provenance.UNKNOWN,
                         construction="none — fewer than two distinct cycle counts",
                         note="no cycle sweep: the simulated-time cost is UNKNOWN")
        notes.append("NO CYCLE SWEEP: the per-cycle term is UNKNOWN.")

    # 4. the FIXED term.
    fixed = _fixed_term(floors, b, c, c_intercept, cycle_pool, origin_reach, notes)

    return CostLaw(
        substrate=substrate, concurrency=concurrency, fixed=fixed, per_cycle=per_cycle,
        per_word=per_word, cycles_only_per_cycle=cycles_only_slope(corpus or cycle_pool),
        n_samples=len(rows), notes=tuple(notes))


def _fixed_term(floors, b, c, c_intercept, cycle_pool, origin_reach, notes) -> Term:
    """The per-query overhead that is neither cycles nor words -- or an honest UNKNOWN."""
    if floors and b is not None and c is not None:
        f = min(floors, key=lambda s: (s.words, s.cycles))
        value = f.seconds - b * f.cycles - c * f.words
        if value < 0:
            notes.append(f"the floor probe implies a NEGATIVE fixed term ({value:.4g} s); recorded "
                         f"UNKNOWN rather than a negative cost")
            return Term(name="fixed", value=None, unit="s", provenance=Provenance.UNKNOWN,
                        construction="floor probe, but it came out negative",
                        note=f"floor probe residual {value:.4g} s")
        return Term(name="fixed", value=value, unit="s", provenance=Provenance.MEASURED,
                    construction=f"floor probe ({f.words} words, {f.cycles} cycles), rate terms removed",
                    n=len(floors), domain=(0.0, 0.0))
    if c_intercept is None or not cycle_pool:
        return Term(name="fixed", value=None, unit="s", provenance=Provenance.UNKNOWN,
                    construction="none — no floor probe and no cycle fit",
                    note="the per-query overhead was never measured; it is NOT zero")
    lo = min(s.cycles for s in cycle_pool)
    hi = max(s.cycles for s in cycle_pool)
    if hi and lo > origin_reach * hi:
        notes.append(
            f"the cycle fit spans {lo}..{hi}, so its intercept ({c_intercept:.4g} s) is an "
            f"extrapolation back across the whole domain, not a measurement of the fixed term")
        return Term(name="fixed", value=None, unit="s", provenance=Provenance.UNKNOWN,
                    construction=f"cycle-fit intercept rejected: domain starts at {lo} of {hi}",
                    note=f"rejected intercept {c_intercept:.6g} s; supply a floor probe to measure it")
    if c_intercept < 0:
        notes.append(f"the fitted fixed term is NEGATIVE ({c_intercept:.4g} s), which is not a cost — "
                     f"recorded UNKNOWN; it is fit curvature, not a discount")
        return Term(name="fixed", value=None, unit="s", provenance=Provenance.UNKNOWN,
                    construction="cycle-fit intercept, rejected as negative",
                    note=f"rejected intercept {c_intercept:.6g} s")
    return Term(name="fixed", value=c_intercept, unit="s", provenance=Provenance.MEASURED,
                construction="intercept of the cycle fit, word term removed",
                n=len(cycle_pool), domain=(float(lo), float(hi)))


# --- concurrency ------------------------------------------------------------------------------------

def concurrency_inflation(*, serial_seconds: float, observed_seconds: float,
                          workers: int) -> ConcurrencyReport:
    """What N-way parallelism did to a per-query number.

    The same query, same program, same cycle count, costs more wall-clock per query when N of them
    contend. Directly measured on one substrate: 3.7 s serial, 23.4 s at 16 workers -- 6.3x. Quoting
    the contended number as a query latency (or the serial one as a throughput) is the error this
    exists to make visible.
    """
    if serial_seconds <= 0:
        raise ValueError("serial_seconds must be > 0")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    x = observed_seconds / serial_seconds
    return ConcurrencyReport(
        workers=workers, serial_seconds=serial_seconds, observed_seconds=observed_seconds,
        inflation_x=x,
        note=(f"a per-query cost of {observed_seconds:.4g} s measured under {workers}-way parallelism "
              f"is {x:.3g}x its {serial_seconds:.4g} s serial latency — it is a throughput figure, not "
              f"a latency figure, and must not be quoted as one"))


# --- measurement driver -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunOutcome:
    """What a substrate reports back about one program it ran."""

    seconds: float
    cycles: int
    words: int


@runtime_checkable
class Substrate(Protocol):
    """Anything that can run a program and say how long it took, how many cycles, how many words.

    Structural, not nominal: any object with a compatible ``run`` and a ``concurrency`` satisfies it,
    so a simulator, an ISA model and a board are priced by the same code without a shared base class.
    """

    name: str
    concurrency: int

    def run(self, program: object) -> RunOutcome:  # pragma: no cover - protocol shape
        ...


@dataclass(frozen=True)
class Probe:
    """A program to run, plus what it was constructed to isolate."""

    program: object
    kind: ProbeKind
    label: str | None = None


def halt_first_probes(build: Callable[[int], object], word_counts: Sequence[int]) -> list[Probe]:
    """The load ladder: programs whose FIRST word is the halt, padded to each of ``word_counts``.

    ``build(n_words)`` is the caller's -- how a halt is encoded and how padding is spelled are facts
    about the target, derived from its own ISA, never assumed here. What is generic is the shape of
    the experiment: every word is still loaded, ~1 instruction retires, so the wall time is the load
    term with the cycle term pinned. The smallest rung doubles as the :attr:`ProbeKind.FLOOR` probe.
    """
    if len(set(word_counts)) < 2:
        raise NotEnoughEvidence("a load ladder needs at least two distinct word counts")
    smallest = min(word_counts)
    probes = []
    for n in word_counts:
        probes.append(Probe(program=build(n), kind=ProbeKind.LOAD, label=f"halt_first_{n}w"))
        if n == smallest:
            probes.append(Probe(program=build(n), kind=ProbeKind.FLOOR, label=f"floor_{n}w"))
    return probes


def measure(substrate: Substrate, probes: Iterable[Probe], *, reps: int = 1) -> list[CostSample]:
    """Run each probe ``reps`` times and stamp every sample with the substrate's concurrency."""
    if reps < 1:
        raise ValueError("reps must be >= 1")
    out: list[CostSample] = []
    for probe in probes:
        for _ in range(reps):
            r = substrate.run(probe.program)
            out.append(CostSample(seconds=r.seconds, cycles=r.cycles, words=r.words,
                                  concurrency=substrate.concurrency, kind=probe.kind,
                                  label=probe.label))
    return out


# --- reporting --------------------------------------------------------------------------------------

def _rate(value: float | None, per: str) -> str:
    if not value:
        return "UNKNOWN"
    return f"{value * 1e3:.6g} ms/{per}  ({1.0 / value:,.1f} {per}s/s)"


def render_law(law: CostLaw, *, validation: Validation | None = None,
               concurrency: ConcurrencyReport | None = None) -> str:
    """A human-readable report. Concurrency is on the first line and in every projection."""
    lines = [
        f"{law.substrate} — oracle cost law   (CONCURRENCY = {law.concurrency})",
        "  seconds = fixed + per_cycle*cycles + per_word*words",
        "",
        f"  per_cycle : {_rate(law.per_cycle.value, 'cycle')}",
        f"              {law.per_cycle.provenance.value}  n={law.per_cycle.n}  "
        f"r2={law.per_cycle.r2:.6g}" if law.per_cycle.r2 is not None else
        f"              {law.per_cycle.provenance.value}  n={law.per_cycle.n}",
        f"              via {law.per_cycle.construction}",
        f"  per_word  : {_rate(law.per_word.value, 'word')}",
        f"              {law.per_word.provenance.value}  n={law.per_word.n}  "
        f"r2={law.per_word.r2:.6g}" if law.per_word.r2 is not None else
        f"              {law.per_word.provenance.value}  n={law.per_word.n}",
        f"              via {law.per_word.construction}",
        f"  fixed     : "
        + (f"{law.fixed.value * 1e3:.6g} ms" if law.fixed.known else "UNKNOWN (not zero)"),
        f"              {law.fixed.provenance.value}  via {law.fixed.construction}",
    ]
    if law.cycles_only_overstatement:
        lines += ["",
                  f"  a CYCLES-ONLY fit would report {law.cycles_only_per_cycle * 1e3:.6g} ms/cycle — "
                  f"{law.cycles_only_overstatement:.3g}x the marginal rate, because it charges the "
                  f"program load to the cycles."]
    dom = law.measured_domain
    if dom:
        lines += ["", "  measured domain: " + ", ".join(f"{k} <= {v:,.0f}" for k, v in dom.items())]
    if validation is not None:
        lines += ["",
                  f"  held out: n={validation.n}  median |rel err| = "
                  f"{validation.median_abs_rel_err:.2%}  max = {validation.max_abs_rel_err:.2%}"]
    if concurrency is not None:
        lines += ["", f"  concurrency: {concurrency.note}"]
    for n in law.notes:
        lines.append(f"  NOTE: {n}")
    return "\n".join(lines)
