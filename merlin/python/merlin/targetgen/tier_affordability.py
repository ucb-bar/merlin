"""Can this target afford ONE TIER for ONE capsule -- the per-(tier, engine) price sheet.

NAMED APART FROM ``cert_cost`` DELIBERATELY. Two modules with this content existed under the same
name on two branches and they answer different questions. ``cert_cost`` is a SIZING instrument: given a
budget, how large may a capsule be (``max_elements_within``, ``predict_seconds``, the measured power
law). This module is an AFFORDABILITY verdict: given a capsule and a tier, is that tier AFFORDABLE,
TOO_EXPENSIVE or UNKNOWN on this target's own measured history, keyed per engine because the two
elaborated-RTL engines answer the same capsule roughly 26x apart.

Merging the two by text would have produced one module with two incompatible APIs and imports
resolving to whichever half won -- and did, briefly: ``tier_policy`` was written against this API,
kept the other module, and every call into it raised ``AttributeError`` at the first use. That is why
the name is different rather than clever.

"""

from __future__ import annotations

import json
import statistics
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AFFORDABLE", "TOO_EXPENSIVE", "UNKNOWN", "EXTRAPOLATION_MARGIN", "MIN_SAMPLES",
           "ENGINE_UNATTRIBUTED", "CycleCostFit", "Affordability", "Sample",
           "fits_for", "fit_for", "measured_cycles", "affordability", "reset_cache"]

#: How far past the largest measured cycle count a prediction is still honest, as a multiple. A fit is
#: a local linearisation of a simulator's behaviour, not a law; beyond this the answer is "unknown".
EXTRAPOLATION_MARGIN = 2.0

#: Fewest measured capsules a fit may rest on. Two points define a line through anything.
MIN_SAMPLES = 5

#: Bucket key for a cycle-accurate sample whose record does not say which engine produced it. Kept as
#: its OWN population rather than merged into a named one: a sample of unknown provenance is not
#: evidence about a particular engine, and folding it into one would silently move that engine's law.
ENGINE_UNATTRIBUTED = "unattributed"

#: The three affordability verdicts. ``UNKNOWN`` is a first-class answer, never a synonym for either
#: of the others -- see the module docstring's third refusal.
AFFORDABLE = "AFFORDABLE"
TOO_EXPENSIVE = "TOO_EXPENSIVE"
UNKNOWN = "UNKNOWN"

#: Fields on a tier record that may carry the engine's identity, most explicit first. Tried in order;
#: whichever answers first becomes the bucket key, VERBATIM. This module attaches no meaning to the
#: value -- it is an opaque discriminator, so a target adding an engine needs no change here.
_ENGINE_FIELDS = ("engine", "sim", "simulator", "oracle", "evidence")


@dataclass(frozen=True)
class Sample:
    """One capsule's measured cost at one tier on one engine, with where it was read from."""

    capsule: str
    tier: str
    engine: str
    seconds: float
    cycles: int
    functional_cycles: int | None
    source: str


@dataclass(frozen=True)
class CycleCostFit:
    """``seconds ~= intercept_s + per_cycle_s * cycles``, with the evidence it rests on."""

    target: str
    tier: str
    engine: str
    intercept_s: float
    per_cycle_s: float
    r2: float
    n_samples: int
    cycles_min: int
    cycles_max: int
    #: median ``cycle_accurate_cycles / functional_cycles`` over capsules that ran both, or None.
    functional_ratio: float | None = None
    n_ratio_samples: int = 0
    sources: tuple[str, ...] = ()

    def predict(self, cycles: int | None) -> "float | None":
        """Predicted seconds for a capsule running ``cycles``, or ``None`` outside the evidence.

        ``None`` past the measured range times :data:`EXTRAPOLATION_MARGIN`. A prediction the fit
        cannot support is an absence, not a big number: the honest answer to "how long would a capsule
        a hundred times bigger than anything we have run take" is that we do not know.
        """
        if not cycles or cycles <= 0:
            return None
        if cycles > self.cycles_max * EXTRAPOLATION_MARGIN:
            return None
        return self.intercept_s + self.per_cycle_s * float(cycles)

    def max_cycles_within(self, budget_s: float) -> "int | None":
        """Most cycles a capsule may run and still certify inside ``budget_s``.

        ``None`` when the fixed floor alone exceeds the budget -- which is a statement about the budget,
        not about any capsule shape. Clamped to the measured range: past the evidence the line is an
        opinion.
        """
        if budget_s <= 0 or self.per_cycle_s <= 0:
            return None
        if budget_s <= self.intercept_s:
            return None
        raw = int((budget_s - self.intercept_s) / self.per_cycle_s)
        return max(1, min(raw, int(self.cycles_max * EXTRAPOLATION_MARGIN)))

    def scale_functional(self, functional_cycles: int | None) -> "tuple[int | None, str]":
        """``(cycle_accurate_cycles, basis)`` estimated from a FUNCTIONAL run's cycle count.

        The cheap path this module exists to enable: the screen tier costs milliseconds and reports a
        cycle count, so a capsule can be priced for certification without ever being certified.
        Refuses -- rather than assuming 1.0 -- when no capsule on this bucket has run at both tiers.
        """
        if not functional_cycles or functional_cycles <= 0:
            return None, "no functional cycle count to scale from"
        if not self.functional_ratio:
            return None, ("no capsule has run at BOTH the screen and cert tiers on this engine, so the "
                          "functional-to-cycle-accurate cycle ratio is unmeasured and cannot be assumed")
        return (int(functional_cycles * self.functional_ratio),
                f"functional {functional_cycles} cycles x measured ratio "
                f"{self.functional_ratio:.2f} (n={self.n_ratio_samples})")

    def to_dict(self) -> dict:
        return {"target": self.target, "tier": self.tier, "engine": self.engine,
                "intercept_s": round(self.intercept_s, 3),
                "per_cycle_s": round(self.per_cycle_s, 6), "r2": round(self.r2, 4),
                "n_samples": self.n_samples,
                "measured_range_cycles": [self.cycles_min, self.cycles_max],
                "functional_to_cycle_accurate_ratio": (round(self.functional_ratio, 3)
                                                       if self.functional_ratio else None),
                "n_ratio_samples": self.n_ratio_samples, "n_sources": len(self.sources)}


@dataclass(frozen=True)
class Affordability:
    """A three-valued price check: ``AFFORDABLE`` / ``TOO_EXPENSIVE`` / ``UNKNOWN``, and why.

    ``reason`` is written to be quoted verbatim into a tier record, so a reader of the record can see
    the budget, the prediction and the evidence behind it without going back to this module.
    """

    verdict: str
    reason: str
    budget_s: float | None = None
    predicted_s: float | None = None
    cycles: int | None = None
    cycles_basis: str | None = None
    fit: CycleCostFit | None = None
    #: Every measured bucket's prediction for this capsule, cheapest first. What makes a cap
    #: actionable: it says which engine WOULD have fitted the budget.
    alternatives: tuple[tuple[str, float], ...] = ()

    @property
    def known(self) -> bool:
        return self.verdict in (AFFORDABLE, TOO_EXPENSIVE)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason,
                "budget_s": self.budget_s,
                "predicted_s": round(self.predicted_s, 1) if self.predicted_s else None,
                "cycles": self.cycles, "cycles_basis": self.cycles_basis,
                "fit": self.fit.to_dict() if self.fit else None,
                "cheaper_engines": [{"engine": e, "predicted_s": round(s, 1)}
                                    for e, s in self.alternatives]}


# --- reading the measurements ---------------------------------------------------------------------

def _engine_key(record: dict) -> str:
    """The opaque engine discriminator for one tier record.

    Structural: the first of :data:`_ENGINE_FIELDS` that carries a value wins and is used VERBATIM.
    No parsing, no meaning attached -- a key is only ever compared with another key. A record naming
    no engine gets :data:`ENGINE_UNATTRIBUTED` rather than being guessed into a named bucket.
    """
    for field in _ENGINE_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ENGINE_UNATTRIBUTED


def _is_cycle_accurate(record: dict) -> bool:
    """Whether the tier record DECLARES itself cycle-accurate.

    Asked of the record, never inferred from the tier's name: one target's L3 is elaborated RTL and
    another's is a model, so a name-based test prices a functional run as a certification.
    ``derived_from_rtl`` is accepted as the older spelling of the same claim.
    """
    return record.get("cycle_accurate") is True or record.get("derived_from_rtl") is True


def _result_roots(target: str, roots: Iterable[Path] | None) -> list[Path]:
    """Where to look for this TARGET's results.

    ⚠️ BOTH BASES ARE TARGET-SCOPED. An unscoped runs root globs every target's runs, which hands
    every target the same fit and silently breaks this module's own stated refusal that a target with
    no certification history has no basis for sizing.
    """
    if roots is not None:
        return [Path(r) for r in roots]
    from merlin.common.paths import artifacts_dir, runs_dir
    return [artifacts_dir() / "capsule-bench" / str(target), runs_dir() / str(target)]


def _samples(target: str, roots: Iterable[Path] | None = None) -> list[Sample]:
    """Every cycle-accurate cost observation on disk for ``target``.

    A run that never reached a cycle-accurate tier contributes NOTHING rather than its functional
    time -- a fit that absorbed those reads a near-zero cost for a capsule nobody certified, which is
    the "zero reads as free" error this module exists to prevent.
    """
    out: list[Sample] = []
    for base in _result_roots(target, roots):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("capsule_result.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue                       # unreadable is not a measurement
            if not isinstance(doc, dict):
                continue
            name = doc.get("capsule")
            tiers = doc.get("tiers")
            if not name or not isinstance(tiers, dict):
                continue
            # The screen tier's cycle count, kept separately: it is the cheap PREDICTOR and never the
            # cost itself. Lowest-named non-cycle-accurate tier that reported cycles.
            functional = None
            for tier_name in sorted(tiers):
                rec = tiers[tier_name]
                if not isinstance(rec, dict) or _is_cycle_accurate(rec):
                    continue
                cycles = rec.get("cycles")
                if isinstance(cycles, int) and cycles > 0:
                    functional = cycles
                    break
            for tier_name, rec in tiers.items():
                if not isinstance(rec, dict) or not _is_cycle_accurate(rec):
                    continue
                timing = rec.get("timing") if isinstance(rec.get("timing"), dict) else {}
                seconds = timing.get("sim_active_s")
                cycles = rec.get("cycles")
                if not (isinstance(seconds, (int, float)) and seconds > 0):
                    continue
                if not (isinstance(cycles, int) and cycles > 0):
                    continue
                out.append(Sample(capsule=str(name), tier=str(tier_name),
                                  engine=_engine_key(rec), seconds=float(seconds),
                                  cycles=int(cycles), functional_cycles=functional,
                                  source=str(path)))
    return out


def _ordinary_least_squares(xs: list[int], ys: list[float]) -> "tuple[float, float, float] | None":
    """``(intercept, slope, r2)`` or ``None`` when the sample cannot support a line."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None                            # a line through one x tells you nothing
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - residual / ss_tot if ss_tot else 0.0
    return intercept, slope, r2


_CACHE: dict[tuple, dict[tuple[str, str], CycleCostFit]] = {}
_CYCLES_CACHE: dict[tuple, dict[tuple[str, str, str], int]] = {}
_LOCK = threading.Lock()


def reset_cache() -> None:
    """Forget every fit read from disk. For tests, and for a caller that wants a re-read."""
    with _LOCK:
        _CACHE.clear()
        _CYCLES_CACHE.clear()


def _cache_key(target: str, roots) -> tuple:
    return (str(target), tuple(sorted(str(r) for r in roots)) if roots is not None else None)


def _build(target: str, roots) -> tuple[dict[tuple[str, str], CycleCostFit],
                                        dict[tuple[str, str, str], int]]:
    """Fit every ``(tier, engine)`` bucket, and index each capsule's largest measured cycle count."""
    buckets: dict[tuple[str, str], list[Sample]] = {}
    cycles: dict[tuple[str, str, str], int] = {}
    for s in _samples(target, roots):
        buckets.setdefault((s.tier, s.engine), []).append(s)
        key = (s.capsule, s.tier, s.engine)
        # LARGEST measured, not the mean: a repeat measurement of the same capsule bounds what the next
        # run will cost, and the binding number is the slow one.
        cycles[key] = max(cycles.get(key, 0), s.cycles)
        if s.functional_cycles:
            fk = (s.capsule, "", "")
            cycles[fk] = max(cycles.get(fk, 0), s.functional_cycles)
    fits: dict[tuple[str, str], CycleCostFit] = {}
    for (tier, engine), samples in buckets.items():
        xs = [s.cycles for s in samples]
        ys = [s.seconds for s in samples]
        if len(xs) < MIN_SAMPLES or len(set(xs)) < 2:
            continue                           # too little to fit; the caller gets UNKNOWN
        line = _ordinary_least_squares(xs, ys)
        if line is None:
            continue
        intercept, slope, r2 = line
        ratios = [s.cycles / s.functional_cycles for s in samples if s.functional_cycles]
        fits[(tier, engine)] = CycleCostFit(
            target=str(target), tier=tier, engine=engine, intercept_s=intercept,
            per_cycle_s=slope, r2=r2, n_samples=len(xs), cycles_min=min(xs), cycles_max=max(xs),
            functional_ratio=statistics.median(ratios) if ratios else None,
            n_ratio_samples=len(ratios),
            sources=tuple(sorted({s.source for s in samples})))
    return fits, cycles


def fits_for(target: str, *, roots=None, tier: str | None = None,
             engine: str | None = None) -> dict[tuple[str, str], CycleCostFit]:
    """``(tier, engine) -> CycleCostFit`` for every bucket with a measured basis.

    An EMPTY mapping is a real answer and the caller must honour it: a target with no certification
    history has no basis for sizing a capsule to a time budget, and the correct response is to refuse
    rather than to size from a default.

    ``engine`` narrows by token containment against the bucket key, because the key is whatever the
    record said (an explicit engine field on newer runs, an evidence filename on older ones) and both
    spellings should answer to the same engine name. Compared as data, never pattern-matched.
    """
    key = _cache_key(target, roots)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is None:
        fits, cycles = _build(str(target), roots)
        with _LOCK:
            _CACHE[key] = fits
            _CYCLES_CACHE[key] = cycles
        cached = fits
    out = dict(cached)
    if tier is not None:
        out = {k: v for k, v in out.items() if k[0] == str(tier)}
    if engine is not None:
        want = str(engine).strip().lower()
        out = {k: v for k, v in out.items() if want and want in k[1].lower()}
    return out


def fit_for(target: str, tier: str, *, engine: str | None = None,
            roots=None) -> "CycleCostFit | None":
    """The single most EXPENSIVE measured fit for ``tier``, or ``None`` when nothing was measured.

    The most expensive rather than the cheapest, because this answers "may this capsule be allowed to
    run here" and the binding cost is the one that decides. When the engine that will serve the tier is
    known, pass it and get that bucket instead of the conservative one.
    """
    fits = fits_for(target, roots=roots, tier=tier, engine=engine)
    if not fits:
        return None
    return max(fits.values(), key=lambda f: (f.per_cycle_s, f.intercept_s, f.engine))


def measured_cycles(target: str, capsule: str, tier: str, *, engine: str | None = None,
                    roots=None) -> "tuple[int | None, str | None]":
    """``(cycles, engine_key)``: the largest cycle count ``capsule`` has measured at ``tier``.

    The strongest cycle basis there is -- this capsule's own history on this tier -- and it is free,
    because the runs that produced it are already on disk.
    """
    fits_for(target, roots=roots)               # populate the cycles index alongside the fits
    with _LOCK:
        index = dict(_CYCLES_CACHE.get(_cache_key(target, roots)) or {})
    want = str(engine).strip().lower() if engine else None
    best: tuple[int, str] | None = None
    for (name, t, eng), cycles in index.items():
        if name != str(capsule) or t != str(tier) or not t:
            continue
        if want and want not in eng.lower():
            continue
        if best is None or cycles > best[0]:
            best = (cycles, eng)
    return best if best else (None, None)


def measured_functional_cycles(target: str, capsule: str, *, roots=None) -> "int | None":
    """The largest SCREEN-tier cycle count ``capsule`` has ever reported, or None."""
    fits_for(target, roots=roots)
    with _LOCK:
        index = dict(_CYCLES_CACHE.get(_cache_key(target, roots)) or {})
    return index.get((str(capsule), "", "")) or None


# --- the decision ---------------------------------------------------------------------------------

def affordability(target: str, tier: str, *, budget_s: float | None, capsule: str | None = None,
                  cycles: int | None = None, functional_cycles: int | None = None,
                  engine: str | None = None, roots=None) -> Affordability:
    """Can ``tier`` be afforded for this capsule inside ``budget_s``?

    Cycle basis, strongest first, and the one used is NAMED in the result:

    1. ``cycles`` passed by the caller -- a cycle-accurate count already in hand;
    2. this capsule's own largest measured count at this tier, read from disk;
    3. ``functional_cycles`` (or this capsule's measured screen-tier count) scaled by the bucket's
       own measured functional-to-cycle-accurate ratio.

    Every failure to establish one of those is ``UNKNOWN``, never ``AFFORDABLE`` and never
    ``TOO_EXPENSIVE``. So is the absence of a fit, and so is a cycle count outside the fitted range.
    """
    if budget_s is None or budget_s <= 0:
        return Affordability(UNKNOWN, "no certification budget was declared, so affordability is not "
                                      "a question this can answer", budget_s=budget_s)
    fit = fit_for(target, tier, engine=engine, roots=roots)
    if fit is None:
        every = fits_for(target, roots=roots)
        return Affordability(
            UNKNOWN,
            (f"no measured certification cost for tier {tier} on this target"
             + (f" for engine {engine!r}" if engine else "")
             + (f"; priced buckets here: {sorted({f'{t}/{e}' for t, e in every})}" if every
                else "; this target has no cycle-accurate run on disk to fit")
             + f". A fit needs at least {MIN_SAMPLES} samples at two distinct cycle counts; UNKNOWN is "
               "reported rather than assuming the tier is affordable."),
            budget_s=budget_s)

    basis: str | None = None
    if cycles and cycles > 0:
        basis = "cycle-accurate cycle count supplied by the caller"
    else:
        if capsule:
            cycles, eng = measured_cycles(target, capsule, tier, engine=engine, roots=roots)
            if cycles:
                basis = f"largest cycle-accurate count this capsule measured at {tier} on {eng!r}"
        if not cycles:
            fc = functional_cycles
            if not fc and capsule:
                fc = measured_functional_cycles(target, capsule, roots=roots)
            cycles, basis = fit.scale_functional(fc)
            if not cycles:
                return Affordability(UNKNOWN,
                                     f"cannot establish a cycle count for this capsule at {tier}: "
                                     f"{basis}. UNKNOWN rather than a guess -- an unpriced capsule is "
                                     f"neither affordable nor capped.",
                                     budget_s=budget_s, fit=fit)

    predicted = fit.predict(cycles)
    if predicted is None:
        return Affordability(
            UNKNOWN,
            (f"{cycles} cycles is beyond the measured range for {tier}/{fit.engine} "
             f"({fit.cycles_min}..{fit.cycles_max} cycles, x{EXTRAPOLATION_MARGIN:g} extrapolation "
             f"margin), so the cost is UNKNOWN rather than extrapolated"),
            budget_s=budget_s, cycles=cycles, cycles_basis=basis, fit=fit)

    others = sorted(
        ((f.engine, p) for (t, _e), f in fits_for(target, roots=roots, tier=tier).items()
         if t == str(tier) and (p := f.predict(cycles)) is not None and f.engine != fit.engine),
        key=lambda pair: pair[1])
    verdict = AFFORDABLE if predicted <= budget_s else TOO_EXPENSIVE
    if verdict == AFFORDABLE:
        reason = (f"predicted {predicted:.0f}s at {tier}/{fit.engine} is within the declared "
                  f"{budget_s:.0f}s budget ({basis}; fit {fit.intercept_s:.0f}s + "
                  f"{fit.per_cycle_s:.5f}s/cycle, r2={fit.r2:.2f}, n={fit.n_samples})")
    else:
        cheaper = ", ".join(f"{e}: {p:.0f}s" for e, p in others if p <= budget_s)
        reason = (f"predicted {predicted:.0f}s at {tier}/{fit.engine} exceeds the declared "
                  f"{budget_s:.0f}s budget ({basis}; fit {fit.intercept_s:.0f}s + "
                  f"{fit.per_cycle_s:.5f}s/cycle, r2={fit.r2:.2f}, n={fit.n_samples})"
                  + (f"; would fit on {cheaper}" if cheaper else ""))
    return Affordability(verdict, reason, budget_s=budget_s, predicted_s=predicted, cycles=cycles,
                         cycles_basis=basis, fit=fit, alternatives=tuple(others))
