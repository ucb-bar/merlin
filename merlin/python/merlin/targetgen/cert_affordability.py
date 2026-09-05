"""Whether a cohort can be certified — priced per (target, ENGINE), because the engine is the cost.

:mod:`merlin.targetgen.cert_cost` fits certification seconds against capsule size for a target. That fit
answers "how big may a capsule be and still certify" only if every sample it rests on came from the same
simulator, and until recently nothing recorded which one did. It matters by more than an order of
magnitude: measured on this repo's systolic target against the identical ELF, GSIM answers a capsule in
3.31 s where Verilator takes 86.83 s. A fit over a mixture of the two prices a capsule at neither
engine's cost, and every "can we afford this cohort at L3" number computed from it is about no machine
that exists.

So this module is the same arithmetic with the engine restored as a first-class axis:

  * a SAMPLE is one capsule's cycle-accurate ``timing.sim_active_s`` together with the engine the record
    says produced it, joined to that capsule's size from the corpus;
  * a FIT is per ``(target, engine)`` and never crosses either;
  * a COHORT price is the sum over the capsules that demand certification, at one engine's measured cost.

THE ENGINE COMES FROM THE RECORD'S OWN ``engine`` FIELD AND FROM NOWHERE ELSE. It is tempting to recover
it from the ``evidence`` filename, which is ``<engine>_console.log`` and is the only trace older records
carry. That inference is WRONG, and the runner says why in its own comment: ``sim_name`` comes from the
contract's static ``tier_sim`` map, which cannot know that a faster RTL-derived engine replaced the
declared one at run time, so a console written by GSIM was filed under Verilator's name. Deriving the
engine from that filename would attribute one engine's seconds to another — the exact defect this module
exists to remove, reintroduced through the back door. A record with no ``engine`` field is
:data:`UNATTRIBUTED`: it is COUNTED and REPORTED, never guessed at and never quietly dropped.

TWO REFUSALS, inherited deliberately from ``cert_cost`` and extended to the engine axis:

* **No measured history, no fit.** A ``(target, engine)`` pair nobody has certified yields ``None``.
  There is no default, no borrowing from the target's other engine, and no borrowing from another
  target: a number nobody measured, driving a decision somebody quotes, is the failure mode.
* **No extrapolation.** A prediction past the measured range (times ``cert_cost``'s own margin) is
  ``None``, not a large number.

The fit form and the size metric are ``cert_cost``'s, read from it rather than restated, so the two
cannot drift; :func:`fit_for` over a target whose whole history is one engine reproduces
``cert_cost.fit_for`` exactly, and a test pins that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["UNATTRIBUTED", "Sample", "EngineFit", "engine_of", "samples_for", "fit_for",
           "fits_for", "cohort_price", "affordability"]

#: A cycle-accurate second whose engine the record does not state. Spelled with delimiters no engine
#: token can produce, so it can never collide with a real engine name.
UNATTRIBUTED = "<unattributed>"

#: Sized corpora, keyed by their roots. Sizing the shipped corpus means parsing ~27k capsule interfaces
#: and takes the better part of a minute, and every caller here asks for the SAME corpus once per target
#: -- so the affordability gate paid that minute four times per audit before this existed. Bounded, and
#: keyed on the roots only: a corpus directory does not change under a running process, while the timing
#: records DO (a broker writes them while it runs) and are therefore never cached.
_SIZE_CACHE: dict = {}


@dataclass(frozen=True)
class Sample:
    """One measured certification: which capsule, how big, how long, on what, from where."""

    capsule: str
    elements: int
    seconds: float
    engine: str
    source: str


@dataclass(frozen=True)
class EngineFit:
    """``seconds ~= intercept_s + per_element_s * elements`` for ONE ``(target, engine)``."""

    target: str
    engine: str
    intercept_s: float
    per_element_s: float
    r2: float
    n_samples: int
    elements_min: int
    elements_max: int
    metric: str
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"target": self.target, "engine": self.engine,
                "intercept_s": round(self.intercept_s, 3),
                "per_element_s": round(self.per_element_s, 6), "r2": round(self.r2, 4),
                "n_samples": self.n_samples,
                "measured_range_elements": [self.elements_min, self.elements_max],
                "metric": self.metric, "n_sources": len(self.sources)}


def engine_of(tier_record) -> "str | None":
    """The engine a tier record STATES it ran on, or ``None`` when it states none.

    Structural: the record's own field, stripped. No filename inference — see the module docstring for
    the measured reason that inference is unsound.
    """
    if not isinstance(tier_record, dict):
        return None
    value = str(tier_record.get("engine") or "").strip()
    return value or None


def _cycle_accurate(rec) -> bool:
    """Whether a tier record DECLARES itself cycle-accurate. ``derived_from_rtl`` is the older spelling
    of the same claim, accepted for the same reason ``cert_cost`` accepts it."""
    return isinstance(rec, dict) and (rec.get("cycle_accurate") is True
                                      or rec.get("derived_from_rtl") is True)


def _seconds(rec) -> "float | None":
    tm = rec.get("timing") if isinstance(rec.get("timing"), dict) else rec
    value = (tm or {}).get("sim_active_s")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _record_roots(target: str, root=None, extra_roots=()):
    """The same two target-scoped bases ``cert_cost`` reads, for the same reason it scopes them: an
    unscoped runs root hands every target every other target's measurements."""
    from merlin.common.paths import artifacts_dir, runs_dir

    if root is not None:
        bases = [Path(root)]
    else:
        bases = [artifacts_dir() / "capsule-bench" / str(target), runs_dir() / str(target)]
    return bases + [Path(r) for r in (extra_roots or ())]


def _timing_by_engine(target: str, root=None, extra_roots=()) -> dict:
    """``{(capsule, engine): (seconds, source)}`` over every run this target has on disk.

    Keyed by the PAIR, not by the capsule: a capsule certified on two engines is two measurements of two
    different machines, and collapsing them to "the most recent" is how a corpus loses the cheap engine's
    evidence the moment the slow one runs. Within a pair, the later file wins (the most recent
    measurement) and the deepest qualifying tier wins within one file (the longer one is the binding
    cost) — both rules taken from ``cert_cost._cycle_accurate_seconds``.
    """
    from merlin.targetgen import cert_cost as CC

    out: dict = {}

    def _absorb(capsule: str, by_tier: dict, path) -> None:
        best: dict = {}
        for tier, rec in (by_tier or {}).items():
            if not _cycle_accurate(rec):
                continue
            secs = _seconds(rec)
            if secs is None:
                continue
            eng = engine_of(rec) or UNATTRIBUTED
            prev = best.get(eng)
            if prev is None or secs > prev[0]:
                best[eng] = (secs, f"{path}#{tier}@{eng}")
        for eng, val in best.items():
            out[(str(capsule), eng)] = val

    for base in _record_roots(target, root, extra_roots):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            if path.name != "capsule_result.json" and not path.name.startswith("score"):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):     # unreadable is not a measurement
                continue
            block = doc.get("timing_diagnostic")
            if isinstance(block, dict) and block:
                for capsule, timing in block.items():
                    if isinstance(timing, dict) and isinstance(timing.get("by_tier"), dict):
                        _absorb(capsule, timing["by_tier"], path)
                continue
            capsule = doc.get("capsule")
            if capsule:
                # Reshaped by cert_cost's own reader, so the engine reaches here through exactly the
                # field the cost model reads and one copy of that mapping exists.
                _absorb(capsule, CC._per_tier_from_result(doc), path)
    return out


def samples_for(target: str, *, corpus_roots=None, timing_root=None, extra_timing_roots=()) -> dict:
    """Every measured certification for ``target``, split by engine, with what could not be used.

    ``{"by_engine": {engine: (Sample, ...)}, "unattributed": n, "unsized": n, "metric": str}``

    ``unattributed`` counts seconds the records do not attribute to an engine; ``unsized`` counts
    measurements whose capsule is not in the corpus (so it has no size to fit against). Both are
    reported rather than dropped: a fit resting on 4 samples out of 800 must not look like a fit resting
    on 800.
    """
    from merlin.common.paths import merlin_dir
    from merlin.targetgen import cert_cost as CC

    metric = CC.CostFit.__dataclass_fields__["metric"].default
    timings = _timing_by_engine(target, timing_root, extra_timing_roots)
    if not timings:
        # Nothing measured: say so without walking the corpus. Sizing 27k capsules to answer "no
        # history" is minutes of work to reach a conclusion the timing records already gave.
        return {"by_engine": {}, "unattributed": 0, "unsized": 0, "metric": metric}
    roots = list(corpus_roots) if corpus_roots else [merlin_dir() / "contract" / "capsules"]
    key = tuple(str(r) for r in roots)
    sizes = _SIZE_CACHE.get(key)
    if sizes is None:
        if len(_SIZE_CACHE) > 4:
            _SIZE_CACHE.clear()
        sizes = _SIZE_CACHE[key] = CC._capsule_sizes(roots)

    by_engine: dict = {}
    unattributed = unsized = 0
    for (capsule, engine), (seconds, source) in sorted(timings.items()):
        if engine == UNATTRIBUTED:
            unattributed += 1
            continue
        size = sizes.get(capsule)
        if not size:
            unsized += 1
            continue
        by_engine.setdefault(engine, []).append(
            Sample(capsule=capsule, elements=int(size), seconds=float(seconds),
                   engine=engine, source=source))
    return {"by_engine": {e: tuple(v) for e, v in sorted(by_engine.items())},
            "unattributed": unattributed, "unsized": unsized, "metric": metric}


def _least_squares(xs, ys) -> "tuple[float, float, float] | None":
    """``(intercept, slope, r2)`` — the same ordinary least squares ``cert_cost`` fits, once."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = (1.0 - sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / ss_tot
          if ss_tot else 0.0)
    return intercept, slope, r2


def fit_for(target: str, engine: str, *, corpus_roots=None, timing_root=None,
            extra_timing_roots=()) -> "EngineFit | None":
    """The cost model for one ``(target, engine)``, or ``None`` when it has no measured history.

    ``None`` is a real answer and the caller must honour it. There is no fallback to the target's other
    engine, to another target, or to a module-level constant: the whole point of the engine axis is that
    one engine's seconds do not describe another's.
    """
    got = samples_for(target, corpus_roots=corpus_roots, timing_root=timing_root,
                      extra_timing_roots=extra_timing_roots)
    return _fit_rows(target, engine, got["by_engine"].get(str(engine)) or (), got["metric"])


def _fit_rows(target: str, engine: str, rows, metric: str) -> "EngineFit | None":
    """The fit over samples already collected. Split out so a caller fitting several engines sizes the
    corpus once rather than once per engine."""
    from merlin.targetgen import cert_cost as CC

    xs = [s.elements for s in rows]
    ys = [s.seconds for s in rows]
    if len(xs) < CC._MIN_SAMPLES or len(set(xs)) < 2:
        return None                                 # a line through one x tells you nothing
    fit = _least_squares(xs, ys)
    if fit is None:
        return None
    intercept, slope, r2 = fit
    return EngineFit(target=str(target), engine=str(engine), intercept_s=intercept,
                     per_element_s=slope, r2=r2, n_samples=len(xs),
                     elements_min=min(xs), elements_max=max(xs), metric=str(metric),
                     sources=tuple(sorted({s.source for s in rows})))


def fits_for(target: str, **kw) -> dict:
    """Every engine this target has measured history on, fitted, plus what could not be attributed.

    ``{"target": t, "engines": {engine: EngineFit|None}, "sample_counts": {engine: n},
      "unattributed_samples": n, "unsized_samples": n}``

    An engine present with a ``None`` fit is "measured, but too little to fit" — a different state from
    an engine that is absent entirely, and the two must not collapse: the first says re-run a few more,
    the second says nothing has ever run there.
    """
    got = samples_for(target, **kw)
    engines = {engine: _fit_rows(target, engine, rows, got["metric"])
               for engine, rows in got["by_engine"].items()}
    return {"target": str(target), "engines": engines,
            "sample_counts": {e: len(v) for e, v in got["by_engine"].items()},
            "unattributed_samples": got["unattributed"], "unsized_samples": got["unsized"],
            "metric": got["metric"]}


def cohort_price(fit: "EngineFit | None", sizes) -> dict:
    """What certifying a whole cohort costs on one engine, and what could not be priced.

    ``sizes`` is ``{capsule: elements}``. Returns ``{"total_s", "priced", "beyond_evidence",
    "unpriceable", "basis"}``. A capsule the fit cannot speak for lands in ``beyond_evidence`` and is
    EXCLUDED from the total rather than extrapolated into it, because a cohort total that silently
    absorbs an unsupported guess is exactly the number nobody should quote. ``total_s`` is ``None`` when
    there is no fit at all — the affordability of an unmeasured engine is unknown, not zero.
    """
    if fit is None:
        return {"total_s": None, "priced": 0, "beyond_evidence": [],
                "unpriceable": sorted(str(c) for c in sizes),
                "basis": "no measured (target, engine) history"}
    from merlin.targetgen import cert_cost as CC

    total, priced, beyond, unpriceable = 0.0, 0, [], []
    ceiling = fit.elements_max * CC._EXTRAPOLATION_MARGIN
    for capsule, elements in sorted(sizes.items()):
        n = int(elements or 0)
        if n <= 0:
            unpriceable.append(str(capsule))
            continue
        if n > ceiling:
            beyond.append(str(capsule))
            continue
        total += fit.intercept_s + fit.per_element_s * float(n)
        priced += 1
    return {"total_s": total, "priced": priced, "beyond_evidence": beyond,
            "unpriceable": unpriceable,
            "basis": (f"{fit.n_samples} measured {fit.engine} certification(s) of {fit.target}, "
                      f"r2 {fit.r2:.2f}, over {fit.elements_min}..{fit.elements_max} "
                      f"{fit.metric}")}


def affordability(target: str, *, budget_s: float, sizes=None, **kw) -> dict:
    """Per-engine answer to "can this target's cohort be certified inside ``budget_s`` per capsule".

    ``sizes`` is the cohort as ``{capsule: elements}``; omit it to get the fits and the evidence without
    a cohort attached. Every engine reports ``max_elements`` (the largest capsule that fits the per-
    capsule budget) and, when a cohort is given, its total. Both are ``None`` for an engine with no fit,
    which the caller must surface rather than fill in.
    """
    from merlin.targetgen import cert_cost as CC

    got = fits_for(target, **kw)
    out = {"target": str(target), "budget_s": float(budget_s), "engines": {},
           "unattributed_samples": got["unattributed_samples"],
           "unsized_samples": got["unsized_samples"], "metric": got["metric"]}
    for engine, fit in got["engines"].items():
        row: dict = {"n_samples": got["sample_counts"].get(engine, 0),
                     "fit": fit.to_dict() if fit is not None else None,
                     "max_elements": None, "cohort": None}
        if fit is not None:
            # Reuse cert_cost's own inverse so the budget rule (floor exceeds budget -> None; clamp to
            # the measured range) exists in one place.
            row["max_elements"] = CC.max_elements_within(
                CC.CostFit(target=fit.target, intercept_s=fit.intercept_s,
                           per_element_s=fit.per_element_s, r2=fit.r2, n_samples=fit.n_samples,
                           elements_min=fit.elements_min, elements_max=fit.elements_max,
                           metric=fit.metric), budget_s)
        if sizes is not None:
            row["cohort"] = cohort_price(fit, sizes)
        out["engines"][engine] = row
    return out
