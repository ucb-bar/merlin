"""What a cycle-accurate certification COSTS, fitted from the runs this repo has already paid for.

A capsule derived at an application's real shape is worthless if nobody can afford to certify it.
GSIM L3 is ~23x verilator and 143 of 183 capsules demand L3, so "derive a bigger capsule" and "the
corpus still runs" are in direct tension — and the sweet spot between "too small to generalize" and
"too big to simulate" is exactly what nobody can pick by eye.

So it is measured. Every graded run already records ``sim_active_s`` per capsule
(``capsule_grade``'s timing block), and the corpus records each capsule's declared operands. Joining
the two gives a cost model per target, and the shape of it is the useful part:

    gemmini, n=32 measured runs:  seconds ~= 114.8 + 0.0605 * elements   (R^2 = 0.70)

A ~115 second FLOOR that a capsule pays for existing, and ~0.06 s per operand element on top. The
floor dominates below ~1900 elements, and today's capsules are 256-512 — so the corpus is paying
almost the whole cost of a certification to exercise a 16x16 tile, and could grow roughly sevenfold
before size is what it is paying for. That is a fact about this hardware and this oracle, not a
guess, and it is why sizing belongs here rather than in a constant someone picked.

HOW WELL IT PREDICTS, measured rather than hoped. Leave-one-out over those 32 runs -- refit without
each capsule, then predict it -- gives a median absolute error of 17.5%, p90 31%, worst case 51%,
with 31 of 32 inside 50%. So it is a sizing instrument, not a stopwatch: budget with margin and
expect a capsule sized to 300 s to sometimes land near 350. That is the honest reading of an R^2 of
0.70, and it is why the fit reports ``r2`` and its sample count rather than presenting a number.

TWO REFUSALS, both deliberate:

* **No measured history, no fit.** A target nobody has certified yields ``None``, and the caller
  must then decline to promote a capsule to the cert tier rather than size it from a default. A
  default here would be a number nobody measured driving a decision somebody quotes.
* **No extrapolation.** A fit built on 256-4096 elements says nothing about 400,000. Predictions
  outside the measured range (with a small margin) return ``None``, because the honest answer to
  "how long would a capsule 100x bigger than anything we have run take" is that we do not know.

WHICH SIZE METRIC. Measured, not assumed: on the first 15-run subset the largest single operand
predicted cost at R^2 0.914 against 0.908 for total operand elements, and declared OUTPUT elements is
degenerate
because a capsule records its inputs and not its result shape. That last point matters, because the
memory-regime module reasons that a deep-K sweep is cheap on the grounds that cost tracks output
size (``memory_regime.deep_k_rows``). It is cheapER -- doubling K moved A3 only 5 s -- but the
largest operand does grow with K, so deep-K is not free and this module does not pretend it is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CostFit", "fit_for", "max_elements_within", "predict_seconds", "capsule_elements"]

#: How far past the largest measured capsule a prediction is still honest, as a multiple. A fit is a
#: local linearisation of a simulator's behaviour, not a law; beyond this the answer is "unknown".
_EXTRAPOLATION_MARGIN = 2.0

#: Fewest measured capsules a fit may rest on. Two points define a line through anything.
_MIN_SAMPLES = 5


@dataclass(frozen=True)
class CostFit:
    """``seconds ~= intercept_s + per_element_s * elements``, with the evidence it rests on."""

    target: str
    intercept_s: float
    per_element_s: float
    r2: float
    n_samples: int
    elements_min: int
    elements_max: int
    metric: str = "max_operand_elements"
    sources: tuple[str, ...] = ()

    @property
    def floor_dominates_below(self) -> int:
        """Elements below which the fixed cost exceeds the size-dependent cost.

        The number that says how much bigger a capsule can get before it is paying for its size
        rather than for existing at all -- i.e. where representativeness is nearly free.
        """
        if self.per_element_s <= 0:
            return 0
        return int(self.intercept_s / self.per_element_s)

    def to_dict(self) -> dict:
        return {
            "target": self.target, "intercept_s": round(self.intercept_s, 3),
            "per_element_s": round(self.per_element_s, 6), "r2": round(self.r2, 4),
            "n_samples": self.n_samples,
            "measured_range_elements": [self.elements_min, self.elements_max],
            "metric": self.metric, "sources": list(self.sources),
            "floor_dominates_below_elements": self.floor_dominates_below,
        }


def capsule_elements(capsule_yaml: dict) -> int:
    """The size metric: the largest single declared operand, in elements.

    Chosen by measurement rather than by argument -- see the module docstring. Operands with no
    shape contribute nothing instead of raising: a capsule that declares one is not thereby
    unmeasurable, it just does not move this metric.
    """
    biggest = 0
    for operand in (capsule_yaml.get("inputs") or ()):
        n = 1
        for dim in (operand.get("shape") or ()):
            try:
                n *= int(dim)
            except (TypeError, ValueError):        # a symbolic dim contributes no size
                n = 0
                break
        biggest = max(biggest, n)
    return biggest


#: What a CERTIFICATION costs is the cycle-accurate tier's own time, never the sum over tiers. The two
#: differ by orders of magnitude on the same capsule -- measured on PC00_k64, spike at L2 took 0.009s
#: while verilator at L3 took 698.2s -- so a model fitted on the sum is fitted on whichever tiers
#: happened to run and cannot answer "how big may this capsule be and still be certifiable".
_CYCLE_ACCURATE_ONLY = "cycle_accurate_tier"
_SUMMED_LEGACY = "summed_over_tiers(legacy score file, no per-tier block)"


def _cycle_accurate_seconds(timing: dict) -> tuple[float | None, str]:
    """``(seconds, basis)`` for the cycle-accurate tier of one capsule's timing entry.

    Prefers the per-tier block and selects the tier that DECLARES itself cycle-accurate, rather than
    assuming a tier name means an oracle kind (a target may certify on any rung its contract
    declares). Falls back to the summed scalar only for score files written before the per-tier block
    existed, and says so in the basis so a fit over mixed provenance is visible rather than implied.
    """
    by_tier = timing.get("by_tier")
    if isinstance(by_tier, dict) and by_tier:
        best = None
        for name, rec in by_tier.items():
            if not isinstance(rec, dict):
                continue
            # `cycle_accurate` is the property the cost question is about; `derived_from_rtl` is
            # accepted as the older spelling of the same claim.
            if not (rec.get("cycle_accurate") is True or rec.get("derived_from_rtl") is True):
                continue
            secs = rec.get("sim_active_s")
            if isinstance(secs, (int, float)) and secs > 0:
                # Deepest reported wins if several qualify; a longer one is the binding cost.
                if best is None or secs > best[0]:
                    best = (float(secs), f"{_CYCLE_ACCURATE_ONLY}:{name}")
        if best:
            return best
        return None, "no cycle-accurate tier ran for this capsule"
    secs = timing.get("sim_active_s")
    if isinstance(secs, (int, float)) and secs > 0:
        return float(secs), _SUMMED_LEGACY
    return None, "no positive sim_active_s"


def _per_tier_from_result(doc: dict) -> dict:
    """A capsule_result's ``tiers`` reshaped into the ``by_tier`` block a score file carries.

    The per-capsule result is the PRIMARY record and always has per-tier timing; a score file's
    ``timing_diagnostic`` is a roll-up of it. Reading both means a cost sample comes from any capsule
    run, not only from a graded batch -- which is what a single calibration run produces, and it was
    otherwise invisible to this model.
    """
    out = {}
    for name, rec in (doc.get("tiers") or {}).items():
        if not isinstance(rec, dict):
            continue
        tm = rec.get("timing")
        if not isinstance(tm, dict):
            continue
        out[str(name)] = {"sim_active_s": tm.get("sim_active_s"),
                          "build_s": tm.get("build_s"),
                          "oracle_wait_s": tm.get("oracle_wait_s"),
                          "cycle_accurate": rec.get("cycle_accurate"),
                          "derived_from_rtl": rec.get("derived_from_rtl"),
                          "evidence": rec.get("evidence")}
    return out


def _timing_records(target: str, root: Path | None = None,
                    extra_roots=()) -> dict[str, tuple[float, str]]:
    """``capsule -> (cycle_accurate_seconds, source)`` from every run this target has on disk.

    Two record kinds are read, because they are written by different paths: a score file's
    ``timing_diagnostic`` (the batch grader's roll-up) and a ``capsule_result.json``'s ``tiers``
    (the per-capsule primary record, which a single-capsule run writes and a score file does not
    exist for). Later files win on a repeat, which is what "the most recent measurement" means when a
    capsule has been certified more than once. A run that never reached a cycle-accurate tier
    contributes NOTHING rather than its functional time -- a fit that absorbed those would read a
    near-zero cost for a capsule nobody certified.
    """
    from merlin.common.paths import artifacts_dir, runs_dir

    bases = [Path(root)] if root else [artifacts_dir() / "capsule-bench" / str(target),
                                       runs_dir()]
    bases += [Path(r) for r in extra_roots]
    out: dict[str, tuple[float, str]] = {}
    for base in bases:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            if path.name != "capsule_result.json" and not path.name.startswith("score"):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):   # unreadable is not a measurement
                continue
            block = doc.get("timing_diagnostic")
            if isinstance(block, dict) and block:
                for name, timing in block.items():
                    if not isinstance(timing, dict):
                        continue
                    seconds, basis = _cycle_accurate_seconds(timing)
                    if seconds is not None:
                        out[str(name)] = (seconds, f"{path}#{basis}")
                continue
            name = doc.get("capsule")
            per_tier = _per_tier_from_result(doc)
            if name and per_tier:
                seconds, basis = _cycle_accurate_seconds({"by_tier": per_tier})
                if seconds is not None:
                    out[str(name)] = (seconds, f"{path}#{basis}")
    return out


def _capsule_sizes(corpus_roots) -> dict[str, int]:
    """``capsule -> size metric`` for every capsule under the given roots."""
    import yaml

    sizes: dict[str, int] = {}
    for root in corpus_roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for cy in base.rglob("capsule.yaml"):
            try:
                doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            name = str(doc.get("name") or cy.parent.name)
            size = capsule_elements(doc)
            if size > 0:
                sizes[name] = size
    return sizes


def fit_for(target: str, *, corpus_roots=None, timing_root=None,
            extra_timing_roots=()) -> "CostFit | None":
    """The cost model for ``target``, or ``None`` when nothing has been measured.

    ``None`` is a real answer and the caller must honour it: a target with no certification history
    has no basis for sizing a capsule to a time budget, and the correct response is to leave its
    capsules at the shallow tier rather than to certify a size nobody has evidence for.
    """
    from merlin.common.paths import merlin_dir

    timings = _timing_records(target, timing_root, extra_roots=extra_timing_roots or ())
    if not timings:
        return None
    roots = list(corpus_roots) if corpus_roots else [merlin_dir() / "contract" / "capsules"]
    sizes = _capsule_sizes(roots)

    xs: list[int] = []
    ys: list[float] = []
    sources: set[str] = set()
    for name, (seconds, source) in sorted(timings.items()):
        size = sizes.get(name)
        if not size:
            continue
        xs.append(size)
        ys.append(seconds)
        sources.add(source)
    if len(xs) < _MIN_SAMPLES or len(set(xs)) < 2:
        return None                                # a line through one x tells you nothing

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - (sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / ss_tot) if ss_tot else 0.0
    return CostFit(target=str(target), intercept_s=intercept, per_element_s=slope, r2=r2,
                   n_samples=n, elements_min=min(xs), elements_max=max(xs),
                   sources=tuple(sorted(sources)))


def predict_seconds(fit: "CostFit | None", elements: int) -> "float | None":
    """Predicted certification seconds, or ``None`` when the question is outside the evidence.

    Refuses below zero elements and above the measured range times the margin. A prediction the fit
    cannot support is not a large number, it is an absence -- reporting one anyway is how a capsule
    nobody could afford gets scheduled on the strength of arithmetic.
    """
    if fit is None or elements <= 0:
        return None
    if elements > fit.elements_max * _EXTRAPOLATION_MARGIN:
        return None
    return fit.intercept_s + fit.per_element_s * float(elements)


def max_elements_within(fit: "CostFit | None", budget_s: float) -> "int | None":
    """The largest capsule whose predicted certification fits ``budget_s``.

    ``None`` when there is no fit, when the budget cannot even cover the fixed floor (no capsule of
    any size fits, which is a statement about the budget rather than about the shape), or when the
    answer would lie outside the measured range -- in which case it is clamped to the range and the
    caller gets the largest size the evidence actually supports.
    """
    if fit is None or budget_s <= 0:
        return None
    if budget_s <= fit.intercept_s:
        return None                                # the floor alone exceeds the budget
    if fit.per_element_s <= 0:
        return int(fit.elements_max)               # size did not move cost over the measured range
    raw = int((budget_s - fit.intercept_s) / fit.per_element_s)
    ceiling = int(fit.elements_max * _EXTRAPOLATION_MARGIN)
    return max(1, min(raw, ceiling))

# ---------------------------------------------------------------------------------------------------
# sizing by WORK rather than by shape
# ---------------------------------------------------------------------------------------------------
# A shape metric is a proxy, and a weak one: measured over 72 real gemmini certifications, seconds vs
# `max_operand_elements` has r2 0.20, and the best of five shape candidates (`output_elements`) only
# reaches 0.33. An RTL simulator's time is not spent on a tensor's declared extent, it is spent
# advancing cycles -- so the honest independent variable is the cycle count, and it happens to be
# something we can obtain almost free. Measured on the calibration ladder: the FUNCTIONAL tier costs
# 0.006-0.008s and reports a cycle count, and the cycle-accurate tier's count tracks it at a roughly
# stable ratio. So a capsule that has only ever run at L2 can still be sized for L3.
#
# This does not replace the shape fit; a capsule that has never run at all has no cycles either, and
# the shape fit is the only thing that can speak for it. Both are offered, and each says what it rests
# on rather than pretending to one authority.


@dataclass(frozen=True)
class CycleCostFit:
    """``seconds ~= intercept_s + per_cycle_s * cycles`` on the cycle-accurate tier."""

    target: str
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

    def to_dict(self) -> dict:
        return {"target": self.target, "intercept_s": round(self.intercept_s, 3),
                "per_cycle_s": round(self.per_cycle_s, 6), "r2": round(self.r2, 4),
                "n_samples": self.n_samples,
                "measured_range_cycles": [self.cycles_min, self.cycles_max],
                "functional_to_cycle_accurate_ratio": (round(self.functional_ratio, 3)
                                                       if self.functional_ratio else None),
                "n_ratio_samples": self.n_ratio_samples, "sources": list(self.sources)}


def _cycle_records(target: str, root: Path | None = None, extra_roots=()) -> dict[str, dict]:
    """``capsule -> {seconds, cycles, functional_cycles, source}`` for cycle-accurate runs.

    Only a tier that DECLARES itself cycle-accurate contributes seconds and cycles; a functional
    tier's cycle count is kept separately, as the cheap predictor, and never as the cost itself.
    """
    from merlin.common.paths import artifacts_dir, runs_dir

    bases = [Path(root)] if root else [artifacts_dir() / "capsule-bench" / str(target), runs_dir()]
    bases += [Path(r) for r in extra_roots]
    out: dict[str, dict] = {}
    for base in bases:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("capsule_result.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = doc.get("capsule")
            if not name:
                continue
            secs = cycles = func_cycles = None
            for rec in (doc.get("tiers") or {}).values():
                if not isinstance(rec, dict):
                    continue
                tm = rec.get("timing") if isinstance(rec.get("timing"), dict) else {}
                accurate = rec.get("cycle_accurate") is True or rec.get("derived_from_rtl") is True
                c = rec.get("cycles")
                if accurate:
                    sv = tm.get("sim_active_s")
                    if isinstance(sv, (int, float)) and sv > 0 and isinstance(c, int) and c > 0:
                        if secs is None or sv > secs:
                            secs, cycles = float(sv), int(c)
                elif isinstance(c, int) and c > 0 and func_cycles is None:
                    func_cycles = int(c)
            if secs is not None:
                out[str(name)] = {"seconds": secs, "cycles": cycles,
                                  "functional_cycles": func_cycles, "source": str(path)}
    return out


def fit_cycles_for(target: str, *, timing_root=None, extra_timing_roots=()) -> "CycleCostFit | None":
    """Seconds-per-cycle for ``target``'s cycle-accurate tier, or None when too little was measured.

    ``None`` is a real answer, honoured the same way :func:`fit_for`'s is: a target nobody has timed
    cannot have its capsules sized to a budget, and the correct response is to leave them shallow
    rather than certify a size on a guess.
    """
    recs = _cycle_records(target, timing_root, extra_timing_roots)
    xs = [r["cycles"] for r in recs.values()]
    ys = [r["seconds"] for r in recs.values()]
    ratios = [r["cycles"] / r["functional_cycles"] for r in recs.values()
              if r.get("functional_cycles")]
    if len(xs) < _MIN_SAMPLES or len(set(xs)) < 2:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    icept = my - slope * mx
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - (sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys)) / sst) if sst else 0.0
    med = None
    if ratios:
        rs = sorted(ratios)
        med = rs[len(rs) // 2] if len(rs) % 2 else (rs[len(rs) // 2 - 1] + rs[len(rs) // 2]) / 2
    return CycleCostFit(target=str(target), intercept_s=icept, per_cycle_s=slope, r2=r2,
                        n_samples=n, cycles_min=min(xs), cycles_max=max(xs),
                        functional_ratio=med, n_ratio_samples=len(ratios),
                        sources=tuple(sorted({r["source"] for r in recs.values()})))


def predict_seconds_from_cycles(fit: "CycleCostFit | None", cycles: int) -> "float | None":
    """Certification seconds for a capsule expected to run ``cycles``, or None with no fit."""
    if fit is None or not cycles or cycles <= 0:
        return None
    return fit.intercept_s + fit.per_cycle_s * float(cycles)


def predict_seconds_from_functional_cycles(fit: "CycleCostFit | None",
                                           functional_cycles: int) -> "tuple[float | None, str]":
    """Estimate the cycle-accurate cost from a FUNCTIONAL run's cycle count.

    This is the cheap path the ladder exists to justify: the functional tier costs milliseconds and
    reports a cycle count, so a capsule can be priced for certification without ever being certified.
    Returns ``(seconds, basis)`` and refuses -- rather than guessing -- when no ratio was measured.
    """
    if fit is None:
        return None, "no cycle cost fit for this target"
    if not functional_cycles or functional_cycles <= 0:
        return None, "no functional cycle count to scale"
    if not fit.functional_ratio:
        return None, ("no capsule has run at BOTH tiers on this target, so the functional-to-"
                      "cycle-accurate cycle ratio is unmeasured and cannot be assumed")
    est = fit.intercept_s + fit.per_cycle_s * functional_cycles * fit.functional_ratio
    return est, (f"functional cycles x measured ratio {fit.functional_ratio:.2f} "
                 f"(n={fit.n_ratio_samples}), then {fit.per_cycle_s:.4f} s/cycle "
                 f"over a {fit.intercept_s:.0f}s floor")


def max_cycles_within(fit: "CycleCostFit | None", budget_s: float) -> "int | None":
    """The most cycles a capsule may run and still certify inside ``budget_s``.

    Clamped to the measured range for the same reason :func:`max_elements_within` is: past the
    evidence the line is an opinion.
    """
    if fit is None or budget_s <= 0 or fit.per_cycle_s <= 0:
        return None
    if budget_s <= fit.intercept_s:
        return None
    raw = int((budget_s - fit.intercept_s) / fit.per_cycle_s)
    return max(1, min(raw, int(fit.cycles_max * _EXTRAPOLATION_MARGIN)))

