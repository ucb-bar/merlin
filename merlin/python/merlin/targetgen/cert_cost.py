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


def _timing_records(target: str, root: Path | None = None) -> dict[str, tuple[float, str]]:
    """``capsule -> (sim_active_s, source)`` from every graded run this target has on disk.

    Later files win on a repeat, which is what "the most recent measurement" means when a capsule
    has been certified more than once.
    """
    from merlin.common.paths import artifacts_dir

    base = Path(root) if root else (artifacts_dir() / "capsule-bench" / str(target))
    out: dict[str, tuple[float, str]] = {}
    if not base.is_dir():
        return out
    for path in sorted(base.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):    # an unreadable score file is not a measurement
            continue
        block = doc.get("timing_diagnostic")
        if not isinstance(block, dict):
            continue
        for name, timing in block.items():
            seconds = (timing or {}).get("sim_active_s") if isinstance(timing, dict) else None
            if isinstance(seconds, (int, float)) and seconds > 0:
                out[str(name)] = (float(seconds), str(path))
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


def fit_for(target: str, *, corpus_roots=None, timing_root=None) -> "CostFit | None":
    """The cost model for ``target``, or ``None`` when nothing has been measured.

    ``None`` is a real answer and the caller must honour it: a target with no certification history
    has no basis for sizing a capsule to a time budget, and the correct response is to leave its
    capsules at the shallow tier rather than to certify a size nobody has evidence for.
    """
    from merlin.common.paths import merlin_dir

    timings = _timing_records(target, timing_root)
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
