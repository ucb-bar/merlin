"""Cumulative spend over a run, priced per token bucket.

WHY A STRAIGHT LINE IS WRONG. Joining (0, 0) to (total hours, total cost) draws a plausible-looking
cost curve that contains exactly two measurements and asserts a constant burn rate between them. Runs
do not spend at a constant rate, and the reason is structural: the four token buckets are billed at
very different prices -- on one model, output is 5x fresh input, cached input is a TENTH of it, and a
cache write carries a 25% premium. A run whose early turns build cache and whose later turns read it
spends fast then slow, and the straight line hides the whole shape.

The cumulative token curve already carries all four buckets over time, so the spend curve is
``sum(bucket(t) * rate(bucket))`` -- a real series, one point per usage report.

WHY IT IS CROSS-CHECKED. Pricing is the easiest thing here to get quietly wrong: a rate table keyed
on the wrong spelling of a model id yields a curve that looks fine and is off by a constant factor.
So the endpoint is compared against the total the harness recorded independently, and a curve that
disagrees is NOT drawn. Measured across the corpus: 40 runs agree to within a rounding error, and 10
disagree -- seven of them by exactly the ratio between two of the same vendor's price tiers, which is
a pricing disagreement to resolve rather than a curve to publish.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from merlin.agentreport.availability import Availability, derived, measured, unavailable

#: How far the priced endpoint may sit from the independently recorded total before the curve is
#: disowned. Generous, because the recorded figure and the table can round differently.
_CROSS_CHECK_TOLERANCE = 0.10

#: Bucket order used everywhere here: (fresh input, output, cache read, cache write).
BUCKETS = ("input", "output", "cache_read", "cache_creation")


@dataclass
class CostPoint:
    t_s: float
    usd: float


@dataclass
class CostCurve:
    points: list[CostPoint] = field(default_factory=list)
    model: str = ""
    rate: tuple[float, float, float, float] | None = None
    final_usd: float | None = None
    recorded_usd: float | None = None
    availability: Availability = field(default_factory=Availability)

    @property
    def ok(self) -> bool:
        return len(self.points) >= 2


def build_cost_curve(token_curve: Sequence[dict], model: str,
                     rate_for: Callable[[str], tuple | None],
                     recorded_usd: float | None) -> CostCurve:
    """Price a cumulative token curve, then refuse it if it disagrees with the recorded total.

    ``rate_for`` maps a model id to ``(input, output, cache_read, cache_write)`` in USD per million
    tokens, or ``None`` when the model is unpriced. It is injected rather than imported so the price
    table stays the caller's choice and this stays testable without one."""
    out = CostCurve(model=model, recorded_usd=recorded_usd)
    if not token_curve:
        out.availability.set("cost_curve", unavailable(
            "this run has no token curve, so spend over time cannot be reconstructed — only its "
            "end-of-run total is known"))
        return out

    rate = rate_for(model)
    if not rate or len(rate) < 4:
        out.availability.set("cost_curve", unavailable(
            f"no per-bucket rate is available for model {model!r}, so the four token buckets cannot "
            f"be priced separately. A single blended rate would draw a straight line, which is the "
            f"shape this exists to avoid."))
        return out
    out.rate = tuple(float(x) for x in rate[:4])

    for sample in token_curve:
        usd = sum(float(sample.get(bucket) or 0) * out.rate[i] for i, bucket in enumerate(BUCKETS))
        out.points.append(CostPoint(float(sample.get("t_s") or 0.0), usd / 1e6))
    out.final_usd = out.points[-1].usd if out.points else None

    if recorded_usd is None or recorded_usd <= 0:
        out.availability.set("cost_curve", derived(
            "priced from this run's own token buckets; the harness recorded no total to check it "
            "against, so the curve's shape is supported but its level is not corroborated",
            source="price_table"))
        return out

    off = abs((out.final_usd or 0.0) - recorded_usd) / recorded_usd
    if off > _CROSS_CHECK_TOLERANCE:
        out.availability.set("cost_curve", unavailable(
            f"the priced curve ends at ${out.final_usd:,.2f} against ${recorded_usd:,.2f} recorded "
            f"by the harness ({off:.0%} apart), so the rate this reader used is not the rate that "
            f"run was billed at. Trust the recorded total; this curve is not drawn."))
        return out
    out.availability.set("cost_curve", measured("price_table"))
    return out
