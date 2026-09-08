"""The two movement terms, measured from a CONTROLLED transfer-size series.

WHY THIS EXISTS. :mod:`merlin.perf.contract` declares both data-movement terms UNKNOWN and names the
construction that would establish them: *"no fact states this engine's beat width or its issue rate;
the structural walk derives feed-forward depth, and a sequenced engine has none. Needs a measurement
at >=2 transfer sizes to separate the rate from the fixed per-transfer cost."* Until they are
measured, :func:`merlin.perf.optimization_ledger.arithmetic_intensity` returns ``bound_by: UNKNOWN``
and :func:`merlin.perf.roofline.empirical_roofline` refuses -- correctly, because a roofline drawn
through a guessed ridge points optimization effort at whichever axis the guess favoured.

WHY A CONTROLLED SERIES AND NOT A CORPUS FIT. This was tried on the existing corpus first and is
REFUTED, measured: 31 pure-movement programs already carried cycle-accurate measurements at 8
distinct declared byte volumes, and the same volume disagreed with itself by up to **12.8x**
(``A1_mvin_mvout`` at 512 B measured 148, 160, 168, 169 and 1899 cycles). Those are different
emitted programs at one declared volume, so the term that varies between them is not the transfer
size, and a fit would charge their difference to bytes. :mod:`merlin.perf.oracle_cost` records the
same lesson at a different seam -- a corpus-only fit landed **1.77x** off the true marginal rate at
r-squared 0.97, which is why a high r-squared is not the acceptance test here. So the samples must
come from ONE compiler over a series whose only intended difference is the transfer size, and
:func:`fit` refuses a series that does not look like one.

WHAT THE RESULT IS LICENSED TO SAY, which is narrower than it looks. The slope is a MARGINAL rate
over the measured domain, and on a domain where the intercept dominates it is a LOWER bound on the
bandwidth a large transfer can achieve: a big transfer amortizes the per-transfer setup and can
pipeline in ways a small one cannot. A lower bound on bandwidth makes ``peak_compute / bandwidth`` an
**UPPER** bound on the ridge point. Therefore:

* an arithmetic intensity ABOVE that bound is **provably compute-bound**;
* an intensity BELOW it proves nothing at all, and must not be reported as memory-bound.

That asymmetry is the whole result and :attr:`MovementBalance.ridge_licence` states it, because the
tempting reading -- "intensity under the ridge, so memory-bound" -- is exactly the inference this
module's own history shows going wrong.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

__all__ = ["MovementSample", "MovementBalance", "fit", "MIN_DISTINCT_SIZES", "MIN_DOMAIN_RATIO"]

#: Two points can fit a line through any two points; the third is what makes a bad fit visible at
#: all. The repo's standing rule is at least two points per fitted parameter, and there are two.
MIN_DISTINCT_SIZES = 3

#: The measured domain must span at least this ratio, or the slope has no leverage: a series whose
#: sizes all sit within a few percent of each other measures the intercept precisely and the rate
#: not at all, while still reporting a high r-squared.
MIN_DOMAIN_RATIO = 4.0


@dataclass(frozen=True)
class MovementSample:
    """One measured transfer: declared bytes moved, and the cycles a cycle-accurate engine counted."""

    program: str
    bytes_moved: int
    cycles: int
    #: The engine that counted the cycles. Recorded so a series cannot silently mix a cycle-accurate
    #: reading with a functional model's instruction count, which is not a cycle count at all.
    engine: str
    #: The compiler package that emitted this program. A series spanning two packages is two
    #: instruments: the schedules differ, so the difference between them is not the transfer size.
    package: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"program": self.program, "bytes_moved": self.bytes_moved, "cycles": self.cycles,
                "engine": self.engine, "package": self.package}


@dataclass(frozen=True)
class MovementBalance:
    """The fitted movement terms, or a refusal naming what the series could not establish."""

    status: str
    reason: str = ""
    peak_bytes_per_cycle: float | None = None
    base_latency_cycles: float | None = None
    r_squared: float | None = None
    domain_bytes: tuple[int, int] | None = None
    n_samples: int = 0
    n_distinct_sizes: int = 0
    residual_cycles: tuple[float, ...] = ()
    samples: tuple[MovementSample, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    #: The one sentence a consumer must carry with the ridge point. See the module docstring.
    ridge_licence: str = (
        "the slope is a MARGINAL rate over the measured domain and, where the intercept dominates, "
        "a LOWER bound on what a large transfer can achieve -- so peak_compute divided by it is an "
        "UPPER bound on the ridge. An intensity above that bound is provably compute-bound; an "
        "intensity below it proves NOTHING and is not evidence of being memory-bound.")

    @property
    def derived(self) -> bool:
        return self.status == "derived"

    def macs_per_byte_upper_bound(self, peak_macs_per_cycle: float) -> float | None:
        """An UPPER bound on the ridge point, or None. Named for what it is, not ``ridge``.

        A method rather than a field because it needs the compute peak, which belongs to the target
        and not to this measurement -- and because a bare ``ridge`` attribute is what would get read
        as a two-sided verdict.
        """
        if not self.derived or not peak_macs_per_cycle or peak_macs_per_cycle <= 0:
            return None
        if not self.peak_bytes_per_cycle or self.peak_bytes_per_cycle <= 0:
            return None
        return float(peak_macs_per_cycle) / float(self.peak_bytes_per_cycle)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_movement_balance_v1", "status": self.status,
                "reason": self.reason, "peak_bytes_per_cycle": self.peak_bytes_per_cycle,
                "base_latency_cycles": self.base_latency_cycles, "r_squared": self.r_squared,
                "domain_bytes": list(self.domain_bytes) if self.domain_bytes else None,
                "n_samples": self.n_samples, "n_distinct_sizes": self.n_distinct_sizes,
                "residual_cycles": list(self.residual_cycles),
                "samples": [s.to_dict() for s in self.samples],
                "notes": list(self.notes), "ridge_licence": self.ridge_licence}


def _refuse(reason: str, samples: Sequence[MovementSample]) -> MovementBalance:
    return MovementBalance(status="unavailable", reason=reason, n_samples=len(samples),
                           n_distinct_sizes=len({s.bytes_moved for s in samples}),
                           samples=tuple(samples))


def fit(samples: Sequence[MovementSample], *, engine: str | None = None,
        package: str | None = None) -> MovementBalance:
    """Separate the per-byte rate from the fixed per-transfer cost, or refuse and say why.

    Exact rational least squares, so nothing rounds on the way to the coefficients (the same choice
    :mod:`merlin.perf.residency_claim` makes for the same reason). ``engine`` and ``package``, when
    given, are required to match every sample: a series that mixes two engines or two compilers is
    two instruments, and the difference between them is not the transfer size the fit charges it to.
    """
    rows = list(samples)
    if not rows:
        return _refuse("no movement samples were supplied", rows)
    for row in rows:
        if row.bytes_moved <= 0 or row.cycles <= 0:
            return _refuse(
                f"{row.program!r} reports {row.bytes_moved} bytes in {row.cycles} cycles; a "
                f"non-positive extent cannot be a measured transfer", rows)
    if engine is not None:
        wrong = sorted({s.engine for s in rows if s.engine != engine})
        if wrong:
            return _refuse(
                f"the series mixes engine(s) {wrong} with the required {engine!r}; a functional "
                f"model's instruction count is not a cycle count and the two must not be fitted "
                f"together", rows)
    if package is not None:
        wrong = sorted({s.package for s in rows if s.package != package})
        if wrong:
            return _refuse(
                f"the series mixes compiler package(s) {wrong} with the required {package!r}; two "
                f"packages emit two schedules, so their difference is not the transfer size", rows)

    # A byte size that disagrees with itself is the corpus failure this module exists to avoid, and
    # it is refused rather than averaged: averaging would turn a 12.8x disagreement into a number.
    by_size: dict[int, set[int]] = {}
    for row in rows:
        by_size.setdefault(row.bytes_moved, set()).add(row.cycles)
    disagreeing = {size: sorted(cycles) for size, cycles in by_size.items() if len(cycles) > 1}
    if disagreeing:
        worst = max(max(c) / min(c) for c in disagreeing.values())
        return _refuse(
            f"{len(disagreeing)} transfer size(s) measured more than one cycle count "
            f"({disagreeing}), disagreeing by up to {worst:.1f}x. These are different emitted "
            f"programs at one declared volume, so what varies between them is not the transfer "
            f"size; averaging them would charge their difference to bytes", rows)

    sizes = sorted(by_size)
    if len(sizes) < MIN_DISTINCT_SIZES:
        return _refuse(
            f"{len(sizes)} distinct transfer size(s); at least {MIN_DISTINCT_SIZES} are needed "
            f"because two points fit a line through any two points and the third is what makes a "
            f"bad fit visible", rows)
    ratio = sizes[-1] / sizes[0]
    if ratio < MIN_DOMAIN_RATIO:
        return _refuse(
            f"the transfer sizes span only {ratio:.2f}x ({sizes[0]}..{sizes[-1]} B), under the "
            f"{MIN_DOMAIN_RATIO}x this fit requires: over a narrow domain the intercept is measured "
            f"precisely and the rate is not, while r-squared stays high", rows)

    xs = [Fraction(s.bytes_moved) for s in rows]
    ys = [Fraction(s.cycles) for s in rows]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    if sxx == 0:
        return _refuse("every sample has the same transfer size, so no rate is separable", rows)
    slope = sxy / sxx
    if slope <= 0:
        return _refuse(
            f"the fitted slope is {float(slope):.6g} cycles/byte, which is not positive: over this "
            f"series a larger transfer did not cost more, so the samples do not measure a transfer "
            f"rate", rows)
    intercept = mean_y - slope * mean_x
    predicted = [intercept + slope * x for x in xs]
    sst = sum((y - mean_y) ** 2 for y in ys)
    sse = sum((y - p) ** 2 for y, p in zip(ys, predicted, strict=True))
    r2 = float(1 - sse / sst) if sst != 0 else None

    notes = [
        "exact rational least squares; the coefficients do not round on the way out",
        f"the fixed per-transfer cost is {float(intercept):.1f} cycles against a smallest measured "
        f"transfer of {sizes[0]} B, so the intercept dominates this domain and the slope is the "
        f"MARGINAL rate over it -- not the bandwidth a large transfer achieves",
    ]
    if intercept <= 0:
        notes.append(
            "the fitted intercept is not positive, which no sequenced engine can be: read the rate "
            "as unreliable rather than the engine as free")
    return MovementBalance(
        status="derived",
        peak_bytes_per_cycle=float(1 / slope), base_latency_cycles=float(intercept),
        r_squared=r2, domain_bytes=(sizes[0], sizes[-1]), n_samples=n,
        n_distinct_sizes=len(sizes),
        residual_cycles=tuple(round(float(y - p), 3) for y, p in zip(ys, predicted, strict=True)),
        samples=tuple(rows), notes=tuple(notes))


def samples_from_capsule_runs(runs_root: Any, *, engine: str = "gsim",
                              package: str = "") -> tuple[list[MovementSample], list[dict]]:
    """``(samples, refusals)`` over the pure-movement capsule runs under ``runs_root``.

    A run contributes only when it PASSED, its cycles came from a cycle-accurate engine, and every
    command in its emitted program is a movement command whose operands price to a byte count. Each
    of those is a separate refusal with its own reason rather than a silent skip: a series that
    quietly dropped the programs it could not price would be fitted over whatever happened to be
    priceable.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from merlin.perf import lane_cost as LC  # noqa: PLC0415

    root = Path(runs_root)
    out: list[MovementSample] = []
    refusals: list[dict] = []
    for result_path in sorted(root.rglob("capsule_result.json")):
        run_dir = result_path.parent
        name = run_dir.name
        buffer_path = run_dir / "generated" / "command_buffer.json"
        if not buffer_path.is_file():
            refusals.append({"program": name, "reason": "the run kept no emitted command buffer, "
                                                        "so its cycles cannot be attributed"})
            continue
        try:
            document = json.loads(result_path.read_text(encoding="utf-8"))
            buffer = json.loads(buffer_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            refusals.append({"program": name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        tier = None
        for row in (document.get("tiers") or {}).values():
            if (isinstance(row, Mapping) and row.get("cycles") is not None
                    and row.get("derived_from_rtl") is True and row.get("engine") == engine):
                tier = row
                break
        if tier is None:
            refusals.append({"program": name,
                             "reason": f"no cycle count from a cycle-accurate {engine} tier; a "
                                       f"functional model's instruction count is not a cycle count"})
            continue
        if tier.get("status") != "pass":
            refusals.append({"program": name,
                             "reason": f"the tier reached status {tier.get('status')!r}; cycles "
                                       f"from a run that did not compute its declared operation "
                                       f"are not poolable with cycles from one that did"})
            continue
        tensors = buffer.get("tensors") or {}
        total, why = 0, ""
        for index, command in enumerate(buffer.get("commands") or []):
            if str(command.get("opcode") or "") != "MOVEMENT":
                why = (f"command {index} is {command.get('opcode')!r}, not a movement command, so "
                       f"this program's cycles are not all transfer cost")
                break
            for role in ("src", "dst"):
                tensor = tensors.get((command.get("operands") or {}).get(role))
                bits = LC.dtype_bits((tensor or {}).get("dtype"))
                elements = LC._elements((tensor or {}).get("shape"))  # noqa: SLF001 -- one pricer
                if bits is None or elements is None:
                    why = f"command {index} {role} has no priceable shape/dtype"
                    break
                total += elements * bits // 8
            if why:
                break
        if why:
            refusals.append({"program": name, "reason": why})
            continue
        out.append(MovementSample(program=name, bytes_moved=total, cycles=int(tier["cycles"]),
                                  engine=str(tier.get("engine") or engine), package=package))
    return out, refusals
