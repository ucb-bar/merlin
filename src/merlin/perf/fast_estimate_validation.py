"""Grouped validation for empirical cycle screening, independent of the target.

Providers select JSON-pointer features and bind observations to exact executable,
input, engine, timing scope and memory-regime evidence. A statistical fit here is
only a screening signal: its coefficients do not establish physical service rates,
memory traffic or overlap. Mechanism calibration remains in
``phase2_feature_calibration`` and resource composition in
``phase2_analytical_provider``.

All variants of one workload are held out together. Predictions outside a training
feature domain remain UNKNOWN. Validation reports both absolute error and the
within-workload ordering the optimizer needs; training error cannot approve a fit.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_sha256
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from . import rank_validation as rank


@dataclass(frozen=True)
class Observation:
    program: str
    workload: str
    group: str
    domain: str
    features: Mapping[str, float | None]
    cycles: float
    evidence_sha256s: tuple[str, ...]

    @property
    def id(self) -> str:
        return canonical_sha256([self.program, self.workload])

    def __post_init__(self) -> None:
        if not all(is_sha256(x) for x in (self.program, self.workload, self.domain)):
            raise ValueError("observation must bind executable, workload and domain digests")
        if not self.group or not self.evidence_sha256s or not all(is_sha256(x) for x in self.evidence_sha256s):
            raise ValueError("observation requires a leakage group and evidence digests")
        if isinstance(self.cycles, bool) or not math.isfinite(self.cycles) or self.cycles <= 0:
            raise ValueError("observed cycles must be positive and finite")
        for pointer, value in self.features.items():
            if not pointer.startswith("/"):
                raise ValueError("features must use absolute provider JSON pointers")
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError("feature values must be finite and nonnegative, or UNKNOWN")


class Predictor(Protocol):
    def predict(self, features: Mapping[str, float | None], *, domain_sha256: str) -> CycleInterval: ...


@dataclass(frozen=True)
class LinearScreen:
    domain_sha256: str
    pointers: tuple[str, ...]
    coefficients: tuple[float, ...]
    fixed_cycles: float
    domains: tuple[tuple[float, float], ...]
    provenance_sha256: str

    def predict(self, features: Mapping[str, float | None], *, domain_sha256: str) -> CycleInterval:
        if domain_sha256 != self.domain_sha256:
            return CycleInterval.unknown("target, timing scope or execution regime differs from fit")
        values: list[float] = []
        for pointer, (lo, hi) in zip(self.pointers, self.domains, strict=True):
            value = features.get(pointer)
            if value is None:
                return CycleInterval.unknown(f"feature {pointer} is UNKNOWN")
            if isinstance(value, bool) or not math.isfinite(value) or value < lo or value > hi:
                return CycleInterval.unknown(f"feature {pointer} is outside the training domain")
            values.append(float(value))
        cycles = self.fixed_cycles + sum(
            x * coefficient for x, coefficient in zip(values, self.coefficients, strict=True)
        )
        if not math.isfinite(cycles) or cycles < 0:
            return CycleInterval.unknown("screening prediction has no finite nonnegative cost")
        return CycleInterval.point(cycles, f"empirical screening fit sha256:{self.provenance_sha256}")


def fit_linear_screen(
    observations: Sequence[Observation],
    *,
    pointers: Sequence[str],
    include_fixed: bool,
    maximum_condition: float,
) -> LinearScreen:
    """Diagnostic least-squares fit, refusing underdetermined or negative rates.

    The fixed term is an explicit caller choice. Features are scaled for numerical
    conditioning. No fitted coefficient is exported as a mechanism calibration.
    """
    pointers = tuple(pointers)
    if not pointers or len(set(pointers)) != len(pointers):
        raise ValueError("select distinct feature pointers")
    if not math.isfinite(maximum_condition) or maximum_condition < 1:
        raise ValueError("maximum condition must be finite and at least one")
    if len({row.domain for row in observations}) != 1:
        raise ValueError("fit requires one exact target, timing scope and execution regime")
    if any(row.features.get(key) is None for row in observations for key in pointers):
        raise ValueError("fit has UNKNOWN features")
    matrix = np.array([[row.features[key] for key in pointers] for row in observations], dtype=float)
    if len({tuple(row) for row in matrix}) < 2 * (len(pointers) + int(include_fixed)):
        raise ValueError("fit needs at least two distinct points per parameter")
    scale = np.maximum(np.max(matrix, axis=0), 1.0)
    scaled = matrix / scale
    if include_fixed:
        scaled = np.column_stack((np.ones(len(scaled)), scaled))
    condition = float(np.linalg.cond(scaled))
    if not math.isfinite(condition) or condition > maximum_condition:
        raise ValueError("fit features are collinear or ill-conditioned")
    coefficients, _, matrix_rank, _ = np.linalg.lstsq(
        scaled, np.array([row.cycles for row in observations]), rcond=None
    )
    if matrix_rank != scaled.shape[1] or any(x < 0 for x in coefficients):
        raise ValueError("fit does not identify nonnegative screening terms")
    fixed = float(coefficients[0]) if include_fixed else 0.0
    slopes = coefficients[1:] if include_fixed else coefficients
    receipt = canonical_sha256(
        {
            "pointers": pointers,
            "include_fixed": include_fixed,
            "maximum_condition": maximum_condition,
            "scaled_coefficients": coefficients.tolist(),
            "observations": [
                {
                    "program": row.program,
                    "workload": row.workload,
                    "domain": row.domain,
                    "features": dict(row.features),
                    "cycles": row.cycles,
                    "evidence": row.evidence_sha256s,
                }
                for row in observations
            ],
        }
    )
    return LinearScreen(
        observations[0].domain,
        pointers,
        tuple(float(x) for x in slopes / scale),
        fixed,
        tuple((float(min(col)), float(max(col))) for col in matrix.T),
        receipt,
    )


def feature_collisions(observations: Sequence[Observation], pointers: Sequence[str]) -> list[dict[str, Any]]:
    """Report schedules indistinguishable to a feature set but different in time."""
    buckets: dict[tuple[Any, ...], list[Observation]] = {}
    for row in observations:
        values = tuple(row.features.get(key) for key in pointers)
        if None not in values:
            buckets.setdefault((row.domain, row.workload, *values), []).append(row)
    collisions = []
    for rows in buckets.values():
        lo, hi = min(r.cycles for r in rows), max(r.cycles for r in rows)
        if hi > lo:
            collisions.append(
                {
                    "programs": [row.program for row in rows],
                    "cycles": [lo, hi],
                    "minimum_point_relative_error": (hi - lo) / (hi + lo),
                    "features": {key: rows[0].features[key] for key in pointers},
                }
            )
    return collisions


def cross_validate(
    observations: Sequence[Observation],
    fit: Callable[[Sequence[Observation]], Predictor],
    *,
    maximum_relative_error: float,
    minimum_predictions: int,
    minimum_rank_rate: float,
    minimum_decided: int,
    minimum_slice_decided: int,
    minimum_slices: int,
) -> dict[str, Any]:
    """Fit each fold without any measured label from its held-out workload group."""
    if len({row.domain for row in observations}) != 1:
        raise ValueError("validation cannot mix target, scope or execution regimes")
    if len({row.id for row in observations}) != len(observations):
        raise ValueError("aggregate executable replicates before validation")
    groups: dict[str, set[str]] = {}
    for row in observations:
        groups.setdefault(row.workload, set()).add(row.group)
    if any(len(g) != 1 for g in groups.values()):
        raise ValueError("all variants of a workload must share one held-out group")
    if not math.isfinite(maximum_relative_error) or maximum_relative_error < 0:
        raise ValueError("maximum relative error must be finite and nonnegative")
    if minimum_predictions < 1:
        raise ValueError("validation must require held-out predictions")
    if not 0.5 < minimum_rank_rate <= 1 or any(
        isinstance(x, bool) or not isinstance(x, int) or x < 1
        for x in (minimum_predictions, minimum_decided, minimum_slice_decided, minimum_slices)
    ):
        raise ValueError("ranking must require positive evidence counts and agreement above chance")
    intervals: dict[str, tuple[float, float]] = {}
    records = []
    errors = []
    for group in sorted({row.group for row in observations}):
        train = [row for row in observations if row.group != group]
        test = [row for row in observations if row.group == group]
        try:
            model = fit(train)
            fit_error = None
        except ValueError as exc:
            model, fit_error = None, str(exc)
        for row in test:
            prediction = (
                model.predict(row.features, domain_sha256=row.domain)
                if model
                else CycleInterval.unknown(fit_error or "no fit")
            )
            record = {
                "id": row.id,
                "program": row.program,
                "workload": row.workload,
                "group": group,
                "measured_cycles": row.cycles,
                "training_programs": [r.program for r in train],
                "prediction": prediction.to_dict(),
            }
            if prediction.resolved:
                lo, hi = float(prediction.lo), float(prediction.hi)
                if not math.isfinite(lo) or not math.isfinite(hi):
                    raise ValueError("predictor returned nonfinite cycles")
                intervals[row.id] = (lo, hi)
                error = abs((lo + hi) / 2 - row.cycles) / row.cycles
                errors.append(error)
                record.update(relative_error=error, contains_observation=lo <= row.cycles <= hi)
            records.append(record)
    programs = [rank.Program(row.workload, row.id, row.cycles, row.group) for row in observations]
    overall = rank.interval_agreement(rank.ordered_pairs(programs), intervals)
    slices = {
        group: rank.interval_agreement(rank.ordered_pairs([p for p in programs if p.group == group]), intervals)
        for group in sorted({p.group for p in programs})
    }
    gate = rank.verdict(
        overall,
        slices,
        minimum_rate=minimum_rank_rate,
        minimum_decided=minimum_decided,
        minimum_slice_decided=minimum_slice_decided,
        minimum_slices=minimum_slices,
    )
    reasons = list(gate["reasons"])
    if len(errors) < minimum_predictions:
        reasons.append(f"only {len(errors)} held-out predictions; requires {minimum_predictions}")
    if errors and max(errors) > maximum_relative_error:
        reasons.append(f"maximum held-out relative error {max(errors):.6g} exceeds {maximum_relative_error}")
    return {
        "schema": "empirical_cycle_screen_validation_v1",
        "domain_sha256": observations[0].domain,
        "exposable": not reasons,
        "reasons": reasons,
        "predictions": records,
        "absolute_error": {
            "n": len(errors),
            "median_relative": statistics.median(errors) if errors else None,
            "maximum_relative": max(errors) if errors else None,
        },
        "ranking": gate,
        "coefficient_scope": "empirical screening only; no physical service-rate inference",
        "thresholds": {"maximum_relative_error": maximum_relative_error, "minimum_predictions": minimum_predictions},
    }
