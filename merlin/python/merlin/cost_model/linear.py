"""Generic linear static cost model — the class the per-target cost models specialize.

Predicts region cycles as a linear combination of per-command counts:

    cycles ≈ const + Σ_e  coeff[e] · n_e        for e in EVENTS

The machinery (fit coefficients + calibration band, load/save a coeff artifact, predict) is
target-agnostic; a concrete backend cost model is an INSTANCE that sets its ``EVENTS`` (its command
vocabulary) and, if needed, folds derived counts (see :class:`~merlin.cost_model.gemmini.
GemminiCostModel`, the systolic/NPU instance). This is the instance→class split for cost models:
the linear regressor is shared; the event set + fitted coefficients are the per-target data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class LinearCostModel:
    const: float = 0.0
    coeff: dict[str, float] = field(default_factory=dict)
    error: dict[str, float] = field(default_factory=dict)  # mape, max_abs_pct, n_points
    meta: dict = field(default_factory=dict)

    EVENTS: ClassVar[tuple[str, ...]] = ()          # the command vocabulary (per-target instance)
    DEFAULT_ARTIFACT: ClassVar[Path | None] = None  # the frozen coeff artifact (per-target instance)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LinearCostModel":
        p = Path(path) if path is not None else cls.DEFAULT_ARTIFACT
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        return cls(const=d["const"], coeff=d["coeff"],
                   error=d.get("error", {}), meta=d.get("meta", {}))

    def save(self, path: str | Path | None = None) -> None:
        p = Path(path) if path is not None else type(self).DEFAULT_ARTIFACT
        Path(p).write_text(json.dumps(
            {"const": self.const, "coeff": self.coeff, "error": self.error,
             "meta": self.meta}, indent=1), encoding="utf-8")

    def priced_events(self) -> tuple[str, ...]:
        """The command vocabulary this model prices.

        A subclass declares ``EVENTS`` for its backend. When it does not -- i.e. this generic class
        is used directly with a loaded calibration artifact, which is how a target-agnostic caller
        reaches a per-target model -- the artifact's own coefficient keys ARE the vocabulary. Without
        this the generic path silently prices nothing and returns the bare intercept, which reads as
        a real (and badly wrong) cycle count rather than as a refusal.
        """
        return self.EVENTS or tuple(sorted(self.coeff))

    def predict(self, events: dict[str, float]) -> float:
        """Predict region cycles from a per-command count dict (over this model's priced events)."""
        cyc = self.const
        for e in self.priced_events():
            cyc += self.coeff.get(e, 0.0) * events.get(e, 0.0)
        return cyc

    def predict_with_band(self, events: dict[str, float]) -> tuple[float, float]:
        """Return (cycles, +/- band) using the calibration MAPE as the relative band."""
        c = self.predict(events)
        return c, c * self.error.get("mape", 0.0)
