"""Gemmini static instruction cost model — the shared currency for Stage-F/G and Autocomp.

Predicts region cycles as a linear combination of Gemmini command counts:

    cycles ≈ const + Σ_e  coeff[e] · n_e        for e in EVENTS

This is the bridge from the L2 event counts we already extract (Spike commit log) to the
L2.5 calibrated cycles we want, *without* running RTL per candidate. Coefficients are fit
once against the cycle-exact Verilator sim (see ``calibrate.py``) and frozen in a JSON
artifact. The model is deliberately linear and serial: it captures the dominant per-command
costs (DMA, systolic compute, the full-drain ``fence``) but NOT load/compute overlap — which
is exactly why ``double_buffering`` is parked to true RTL and never costed here.

Honest by construction: ``predict`` carries the calibration error band; callers must not act
on a margin thinner than the band (same rule as the L2.5 fidelity level in the slate).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Regressors. ``preload`` is folded into ``compute`` (always 1:1 in real WS code). ``mvin3``
# (bias into the i32 accumulator) is byte-proportional to ``mvin`` and modeled as a multiple.
EVENTS = ("config", "mvin_A", "mvin2_B", "compute", "mvout", "fence")
MVIN3_BYTE_RATIO = 4  # acc_t (4B) vs elem_t (1B) per element

_DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "gemmini_cost_coeffs.json"


@dataclass
class GemminiCostModel:
    const: float = 0.0
    coeff: dict[str, float] = field(default_factory=dict)
    error: dict[str, float] = field(default_factory=dict)  # mape, max_abs_pct, n_points
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_ARTIFACT) -> "GemminiCostModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(const=d["const"], coeff=d["coeff"],
                   error=d.get("error", {}), meta=d.get("meta", {}))

    def save(self, path: str | Path = _DEFAULT_ARTIFACT) -> None:
        Path(path).write_text(json.dumps(
            {"const": self.const, "coeff": self.coeff, "error": self.error,
             "meta": self.meta}, indent=1), encoding="utf-8")

    def predict(self, events: dict[str, float]) -> float:
        """Predict region cycles from a Gemmini command-count dict.

        Accepts the event keys this module emits (``mvin_A``/``mvin2_B``/...) and tolerates
        the raw Spike-decoder keys (``mvin3_bias``), folding mvin3 into mvin by byte ratio.
        """
        cyc = self.const
        for e in EVENTS:
            cyc += self.coeff.get(e, 0.0) * events.get(e, 0.0)
        # mvin3 (bias load) costs ~MVIN3_BYTE_RATIO mvins of DMA.
        cyc += self.coeff.get("mvin_A", 0.0) * MVIN3_BYTE_RATIO * events.get("mvin3_bias", 0.0)
        return cyc

    def predict_with_band(self, events: dict[str, float]) -> tuple[float, float]:
        """Return (cycles, +/- band) using the calibration MAPE as the relative band."""
        c = self.predict(events)
        return c, c * self.error.get("mape", 0.0)
