"""Gemmini (systolic-array / NPU) instance of the linear static cost model.

The linear-regressor machinery lives in :class:`merlin.cost_model.linear.LinearCostModel` (target
class); this module is the Gemmini INSTANCE — it sets the Gemmini command vocabulary (``EVENTS``), the
frozen coefficient artifact, and the one Gemmini-specific fold (``mvin3`` bias-load into ``mvin``).
A different systolic/NPU target reuses ``LinearCostModel`` with its own EVENTS + calibrated coeffs.

Predicts region cycles as ``const + Σ_e coeff[e]·n_e`` — the bridge from L2 event counts (Spike commit
log) to L2.5 calibrated cycles without running RTL per candidate. Coefficients are fit once against the
cycle-exact Verilator sim (see ``calibrate.py``). Honest by construction: ``predict_with_band`` carries
the calibration error band; callers must not act on a margin thinner than the band.
"""
from __future__ import annotations

from pathlib import Path

from .linear import LinearCostModel

# Regressors = the Gemmini command counts. ``preload`` is folded into ``compute`` (always 1:1 in real
# WS code); ``mvin3`` (bias into the i32 accumulator) is byte-proportional to ``mvin`` (see predict).
EVENTS = ("config", "mvin_A", "mvin2_B", "compute", "mvout", "fence")
MVIN3_BYTE_RATIO = 4  # acc_t (4B) vs elem_t (1B) per element

_DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "gemmini_cost_coeffs.json"


class GemminiCostModel(LinearCostModel):
    """The Gemmini instance: the systolic EVENTS + coeff artifact + the mvin3 bias-load fold."""

    EVENTS = EVENTS
    DEFAULT_ARTIFACT = _DEFAULT_ARTIFACT

    def predict(self, events: dict[str, float]) -> float:
        """Predict region cycles from a Gemmini command-count dict. Tolerates the raw Spike-decoder
        key ``mvin3_bias``, folding it into ``mvin`` by byte ratio."""
        cyc = super().predict(events)
        cyc += self.coeff.get("mvin_A", 0.0) * MVIN3_BYTE_RATIO * events.get("mvin3_bias", 0.0)
        return cyc
