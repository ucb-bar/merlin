"""Interface plans: the 4 cost-model variants and the I0–I3 contract ladder.

Two related views of "which interface is exposed", both expressed as cost-model *plans*
(flag-sets consumed by ``cost_model.evaluate_cost``):

  * ``build_variants(rpv, feature)`` — the 4 named variants (baseline / software_visible /
    hardware_managed / oracle) for a single feature, isolating that feature's benefit. Drives
    ``dse_result`` and the ``exploitability_report``.
  * ``contract_plans(rpv)`` — one plan per I0–I3 contract (features combined), driving the
    phase-transition experiment.
"""
from __future__ import annotations

from merlin.design_pressure.synthesize import FEATURE_ACCUMULATOR, FEATURE_RESIDENT

VARIANTS = ["baseline", "software_visible", "hardware_managed", "oracle"]


def _steps(rpv: dict) -> int:
    return max(int(rpv["metrics"].get("steps", 1)), 1)


def _epi(rpv: dict) -> bool:
    return bool(rpv["metrics"].get("has_epilogue", False))


def build_variants(rpv: dict, feature: str) -> dict[str, dict]:
    """The 4 variant plans for ``feature``, holding the other feature constant.

    For ``resident_packed_tensor`` the variants vary weight pack/load amortisation (epilogue
    held constant). For ``accumulator_commit`` they vary whether the i32 intermediate is
    materialised per step (residency held at its 'on' setting so the commit effect is
    isolated).
    """
    steps = _steps(rpv)
    epi = _epi(rpv)
    if feature == FEATURE_RESIDENT:
        # software_visible exposes residency (pays the make_resident+evict setup);
        # hardware_managed reuses the loaded weight implicitly (no exposed setup) but cannot
        # hoist the pack; oracle is idealised (residency with zero setup, perfect batching).
        return {
            "baseline":         _plan(steps, steps, epi, steps),
            "hardware_managed": _plan(steps, 1,     epi, steps),
            "software_visible": _plan(1,     1,     epi, steps, resident_setup=True),
            "oracle":           _plan(1,     1,     epi, 1),
        }
    if feature == FEATURE_ACCUMULATOR:
        # Residency held 'on' (pack once, load once) to isolate the commit effect.
        return {
            "baseline":         _plan(1, 1, True,  steps),
            "hardware_managed": _plan(1, 1, False, steps),
            "software_visible": _plan(1, 1, False, steps, accumulator_setup=True),
            "oracle":           _plan(1, 1, False, 1),
        }
    raise ValueError(f"unknown feature: {feature}")


def contract_plans(rpv: dict) -> dict[str, dict]:
    """One plan per I0–I3 contract (features combined) for the phase-transition plot.

    I0/I1 pay repeated pack+load and (with an epilogue) per-step i32 intermediates; I2 adds
    residency; I3 adds accumulator-commit (single low-precision output, no i32 intermediate).
    I0 and I1 have equal M1 cost — I1's explicit-DMA overlap benefit is an M2 (overlap) story.
    """
    steps = _steps(rpv)
    epi = _epi(rpv)
    return {
        "I0": _plan(steps, steps, epi,   steps),
        "I1": _plan(steps, steps, epi,   steps),
        "I2": _plan(1,     1,     epi,   steps, resident_setup=True),
        "I3": _plan(1,     1,     False, steps, resident_setup=True, accumulator_setup=True),
    }


def _plan(pack_count: int, weight_loads: int, per_step_intermediate: bool,
          dispatch_count: int, resident_setup: bool = False,
          accumulator_setup: bool = False) -> dict:
    return {
        "pack_count": pack_count,
        "weight_loads": weight_loads,
        "per_step_intermediate": per_step_intermediate,
        "dispatch_count": dispatch_count,
        "resident_setup": resident_setup,
        "accumulator_setup": accumulator_setup,
    }
