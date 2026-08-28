"""The localized-error veto for the COSINE-ONLY tier, derived per run instead of tuned per model.

``t3`` accepts a whole-model output on cosine alone. Its rationale is sound but incomplete: a
whole-model output legitimately carries large per-element RELATIVE error on its many small elements,
so the ``MERLIN_GATE_MAX_REL`` ceiling cannot gate one. Measured on the tracked recaptures, the
deviation of a CORRECT host int8 reference (``golden_w8a8``) from the fp32 golden is:

    small_llama_int8   max|d|/rms = 0.027    per-element max-rel =  1.3
    spectformer_int8   max|d|/rms = 0.365    per-element max-rel =  9.3
    gemma2_2b_int8     max|d|/rms = 1.880    per-element max-rel = 99.0

So the 0.05 relative ceiling rejects every correct implementation, and no FIXED absolute bound works
either -- correct references span 0.027 to 1.88. Any constant would be fitted to whichever model was
looked at first. What is comparable is the run against THE SAME MODEL's own quantization noise: a
conformant accelerator computes the same quantized math in a different but equally valid rounding and
accumulation order, so it lands within a small multiple of the floor its own references already cost;
a localized blow-up lands far outside it.

The consequence that matters: t3 used to apply NO per-element veto at all, so the 1209%-style
localized blow-up -- the exact hole the per-element term was introduced to close -- passed t3
untouched whenever it reached that tier. These tests pin the veto in both directions, and pin that it
stays OFF (rather than guessing) when the floor cannot be measured.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.runtime.backends.zephyr_model import _gate


def _model(n=4096, seed=0, quant_noise=0.006, run_noise=0.009, scale=1.0):
    """fp32 golden, a host-int8 golden offset from it by `quant_noise`, and a run offset by
    `run_noise`. `scale` moves the whole model's magnitude so the bound cannot be a constant."""
    rng = np.random.default_rng(seed)
    f = (rng.standard_normal(n) * scale).astype(np.float32)
    q = (f + rng.standard_normal(n) * quant_noise * scale).astype(np.float32)
    p = (f + rng.standard_normal(n) * run_noise * scale).astype(np.float32)
    return p, f, q


def test_a_run_within_its_own_quantization_floor_is_guarded_not_cosine_only():
    p, f, q = _model()
    g = _gate(p, {"fp32": f, "w8a8": q})
    assert g["ok"] is True
    assert g["per_element_guarded"] is True, "the derived veto DID apply -- this is not cosine-only"
    assert g["per_element_basis"] == "quantization_excess"
    assert g["quant_excess"] < 4.0


def test_a_localized_blowup_is_caught_though_cosine_stays_near_perfect():
    """The 1209% shape at the tier that previously had no per-element term."""
    p, f, q = _model()
    p = p.copy()
    p[7] += 12.0 * float(np.abs(q - f).max())      # one element far outside the floor, yet invisible to cosine
    g = _gate(p, {"fp32": f, "w8a8": q})
    assert g["fp32_cos"] > 0.9999, f"cosine must still look fine: {g['fp32_cos']}"
    assert g["ok"] is False, "a localized blow-up must not pass on cosine alone"
    assert g["quant_excess"] > 4.0


def test_the_bound_is_derived_so_models_with_very_different_floors_both_pass():
    """A constant cannot do this: these two models' ABSOLUTE floors differ by ~1000x, and both
    runs are conformant relative to their OWN floor."""
    for scale, qn in ((1.0, 0.001), (1000.0, 0.001)):
        p, f, q = _model(quant_noise=qn, run_noise=qn * 1.5, scale=scale)
        g = _gate(p, {"fp32": f, "w8a8": q})
        assert g["ok"] is True, f"scale={scale} floor={g['quant_floor_abs']:.4g} rejected"
        assert g["per_element_guarded"] is True


def test_an_unmeasurable_floor_leaves_the_veto_off_rather_than_guessing():
    """Identical references (floor 0) and a single reference both make the floor unmeasurable. The
    gate must then report itself as UNguarded -- never manufacture a bound it could not measure."""
    p, f, _ = _model()
    for refs in ({"fp32": f, "w8a8": f}, {"fp32": f}):
        g = _gate(p, refs)
        assert "quant_excess" not in g, f"no floor was measurable, so no excess may be reported: {refs.keys()}"
        assert g["per_element_guarded"] is False
        assert g["per_element_basis"] is None


def test_the_relative_veto_still_owns_the_tiers_where_it_is_meaningful():
    """A bit-close run is guarded by the RELATIVE veto (t1), not by the derived one -- the derived
    bound is the fallback for outputs the relative bound cannot gate, not a replacement."""
    _, f, _ = _model(quant_noise=0.0005, run_noise=0.0)
    g = _gate(f.copy(), {"fp32": f, "w8a8": f * np.float32(1.0)})
    assert g["per_element_basis"] == "relative"


@pytest.mark.parametrize("excess", [0.0, ""])
def test_the_veto_can_be_disabled(monkeypatch, excess):
    monkeypatch.setattr("merlin.runtime.backends.zephyr_model._GATE_QUANT_EXCESS", float(excess or 0))
    p, f, q = _model()
    p = p.copy(); p[7] += 12.0 * float(np.abs(q - f).max())
    g = _gate(p, {"fp32": f, "w8a8": q})
    assert g["ok"] is True, "disabled veto must restore the previous cosine-only behaviour"
    assert g["per_element_guarded"] is False
