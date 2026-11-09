from __future__ import annotations

from verification_witnesses.gate_tiers import _gate, _ramp, np, pytest


def test_exact_match_against_w8a8_passes_even_when_fp32_tier_fails():
    """A faithful W8A8 run legitimately diverges from the fp32 golden; that is not a defect.

    This is the measured TinyLlama case: exact against the W8A8 reference, far from the
    weight-only one. `ok` must come from the w8a8 tier.
    """
    w8a8 = _ramp(seed=1)
    fp32 = _ramp(seed=2)
    g = _gate(w8a8, {"fp32": fp32, "w8a8": w8a8})
    assert g["w8a8_rel"] == 0.0
    assert g["w8a8_max_rel"] == 0.0
    assert g["fp32_cos"] < 0.99
    assert g["ok"]
    assert g["tier_ok"] == "w8a8"
    ref = _ramp()
    g = _gate(ref, {"fp32": ref})
    assert g["tiers"] == ["fp32"]
    assert "w8a8_cos" not in g
    assert g["tier_ok"] != "w8a8"
