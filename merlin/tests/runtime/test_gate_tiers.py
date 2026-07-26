"""The multi-tier accuracy gate must say WHICH reference decided the verdict.

Regression for a real incident: `tiny_llama_int8_full` shipped without a
`golden_w8a8.npy`, so a W8A8 board run was graded only against the WEIGHT-ONLY-int8
`golden.npy`. The ordinary weight-only-vs-W8A8 divergence then read as `cos = 0.484`
and was chased for hours as a codegen defect. The board was in fact exact against its
own W8A8 reference (rel 0.0). The gate had all the information needed to say "you are
grading against the wrong yardstick" and said nothing.
"""
import numpy as np

from merlin.runtime.backends import zephyr_model as zm


def _ramp(n=512, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n).astype(np.float32)


def test_w8a8_tier_reported_when_its_reference_is_present():
    ref = _ramp()
    g = zm._gate(ref, {"fp32": ref, "w8a8": ref})
    assert g["ok"]
    assert g["tiers"] == ["fp32", "w8a8"]
    assert g["tier_ok"] == "w8a8"


def test_missing_w8a8_reference_is_visible_in_the_result():
    """The exact shape of the incident: only the fp32 (weight-only) golden is available."""
    ref = _ramp()
    g = zm._gate(ref, {"fp32": ref})
    # the w8a8 tier is absent, and the result SAYS so rather than implying it was checked
    assert g["tiers"] == ["fp32"]
    assert "w8a8_cos" not in g
    assert g["tier_ok"] != "w8a8"


def test_exact_match_against_w8a8_passes_even_when_fp32_tier_fails():
    """A faithful W8A8 run legitimately diverges from the fp32 golden; that is not a defect.

    This is the measured TinyLlama case: exact against the W8A8 reference, far from the
    weight-only one. `ok` must come from the w8a8 tier.
    """
    w8a8 = _ramp(seed=1)
    fp32 = _ramp(seed=2)          # unrelated -> the fp32 tier cannot pass
    g = zm._gate(w8a8, {"fp32": fp32, "w8a8": w8a8})
    assert g["w8a8_rel"] == 0.0
    assert g["w8a8_max_rel"] == 0.0
    assert g["fp32_cos"] < 0.99
    assert g["ok"]
    assert g["tier_ok"] == "w8a8"


def test_tier_ok_is_none_when_nothing_passes():
    g = zm._gate(_ramp(seed=3), {"fp32": _ramp(seed=4), "w8a8": _ramp(seed=5)})
    assert not g["ok"]
    assert g["tier_ok"] is None
