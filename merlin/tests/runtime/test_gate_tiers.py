"""The multi-tier accuracy gate must say WHICH reference decided the verdict.

Regression for a real incident: `tiny_llama_int8_full` shipped without a
`golden_w8a8.npy`, so a W8A8 board run was graded only against the WEIGHT-ONLY-int8
`golden.npy`. The ordinary weight-only-vs-W8A8 divergence then read as `cos = 0.484`
and was chased for hours as a codegen defect. The board was in fact exact against its
own W8A8 reference (rel 0.0). The gate had all the information needed to say "you are
grading against the wrong yardstick" and said nothing.

A whole-model PASS must carry the strength of the gate that produced it.

The numeric gate has a per-element ceiling for exactly the failure aggregates hide -- a measured
fp16-accumulate GEMM was 1209% wrong on one element at cos 0.9999986. Its cosine-only tier deliberately
bypasses that ceiling, because a whole-model regression output legitimately carries high per-element
relative error on its many small elements. Both are real passes. They are not the same claim, and `ok`
alone cannot tell them apart.
"""

import numpy as np

from merlin.runtime.backends.zephyr_model import _gate


def _ramp(n=512, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n).astype(np.float32)


def test_w8a8_tier_reported_when_its_reference_is_present():
    ref = _ramp()
    g = _gate(ref, {"fp32": ref, "w8a8": ref})
    assert g["ok"]
    assert g["tiers"] == ["fp32", "w8a8"]
    assert g["tier_ok"] == "w8a8"


def test_missing_w8a8_reference_is_visible_in_the_result():
    """The exact shape of the incident: only the fp32 (weight-only) golden is available."""
    ref = _ramp()
    g = _gate(ref, {"fp32": ref})
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
    fp32 = _ramp(seed=2)  # unrelated -> the fp32 tier cannot pass
    g = _gate(w8a8, {"fp32": fp32, "w8a8": w8a8})
    assert g["w8a8_rel"] == 0.0
    assert g["w8a8_max_rel"] == 0.0
    assert g["fp32_cos"] < 0.99
    assert g["ok"]
    assert g["tier_ok"] == "w8a8"


def test_tier_ok_is_none_when_nothing_passes():
    g = _gate(_ramp(seed=3), {"fp32": _ramp(seed=4), "w8a8": _ramp(seed=5)})
    assert not g["ok"]
    assert g["tier_ok"] is None


def test_an_exact_run_is_carried_by_a_per_element_guarded_tier():
    r = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    g = _gate(r.copy(), {"fp32": r, "w8a8": r})
    assert g["ok"] is True
    assert g["per_element_guarded"] is True
    assert g["tier_ok"] == "w8a8"


def test_a_localized_blowup_is_still_rejected():
    """The 1209% shape: aggregates fine, one element badly wrong."""
    r = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bad = r.copy()
    bad[0] = 3.0
    g = _gate(bad, {"fp32": r})
    assert g["ok"] is False, "a per-element blow-up must not pass"
    assert g["per_element_guarded"] is False


def test_a_cosine_only_pass_declares_that_no_per_element_bound_applied():
    """The shape that actually carried a whole-model verdict: cos 0.99993 with per-element 2.82 and
    global rel 0.0148 -- t1 rejected on rel, t2 rejected on per-element, t3 passed on cosine alone.
    Both references present, so the `legacy` tier (which only applies when w8a8 is absent) is out."""
    rng = np.random.default_rng(0xC05)
    r = (rng.standard_normal(4096) * 10.0).astype(np.float32)
    rms = float(np.sqrt(np.mean(r.astype(np.float64) ** 2)))
    i = int(np.argmin(np.abs(np.abs(r) - rms)))  # a SIGNIFICANT element, above the RMS mask
    p_ = r.copy()
    p_[i] = r[i] * 1.30  # 30% off -> per-element veto would fire

    g = _gate(p_, {"fp32": r, "w8a8": r})
    assert g["fp32_cos"] > 0.9999, f"cosine must still clear the t3 bar: {g['fp32_cos']}"
    assert g["fp32_max_rel"] > 0.05, f"the per-element veto WOULD have fired: {g['fp32_max_rel']}"
    assert g["ok"] is True, "t3 passes it on cosine alone -- that is the behaviour under test"
    assert g["tier_ok"] == "fp32_cos_only"
    assert g["per_element_guarded"] is False, (
        "a pass that no per-element bound vetted must not report itself as guarded"
    )


def test_the_model_capsule_records_the_gate_strength():
    import inspect

    from merlin.targetgen import capsule_runner

    # the grade itself; `_grade_model_capsule` is the wall-clock budget wrapper around it
    src = inspect.getsource(capsule_runner._grade_model_capsule_inline)
    assert "per_element_guarded" in src, "the capsule must record whether a per-element bound applied"
    assert "AGGREGATE ONLY" in src, "a cosine-only verdict must say so in words, not only as a flag"
