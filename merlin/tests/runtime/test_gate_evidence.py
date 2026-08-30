"""A whole-model PASS must carry the strength of the gate that produced it.

The numeric gate has a per-element ceiling for exactly the failure aggregates hide -- a measured
fp16-accumulate GEMM was 1209% wrong on one element at cos 0.9999986. Its cosine-only tier deliberately
bypasses that ceiling, because a whole-model regression output legitimately carries high per-element
relative error on its many small elements. Both are real passes. They are not the same claim, and `ok`
alone cannot tell them apart.
"""
from __future__ import annotations

import numpy as np

from merlin.runtime.backends.zephyr_model import _gate


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
    i = int(np.argmin(np.abs(np.abs(r) - rms)))            # a SIGNIFICANT element, above the RMS mask
    p_ = r.copy()
    p_[i] = r[i] * 1.30                                     # 30% off -> per-element veto would fire

    g = _gate(p_, {"fp32": r, "w8a8": r})
    assert g["fp32_cos"] > 0.9999, f"cosine must still clear the t3 bar: {g['fp32_cos']}"
    assert g["fp32_max_rel"] > 0.05, f"the per-element veto WOULD have fired: {g['fp32_max_rel']}"
    assert g["ok"] is True, "t3 passes it on cosine alone -- that is the behaviour under test"
    assert g["tier_ok"] == "fp32_cos_only"
    assert g["per_element_guarded"] is False, \
        "a pass that no per-element bound vetted must not report itself as guarded"


def test_the_model_capsule_records_the_gate_strength():
    import inspect

    from merlin.targetgen import capsule_runner
    # the grade itself; `_grade_model_capsule` is the wall-clock budget wrapper around it
    src = inspect.getsource(capsule_runner._grade_model_capsule_inline)
    assert "per_element_guarded" in src, "the capsule must record whether a per-element bound applied"
    assert "AGGREGATE ONLY" in src, "a cosine-only verdict must say so in words, not only as a flag"
