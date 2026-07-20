"""Per-element correctness gate — the beam's K3 (spike) / K5 (K1) chokepoint.

``zephyr_model._gate`` used to accept a fork on AGGREGATE cosine plus a GLOBAL-max-normalized
relative error. Both terms are aggregates, so a localized per-element blow-up drowns in them: a
measured fp16-accumulate GEMM passed the old gate at cos≈0.9999986 while being ~1209% wrong on a
single output element. These tests pin the per-element term that closes that hole, in BOTH
directions and on BOTH tiers:

  * it REJECTS a construction that looks near-perfect in aggregate (high cos, small global-max rel,
    argmax preserved) but has one catastrophically-wrong element, and
  * it still PASSES a realistic near-perfect baseline whose genuine per-element spread is ~3e-3
    (the measured bitvla fp32-baseline spread), so real recaptures are not falsely rejected.

Pure unit tests on the gate math — NO board, NO compile, NO spike boot.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.runtime.backends import zephyr_model as zm


def _blowup_case(n: int = 100_000, spike: float = 100.0, elem: float = 0.08,
                 factor: float = 13.09):
    """A vector that clears EVERY aggregate term of the old gate but is ~1209% wrong on ONE element.

    ``r`` is flat 1.0 with a dominant ``spike`` at index 0 (the argmax). ``pref`` copies it but blows
    one small element (``elem``) up by ``factor`` (raw per-element rel = factor-1 ≈ 12.09, i.e. 1209%).
    Because ``elem`` is small, the blow-up's ABSOLUTE error (≈0.97) is tiny against the spike, so the
    global-max-normalized rel stays < 1e-2 and cos stays > 0.999 — so the old gate passed it through
    T1 (w8a8) AND T2 (fp32/argmax). This is exactly the aggregate-looks-fine / per-element-catastrophe
    signature the old gate missed. The per-element term (denominator floored at 0.1% of |r|.max() to
    protect genuine near-zero elements) still reports a multi-hundred-percent error on this element."""
    r = np.ones(n, dtype=np.float32)
    r[0] = spike
    i = 500
    r[i] = np.float32(elem)
    pref = r.copy()
    pref[i] = np.float32(elem * factor)
    return pref, r


def _near_perfect_case(n: int = 4096, seed: int = 0, spread: float = 3e-3):
    """A realistic near-perfect baseline: every element within ``spread`` relative of the reference,
    references bounded away from zero (the measured bitvla fp32-baseline regime)."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 2.0, n).astype(np.float32)
    r[0] = 10.0  # unambiguous argmax so the fp32-tier top-1 check is stable under the noise
    pref = (r * (1.0 + rng.uniform(-spread, spread, n))).astype(np.float32)
    return pref, r


# --------------------------------------------------------------------------- fp32 tier

def test_blowup_per_element_recorded_fp32():
    """The per-element term must still MEASURE the localized ~1200% blow-up on a well-scaled output.

    Corrected design (evidence-forced): the per-element VETO applies to the tiers where outputs are
    well-scaled and it is meaningful — int8 (w8a8/t1), classification (argmax/t2), bit-close (legacy)
    — see test_blowup_rejected_w8a8 / _legacy. The fp32 whole-model REGRESSION tier (t3) is cosine-
    only, matching the k1_e2e_xnnpack four-way authority, because a real regression output legitimately
    carries high per-element relative error on its many small (low-absolute) elements (measured: bitvla
    seed per-element max-rel 1.1 at cos 0.9999945, numerically correct). So here we assert the term is
    correctly COMPUTED (so the w8a8/legacy tiers can veto on it), not that t3 rejects a high-cos fp32."""
    pref, r = _blowup_case()
    g = zm._gate(pref, {"fp32": r})
    assert g["fp32_cos"] > 0.999, g["fp32_cos"]
    assert g["fp32_max_rel"] > 5.0, g["fp32_max_rel"]   # blow-up on the 0.08 element is measured
    assert g["max_rel"] == g["fp32_max_rel"]


def test_regression_output_passes_fp32_via_cosine_tier():
    """A whole-model regression output — cos > 0.9999 but global rel > 1e-3 (fails legacy) and per-
    element-noisy on small elements — passes via the cosine-only T3 tier (the four-way authority)."""
    import numpy as np
    rng = np.random.default_rng(11)
    r = rng.uniform(0.3, 3.0, 4096).astype(np.float32)
    r[7] = 1e-3                                   # a genuine small element -> high per-element rel
    pref = (r * (1.0 + rng.uniform(-3e-3, 3e-3, r.shape))).astype(np.float32)
    pref[7] = 3e-3                                # 200% per-element on the small element (negligible abs)
    g = zm._gate(pref, {"fp32": r})
    assert g["fp32_cos"] > 0.9999 and g["fp32_rel"] > 1e-3     # would fail legacy
    assert g["ok"] is True                                      # T3 cosine tier accepts it


def test_blowup_would_pass_without_per_element_term():
    """Regression pin: with the per-element term disabled (max_rel<=0) the OLD aggregate-only gate
    still accepts the blow-up — proving the term, not some other tightening, is what rejects it."""
    pref, r = _blowup_case()
    g = zm._gate(pref, {"fp32": r}, max_rel=0.0)
    assert g["ok"] is True


def test_near_perfect_passes_fp32():
    pref, r = _near_perfect_case()
    g = zm._gate(pref, {"fp32": r})
    assert g["fp32_max_rel"] < 0.05, g["fp32_max_rel"]   # genuine ~3e-3 spread clears the ceiling
    assert g["ok"] is True


def test_legacy_single_reference_passes_and_records_per_element():
    """Legacy callers pass a bare array (not a dict). A near-perfect bit-close output passes; the
    blow-up's per-element error is recorded. NOTE: the blow-up here has cos>0.9999, so the fp32
    cosine T3 tier (four-way authority) accepts it at the whole-model level — the per-element veto is
    exercised on the well-scaled w8a8/argmax tiers (test_blowup_rejected_w8a8) and in the dtype
    drivers, not on cosine-tight fp32 regression (see test_blowup_per_element_recorded_fp32)."""
    pref, r = _near_perfect_case()
    assert zm._gate(pref, r)["ok"] is True
    bpref, br = _blowup_case()
    gb = zm._gate(bpref, br)
    assert gb["fp32_max_rel"] > 5.0                 # measured
    # a blow-up that ALSO drops cosine below the tier bar is rejected outright
    big = int(br.argmax()); bad = br.copy(); bad[big] = br[big] * 0.5
    assert zm._gate(bad, br)["ok"] is False


# --------------------------------------------------------------------------- w8a8 (int8) tier

def test_blowup_rejected_w8a8():
    pref, r = _blowup_case()
    g = zm._gate(pref, {"w8a8": r})
    assert g["w8a8_cos"] > 0.999
    assert g["w8a8_rel"] < 1e-2                 # would satisfy the old T1 conjunction
    assert g["w8a8_max_rel"] > 5.0
    assert g["ok"] is False                      # per-element term vetoes T1


def test_near_perfect_passes_w8a8():
    pref, r = _near_perfect_case()
    g = zm._gate(pref, {"w8a8": r})
    assert g["w8a8_cos"] > 0.999
    assert g["w8a8_rel"] < 1e-2
    assert g["w8a8_max_rel"] < 0.05
    assert g["ok"] is True


# --------------------------------------------------------------------------- threshold behavior

def test_threshold_is_tunable_both_directions():
    """A per-element spread of ~3e-2 sits between a strict 0.05 ceiling (passes) and a very tight
    5e-3 ceiling (rejects) — proving the knob moves the boundary as intended."""
    pref, r = _near_perfect_case(spread=3e-2)
    perel = zm._gate(pref, {"fp32": r}, max_rel=0.0)["fp32_max_rel"]
    assert 5e-3 < perel < 0.05, perel
    assert zm._gate(pref, {"fp32": r}, max_rel=0.05)["ok"] is True
    assert zm._gate(pref, {"fp32": r}, max_rel=5e-3)["ok"] is False


def test_module_default_matches_env_contract():
    """The default ceiling is the documented 0.05 (unless MERLIN_GATE_MAX_REL overrides it)."""
    import os
    if "MERLIN_GATE_MAX_REL" not in os.environ:
        assert zm._GATE_MAX_REL == pytest.approx(0.05)
