"""The forensics are tested on SYNTHESISED failures, and on the real one it was built to explain.

Every test here constructs a wrong result by a known mechanism and requires the analysis to name that
mechanism. That is the only way to trust it: a tool that explains a failure after the fact, having been
written while looking at that failure, has fit one data point.

The last class reproduces the actual defect this was written for — an 8x32x24 contraction that came back
missing exactly one reduction step's contribution in the columns of the second physical subtile.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.kernels import opu_corpus as K
from merlin.kernels import opu_forensics as F


def _case(name="asymmetric_n_gt_m"):
    case = [c for c in K.select(32)[0] if c.name == name][0]
    lhs, rhs, bias = case.operands()
    return case, lhs, rhs, K.reference(lhs, rhs, bias)


class TestASynthesisedDroppedStep:
    def test_dropping_one_step_is_named_with_its_index(self):
        _, lhs, rhs, ref = _case()
        drop_k = 7
        dev = ref.astype(np.int64) - np.outer(lhs[drop_k].astype(np.int64),
                                              rhs[drop_k].astype(np.int64))
        got = [e for e in F.explain_deltas(lhs, rhs, dev, ref) if e.kind == "dropped_step"]
        assert got, "a dropped step must be recognised"
        assert any(e.detail["k"] == drop_k for e in got)

    def test_dropping_a_step_in_only_some_columns_is_still_named(self):
        # The real failure only lost the step in one subtile's columns, so a partial-column drop must be
        # recognised too -- otherwise the tool would miss the case it exists for.
        _, lhs, rhs, ref = _case()
        drop_k, half = 15, ref.shape[1] // 2
        dev = ref.astype(np.int64).copy()
        dev[:, half:] -= np.outer(lhs[drop_k].astype(np.int64), rhs[drop_k].astype(np.int64))[:, half:]
        got = [e for e in F.explain_deltas(lhs, rhs, dev, ref) if e.kind == "dropped_step"]
        assert any(e.detail["k"] == drop_k for e in got)

    def test_double_counting_is_distinguished_from_dropping(self):
        _, lhs, rhs, ref = _case()
        k = 3
        dev = ref.astype(np.int64) + np.outer(lhs[k].astype(np.int64), rhs[k].astype(np.int64))
        kinds = {e.kind for e in F.explain_deltas(lhs, rhs, dev, ref)}
        assert "double_counted_step" in kinds and "dropped_step" not in kinds

    def test_a_correct_result_yields_no_explanation(self):
        _, lhs, rhs, ref = _case()
        assert F.explain_deltas(lhs, rhs, ref, ref) == []

    def test_a_partial_match_is_not_reported(self):
        # Only complete explanations are returned; a partial one is usually a coincidence of int8 products.
        _, lhs, rhs, ref = _case()
        dev = ref.astype(np.int64).copy()
        dev[0, 0] += 1                      # a delta no single step can explain
        assert all(e.complete for e in F.explain_deltas(lhs, rhs, dev, ref))

    def test_mismatched_shapes_raise(self):
        _, lhs, rhs, ref = _case()
        with pytest.raises(ValueError):
            F.explain_deltas(lhs, rhs, ref[:, :4], ref)

    def test_non_k_major_operands_raise(self):
        _, lhs, rhs, ref = _case()
        with pytest.raises(ValueError, match="K-major"):
            F.find_dropped_step(lhs.T, rhs, {(0, 0): 1})


class TestTheUninitialisedSignature:
    def test_a_constant_wrong_region_is_named_uninitialised(self):
        _, lhs, rhs, ref = _case()
        dev = ref.astype(np.int64).copy()
        dev[:, 16:] = 0
        got = [e for e in F.explain_deltas(lhs, rhs, dev, ref) if e.kind == "uninitialised"]
        assert got and got[0].detail["value"] == 0

    def test_a_varied_wrong_region_is_not(self):
        # Plausible-but-wrong sums are NOT the uninitialised signature, and conflating them is what sent
        # the real diagnosis down two dead ends.
        _, lhs, rhs, ref = _case()
        dev = ref.astype(np.int64).copy()
        dev[:, 16:] += np.arange(1, dev.shape[1] - 15)
        assert not [e for e in F.explain_deltas(lhs, rhs, dev, ref) if e.kind == "uninitialised"]

    def test_a_clean_result_is_not_flagged(self):
        _, lhs, rhs, ref = _case()
        assert F.uninitialised_columns(ref, ref) is None


class TestTheRealDefect:
    """The measured device values from the failing RTL run, verbatim."""

    #: Row 0, columns 16..23 of `asymmetric_n_gt_m` as the unit actually produced them. Column 22 is
    #: absent from the device's mismatch dump because it AGREED -- which the explanation has to survive.
    _DEVICE_ROW0 = {16: 86, 17: -104, 18: -87, 19: 9, 20: -31, 21: 6, 23: -53}

    def test_the_measured_deltas_are_exactly_one_dropped_step(self):
        _, lhs, rhs, ref = _case()
        deltas = {(0, j): v - int(ref[0, j]) for j, v in self._DEVICE_ROW0.items()}
        deltas[(0, 22)] = 0                 # it agreed, so its delta is zero and must also be explained
        got = F.find_dropped_step(lhs, rhs, deltas)
        assert got, "the measured deltas must have a single-step explanation"
        assert all(e.kind == "dropped_step" for e in got)
        assert {e.detail["k"] for e in got} == {15}
        # lhs[15, 0] = -6, and delta = +6 * rhs[15, col] follows from the sign.
        assert {v for e in got for v in e.detail["lhs_values"].values()} == {-6}

    def test_the_agreeing_column_is_explained_by_a_zero_operand(self):
        # Column 22 matched not by luck but because rhs[15, 22] == 0, so the dropped term was zero there.
        _, lhs, rhs, _ = _case()
        assert int(rhs[15, 22]) == 0

    def test_it_is_not_the_uninitialised_signature(self):
        # The values are plausible sums, not a constant -- which is why "uninitialised" was the wrong
        # diagnosis and had to be retracted.
        _, lhs, rhs, ref = _case()
        dev = ref.astype(np.int64).copy()
        for j, v in self._DEVICE_ROW0.items():
            dev[0, j] = v
        assert F.uninitialised_columns(dev, ref) is None
