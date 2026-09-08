"""A capsule may declare the range its deterministic stimulus is drawn from.

WHY IT MATTERS, proven on hardware. The historical default is 0..3 — non-negative — so with
non-negative weights and inputs the accumulator is never negative and ``max(0, x)`` is the identity
on every value such a program can produce. ``B1_linear_relu_i8`` declares a fused ReLU and PASSES
whether or not the device applies it.

The same program with ``stimulus_range: [-3, 3]`` makes 49% of the accumulator negative, and it
FAILS on both spike (L2) and gsim (L3) with 126 negative outputs and ``min = -85`` — exactly the raw
accumulator — while L0 and L1 pass. The compiler's intent is right; the device discards the
activation, because the readout is full-width ``i32`` and that path writes the raw accumulator
(``gemmini.cc:271-282`` computes the scaled, activated value and throws it away).

Everything here also pins the property that makes this safe to land: a capsule that declares nothing
materializes byte-identically to before.
"""
from __future__ import annotations

import pytest

from merlin.runtime.commandbuffer import (DEFAULT_STIMULUS_RANGE, STIMULUS_RANGE_KEY,
                                          materialize_inputs, stimulus_range)
from merlin.runtime.tensor import Tensor
from merlin.targetgen.capsule_golden import capsule_stimulus_range, materialize_capsule_leaves


def _cb(*, params=None, data=None):
    spec = {"shape": [4, 4], "dtype": "i8", "role": "input"}
    if data is not None:
        spec["data"] = data
    return {"tensors": {"X": spec}, "commands": [], "params": params or {}}


class TestTheDefaultIsUnchanged:
    def test_an_undeclared_range_is_the_historical_default(self):
        assert stimulus_range({}) == DEFAULT_STIMULUS_RANGE == (0, 3)
        assert stimulus_range({"params": {}}) == (0, 3)
        assert capsule_stimulus_range({}) == (0, 3)

    def test_an_undeclared_capsule_materializes_byte_identically(self):
        """The safety property that makes this landable: no existing golden moves."""
        cap = {"inputs": [{"name": "W", "role": "weight", "shape": [8, 8], "dtype": "i8"}]}
        got = materialize_capsule_leaves(cap)["W"]
        assert list(got.data) == list(Tensor.deterministic("W", (8, 8), "i8").data)

    def test_an_undeclared_buffer_materializes_byte_identically(self):
        got = materialize_inputs(_cb())["X"]
        assert list(got.data) == list(Tensor.deterministic("X", (4, 4), "i8").data)


class TestADeclaredRangeReachesBothMaterializers:
    def test_the_buffer_side_honours_it(self):
        got = materialize_inputs(_cb(params={STIMULUS_RANGE_KEY: [-3, 3]}))["X"]
        assert min(got.data) < 0 and set(got.data) <= set(range(-3, 4))

    def test_the_capsule_side_honours_it(self):
        cap = {"stimulus_range": [-3, 3],
               "inputs": [{"name": "X", "role": "input", "shape": [4, 4], "dtype": "i8"}]}
        got = materialize_capsule_leaves(cap)["X"]
        assert min(got.data) < 0 and set(got.data) <= set(range(-3, 4))

    def test_both_sides_produce_THE_SAME_values_for_one_declared_range(self):
        """The golden reads the capsule and the device reads the buffer; they must not diverge."""
        cap = {"stimulus_range": [-3, 3],
               "inputs": [{"name": "X", "role": "input", "shape": [4, 4], "dtype": "i8"}]}
        from_capsule = materialize_capsule_leaves(cap)["X"]
        from_buffer = materialize_inputs(_cb(params={STIMULUS_RANGE_KEY: [-3, 3]}))["X"]
        assert list(from_capsule.data) == list(from_buffer.data)

    def test_one_validator_serves_both_so_they_cannot_disagree(self):
        """A range read two ways is a range that will eventually be read two different ways."""
        for bad in ([1], [3, 1], "x", [1.5, 2], [True, 3]):
            with pytest.raises(ValueError):
                stimulus_range({"params": {STIMULUS_RANGE_KEY: bad}})
            with pytest.raises(ValueError):
                capsule_stimulus_range({STIMULUS_RANGE_KEY: bad})


class TestAMalformedRangeIsRefusedNotDefaulted:
    def test_a_wrong_shape_raises_with_the_reason(self):
        with pytest.raises(ValueError, match="two-element"):
            stimulus_range({"params": {STIMULUS_RANGE_KEY: [1, 2, 3]}})

    def test_an_empty_range_raises(self):
        with pytest.raises(ValueError, match="empty"):
            stimulus_range({"params": {STIMULUS_RANGE_KEY: [5, 4]}})

    def test_a_bool_is_not_an_integer_bound(self):
        with pytest.raises(ValueError, match="two-element"):
            stimulus_range({"params": {STIMULUS_RANGE_KEY: [False, 3]}})

    def test_the_refusal_says_why_a_default_would_be_worse(self):
        try:
            stimulus_range({"params": {STIMULUS_RANGE_KEY: "nope"}})
        except ValueError as exc:
            assert "reports a stimulus difference as a datapath failure" in str(exc)
        else:
            pytest.fail("a malformed range must not be silently defaulted")

    def test_a_single_point_range_is_allowed(self):
        """lo == hi is a constant stimulus, which is degenerate but not malformed."""
        assert stimulus_range({"params": {STIMULUS_RANGE_KEY: [7, 7]}}) == (7, 7)
        assert set(materialize_inputs(_cb(params={STIMULUS_RANGE_KEY: [7, 7]}))["X"].data) == {7}


class TestValuesOnTheDeclarationWinOverAnyFill:
    def test_declared_data_is_used_verbatim(self):
        got = materialize_inputs(_cb(data=[[1, 2, 3, 4]] * 4))["X"]
        assert list(got.data) == [1, 2, 3, 4] * 4

    def test_declared_data_beats_a_declared_range(self):
        """Injected values are the strongest form: both sides compute on the same bytes."""
        got = materialize_inputs(_cb(params={STIMULUS_RANGE_KEY: [-3, 3]},
                                     data=[[9, 9, 9, 9]] * 4))["X"]
        assert set(got.data) == {9}

    def test_an_explicit_inputs_argument_still_wins_over_both(self):
        got = materialize_inputs(_cb(params={STIMULUS_RANGE_KEY: [-3, 3]},
                                     data=[[9, 9, 9, 9]] * 4),
                                 {"X": [[5, 5, 5, 5]] * 4})["X"]
        assert set(got.data) == {5}


class TestTheSignedReluCapsuleIsLoadBearing:
    """The capsule this feature exists for: its ReLU must actually be able to fail."""

    def _capsule(self):
        import yaml
        from pathlib import Path
        from merlin.common.paths import repo_root
        path = (Path(repo_root()) / "merlin/contract/capsules/layers"
                / "B1s_linear_relu_signed_i8" / "capsule.yaml")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_it_declares_a_signed_range(self):
        assert capsule_stimulus_range(self._capsule()) == (-3, 3)

    def test_its_accumulator_is_negative_for_a_large_share_of_outputs(self):
        cap = self._capsule()
        env = materialize_capsule_leaves(cap)
        acc = list(env["X"].matmul(env["W"]).data)
        negatives = [v for v in acc if v < 0]
        assert negatives, "with no negative accumulator the relu is the identity and this is vacuous"
        assert len(negatives) / len(acc) > 0.25, "the relu must bind on a large share, not a corner"

    def test_the_default_stimulus_would_have_made_it_vacuous(self):
        """Which is precisely why B1_linear_relu_i8 passes whether or not the relu is applied."""
        cap = dict(self._capsule())
        cap.pop("stimulus_range")
        env = materialize_capsule_leaves(cap)
        acc = list(env["X"].matmul(env["W"]).data)
        assert min(acc) >= 0, "the default 0..3 stimulus cannot produce a negative accumulator"
