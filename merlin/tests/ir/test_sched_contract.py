"""Numerics contracts and the exhaustive epilogue-enumeration gate (merlin.sched)."""

import numpy as np
import pytest

from merlin.runtime.tensor import Tensor
from merlin.sched.check import enumerate_readout_flips, reachable_accumulator_range
from merlin.sched.contract import ContractError, NumericsContract, contract, from_readout_facts

# The shape of a target's derived readout facts (a header-verified scalar readout contract). Values are
# the i32 -> i8 case; the tests below exercise the arithmetic, not any particular target.
READOUT = {
    "schema": "scalar_narrow_readout_contract_v1",
    "accumulator_dtype": "i32",
    "output_dtype": "i8",
    "scale_dtype": "f32",
    "clamp_min": -128,
    "clamp_max": 127,
    "provenance": {"params_header_sha256": "test"},
}


def _reference(acc: np.ndarray, scale: float) -> np.ndarray:
    """The existing scalar Tensor readout (round half even in f32), then the i8 clamp."""
    t = Tensor((acc.size,), [int(x) for x in acc.ravel()], "i32").requant_acc_scale(scale)
    return np.clip(np.asarray(t.data, dtype=np.int64), -128, 127).reshape(acc.shape)


def test_per_tensor_readout_matches_scalar_reference():
    c = contract("per_tensor_readout_v1", READOUT)
    rng = np.random.default_rng(0)
    acc = rng.integers(-(1 << 20), 1 << 20, size=4096, dtype=np.int64)
    ties = np.array([1, 3, 5, -1, -3, -5, 255, -255, 257], dtype=np.int64)  # x * 0.5 lands on .5
    for scale in (0.5, 0.0078125, 1.0 / 3.0, 0.01234):
        for a in (acc, ties):
            np.testing.assert_array_equal(c.readout(a, scale).astype(np.int64), _reference(a, scale))


def test_relu_and_clamp_are_applied():
    c = contract("per_tensor_readout_v1", READOUT)
    out = c.readout(np.array([-1000, -3, 0, 3, 1000]), 1.0, activation="relu")
    assert out.tolist() == [0, 0, 0, 3, 127]
    assert out.dtype == np.int8


def test_row_and_column_scales_broadcast_on_the_right_axis():
    acc = np.arange(12, dtype=np.int64).reshape(3, 4) * 7 - 40
    rows = np.array([0.5, 0.25, 2.0], dtype=np.float32)
    cols = np.array([1.0, 0.5, 0.125, 3.0], dtype=np.float32)
    per_row = contract("per_row_readout_v1", READOUT).readout(acc, rows)
    per_col = contract("per_column_readout_v1", READOUT).readout(acc, cols)
    for i in range(3):
        np.testing.assert_array_equal(per_row[i].astype(np.int64), _reference(acc[i], float(rows[i])))
    for j in range(4):
        np.testing.assert_array_equal(per_col[:, j].astype(np.int64), _reference(acc[:, j], float(cols[j])))


def test_wrong_scale_shape_is_refused():
    with pytest.raises(ContractError):
        contract("per_row_readout_v1", READOUT).readout(np.zeros((3, 4), dtype=np.int64), np.ones(4))
    with pytest.raises(ContractError):
        contract("per_tensor_readout_v1", READOUT).readout(np.zeros(4, dtype=np.int64), np.ones(2))


def test_rank1_is_declared_but_refused_until_its_arithmetic_is_fixed():
    c = from_readout_facts(READOUT, contract_id="rank1_probe", granularity="rank1")
    with pytest.raises(ContractError):
        c.readout(np.zeros((2, 2), dtype=np.int64), np.ones(2))


def test_out_of_range_accumulator_is_refused():
    with pytest.raises(ContractError):
        contract("per_tensor_readout_v1", READOUT).readout(np.array([1 << 40]), 1.0)


def test_digest_tracks_semantics_not_provenance():
    a = contract("per_tensor_readout_v1", READOUT)
    b = contract("per_tensor_readout_v1", {**READOUT, "provenance": {"params_header_sha256": "other"}})
    assert a.digest() == b.digest()
    assert a.digest() != contract("per_row_readout_v1", READOUT).digest()
    narrower = {**READOUT, "clamp_min": -127}
    assert a.digest() != contract("per_tensor_readout_v1", narrower).digest()


def test_unknown_schema_and_contract_fail_closed():
    with pytest.raises(ContractError):
        contract("per_tensor_readout_v1", {**READOUT, "schema": "something_else"})
    with pytest.raises(ContractError):
        contract("no_such_contract", READOUT)
    with pytest.raises(ContractError):
        NumericsContract("x", "i32", "i8", "f32", "tensor", "half_up", -128, 127)


def test_enumeration_finds_exactly_the_rounding_ties():
    """Mutation control: a half-up readout differs from half-even only on ties, and on exactly these."""
    c = contract("per_tensor_readout_v1", READOUT)
    half_even = lambda acc, s: c.readout(acc, s)
    half_up = lambda acc, s: np.clip(np.floor(acc.astype(np.float32) * np.float32(s) + 0.5), -128, 127)
    report = enumerate_readout_flips(half_even, half_up, scale=0.5, lo=-8, hi=8, chunk=5)
    assert report.evaluated == 17
    assert {e[0] for e in report.examples} == {1, 5, -3, -7}
    assert report.flips == 4 and not report.exact
    same = enumerate_readout_flips(half_even, half_even, scale=0.37, lo=-70000, hi=70000)
    assert same.exact


def test_reachable_range_uses_all_operand_corners():
    assert reachable_accumulator_range(2, (-128, 127), (-128, 127)) == (-32512, 32768)
    assert reachable_accumulator_range(3, (0, 3), (0, 3), bias_range=(-5, 5)) == (-5, 32)
