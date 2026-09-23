"""The off-device reference for layer-bench programs: fill stream, exact layers, digest."""

import numpy as np

from merlin.perf.layer_bench import reference as ref
from merlin.sched.contract import contract

READOUT = {
    "schema": "scalar_narrow_readout_contract_v1",
    "accumulator_dtype": "i32",
    "output_dtype": "i8",
    "scale_dtype": "f32",
    "clamp_min": -128,
    "clamp_max": 127,
}


def _serial_draws(seed, n):
    s, out = seed, []
    for _ in range(n):
        s = (s * ref.LCG_MUL + ref.LCG_INC) % (1 << 64)
        out.append(s)
    return out


def test_vectorized_stream_equals_the_serial_recurrence_and_continues():
    stream = ref.LcgStream(7)
    got = list(map(int, stream.draws(37))) + list(map(int, stream.draws(29)))
    assert got == _serial_draws(7, 66)


def test_fills_match_the_c_casts():
    draws = _serial_draws(3, 10)
    s = ref.LcgStream(3)
    i8 = s.fill_i8(5)
    assert i8.tolist() == [((d >> 56) + 128) % 256 - 128 for d in draws[:5]]
    acc = s.fill_acc(5, 100)
    assert acc.tolist() == [(d >> 33) % 201 - 100 for d in draws[5:10]]


def test_fnv1a64_standard_vectors():
    assert ref.fnv1a64(b"") == 0xCBF29CE484222325
    assert ref.fnv1a64(b"a") == 0xAF63DC4C8601EC8C


def test_conv_accumulator_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, (2, 5, 5, 3), dtype=np.int64)
    w = rng.integers(-128, 128, (3, 3, 3, 4), dtype=np.int64)
    bias = rng.integers(-1000, 1000, 4, dtype=np.int64)
    for stride, pad in ((1, 1), (2, 1), (2, 0)):
        got = ref.conv2d_accumulator(x, w, bias, stride=stride, padding=pad)
        oh = (5 + 2 * pad - 3) // stride + 1
        want = np.zeros((2, oh, oh, 4), dtype=np.int64)
        for b in range(2):
            for oy in range(oh):
                for ox in range(oh):
                    for co in range(4):
                        acc = int(bias[co])
                        for kh in range(3):
                            for kw in range(3):
                                iy, ix = oy * stride + kh - pad, ox * stride + kw - pad
                                if 0 <= iy < 5 and 0 <= ix < 5:
                                    acc += int((x[b, iy, ix, :] * w[kh, kw, :, co]).sum())
                        want[b, oy, ox, co] = acc
        np.testing.assert_array_equal(got, want)


def test_word_digest_matches_a_serial_definition():
    data = bytes(range(21))  # not a multiple of 8: the tail is zero-padded
    h = ref.FNV_OFFSET
    padded = data + b"\0" * 3
    for i in range(0, len(padded), 8):
        h ^= int.from_bytes(padded[i : i + 8], "little")
        h = (h * ref.FNV_PRIME) % (1 << 64)
    assert ref.fnv1a64_words(data) == h


def test_packed_operands_are_aligned_and_round_trip():
    spec = {
        "op": "conv2d",
        "batch": 1,
        "in_dim": 5,
        "in_channels": 3,
        "out_channels": 4,
        "kernel": 3,
        "stride": 1,
        "padding": 1,
        "seed": 11,
    }
    blob, off = ref.pack_operands(spec, accumulator_dtype="i32")
    assert list(off) == ["input", "weights", "bias"]
    assert all(v % ref.OPERAND_ALIGN == 0 for v in off.values())
    arrays = dict(ref.operand_arrays(spec))
    x = np.frombuffer(blob, dtype=np.int8, count=arrays["input"].size, offset=off["input"])
    np.testing.assert_array_equal(x, arrays["input"].ravel())
    b = np.frombuffer(blob, dtype="<i4", count=4, offset=off["bias"])
    np.testing.assert_array_equal(b, arrays["bias"])
    import pytest

    with pytest.raises(ValueError):
        ref.pack_operands({**spec, "bias_span": 1 << 40}, accumulator_dtype="i32")


def test_expected_output_is_deterministic_and_seed_sensitive():
    c = contract("per_tensor_readout_v1", READOUT)
    spec = {"op": "matmul", "m": 4, "n": 16, "k": 32, "scale": 0.01, "relu": True, "seed": 5}
    a, b = ref.expected_digest(spec, c), ref.expected_digest(spec, c)
    assert a == b and 0 <= a <= ref.DIGEST_MASK
    assert ref.expected_digest({**spec, "seed": 6}, c) != a
    out = ref.expected_output(spec, c)
    assert out.dtype == np.int8 and out.shape == (4, 16) and out.min() >= 0


def test_the_operand_mode_changes_the_draw_and_defaults_to_the_one_receipts_were_keyed_on():
    """``elem_mode`` selects how int8 operands are drawn, and its absence means what it always meant.

    An existing receipt was keyed on a spec that names no mode, so the default has to stay the
    full-range draw or every cached row would silently be about different operands.
    """
    import pytest

    spec = {"op": "matmul", "m": 8, "n": 16, "k": 32, "scale": 1.0, "relu": False, "seed": 1}
    full = dict(ref.operand_arrays(spec))
    assert "elem_mode" not in spec
    np.testing.assert_array_equal(full["a"], dict(ref.operand_arrays({**spec, "elem_mode": "i8_full"}))["a"])

    sparse = dict(ref.operand_arrays({**spec, "elem_mode": "sparse_binary"}))
    # 1/8-dense {0, 1}: every value is a 0 or a 1, and it is not the full-range draw.
    assert set(np.unique(sparse["a"]).tolist()) <= {0, 1}
    assert not np.array_equal(sparse["a"], full["a"])
    assert 0.05 < float(sparse["a"].mean()) < 0.22

    # The mode is part of the spec, so the oracle moves with it rather than staying put.
    c = contract("per_tensor_readout_v1", READOUT)
    assert ref.expected_digest({**spec, "elem_mode": "sparse_binary"}, c) != ref.expected_digest(spec, c)
    with pytest.raises(ValueError, match="unknown elem_mode"):
        ref.operand_arrays({**spec, "elem_mode": "uniform"})
