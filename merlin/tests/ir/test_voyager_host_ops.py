"""Host ops of a lowered Voyager program against Voyager's own op library (concession C6).

``host_ops_golden/`` holds Voyager's outputs for each op the bridge leaves on the host, recorded in
Voyager's own environment by ``merlin/experiments/voyager_h2h/scripts/voyager_host_ops_golden.py``
(compiler f9d4c498, the calls are in its manifest). The numpy reference
(``voyager_schedule.host_op_reference``) must reproduce each one bit for bit, and each test carries a
control: the nearest wrong rounding or padding must NOT reproduce it, so the comparison can fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from merlin.baselines.voyager_ir import UnsupportedConstruct
from merlin.baselines.voyager_schedule import HostOp, host_op_reference
from merlin.common.paths import merlin_dir

GOLDEN = merlin_dir() / "tests" / "data" / "voyager_ir" / "host_ops_golden"


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    return dict(np.load(GOLDEN / "golden.npz"))


def _run(target: str, inputs: dict, attrs: dict, out_shape: tuple, out_dtype: str) -> np.ndarray:
    op = HostOp(target, {key: key for key in inputs}, {"output": "y"}, attrs)
    return host_op_reference(op, dict(inputs), tuple(out_shape), out_dtype)


def _bf16(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def test_quantize_divides_in_bfloat16_then_rounds_half_to_even(golden) -> None:
    x, scale, want = golden["quantize_x"], float(golden["quantize_scale"]), golden["quantize_y"]
    got = _run("quantized_ops::quantize", {"input": x}, {"scale": scale}, want.shape, "int8")
    assert np.array_equal(got, want)
    assert (want == 127).any() and (want == -128).any()  # saturation is exercised
    float32_quotient = np.clip(np.rint(x / np.float32(scale)), -128, 127)
    assert not np.array_equal(float32_quotient, want)


def test_dequantize_rounds_the_integer_to_bfloat16_before_scaling(golden) -> None:
    # Integers past bfloat16's precision tell the readings apart: type promotion casts the int32
    # input to the bfloat16 scale's dtype first, so a single float32 product is NOT Voyager's result.
    x, scale, want = golden["dequantize_x"], float(golden["dequantize_scale"]), golden["dequantize_y"]
    got = _run("quantized_ops::dequantize", {"input": x}, {"scale": scale}, want.shape, "bfloat16")
    assert np.array_equal(got, want)
    single_product = _bf16(x.astype(np.float32) * np.float32(scale))
    assert not np.array_equal(single_product, want)


def test_max_pool_reads_the_tile_its_load_padded(golden) -> None:
    x, want = golden["max_pool_x"], golden["max_pool_y"]
    attrs = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "ceil_mode": False,
        "input_pad_before": (0, 1, 1, 0),
        "input_pad_value": float("-inf"),
    }
    got = _run("quantized_ops::max_pool2d", {"input": x}, attrs, want.shape, "bfloat16")
    assert np.array_equal(got, want)
    zero_padded = _run(
        "quantized_ops::max_pool2d", {"input": x}, {**attrs, "input_pad_value": 0.0}, want.shape, "bfloat16"
    )
    assert not np.array_equal(zero_padded, want)
    with pytest.raises(UnsupportedConstruct, match="ceil_mode"):
        _run("quantized_ops::max_pool2d", {"input": x}, {**attrs, "ceil_mode": True}, want.shape, "bfloat16")


@pytest.mark.parametrize("key, size", [("avg_pool", (1, 1)), ("avg_pool23", (2, 3))])
def test_adaptive_average_pool_matches_voyager(golden, key: str, size: tuple) -> None:
    x, want = golden[f"{key}_x"], golden[f"{key}_y"]
    got = _run("quantized_ops::adaptive_avg_pool2d", {"input": x}, {"output_size": size}, want.shape, "bfloat16")
    assert np.array_equal(got, want)
    assert not np.array_equal(got, np.floor(want))  # the values are not trivial


def test_the_classifier_linear_rounds_once_to_bfloat16(golden) -> None:
    x, w = golden["linear_x"], golden["linear_w"]
    small = _run(
        "aten::linear",
        {"input": x, "weight": w, "bias": golden["linear_b_small"]},
        {},
        golden["linear_y_small"].shape,
        "bfloat16",
    )
    assert np.array_equal(small, golden["linear_y_small"])
    # The IR's bias is an int32 tile, added exactly before the one rounding: Voyager's float32 path.
    # Rounding the bias to bfloat16 first is the other reading, and it differs on a large bias.
    big = _run(
        "aten::linear",
        {"input": x, "weight": w, "bias": golden["linear_b_big"]},
        {},
        golden["linear_y_big_fp32"].shape,
        "bfloat16",
    )
    assert np.array_equal(big, golden["linear_y_big_fp32"])
    assert not np.array_equal(golden["linear_y_big_fp32"], golden["linear_y_big_bf16bias"])
    exact = x.astype(np.float64) @ w.astype(np.float64).T + golden["linear_b_small"]
    assert not np.array_equal(exact, golden["linear_y_small"])  # bfloat16 rounding is real
