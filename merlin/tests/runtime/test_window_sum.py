# ruff: noqa: E501 -- attribute dictionaries are kept one per line, matching the ABI entries
"""WINDOW_SUM: exact int32 window sums and one readout per output, for every pool-family op.

WHY THIS EXISTS. mean.dim, the six average/adaptive pools, the two pool backwards and col2im had no
command-buffer opcode: the readout of a pooled window was a fused max only, and nothing could state a
divisor that varies by window (a padded border, an adaptive window) or a backward that spreads each
window over the positions it covers. WINDOW_SUM states all of them with integer sums and one binary32
readout per output; a backward's per-window divisors become exact integer weights L/count.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.runtime import integer_ops as IO
from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate
from merlin.runtime.tensor import Tensor

_RNG = np.random.default_rng(3)


def _t(values) -> Tensor:
    return Tensor(tuple(values.shape), [int(v) for v in values.reshape(-1)], "i8")


def _np(t: Tensor) -> np.ndarray:
    return np.asarray(t.data, dtype=np.int64).reshape(t.shape)


def _read(acc: float, scale: float) -> int:
    return int(np.clip(np.rint(np.float32(acc) * np.float32(scale)), -128, 127))


def test_a_padded_pool_divides_each_window_by_its_own_count():
    x = _RNG.integers(-128, 128, size=(1, 5, 5, 2))  # nhwc
    attrs = {"mode": "window", "kernel": [3, 3], "stride": [2, 2], "padding": [1, 1], "count_include_pad": False,
             "readout_scale": 0.8, "output_dtype": "i8"}  # fmt: skip
    out = _np(IO.window_sum(_t(x), attrs, [1, 3, 3, 2]))
    for oy in range(3):
        for ox in range(3):
            ys = range(max(2 * oy - 1, 0), min(2 * oy + 2, 5))
            xs = range(max(2 * ox - 1, 0), min(2 * ox + 2, 5))
            for c in range(2):
                acc = sum(int(x[0, y, xx, c]) for y in ys for xx in xs)
                assert out[0, oy, ox, c] == _read(acc, 0.8 / (len(ys) * len(xs)))
    attrs["count_include_pad"] = True
    padded = _np(IO.window_sum(_t(x), attrs, [1, 3, 3, 2]))
    acc = sum(int(x[0, y, xx, 0]) for y in range(0, 2) for xx in range(0, 2))
    assert padded[0, 0, 0, 0] == _read(acc, 0.8 / 9)


def test_adaptive_and_reduce_and_their_divisors():
    x = _RNG.integers(-128, 128, size=(1, 2, 7))  # nchw, one spatial dim
    out = _np(
        IO.window_sum(
            _t(x), {"mode": "adaptive", "layout": "nchw", "readout_scale": 1.0, "output_dtype": "i8"}, [1, 2, 3]
        )
    )
    for o, (lo, hi) in enumerate([(0, 3), (2, 5), (4, 7)]):
        assert out[0, 0, o] == _read(int(x[0, 0, lo:hi].sum()), 1.0 / (hi - lo))
    mean = _np(
        IO.window_sum(
            _t(x),
            {"mode": "reduce", "dims": [-1], "average": True, "readout_scale": 2.0, "output_dtype": "i8"},
            [1, 2, 1],
        )
    )
    assert mean[0, 1, 0] == _read(int(x[0, 1].sum()), 2.0 / 7)
    total = IO.window_sum(
        _t(x),
        {"mode": "reduce", "dims": [-1], "average": False, "readout_scale": 1.0, "output_dtype": "i32"},
        [1, 2, 1],
    )
    assert _np(total)[0, 0, 0] == int(x[0, 0].sum()) and total.dtype == "i32"


def test_a_pool_backward_uses_integer_weights_lcm_over_count():
    grad = _RNG.integers(-128, 128, size=(1, 1, 3, 3))
    attrs = {"mode": "window_transpose", "kernel": [3, 3], "stride": [2, 2], "padding": [1, 1],
             "count_include_pad": False, "layout": "nchw", "readout_scale": 1.0, "output_dtype": "i8"}  # fmt: skip
    out = _np(IO.window_sum(_t(grad), attrs, [1, 1, 5, 5]))
    counts = [[(min(2 * o + 2, 5) - max(2 * o - 1, 0)) for o in range(3)]] * 2
    lcm = int(np.lcm.reduce([cy * cx for cy in counts[0] for cx in counts[1]]))
    acc = np.zeros((5, 5), dtype=np.int64)
    for oy in range(3):
        for ox in range(3):
            weight = lcm // (counts[0][oy] * counts[1][ox])
            acc[max(2 * oy - 1, 0) : min(2 * oy + 2, 5), max(2 * ox - 1, 0) : min(2 * ox + 2, 5)] += (
                weight * grad[0, 0, oy, ox]
            )
    assert out[0, 0].tolist() == [[_read(v, 1.0 / lcm) for v in row] for row in acc]


def test_fold_sums_overlapping_patches():
    cols = _RNG.integers(-20, 20, size=(1, 1 * 2 * 2, 2 * 2))  # C=1, kernel 2x2, 2x2 blocks
    out = _np(IO.window_sum(_t(cols), {"mode": "fold", "kernel": [2, 2], "stride": [1, 1], "padding": [0, 0],
                                        "dilation": [1, 1], "readout_scale": 1.0, "output_dtype": "i8"}, [1, 1, 3, 3]))  # fmt: skip
    patches = cols.reshape(2, 2, 2, 2)  # (ky, kx, by, bx)
    expected = np.zeros((3, 3), dtype=np.int64)
    for ky in range(2):
        for kx in range(2):
            expected[ky : ky + 2, kx : kx + 2] += patches[ky, kx]
    assert out[0, 0].tolist() == np.clip(expected, -128, 127).tolist()


@pytest.mark.parametrize(
    "attrs, match",
    [
        (
            {
                "mode": "window",
                "kernel": [2],
                "stride": [2],
                "padding": [0],
                "readout_scale": 1.0,
                "output_dtype": "i8",
            },
            "count_include_pad",
        ),
        (
            {"mode": "reduce", "dims": [1], "average": True, "readout_scale": 1.0, "output_dtype": "i8", "kernel": [2]},
            "belong",
        ),
        ({"mode": "reduce", "dims": [1], "average": True, "readout_scale": 1.0, "output_dtype": "i32"}, "no readout"),
        ({"mode": "pool", "readout_scale": 1.0, "output_dtype": "i8"}, "mode"),
    ],
)
def test_window_sum_refuses_what_it_cannot_state(attrs, match):
    with pytest.raises(IO.IntegerOpError, match=match):
        IO.window_sum(
            _t(np.ones((1, 4, 2), dtype=np.int64)), attrs, [1, 1, 2] if attrs.get("mode") == "reduce" else [1, 2, 2]
        )


def test_the_engines_agree_on_a_window_sum_command():
    cb = {
        "abi_version": "0.1", "target": "t", "backend": "simulator",
        "tensors": {"X": {"shape": [1, 4, 4, 2], "dtype": "i8", "role": "input"},
                    "Y": {"shape": [1, 2, 2, 2], "dtype": "i8", "role": "output"}},
        "commands": [{"opcode": "WINDOW_SUM", "operands": {"src": "X", "dst": "Y"},
                      "attributes": {"mode": "window", "kernel": [2, 2], "stride": [2, 2], "padding": [0, 0],
                                     "count_include_pad": True, "readout_scale": 0.5, "output_dtype": "i8"}}],
        "resources": {"buffers": [], "handles": []}, "metrics_requested": [],
    }  # fmt: skip
    assert reference_outputs(cb) == simulate(cb)["outputs"]


_TORCH = r"""
import numpy as np, torch
import torch.nn.functional as F
from merlin.runtime.integer_ops import window_sum
from merlin.runtime.tensor import Tensor
torch.manual_seed(0)
T = lambda a: Tensor(tuple(a.shape), [int(v) for v in a.reshape(-1)], "i8")
s_in, s_out = 0.05, 0.04
worst, exact = 0, []
def check(ref, got):
    global worst
    r = torch.clamp(torch.round(ref.detach() / s_out), -128, 127).numpy()
    d = np.abs(r - np.asarray(got.data).reshape(got.shape)); worst = max(worst, int(d.max())); exact.append(float((d == 0).mean()))
x = torch.randint(-128, 128, (2, 3, 9, 7))
base = {"layout": "nchw", "readout_scale": s_in / s_out, "output_dtype": "i8"}
for cip in (False, True):
    ref = F.avg_pool2d(x.float() * s_in, 3, 2, 1, count_include_pad=cip)
    check(ref, window_sum(T(x.numpy()), {**base, "mode": "window", "kernel": [3, 3], "stride": [2, 2], "padding": [1, 1], "count_include_pad": cip}, list(ref.shape)))
ref = F.adaptive_avg_pool2d(x.float() * s_in, (4, 3))
check(ref, window_sum(T(x.numpy()), {**base, "mode": "adaptive"}, list(ref.shape)))
g = torch.randint(-128, 128, (2, 3, 5, 4))
ref = torch.ops.aten.avg_pool2d_backward(g.float() * s_in, torch.zeros(2, 3, 9, 7), [3, 3], [2, 2], [1, 1], False, False, None)
check(ref, window_sum(T(g.numpy()), {**base, "mode": "window_transpose", "kernel": [3, 3], "stride": [2, 2], "padding": [1, 1], "count_include_pad": False}, list(ref.shape)))
ga = torch.randint(-128, 128, (2, 3, 4, 3))
ref = torch.ops.aten._adaptive_avg_pool2d_backward(ga.float() * s_in, torch.zeros(2, 3, 9, 7))
check(ref, window_sum(T(ga.numpy()), {**base, "mode": "adaptive_transpose"}, list(ref.shape)))
cols = torch.randint(-128, 128, (2, 12, 12))
ref = F.fold(cols.float() * s_in, output_size=(5, 4), kernel_size=2)
check(ref, window_sum(T(cols.numpy()), {**base, "mode": "fold", "kernel": [2, 2], "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1]}, list(ref.shape)))
print(worst, min(exact))
"""


def test_window_sum_matches_torch_up_to_rounding_ties():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    env = {**os.environ, "PYTHONPATH": str(repo_root() / "src")}
    result = subprocess.run([python, "-c", _TORCH], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stderr[-4000:]
    worst, exact = result.stdout.split()[-2:]
    # A one-step difference only at an exact rational tie, which the float reference breaks by noise.
    assert int(worst) <= 1 and float(exact) >= 0.94
