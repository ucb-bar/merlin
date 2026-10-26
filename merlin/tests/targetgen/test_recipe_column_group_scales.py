"""Column-group weight scales and accumulator-grid biases in the PT2E recipe path.

A derived output path can apply one scale to ``columns_per_scale`` output columns. A recipe asks for
that with ``weight.channels_per_scale``: every run of that many output channels shares the largest
range in it, so the per-channel scale is uniform per column group and never clips a channel. A recipe
with ``bias_domain: accumulator`` rounds each quantized contraction's bias onto its accumulator grid,
``k = round(b / (s_x * s_w))``, stored as the integer rewrite's own rescale of ``k``.

Torch lives in the capture interpreter (``MERLIN_M2M_PYTHON``), as for the sibling family test; the
structural checks run without it.
"""

# ruff: noqa: E501 - the interpreter script keeps its graph-building calls on one line.

from __future__ import annotations

import os
import subprocess

import pytest

from merlin.common.paths import merlin_dir

_DIR = merlin_dir() / "python" / "merlin" / "targetgen"


def test_the_recipe_fields_are_read_by_the_quantizer() -> None:
    quantizer = (_DIR / "_recipe_quantizer.py").read_text(encoding="utf-8")
    observers = (_DIR / "_recipe_observers.py").read_text(encoding="utf-8")
    assert 'tensor.get("channels_per_scale")' in quantizer
    assert 'recipe.get("bias_domain") == "accumulator"' in quantizer
    assert "def grouped_channel_observer" in observers


_SCRIPT = r"""
import sys
sys.path.insert(0, sys.argv[1])
import torch
import torchao  # noqa: F401  (registers the quantized_decomposed ops)
import _recipe_observers as RO
import _recipe_quantizer as RQ

# 1. grouped observer: uniform per group, never below a channel's own range, last group shorter.
Observer = RO.grouped_channel_observer()
obs = Observer(channels_per_scale=4, dtype=torch.int8, qscheme=torch.per_channel_symmetric,
               quant_min=-127, quant_max=127, ch_axis=0)
w = torch.randn(10, 3) * torch.arange(1, 11).reshape(-1, 1)
obs(w)
scale, _ = obs.calculate_qparams()
assert torch.all(scale[0:4] == scale[0]) and torch.all(scale[4:8] == scale[4]) and torch.all(scale[8:] == scale[8])
own = w.abs().amax(1) / 127
assert torch.all(scale >= own * (1 - 1e-6)), (scale, own)
assert not torch.equal(scale[0], scale[4])
for bad in (0, -1, 2.0):
    try:
        Observer(channels_per_scale=bad, qscheme=torch.per_channel_symmetric)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)

# Explicit reduction policies differ on unequal ranges and preserve a short final block.
for policy, expected in (("max_abs", [8., 8., 8., 12., 12.]),
                         ("mean_abs", [4., 4., 4., 8., 8.])):
    obs = Observer(channels_per_scale=3, block_scale_reduction=policy, dtype=torch.int8,
                   qscheme=torch.per_channel_symmetric, quant_min=-127, quant_max=127, ch_axis=0)
    obs(torch.tensor([[-1., 1.], [-3., 3.], [-8., 8.], [-4., 4.], [-12., 12.]]))
    scales, zeros = obs.calculate_qparams()
    torch.testing.assert_close(scales, torch.tensor(expected) / 127, rtol=0, atol=0)
    assert torch.count_nonzero(zeros) == 0
try:
    Observer(block_scale_reduction="unknown")
except ValueError:
    pass
else:
    raise AssertionError("unknown reduction accepted")

# 2. the spec builds it only for symmetric weight channels
spec = RQ._spec({"granularity": "channel", "symmetric": True, "dtype": "int8", "quant_min": -127,
                 "quant_max": 127, "observer": "minmax", "channels_per_scale": 16}, is_weight=True, eps=1e-8)
built = spec.observer_or_fake_quant_ctr(dtype=spec.dtype, qscheme=spec.qscheme, quant_min=spec.quant_min,
                                       quant_max=spec.quant_max, ch_axis=spec.ch_axis)
assert type(built).__name__ == "GroupedChannelObserver" and built.channels_per_scale == 16, built
spec = RQ._spec({"granularity": "channel", "symmetric": True, "dtype": "int8", "channels_per_scale": 3,
                 "block_scale_reduction": "mean_abs"}, is_weight=True, eps=1e-8)
built = spec.observer_or_fake_quant_ctr(dtype=spec.dtype, qscheme=spec.qscheme)
assert built.block_scale_reduction == "mean_abs"
for bad in ({"channels_per_scale": 0}, {"channels_per_scale": 16, "symmetric": False}):
    try:
        RQ._spec({"granularity": "channel", "symmetric": True, "dtype": "int8", "quant_min": -127,
                  "quant_max": 127, "observer": "minmax", **bad}, is_weight=True, eps=1e-8)
    except RQ.RecipeError:
        pass
    else:
        raise AssertionError(bad)

# 3. bias onto the accumulator grid, on a converted-form graph
qd = torch.ops.quantized_decomposed
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randint(-127, 128, (3, 4), dtype=torch.int8)
        self.s = torch.tensor([0.01, 0.02, 0.04])
        self.z = torch.zeros(3, dtype=torch.int64)
        self.b = torch.tensor([0.1234, -0.5, 3.0])
g = torch.fx.Graph()
x = g.placeholder("x")
xdq = g.call_function(qd.dequantize_per_tensor.default, (x, 0.5, 0, -128, 127, torch.int8))
wdq = g.call_function(qd.dequantize_per_channel.default, (g.get_attr("w"), g.get_attr("s"), g.get_attr("z"), 0, -127, 127, torch.int8))
y = g.call_function(torch.ops.aten.linear.default, (xdq, wdq, g.get_attr("b")))
g.output(y)
gm = torch.fx.GraphModule(M(), g)
original = gm.b.clone()
census = RQ._bias_onto_accumulator_grid(gm)
step = 0.5 * torch.tensor([0.01, 0.02, 0.04])
codes = torch.round(original.double() / step.double())
assert census["sites"] == 1 and census["channels"] == 3 and census["max_change_in_steps"] <= 0.5 + 1e-6
assert torch.equal(gm.b, (codes.float() * torch.tensor(0.5)) * torch.tensor([0.01, 0.02, 0.04]))
# an i32 overflow is refused
gm.b.copy_(torch.tensor([1e12, 0.0, 0.0]))
try:
    RQ._bias_onto_accumulator_grid(gm)
except RQ.RecipeError:
    pass
else:
    raise AssertionError("overflow accepted")
print("OK")
"""


def test_grouped_scales_and_grid_biases_in_the_capture_interpreter() -> None:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not os.path.exists(python):
        pytest.skip("the model2MLIR torch interpreter is not configured")
    completed = subprocess.run([python, "-c", _SCRIPT, str(_DIR)], capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0 and completed.stdout.strip().endswith("OK"), completed.stderr[-3000:]
