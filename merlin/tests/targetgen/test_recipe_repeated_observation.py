"""A stored weight is observed once per calibration, with bit-identical scales.

WHY THIS EXISTS. PT2E feeds every stored weight to its min/max observer on every calibration sample.
For TinyLlama's decode stage that is 1024 passes over all weights, a quarter of the stage's recipe
time, and every pass after the first leaves the running min and max unchanged. The calibration loop
now skips an observer call whose input is the exact tensor (same object, version, storage and
geometry) it observed last. These checks run TorchAO itself, so they need the capture interpreter.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import _m2m_capture_worker as worker

_QUANTIZER = Path(worker.__file__).with_name("_recipe_quantizer.py")

SCRIPT = r"""
import importlib.util, sys
from pathlib import Path
import torch
from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver

sys.path.insert(0, str(Path(sys.argv[1]).parent))
import _recipe_observers as RO
Grouped = RO.grouped_channel_observer()
spec = importlib.util.spec_from_file_location("rq", sys.argv[1])
RQ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RQ)
torch.manual_seed(0)


class StatefulChannel(PerChannelMinMaxObserver):
    def __init__(self):
        super().__init__(dtype=torch.int8, qscheme=torch.per_channel_symmetric)
        self.register_buffer('visits', torch.tensor(0))

    def forward(self, x):
        self.visits += 1
        return super().forward(x)


class Holder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.per_tensor = MinMaxObserver(dtype=torch.int8, qscheme=torch.per_tensor_symmetric)
        self.per_channel = PerChannelMinMaxObserver(dtype=torch.int8, qscheme=torch.per_channel_symmetric)
        self.grouped = Grouped(dtype=torch.int8, qscheme=torch.per_channel_symmetric, channels_per_scale=3)
        self.stateful = StatefulChannel()
        self.histogram = HistogramObserver(dtype=torch.int8)

    def forward(self, values):
        for value in values:
            self.per_tensor(value)
            self.per_channel(value)
            self.grouped(value)
            self.stateful(value)
            self.histogram(value)


def calls(weight, other):
    # the same weight again and again, an in-place update of it, a fresh activation, another
    # tensor and back to the first: each case the skip must either prove unchanged or observe
    yield [weight, weight, weight]
    with torch.no_grad():
        weight[0, 0] = 7.5
    yield [weight, weight]
    yield [torch.randn(4, 6) * 3, weight]
    yield [other, weight, other]
    yield [weight.t().contiguous().t(), weight]


def state(holder):
    result = {k: v.clone() for k, v in holder.state_dict().items()}
    for name in ('per_tensor', 'per_channel', 'grouped', 'histogram'):
        scale, zero = getattr(holder, name).calculate_qparams()
        result[name + '.scale'] = scale.clone()
        result[name + '.zero'] = zero.clone()
    return result


def run(skip):
    torch.manual_seed(1)
    weight = torch.nn.Parameter(torch.randn(4, 6), requires_grad=False)
    other = torch.randn(4, 6) * 10
    holder = Holder()
    counts = []
    calls_seen = {'channel': [], 'grouped': [], 'stateful': []}
    original = PerChannelMinMaxObserver._forward

    def counting(self, x):
        key = {PerChannelMinMaxObserver: 'channel', Grouped: 'grouped', StatefulChannel: 'stateful'}[type(self)]
        calls_seen[key].append(1)
        return original(self, x)

    PerChannelMinMaxObserver._forward = counting
    try:
        context = RQ._RepeatedObservationSkip(holder) if skip else None
        if context is not None:
            context.__enter__()
        for batch in calls(weight, other):
            torch.manual_seed(2)
            holder(batch)
            counts.append({name: len(values) for name, values in calls_seen.items()})
        if context is not None:
            context.__exit__(None, None, None)
    finally:
        PerChannelMinMaxObserver._forward = original
    assert "forward" not in holder.per_tensor.__dict__ and "forward" not in holder.per_channel.__dict__
    assert "forward" not in holder.grouped.__dict__ and "forward" not in holder.stateful.__dict__
    return state(holder), counts


reference, full = run(False)
skipped, partial = run(True)
assert reference.keys() == skipped.keys()
for key in reference:
    assert reference[key].dtype == skipped[key].dtype and torch.equal(reference[key], skipped[key]), key
    assert reference[key].numpy().tobytes() == skipped[key].numpy().tobytes(), key
complete = [3, 5, 7, 10, 12]
for name in ('channel', 'grouped', 'stateful'):
    assert [row[name] for row in full] == complete, (name, full)
# 1 of 3, then the update is observed once, then fresh/weight, other/weight/other, view/weight
for name in ('channel', 'grouped'):
    assert [row[name] for row in partial] == [1, 2, 4, 7, 9], (name, partial)
assert [row['stateful'] for row in partial] == complete, partial
# A future/custom replacement of the known grouped forward must lose the optimization.
original_grouped = Grouped.forward
def changed_forward(self, x):
    return PerChannelMinMaxObserver.forward(self, x)
Grouped.forward = changed_forward
try:
    unchanged, guarded = run(True)
    assert [row['grouped'] for row in guarded] == complete, guarded
    for key in reference:
        assert torch.equal(reference[key], unchanged[key]), key
finally:
    Grouped.forward = original_grouped
print("OK")
"""


def test_skipped_repeated_observations_leave_observer_state_bit_identical(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("an explicit capture interpreter with TorchAO is required")
    script = tmp_path / "check.py"
    script.write_text(SCRIPT)
    result = subprocess.run(
        [python, str(script), str(_QUANTIZER)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]
    assert result.stdout.strip().endswith("OK")
