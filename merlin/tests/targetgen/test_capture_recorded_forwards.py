"""The agreement receipt reuses the floating outputs a calibration recurrence already computed.

WHY THIS EXISTS. A causal decode stream is the floating recurrence: it runs the stage on every sample
it yields, and the agreement receipt runs the same stage on the first 63 of those samples again. The
worker now keeps those outputs while the stream is read and hands them back only for the same
argument objects, unmodified, on the same unmodified module; anything else runs the module.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from merlin.targetgen import _m2m_capture_worker as worker

SCRIPT = r"""
import importlib.util, sys
import torch

spec = importlib.util.spec_from_file_location("worker_under_test", sys.argv[1])
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


class Step(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 3), requires_grad=False)
        self.calls = 0

    def forward(self, x, state):
        self.calls += 1
        out = x @ self.weight + state
        return out.sum(dim=-1, keepdim=True), out


def stream(step, count):
    state = torch.zeros(1, 3)
    with torch.no_grad():
        for i in range(count):
            x = torch.full((1, 3), float(i))
            yield (x, state)
            _logits, state = step(x, state)


torch.manual_seed(0)
step = Step().eval()
recorded = W._RecordedForwards(step, limit=4)
with recorded:
    samples = list(stream(step, 6))
assert step.calls == 6 and len(recorded.kept) == 4
assert not step._forward_hooks, "the hook is removed when the stream is read"

before = step.calls
for sample in samples[:4]:
    reused = recorded(*sample)
    fresh = step(*sample)
    assert torch.equal(reused[0], fresh[0]) and torch.equal(reused[1], fresh[1])
assert step.calls == before + 4, "recorded samples are not recomputed"

before = step.calls
recorded(*samples[4])                               # beyond the limit
recorded(samples[0][0].clone(), samples[0][1])      # equal values, another object
recorded(samples[1][1], samples[1][0])              # same objects, other order
assert step.calls == before + 3

samples[2][0].add_(0)                               # an in-place update of an argument
before = step.calls
recorded(*samples[2])
assert step.calls == before + 1

with torch.no_grad():
    step.weight.mul_(1)                             # an in-place update of the module
before = step.calls
recorded(*samples[0])
assert step.calls == before + 1

step.weight = torch.nn.Parameter(step.weight.detach().clone(), requires_grad=False)
before = step.calls
recorded(*samples[1])
assert step.calls == before + 1

step2 = Step().eval()
recorded2 = W._RecordedForwards(step2, limit=4)
with recorded2:
    samples2 = list(stream(step2, 3))
step2.train()
before = step2.calls
recorded2(*samples2[0])
assert step2.calls == before + 1, "a train/eval change is a different computation"
print("OK")
"""


def test_reused_floating_outputs_are_only_the_same_computation(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("an explicit capture interpreter with torch is required")
    script = tmp_path / "check.py"
    script.write_text(SCRIPT)
    result = subprocess.run(
        [python, str(script), worker.__file__],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]
    assert result.stdout.strip().endswith("OK")
