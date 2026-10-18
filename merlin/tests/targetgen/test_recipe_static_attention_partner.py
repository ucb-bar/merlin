"""The operand a softmax output multiplies takes its own digit count: an attention sink stops costing accuracy.

WHY THIS EXISTS. Measured on TinyLlama's pretrained middle layers, decode stage: 87% of the attention mass
sits on the first cache slot, and that slot's value rows are about 15x smaller than the head's largest.
``P V`` is a convex combination of V's rows, so the output is mostly that small row, while one static int8
scale per head sizes V's grid for the largest rows. The P.V sites carried a 23-29% relative error (cosine
0.956-0.974) where every other site stayed near 2%; the probability planes were not at fault (exact V: 0.4%).

Measured remedies on held-out calibration segments: one scale per (head, cache slot) folded into P clipped
held-out rows (up to 21% error); per-column-group scales helped little (22%). Two int8 digits for V alone
(a 15-bit static per-head grid) gave at most 0.7% on every layer. That is ``planes[SOFTMAX_PARTNER] = 2``:
the P.V device calls double, every attention contraction stays int8, and Q and K keep one digit.

The rewrite runs in the capture venv (``MERLIN_M2M_PYTHON``).
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import quant_recipe as QR

_DIR = merlin_dir() / "python/merlin/targetgen"


def _recipe() -> dict:
    body = QR.QuantRecipe(
        target="t",
        unit="u",
        weight=QR.TensorSpec("int8", "tensor", True, -127, 127),
        activation=QR.TensorSpec("int8", "tensor", True, -128, 127),
    ).to_dict() | {"accumulator_dtype": "int32"}
    body["recipe_sha256"] = QR.digest(body)
    return body


def _static(**planes: int) -> dict:
    return QR.with_dynamic_contractions(
        _recipe(), planes={QR.SOFTMAX_OUTPUT: 2, **planes}, static_families=[], scales=QR.STATIC_HEADS
    )


def test_the_selector_is_stated_and_bounded():
    partner = _static(**{QR.SOFTMAX_PARTNER: 2})
    assert partner[QR.DYNAMIC]["planes"] == {QR.SOFTMAX_OUTPUT: 2, QR.SOFTMAX_PARTNER: 2}
    assert partner["recipe_sha256"] == QR.digest(partner) != _static()["recipe_sha256"]
    with pytest.raises(ValueError, match="at most 3"):
        _static(**{QR.SOFTMAX_PARTNER: 4})
    with pytest.raises(ValueError, match="belong to static per-head scales"):
        QR.with_dynamic_contractions(_recipe(), planes={QR.SOFTMAX_PARTNER: 2}, static_families=[])


_SCRIPT = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
import torch
from torch.nn import functional as F
import _recipe_quantizer as RQ


class Sink(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = (torch.nn.Linear(64, 64) for _ in range(3))

    def forward(self, x):
        q, k, v = (f(x).view(1, 32, 2, 32).transpose(1, 2) for f in (self.q, self.k, self.v))
        mask = torch.ones(32, 32, dtype=torch.bool)
        sink = torch.zeros(1, 1, 32, 32)
        sink[..., 0] = 8.0  # every query attends mostly to the first key ...
        v = torch.cat([0.03 * v[:, :, :1], v[:, :, 1:]], dim=2)  # ... whose values are small
        return F.scaled_dot_product_attention(q, k, v, attn_mask=sink.masked_fill(~mask, 0.0), scale=0.1)


out = {}
for name, recipe in json.loads(sys.argv[2]).items():
    torch.manual_seed(0)
    model = Sink().eval()
    calibration = [(torch.randn(1, 32, 64),) for _ in range(16)]
    graph = torch.export.export(model, calibration[0]).module()
    RQ.spell_attention(graph)
    quantizer = RQ.build_quantizer(recipe, layer_plan=RQ._plan_layers(recipe, model))
    rows = quantizer.rewrite_dynamic(graph, calibration, 100)
    held = [(torch.randn(1, 32, 64),) for _ in range(4)]
    with torch.no_grad():
        errors = [(graph(*x) - model(*x)).double().pow(2).mean().sqrt() for x in held]
        rel = max(float(e / model(*x).double().pow(2).mean().sqrt()) for e, x in zip(errors, held))
    out[name] = {"rel": rel, "planes": [r.get("planes") for r in rows], "calls": [r.get("device_calls") for r in rows]}
print(json.dumps(out))
"""


def test_two_digits_for_the_softmax_partner_rescue_a_sink(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not os.path.exists(python):
        pytest.skip("the model2MLIR torch interpreter is not configured")
    recipes = json.dumps({"head": _static(), "partner": _static(**{QR.SOFTMAX_PARTNER: 2})})
    done = subprocess.run([python, "-c", _SCRIPT, str(_DIR), recipes], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    head, partner = got["head"], got["partner"]
    assert head["planes"] == [[1, 1], [2, 1]] and partner["planes"] == [[1, 1], [2, 2]]  # Q.K keeps one digit
    assert partner["calls"][0] == head["calls"][0] and partner["calls"][1] == 2 * head["calls"][1]
    assert partner["rel"] < head["rel"] / 4, got
