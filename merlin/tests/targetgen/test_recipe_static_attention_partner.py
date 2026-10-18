"""A softmax partner gets extra static residual planes when attention concentrates on a small row.

P.V can be small against the value operand's per-head grid. The independent synthetic network below
constructs that condition and checks the extra plane on unseen inputs. Each plane stays on the
existing signed-int8 round-and-clamp producer domain; the rewrite adds no wide clamp or floor.
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
    for selector in (QR.SOFTMAX_PARTNER, QR.RESIDUAL_OPERAND):
        with pytest.raises(ValueError, match="belong to static per-head scales"):
            QR.with_dynamic_contractions(_recipe(), planes={selector: 2}, static_families=[])
    residual = _static(**{QR.RESIDUAL_OPERAND: 2})
    assert residual["recipe_sha256"] != _static(**{QR.OPERAND: 2})["recipe_sha256"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        _static(**{QR.OPERAND: 2, QR.RESIDUAL_OPERAND: 2})
    with pytest.raises(ValueError, match="at most 3"):
        _static(**{QR.RESIDUAL_OPERAND: 4})


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
    RQ._verify_dynamic_sites(graph, rows, held[0], held[1:])
    floors = sum(n.target == torch.ops.aten.floor.default for n in graph.graph.nodes)
    clamps = [n.args[1:3] for n in graph.graph.nodes if n.target == torch.ops.aten.clamp.default and len(n.args) == 3]
    out[name] = {"rel": rel, "planes": [r.get("planes") for r in rows],
                 "calls": [r.get("device_calls") for r in rows], "floors": floors, "clamps": clamps,
                 "verified": all(r["verification"]["finite"] and r["verification"]["samples"] == 4 for r in rows)}
print(json.dumps(out))
"""


def test_two_digits_for_the_softmax_partner_rescue_a_sink(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not os.path.exists(python):
        pytest.skip("the model2MLIR torch interpreter is not configured")
    recipes = json.dumps(
        {"head": _static(), "partner": _static(**{QR.SOFTMAX_PARTNER: 2}),
         "legacy": _static(**{QR.OPERAND: 2}), "residual": _static(**{QR.RESIDUAL_OPERAND: 2})}
    )
    done = subprocess.run([python, "-c", _SCRIPT, str(_DIR), recipes], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    head, partner = got["head"], got["partner"]
    assert head["planes"] == [[1, 1], [2, 1]] and partner["planes"] == [[1, 1], [2, 2]]  # Q.K keeps one digit
    assert partner["calls"][0] == head["calls"][0] and partner["calls"][1] == 2 * head["calls"][1]
    assert partner["rel"] < head["rel"] / 4, got
    assert partner["floors"] == 0
    assert partner["clamps"] and all(clamp == [-127, 127] for clamp in partner["clamps"])
    assert got["legacy"]["floors"] > 0
    assert [-16384, 16383] in got["legacy"]["clamps"]
    residual = got["residual"]
    assert residual["planes"] == [[2, 2], [2, 2]]
    assert residual["calls"] == [4 * head["calls"][0], 2 * head["calls"][1]]
    assert residual["floors"] == 0 and all(clamp == [-127, 127] for clamp in residual["clamps"])
    assert residual["rel"] < head["rel"] / 4, got
    assert all(row["verified"] for row in got.values())
