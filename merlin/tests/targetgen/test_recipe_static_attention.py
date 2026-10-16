"""The ``static_heads`` mode of a recipe's ``dynamic`` block: attention with one calibrated scale per head.

WHY THIS EXISTS. Run-time per-row scales (the block's default) make every attention scale host work. A
readout applies ONE scale per device call, and the unit runs attention one head per call, so one static
scale per head of each operand is the form the hardware applies itself. Measured on SmolVLA (attention
only, fp32, 8 visible inputs): per-head static Q/K/V with keys smoothed by a calibrated mean and the
probabilities as ``exp(s - rowmax)`` in two int8 planes of fixed scales 1/127 and 1/(127*128) gives 0.049 /
0.068 worst-dim relRMSE against the fp32 model; the user chose it for SmolVLA v2.

What this pins:

* ``quant_recipe.with_dynamic_contractions(scales=STATIC_HEADS)`` states the mode and refuses what it
  cannot state (weights, dynamic recipes, a weight plane selector);
* with the block absent the graph is unchanged;
* with it, each attention becomes per-head ``_int_mm`` calls whose result equals an independent statement
  of the measurement's arithmetic (per-head minmax scales over the calibration samples, the smoothed keys,
  the fixed-scale probability planes, the row-sum divide);
* the key shift is folded into the key projection's bias when only layout steps lie between (the folded
  bias gives the shifted keys within float rounding) and is a host subtraction otherwise.

The quantizer runs in the capture venv (``MERLIN_M2M_PYTHON``); that half is skipped without it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import quant_recipe as QR

_QUANTIZER = merlin_dir() / "python/merlin/targetgen/_recipe_quantizer.py"


def _recipe() -> dict:
    body = {
        "schema": QR.SCHEMA,
        "target": "t",
        "unit": "u",
        "status": QR.DERIVED,
        "families": [QR.CONTRACTION],
        "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True, "quant_min": -127, "quant_max": 127},
        "activation": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -128,
            "quant_max": 127,
            "mode": "static",
        },
        "bias_domain": None,
        "unquantized": {},
        "why": {},
        "underivable": {},
    }
    body["recipe_sha256"] = QR.digest(body)
    return body


def _static(recipe: dict | None = None) -> dict:
    return QR.with_dynamic_contractions(
        recipe or _recipe(), planes={QR.SOFTMAX_OUTPUT: 2}, smooth=True, static_families=[], scales=QR.STATIC_HEADS
    )


def test_the_mode_is_stated_and_changes_the_digest():
    body = _static()
    assert body[QR.DYNAMIC]["scales"] == QR.STATIC_HEADS and body["families"] == [QR.ACTIVATION_CONTRACTION]
    dynamic = QR.with_dynamic_contractions(_recipe(), planes={QR.SOFTMAX_OUTPUT: 2}, smooth=True, static_families=[])
    assert body["recipe_sha256"] == QR.digest(body) != dynamic["recipe_sha256"]


@pytest.mark.parametrize(
    ("kwargs", "change"),
    [
        ({"families": [QR.CONTRACTION]}, {}),
        ({"planes": {QR.WEIGHT: 2}}, {}),
        ({}, {"mode": "dynamic"}),
        ({"scales": "per_tile"}, {}),
    ],
)
def test_what_the_mode_cannot_state_is_refused(kwargs, change):
    recipe = _recipe()
    recipe["activation"] = {**recipe["activation"], **change}
    with pytest.raises(ValueError, match="static per-head|scales are"):
        QR.with_dynamic_contractions(recipe, **{"scales": QR.STATIC_HEADS, **kwargs})


def test_the_mode_has_a_command_line(tmp_path):
    source, out = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text(json.dumps(_recipe()))
    argv = ["dynamic-contractions", str(source), str(out), "--planes", "softmax_output=2", "--smooth"]
    assert QR._main([*argv, "--scales", QR.STATIC_HEADS, "--static-families", ""]) == 0
    assert json.loads(out.read_text())[QR.DYNAMIC]["scales"] == QR.STATIC_HEADS


# --- the rewrite (capture venv) --------------------------------------------------------------------

_SCRIPT = r"""
import importlib.util
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source.parent))
sys.path.insert(0, str(source.parents[2]))
import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location("recipe_quantizer", source)
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)
recipe = json.loads(sys.argv[2])
Q, EPS, LOW = 127, 2.0**-12, -float("inf")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 64)
        self.kv = nn.Linear(64, 128)
        self.out = nn.Linear(64, 64)

    def heads(self, x):
        return x.view(1, 32, 2, 32).transpose(1, 2)

    def forward(self, x, keep):
        q, k, v = self.heads(self.q(x)), self.heads(self.k(x)), self.heads(self.v(x))
        # SigLIP-style: the key projection seen only through layout steps (its bias can take the shift).
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, scale=0.25)
        # Eager-style: keys and values unpacked from one projection (the shift stays a host subtraction).
        k2, v2 = self.kv(x).view(1, 32, 2, 2, 32).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k2.transpose(-2, -1))
        scores *= 0.25
        scores = torch.where(keep, scores, torch.finfo(torch.float32).min)
        eager = torch.matmul(torch.softmax(scores, dim=-1), v2)
        gate = torch.relu(torch.matmul(v, q.transpose(-2, -1)))  # no softmax reads it: no shift
        return self.out((context + eager).transpose(1, 2).reshape(1, 32, 64)) + gate.sum()


def grid(x, scale):
    return torch.clamp(torch.round(x / scale), -Q, Q)


def head_scale(tensors):
    absmax = torch.stack([t.abs().amax(dim=(0, 2, 3)) for t in tensors]).amax(0)
    return (absmax / 127.5).clamp(min=EPS).reshape(1, -1, 1, 1)


def scores_of(q, k, sq, sk):
    return (grid(q, sq).double() @ grid(k, sk).double().transpose(-2, -1) * sq.double() * sk.double()).float()


def context_of(s, v, sv):
    e = torch.exp(s - s.amax(dim=-1, keepdim=True))
    total = e.sum(dim=-1, keepdim=True)
    vi = grid(v, sv).double()
    p1 = torch.clamp(torch.round(e * Q), -Q, Q)
    p2 = torch.clamp(torch.round((e - p1 / Q) * (Q * 128)), -Q, Q)
    acc = (p1.double() @ vi) / Q + (p2.double() @ vi) / (Q * 128)
    return (acc * sv.double() / total.double()).float()


torch.manual_seed(0)
model = Block().eval()
keep = torch.ones(1, 1, 32, 32, dtype=torch.bool)
keep[..., :, 24:] = False
samples = [(torch.randn(1, 32, 64), keep) for _ in range(4)]
held = (torch.randn(1, 32, 64), keep)
graph = torch.export.export(model, samples[0]).module()
quant.spell_attention(graph)
before = graph.code
q = quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, model))
rows = q.rewrite_dynamic(graph, samples, 100)
result = {"rows": rows, "unchanged": graph.code == before}
result["int_mm"] = sum(n.target == torch.ops.aten._int_mm.default for n in graph.graph.nodes)
if rows:
    with torch.no_grad():
        # The measurement's arithmetic, stated independently: statistics of the float model over the samples.
        parts = [(model.heads(model.q(x)), model.heads(model.k(x)), model.heads(model.v(x)),
                  *model.kv(x).view(1, 32, 2, 2, 32).permute(2, 0, 3, 1, 4)) for x, _ in samples]
        mean_k = torch.cat([p[1] for p in parts], 2).double().mean(dim=(0, 2)).float()[None, :, None, :]
        mean_k2 = torch.cat([p[3] for p in parts], 2).double().mean(dim=(0, 2)).float()[None, :, None, :]
        sq = head_scale([p[0] for p in parts])
        sk = head_scale([p[1] - mean_k for p in parts])
        sv = head_scale([p[2] for p in parts])
        sk2 = head_scale([p[3] - mean_k2 for p in parts])
        sv2 = head_scale([p[4] for p in parts])
        x = held[0]
        qh, kh, vh = model.heads(model.q(x)), model.heads(model.k(x)), model.heads(model.v(x))
        k2, v2 = model.kv(x).view(1, 32, 2, 2, 32).permute(2, 0, 3, 1, 4)
        s = (scores_of(qh, kh - mean_k, sq, sk) * 0.25).masked_fill(~keep, LOW)
        context = context_of(s, vh, sv)
        s2 = torch.where(keep, scores_of(qh, k2 - mean_k2, sq, sk2) * 0.25, torch.finfo(torch.float32).min)
        eager = context_of(s2, v2, sv2)
        gate = torch.relu(scores_of(vh, qh, sv, sq))
        want = model.out((context + eager).transpose(1, 2).reshape(1, 32, 64)) + gate.sum()
        got = graph(*held)
        result["max_error"] = float((got - want).abs().max() / want.abs().max())
        # The folded bias gives the shifted keys within float rounding.
        folded = [n for n in graph.graph.nodes if n.op == "get_attr" and "_static_shift_" in str(n.target)]
        result["folded_biases"] = len(folded)
        if folded:
            bias = getattr(graph, folded[0].target)
            shifted = model.heads(F.linear(x, model.k.weight, bias))
            result["fold_error"] = float((shifted - (kh - mean_k)).abs().max() / kh.abs().max())
print(json.dumps(result, default=str))
"""


def _run(recipe: dict) -> dict:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    completed = subprocess.run(
        [python, "-c", _SCRIPT, str(_QUANTIZER), json.dumps(recipe)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_without_the_block_the_graph_is_unchanged():
    data = _run(QR.with_activation_contractions(_recipe()))
    assert data["rows"] == [] and data["unchanged"] and data["int_mm"] == 0


def test_attention_takes_static_per_head_scales_and_fixed_probability_planes():
    data = _run(_static())
    rows = data["rows"]
    assert len(rows) == 5 and all(r["status"] == "rewritten" and r["scales"] == QR.STATIC_HEADS for r in rows)
    smoothed = [r for r in rows if r["smooth"]]
    assert len(smoothed) == 2, "the two softmax-read Q.K^T are shifted, the relu-read one is not"
    assert sorted(r["shift_folded"] for r in smoothed) == [False, True], "folded where only layout steps lie between"
    assert sorted(r["planes"] for r in rows) == [[1, 1]] * 3 + [[2, 1]] * 2
    assert data["int_mm"] == 2 * (1 + 2 + 1 + 2 + 1), "per head: two Q.K^T, two P.V of two planes, the gate"
    assert all(r["host_elements"] > 0 and r["int8_macs"] > 0 for r in rows)
    assert data["max_error"] < 1e-5, data["max_error"]
    assert data["folded_biases"] == 1 and data["fold_error"] < 1e-6, data.get("fold_error")
