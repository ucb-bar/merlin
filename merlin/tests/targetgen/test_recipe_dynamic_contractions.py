"""A recipe's ``dynamic`` block quantizes contractions with run-time scales on non-reduction axes.

WHY THIS EXISTS. Static per-tensor scales cannot represent attention's probabilities: measured on SmolVLA's
8 visible inputs (attention only, linears in float), the static recipe put the per-input worst action dim
at 0.85 relRMSE mean, against a float noise of 0.026. Run-time scales on axes a contraction does not
reduce (one per row of the streamed operand, one per column of the other) keep each int8 product exact
in the i32 accumulator; the host rescales by their outer product. With the softmax probabilities split
into two int8 planes and the keys smoothed (their mean over the keys removed: a per-row shift the softmax
ignores), the same measurement gives 0.044 mean / 0.075 max, inside twice the float noise.

What this pins:

* ``quant_recipe.with_dynamic_contractions`` states the block (families, scales, planes, smooth), refuses
  what it cannot state, and changes the content digest;
* with the block absent, the quantizer leaves the exported graph exactly as it was;
* with it, each selected contraction becomes per-slice ``_int_mm`` calls whose result equals an
  independent statement of the same arithmetic (the measurement harness's), the smooth shift is applied
  only where a row softmax is the sole reader, and a softmax-produced operand gets its planes;
* a Linear's stored weight takes one static scale per output channel, its input run-time row scales.

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
        "families": [QR.CONTRACTION, QR.WINDOW_MEAN],
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


# --- the recipe block ------------------------------------------------------------------------------


def test_the_block_states_families_scales_planes_and_smooth_and_changes_the_digest():
    base = _recipe()
    body = QR.with_dynamic_contractions(base, planes={QR.SOFTMAX_OUTPUT: 2}, smooth=True, static_families=[])
    assert body["families"] == [QR.ACTIVATION_CONTRACTION]
    assert body[QR.DYNAMIC] == {
        "schema": QR.DYNAMIC_SCHEMA,
        "families": [QR.ACTIVATION_CONTRACTION],
        "scales": QR.NON_REDUCTION,
        "planes": {QR.SOFTMAX_OUTPUT: 2},
        "smooth": QR.SOFTMAX_AXIS,
    }
    assert body["recipe_sha256"] == QR.digest(body) != base["recipe_sha256"]
    kept = QR.with_dynamic_contractions(base, families=[QR.CONTRACTION])
    assert kept["families"] == [QR.CONTRACTION, QR.WINDOW_MEAN] and kept[QR.DYNAMIC]["smooth"] is None
    assert base["families"] == [QR.CONTRACTION, QR.WINDOW_MEAN], "the input recipe is not mutated"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"families": [QR.WINDOW_MEAN]}, "contraction families only"),
        ({"planes": {"keys": 2}}, "planes take"),
        ({"planes": {QR.SOFTMAX_OUTPUT: 9}}, "planes take"),
        ({"static_families": ["operand_sum"]}, "not the recipe's own"),
    ],
)
def test_what_the_block_cannot_state_is_refused(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        QR.with_dynamic_contractions(_recipe(), **kwargs)
    floating = _recipe()
    floating["activation"] = {**floating["activation"], "dtype": "fp8_e4m3"}
    with pytest.raises(ValueError, match="symmetric integer"):
        QR.with_dynamic_contractions(floating)


def test_the_block_has_a_command_line(tmp_path):
    source, out = tmp_path / "recipe.json", tmp_path / "out.json"
    source.write_text(json.dumps(_recipe()))
    argv = ["dynamic-contractions", str(source), str(out), "--planes", "softmax_output=2", "--smooth"]
    assert QR._main([*argv, "--static-families", ""]) == 0  # noqa: SLF001
    expected = QR.with_dynamic_contractions(_recipe(), planes={QR.SOFTMAX_OUTPUT: 2}, smooth=True, static_families=[])
    assert json.loads(out.read_text()) == expected


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
QMAX = 127


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(64, 192)
        self.out = nn.Linear(64, 64)

    def forward(self, x, mask):
        q, k, v = self.qkv(x).reshape(1, 32, 3, 2, 32).permute(2, 0, 3, 1, 4)
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.25)
        gate = torch.relu(torch.matmul(q, k.transpose(-2, -1)))  # read by a relu: never smoothed
        return self.out(context.transpose(1, 2).reshape(1, 32, 64)) + gate.sum()


def grid(x, scale):
    return torch.clamp(torch.round(x / scale), -QMAX, QMAX)


def absmax(x, dim):
    return (x.abs().amax(dim=dim, keepdim=True) / QMAX).clamp(min=1.1754943508222875e-38)


def reference(model, x, mask):
    # The measurement harness's arithmetic (iso2), stated independently of the module under test.
    q, k, v = model.qkv(x).reshape(1, 32, 3, 2, 32).permute(2, 0, 3, 1, 4)
    ks = k - k.mean(dim=-2, keepdim=True)
    sq, sk = absmax(q, -1), absmax(ks, -1)
    scores = (grid(q, sq).double() @ grid(ks, sk).double().transpose(-2, -1)) * sq.double() * sk.double().transpose(-2, -1)
    p = torch.softmax(scores.float() * 0.25 + mask, dim=-1)
    sv = absmax(v, -2)
    vi = grid(v, sv).double()
    out, residual = 0, p.double()
    for _ in range(2):
        sp = absmax(residual, -1)
        pi = grid(residual, sp)
        out = out + (pi @ vi) * sp * sv.double()
        residual = residual - pi * sp
    sg, kg = absmax(q, -1), absmax(k, -1)  # the relu-read matmul: no smoothing, one plane
    gate = torch.relu((grid(q, sg).double() @ grid(k, kg).double().transpose(-2, -1)) * sg.double() * kg.double().transpose(-2, -1))
    return model.out(out.float().transpose(1, 2).reshape(1, 32, 64)) + gate.float().sum()


torch.manual_seed(0)
model = Block().eval()
inputs = (torch.randn(1, 32, 64), torch.zeros(1, 1, 32, 32))
graph = torch.export.export(model, inputs).module()
quant.spell_attention(graph)
before = graph.code
q = quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, model))
rows = q.rewrite_dynamic(graph)
int_mm = sum(n.target == torch.ops.aten._int_mm.default for n in graph.graph.nodes)
result = {"rows": rows, "unchanged": graph.code == before, "int_mm": int_mm}
if rows:
    with torch.no_grad():
        got, want = graph(*inputs), reference(model, *inputs)
    result["max_error"] = float((got - want).abs().max() / want.abs().max())
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    prepared = prepare_pt2e(graph, q)
    with torch.no_grad():
        prepared(*inputs)
        converted = convert_pt2e(prepared, fold_quantize=True)
        result["after_pt2e_error"] = float((converted(*inputs) - got).abs().max())
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


def test_attention_takes_run_time_scales_smoothed_keys_and_two_probability_planes():
    recipe = QR.with_dynamic_contractions(_recipe(), planes={QR.SOFTMAX_OUTPUT: 2}, smooth=True, static_families=[])
    data = _run(recipe)
    assert len(data["rows"]) == 3 and all(r["status"] == "rewritten" for r in data["rows"])
    scores = [r for r in data["rows"] if r["smooth"]]
    assert len(scores) == 1 and scores[0]["planes"] == [1, 1], "only the softmax-read Q.K^T is smoothed"
    assert sorted(r["planes"] for r in data["rows"]) == [[1, 1], [1, 1], [2, 1]], "P.V's P has two planes"
    # Two heads: Q.K^T and the relu-read product one call each per head, P.V two (one per plane).
    assert data["int_mm"] == 2 * (1 + 1 + 2)
    assert all(r["host_elements"] > 0 and r["int8_macs"] > 0 for r in data["rows"])
    assert data["max_error"] < 1e-5, data["max_error"]
    assert data["after_pt2e_error"] == 0.0, "nothing left to annotate: PT2E leaves the integer form alone"


def test_a_linear_takes_run_time_row_scales_and_a_stored_per_channel_weight():
    recipe = QR.with_dynamic_contractions(_recipe(), families=[QR.CONTRACTION], static_families=[])
    data = _run(recipe)
    linears = [r for r in data["rows"] if r["family"] == QR.CONTRACTION]
    assert len(linears) == 2 and all(r["status"] == "rewritten" and r["planes"] == [1, 1] for r in linears)
    assert data["int_mm"] == 2
