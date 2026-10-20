"""A per-channel weight scale sits on the weight's OUTPUT axis, whichever contraction reads it.

WHY THIS EXISTS. The static quantizer built one weight spec with ``ch_axis=0``. That is the output axis of a
Linear's ``[N, K]`` weight and of a convolution's ``[Cout, ...]``, but a ``matmul`` against a stored
``[K, N]`` operand has its REDUCTION on axis 0, so a per-channel recipe gave it one scale per K. No readout
can apply a scale inside the sum: the integerizer refuses such a contraction ("weight channel axis must be
the output dimension") and it stays a floating dequantize-then-contract. Now a matmul's stored operand
takes its scales on its last axis.

What this pins (capture venv, ``MERLIN_M2M_PYTHON`` + ``MERLIN_M2M_DIR``; skipped without them): under a
per-channel recipe the converted program dequantizes a Linear's weight along axis 0 and a matmul's stored
``[K, N]`` operand along axis 1 with N scales, and ``integerize_pt2e`` turns both into integer contractions
with nothing refused.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import quant_recipe as QR

_TARGETGEN = Path(QR.__file__).parent

_SCRIPT = r"""
import json
import sys
from pathlib import Path

targetgen, m2m_dir = Path(sys.argv[1]), sys.argv[2]
for path in (str(targetgen), str(targetgen.parents[1]), m2m_dir):
    sys.path.insert(0, path)
import torch
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

import _recipe_quantizer as RQ
from m2m.capture.pt2e_integerize import integerize_pt2e


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(48, 40)
        self.table = nn.Parameter(torch.randn(40, 24))  # a stored [K, N] operand

    def forward(self, x):
        return torch.matmul(self.proj(x), self.table)


torch.manual_seed(0)
recipe = json.loads(sys.argv[3])
model = Model().eval()
samples = [(torch.randn(8, 48),) for _ in range(4)]
graph = torch.export.export(model, samples[0]).module()
prepared = prepare_pt2e(graph, RQ.build_quantizer(recipe, layer_plan=RQ._plan_layers(recipe, model)))
with torch.no_grad():
    for sample in samples:
        prepared(*sample)
converted = convert_pt2e(prepared, fold_quantize=True)
channel = torch.ops.quantized_decomposed.dequantize_per_channel.default
weights = {}
for node in converted.graph.nodes:
    if node.target == channel:
        reader = next(iter(node.users)).target
        scales = getattr(converted, str(node.args[1].target))
        weights[str(reader).split(".")[1]] = {"axis": node.args[3], "scales": int(scales.numel())}
_, receipt = integerize_pt2e(converted, samples[0])
print(json.dumps({"weights": weights, "integerized": receipt["quantized_contractions_integerized"],
                  "refusals": receipt["refusals"]}))
"""


def _recipe() -> dict:
    body = {
        "schema": QR.SCHEMA,
        "target": "t",
        "unit": "u",
        "status": QR.DERIVED,
        "families": [QR.CONTRACTION],
        "weight": {"dtype": "int8", "granularity": "channel", "symmetric": True, "quant_min": -127, "quant_max": 127},
        "activation": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -128,
            "quant_max": 127,
            "mode": "static",
            "observer": "minmax",
        },
        "bias_domain": None,
        "unquantized": {},
        "why": {},
        "underivable": {},
    }
    body["recipe_sha256"] = QR.digest(body)
    return body


def test_a_matmul_stored_operand_takes_its_scales_on_the_output_axis():
    python, m2m_dir = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m_dir:
        pytest.skip("no capture interpreter and model2MLIR checkout are configured")
    completed = subprocess.run(
        [python, "-c", _SCRIPT, str(_TARGETGEN), m2m_dir, json.dumps(_recipe())],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    data = json.loads(completed.stdout.strip().splitlines()[-1])
    assert data["weights"] == {"linear": {"axis": 0, "scales": 40}, "matmul": {"axis": 1, "scales": 24}}
    assert data["integerized"] == 2 and data["refusals"] == []
