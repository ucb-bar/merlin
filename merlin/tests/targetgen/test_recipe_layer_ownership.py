"""A supported ATen spelling inside an unsupported module is still host work."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


def test_static_pt2e_annotates_only_placed_operations() -> None:
    python = os.environ.get("MERLIN_M2M_PYTHON") or (
        str(Path(os.environ["MERLIN_M2M_VENV"]) / "bin/python") if os.environ.get("MERLIN_M2M_VENV") else ""
    )
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    source = merlin_dir() / "python/merlin/targetgen/_recipe_quantizer.py"
    script = r"""
import importlib.util
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location("recipe_quantizer", sys.argv[1])
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)
recipe = {
    "schema": "quant_recipe_v1", "status": "derived", "families": ["contraction"],
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
               "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}

class Unsupported(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        return F.linear(x, self.weight)

class RuntimeMatmul(nn.Linear):
    def forward(self, x):
        # This module has Linear's stored weight, but this operation does not use it.
        return torch.matmul(x, x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.unsupported = Unsupported()
        self.supported = nn.Linear(4, 4)
        self.root_weight = nn.Parameter(torch.randn(4, 4))
        self.embedding = nn.Embedding(8, 4)
        self.lstm = nn.LSTM(4, 4, batch_first=True)
        self.runtime = RuntimeMatmul(4, 4)

    def forward(self, x):
        recurrent, _ = self.lstm(x.unsqueeze(1))
        return (self.supported(self.unsupported(x)) + torch.matmul(x, self.root_weight)
                + recurrent[:, 0, :] + self.runtime(x))

model = Net().eval()
graph = torch.export.export(model, (torch.randn(4, 4),)).module()
plan = quant._plan_layers(recipe, model)
q = quant.build_quantizer(recipe, layer_plan=plan)
q.annotate(graph)
annotated = []
for node in graph.graph.nodes:
    if node.meta.get("quantization_annotation"):
        annotated.append((str(node.target), list(node.meta["nn_module_stack"].values())[-1][0]))
print(json.dumps({"annotated": annotated, "refused": q.refused_by_placement,
                  "host": [(d["fqn"], d["refusal"]) for d in plan["layers"]
                           if d["placement"] == "host"], "count": q.annotated}))
"""
    result = subprocess.run([python, "-c", script, str(source)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["count"] == 2, data
    assert ["aten.linear.default", "supported"] in data["annotated"]
    assert ["aten.matmul.default", ""] in data["annotated"]
    assert ["unsupported", "unmapped_module"] in data["host"]
    assert ["embedding", "unmapped_module"] in data["host"]
    assert ["lstm", "unmapped_module"] in data["host"]
    assert any(d["owner"] == "unsupported" for d in data["refused"]), data
    assert any(d["owner"] == "runtime" for d in data["refused"]), data
    assert not any(owner in ("lstm", "runtime") for _op, owner in data["annotated"]), data
