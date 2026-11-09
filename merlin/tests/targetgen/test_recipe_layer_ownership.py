"""A supported ATen spelling inside an unsupported module is still host work."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import module_source_path


def test_static_pt2e_annotates_only_placed_operations() -> None:
    python = os.environ.get("MERLIN_M2M_PYTHON") or (
        str(Path(os.environ["MERLIN_M2M_VENV"]) / "bin/python") if os.environ.get("MERLIN_M2M_VENV") else ""
    )
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    source = module_source_path("merlin.targetgen._recipe_specs").with_name("_recipe_quantizer.py")
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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_static_pt2e_places_closed_functional_sum_in_container(nested, inplace) -> None:
    """Container-owned arithmetic needs the same operation plan as root arithmetic."""
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    source = module_source_path("merlin.targetgen._recipe_specs").with_name("_recipe_quantizer.py")
    script = r"""
import importlib.util
import json
import sys
import torch
from torch import nn

spec = importlib.util.spec_from_file_location("recipe_quantizer", sys.argv[1])
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)
recipe = {
    "schema": "quant_recipe_v1", "status": "derived",
    "families": ["contraction", "operand_sum"],
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
               "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}
inplace = sys.argv[3] == "True"

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.reader = nn.Linear(4, 4)

    def forward(self, x):
        y = self.first(x)
        if inplace:
            y += x
        else:
            y = y + x
        return self.reader(torch.relu(y))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Residual()

    def forward(self, x):
        return self.block(x)

torch.manual_seed(0)
model = (Net() if sys.argv[2] == "True" else Residual()).eval()
graph = torch.export.export(model, (torch.randn(2, 4),)).module()
q = quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, model))
q.annotate(graph)
print(json.dumps({"contractions": q.annotated, "sums": q.annotated_sums,
                  "refused": q.refused_by_placement}))

# An unknown leaf is still a refusal, even with a placed reader. A result
# returned in float remains open and must not acquire operand quantizers.
class UnknownSum(nn.Module):
    def forward(self, x):
        return x + x

class LeafNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sum = UnknownSum()
        self.reader = nn.Linear(4, 4)

    def forward(self, x):
        return self.reader(self.sum(x))

class OpenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4)

    def forward(self, x):
        return self.first(x) + x

for negative in (LeafNet(), OpenNet()):
    graph = torch.export.export(negative.eval(), (torch.randn(2, 4),)).module()
    rejected = quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, negative))
    rejected.annotate(graph)
    assert rejected.annotated_sums == 0, rejected.refused_by_placement
"""
    result = subprocess.run(
        [python, "-c", script, str(source), str(nested), str(inplace)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["contractions"] == 2, data
    assert data["sums"] == 1, data
    assert data["refused"] == [], data
