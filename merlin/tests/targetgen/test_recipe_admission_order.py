"""Recipe admission metadata must replay byte-identically across Python processes."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import module_source_path


def test_closed_sum_admission_rows_are_stable_across_hash_seeds() -> None:
    """Exercise the unordered closed-sum fixpoint, not a pre-sorted mock record."""
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    source = module_source_path("merlin.targetgen._recipe_specs").with_name("_recipe_quantizer.py")
    script = r"""
import importlib.util
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source.parents[2]))
import torch
from torch import nn

# Keep unrelated FX nodes alive to vary Node identity/hash placement while
# preserving the exported graph and every semantic input across processes.
noise = torch.fx.Graph()
noise_input = noise.placeholder("noise")
for _ in range(int(sys.argv[2]) * 37):
    noise.call_function(torch.ops.aten.add.Tensor, (noise_input, noise_input))

spec = importlib.util.spec_from_file_location("recipe_quantizer", source)
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)

recipe = {
    "schema": "quant_recipe_v1", "status": "derived",
    "families": ["contraction", "operand_sum"],
    "accumulator_dtype": "int32",
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
               "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
    "software_admission": {
        "status": "reviewed",
        "operations": [
            {"id": family, "families": [family], "placement": "accelerator", "signature": {}}
            for family in ("contraction", "operand_sum")
        ],
    },
}

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(8, 8)
        self.reader = nn.Linear(8, 8)

    def forward(self, x):
        return self.reader(self.first(x) + x)

class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(12))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

torch.manual_seed(9)
model = Chain().eval()
graph = torch.export.export(model, (torch.randn(2, 8),)).module()
q = quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, model))
q.annotate(graph)
print(json.dumps({"rows": q.software_decisions, "closed_sums": q.annotated_sums,
                  "contractions": q.annotated, "hash_probe": hash("admission report")}, sort_keys=True))
"""
    observations = []
    for seed in ("1", "2", "3", "4", "5", "6"):
        env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [python, "-P", "-B", "-c", script, str(source), seed],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-3000:]
        observations.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    assert len({row["hash_probe"] for row in observations}) == len(observations), (
        "child interpreter ignored the independently selected Python hash seeds"
    )
    assert all(row["closed_sums"] == 12 and row["contractions"] == 24 for row in observations)
    rows = observations[0]["rows"]
    assert len(rows) > len({json.dumps(row, sort_keys=True) for row in rows}), (
        "the closed-sum fixpoint did not revisit admission; this would not test its unordered path"
    )
    expected_multiset = sorted(json.dumps(row, sort_keys=True) for row in rows)
    assert all(
        sorted(json.dumps(row, sort_keys=True) for row in observation["rows"]) == expected_multiset
        for observation in observations[1:]
    ), "the neutral graph or its actual admission decisions changed, not merely report order"
    assert all(observation["rows"] == rows for observation in observations[1:])
