"""Selected static recipe quantizes string-padding Conv2d with exact ancestry."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

_SOURCE = merlin_dir() / "python/merlin/targetgen/_recipe_quantizer.py"
_PROBE = r"""
import importlib.util
import json
import pathlib
import sys

sys.path.insert(0, sys.argv[2])
# The quantizer imports its own Merlin package (as the capture worker arranges), so
# the probe puts the package that owns the selected source on the path, never another checkout.
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve().parents[2]))
import m2m
import torch

spec = importlib.util.spec_from_file_location("recipe_quantizer", sys.argv[1])
quantizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantizer)

recipe = {
    "schema": "quant_recipe_v1", "status": "derived", "families": ["contraction"],
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
               "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}


class Neutral(torch.nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, (4, 3), padding=padding, bias=False)
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, x):
        return self.linear(self.conv(x).mean((2, 3)))


class NeutralBN(Neutral):
    def __init__(self):
        super().__init__("same")
        self.bn = torch.nn.BatchNorm2d(2)

    def forward(self, x):
        return self.linear(self.bn(self.conv(x)).mean((2, 3)))


torch.manual_seed(11)
input_tensor = torch.randn(1, 1, 6, 6)
rows = []
for label, padding in (("numeric", (1, 1)), ("valid", "valid"), ("same_asymmetric", "same")):
    model = Neutral(padding).eval()
    original = m2m.capture_frontend_snapshot(model, (input_tensor,))
    quantized = quantizer.apply_recipe(
        model, recipe, example_inputs=(input_tensor,), original_frontend_snapshot=original
    )
    rows.append({
        "case": label,
        "source_conv": [node["target"] for node in original["nodes"] if "conv2d" in node["target"]],
        "annotated": quantized._recipe_quantization_stats["annotated_contractions"],
        "normalizations": quantized._recipe_quantization_stats.get("conv2d_padding_normalizations"),
        "refused": quantized._recipe_quantization_stats["operators_refused_by_placement"],
    })

bn_model = NeutralBN().eval()
bn_original = m2m.capture_frontend_snapshot(bn_model, (input_tensor,))
bn_quantized = quantizer.apply_recipe(
    bn_model, recipe, example_inputs=(input_tensor,), original_frontend_snapshot=bn_original
)
bn_ids = {node["id"] for node in bn_original["nodes"] if node["target"] == "aten.batch_norm.default"}
bn_carried = {origin for node in bn_quantized.graph.nodes
              if node.target == torch.ops.aten.conv2d.default
              for origin in (node.meta.get("custom") or {}).get("m2m_lineage", ()) if origin in bn_ids}

bf16_source = Neutral("same").eval().to(torch.bfloat16)
bf16_input = input_tensor.to(torch.bfloat16)
staged_model, staged_inputs, staged_original, staging = m2m.materialize_frontend_precision(
    bf16_source, (bf16_input,), dtype=torch.float32, retarget_float_dtype_arguments=True,
)
staged_quantized = quantizer.apply_recipe(
    staged_model, recipe, example_inputs=staged_inputs, original_frontend_snapshot=staged_original,
    source_layer_inventory=quantizer.layer_inventory(bf16_source),
)
staged_ids = {node["id"] for node in staged_original["nodes"] if node["target"] == "aten.conv2d.padding"}
staged_rows = staged_quantized._recipe_quantization_stats["conv2d_padding_normalizations"]
broken_source = m2m.capture_frontend_snapshot(Neutral("valid").eval(), (input_tensor,))
try:
    quantizer.apply_recipe(
        staged_model, recipe, example_inputs=staged_inputs,
        original_frontend_snapshot=broken_source,
        source_layer_inventory=quantizer.layer_inventory(bf16_source),
    )
except quantizer.RecipeError as error:
    broken_source_refused = "lost original source ancestry" in str(error)
else:
    broken_source_refused = False
print(json.dumps({
    "rows": rows,
    "bn": {"fold_api": bn_quantized._recipe_quantization_stats["fold_provenance_api"],
           "original_bn": len(bn_ids), "carried_bn": len(bn_carried)},
    "staged": {"dtype": staging["target_dtype"], "original_dtype": str(bf16_source.conv.weight.dtype),
               "annotated": staged_quantized._recipe_quantization_stats["annotated_contractions"],
               "source_conv": len(staged_ids), "normalizations": len(staged_rows),
               "ancestry": len(set(staged_rows[0]["source_ids"]) & staged_ids),
               "broken_source_refused": broken_source_refused},
}))
"""


def test_selected_recipe_normalizes_string_conv_before_pt2e() -> None:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not root:
        pytest.skip("selected TorchAO capture interpreter and source are required")
    completed = subprocess.run(
        [python, "-I", "-B", "-c", _PROBE, str(_SOURCE), root],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    rows = result["rows"]
    assert [row["source_conv"] for row in rows] == [
        ["aten.conv2d.default"],
        ["aten.conv2d.padding"],
        ["aten.conv2d.padding"],
    ]
    assert rows[2]["refused"] == []
    assert [row["annotated"] for row in rows] == [2, 2, 2], rows
    assert rows[0]["normalizations"] == []
    assert [row["padding"] for row in rows[1]["normalizations"]] == ["valid"]
    assert [row["padding"] for row in rows[2]["normalizations"]] == ["same"]
    assert rows[2]["normalizations"][0]["explicit_input_padding_lrtb"] == [1, 1, 1, 2]
    assert result["bn"] == {
        "fold_api": "selected_model2mlir",
        "original_bn": 1,
        "carried_bn": 1,
    }
    assert result["staged"] == {
        "dtype": "torch.float32",
        "original_dtype": "torch.bfloat16",
        "annotated": 2,
        "source_conv": 1,
        "normalizations": 1,
        "ancestry": 1,
        "broken_source_refused": True,
    }
