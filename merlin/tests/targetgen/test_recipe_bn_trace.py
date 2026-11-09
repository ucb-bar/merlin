"""A recipe PT2E fold carries the exact captured BatchNorm source identity."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import module_source_path
from merlin.targetgen._recipe_quantizer import _no_batch_norm_fold_needed


def test_missing_fold_api_requires_complete_plain_aten_graph_without_batch_norm() -> None:
    from types import SimpleNamespace

    original = {
        "schema": "m2m.frontend_graph.v1",
        "status": "complete",
        "nodes": [{"op": "call_function", "classification": "aten", "target": "aten.matmul.default"}],
    }
    exported = SimpleNamespace(
        graph=SimpleNamespace(nodes=[SimpleNamespace(op="call_function", target="aten.matmul.default")])
    )
    assert _no_batch_norm_fold_needed(original, exported)
    assert not _no_batch_norm_fold_needed({**original, "status": "incomplete"}, exported)
    assert not _no_batch_norm_fold_needed(
        {**original, "nodes": [{"op": "call_function", "classification": "aten", "target": "aten.batch_norm.default"}]},
        exported,
    )
    assert not _no_batch_norm_fold_needed(
        {**original, "nodes": [{"op": "call_function", "classification": "custom", "target": "opaque"}]},
        exported,
    )


def test_static_recipe_conv_bn_fold_preserves_frontend_trace() -> None:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    source = module_source_path("merlin.targetgen._recipe_specs").with_name("_recipe_quantizer.py")
    script = r"""
import importlib.util
import json
import sys

if sys.argv[2]:
    sys.path.insert(0, sys.argv[2])
import torch
import m2m
from m2m.capture.torchao_pipeline import QuantizationConfig

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

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x)).relu()

model, inputs = Model().eval(), (torch.randn(1, 3, 8, 8),)
original = m2m.capture_frontend_snapshot(model, inputs)
quantized = quantizer.apply_recipe(model, recipe, example_inputs=inputs,
                                   original_frontend_snapshot=original)
result = m2m.convert(quantized, inputs, backend="fx_importer", capture_trace=True,
                     quantization=QuantizationConfig(scheme="int8_static_act_int8_weight"),
                     quantization_preapplied=True, original_frontend_snapshot=original)
trace = result.capture_trace
bn_ids = {node["id"] for node in original["nodes"] if node["target"] == "aten.batch_norm.default"}
carried = {origin for node in trace["graphs"]["quantized"]["nodes"]
           if node["target"] == "aten.conv2d.default"
           for origin in node["origin_node_ids"] if origin in bn_ids}
print(json.dumps({"ok": result.ok, "status": trace["status"], "blockers": trace["blockers"],
                  "bn_count": len(bn_ids), "carried": len(carried),
                  "unresolved": trace["transformations"][0]["unresolved_source_ids"]}))
"""
    completed = subprocess.run(
        [python, "-c", script, str(source), os.environ.get("MERLIN_M2M_DIR", "")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {"ok": True, "status": "complete", "blockers": [], "bn_count": 1, "carried": 1, "unresolved": []}
