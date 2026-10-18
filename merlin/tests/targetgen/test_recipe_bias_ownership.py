"""Bias snapping owns its transformed storage and preserves the original model."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen import quant_recipe as QR

_SCRIPT = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
import torch
import _recipe_quantizer as RQ
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_integerize import integerize_pt2e

torch.manual_seed(3)
torch.set_num_threads(1)
source = torch.nn.Linear(7, 13).eval()
original = {name: (value, value._version, value.detach().clone()) for name, value in source.named_parameters()}
sample = (torch.randn(3, 7),)
calibration = [(torch.randn(3, 7),) for _ in range(4)]
quantized = RQ.apply_recipe(source, json.loads(sys.argv[2]), example_inputs=sample, calibration_inputs=calibration)
assert quantized._recipe_quantization_stats['bias_on_accumulator_grid']['channels'] == 13
for name, value in source.named_parameters():
    tensor, version, before = original[name]
    assert value is tensor and value._version == version, name
    assert torch.equal(before, value) and before.numpy().tobytes() == value.detach().numpy().tobytes(), name
with torch.no_grad():
    reference = run_pt2e_integer_reference(quantized, sample)
    integer, _ = integerize_pt2e(quantized, sample)
    assert torch.equal(reference.output, integer(*sample))
print('OK')
"""


def test_bias_grid_owns_storage_and_still_matches_the_integer_reference(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("an explicit capture interpreter with TorchAO is required")
    recipe = QR.QuantRecipe(
        target="t", unit="u", weight=QR.TensorSpec("int8", "channel", True, -127, 127),
        activation=QR.TensorSpec("int8", "tensor", True, -128, 127), families=(QR.CONTRACTION,),
    ).to_dict() | {"bias_domain": "accumulator", "accumulator_dtype": "int32"}
    recipe["recipe_sha256"] = QR.digest(recipe)
    result = subprocess.run(
        [python, "-c", _SCRIPT, str(Path(worker.__file__).parent), json.dumps(recipe)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]
    assert result.stdout.strip().endswith("OK")
