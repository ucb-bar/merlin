"""Direct diagnostic recipe captures retain the original floating trajectory."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen.quant_recipe import digest


def test_direct_recipe_captures_independent_nonpaper_trajectory(tmp_path):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text('''import torch

def get_model_and_inputs():
    model = torch.nn.Linear(4, 3).eval()
    with torch.no_grad():
        model.weight.copy_(torch.arange(12).reshape(3,4) * 0.0713 - 0.42)
        model.bias.copy_(torch.tensor([0.013, -0.092, 0.177]))
    return model, (torch.tensor([[0.07, -0.31, 0.83, 0.41]]),)

def get_session_spec(model, inputs):
    return dict(kind="image_stream", paper_ready=False, stages=["compute"], states=[],
                streams=[dict(name="x", input_index=0, key="x",
                              values=torch.stack([inputs[0], inputs[0]*0.713]))],
                quality=dict(output_index=0, key="output0"))
''')
    tensor = dict(dtype="int8", granularity="tensor", symmetric=True, block=None, mode="static")
    recipe = dict(schema="quant_recipe_v1", target="t", unit="matrix", status="derived",
                  weight={**tensor, "quant_min": -127, "quant_max": 127},
                  activation={**tensor, "quant_min": -128, "quant_max": 127},
                  families=["contraction"], bias_domain="accumulator",
                  software_numerical_engine="integer_reference")
    recipe["recipe_sha256"] = digest(recipe)
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    result = subprocess.run(
        [python, str(Path(worker.__file__)), "--m2m-dir", m2m, "--loader", str(loader),
         "--dtype", "int8", "--recipe", str(tmp_path / "recipe.json"),
         "--materialize-bundle", "--out", str(tmp_path / "capture")],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-4000:]
    import yaml
    out = tmp_path / "capture"
    contract = yaml.safe_load((out / "session_contract.yaml").read_text())
    assert contract["paper_ready"] is False
    assert contract["quality"]["reference"] == "eager_fp32"
    assert contract["quality"]["observation"]["origin"] == "provided_quality_reference"
    with np.load(out / "session_quality_fp32.npz") as ref, np.load(out / "session_goldens.npz") as gold:
        actual, quantized = ref['output0'], gold['output0']
    weight = np.arange(12, dtype=np.float32).reshape(3,4) * np.float32(0.0713) - np.float32(0.42)
    x = np.array([[0.07, -0.31, 0.83, 0.41]], dtype=np.float32)
    expected = np.stack([v @ weight.T + np.array([0.013,-0.092,0.177], dtype=np.float32)
                         for v in (x, x*np.float32(0.713))])
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert not np.array_equal(actual, quantized)
