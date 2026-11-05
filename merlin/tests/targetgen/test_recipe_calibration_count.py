"""How many calibration samples a static recipe observes is part of the recipe.

WHY THIS EXISTS. A static recipe's activation scales are whatever its observers saw, and the capture
observed at most a hard-coded 100 samples of the loader's stream -- or, for a session stage with no
calibration hook, only the stage's own capture example. Measured on the Phase 0 int8 captures: TinyLlama
and SmolVLA were calibrated on one sample each, and the int8 model disagreed with the float one on
80-94% of TinyLlama's next tokens. A 260-image calibration set was silently cut to its first 100 (four
of its ten classes, as the set is stored class by class).

``framework_capture_policy.calibration_count`` now states the count. It is authored as a framework
parameter of the SW quantization format (``framework.calibration_count``), carried into the recipe, and
so into the recipe digest a capture receipt is keyed on. A loader hook may also decline (return None),
which leaves a stage on its own example exactly as a loader without a hook does.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen import quant_recipe

_QUANTIZER = Path(worker.__file__).with_name("_recipe_quantizer.py")


def _quantizer():
    spec = importlib.util.spec_from_file_location("_recipe_quantizer_under_test", _QUANTIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recipe(**policy):
    tensor = dict(dtype="int8", granularity="tensor", symmetric=True, block=None, mode="static")
    recipe = dict(
        schema="quant_recipe_v1",
        target="t",
        unit="matrix",
        status="derived",
        weight={**tensor, "quant_min": -127, "quant_max": 127},
        activation={**tensor, "quant_min": -128, "quant_max": 127},
        families=["contraction"],
        bias_domain="accumulator",
        software_numerical_engine="integer_reference",
    )
    if policy:
        recipe["framework_capture_policy"] = dict(policy)
    recipe["recipe_sha256"] = quant_recipe.digest(recipe)
    return recipe


class TestTheRecipeStatesItsCalibrationCount:
    def test_absent_keeps_the_adapter_default(self):
        assert _quantizer().calibration_limit(_recipe(), 100) == 100
        assert _quantizer().calibration_limit(_recipe(observer_epsilon=2**-12), 100) == 100

    def test_present_overrides_the_default(self):
        assert _quantizer().calibration_limit(_recipe(calibration_count=260), 100) == 260

    @pytest.mark.parametrize("bad", [0, -3, 2.5, "260", True, None])
    def test_a_non_positive_or_non_integer_count_is_refused(self, bad):
        quantizer = _quantizer()
        with pytest.raises(quantizer.RecipeError, match="calibration_count"):
            quantizer.calibration_limit(_recipe(calibration_count=bad), 100)

    def test_the_worker_reads_far_enough_for_the_recipe_and_the_receipt(self):
        quantizer = _quantizer()
        assert worker._calibration_stream_limit(quantizer, _recipe()) == 100  # noqa: SLF001
        assert worker._calibration_stream_limit(quantizer, _recipe(calibration_count=8)) == 64  # noqa: SLF001
        assert worker._calibration_stream_limit(quantizer, _recipe(calibration_count=512)) == 512  # noqa: SLF001


STREAM_LOADER = """
import os
import torch
from torch import nn

class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2))

    def forward(self, x):
        return self.projection(x)

def get_model_and_inputs():
    return Projection().eval(), (torch.tensor([[1., 2.]]),)

def get_calibration_inputs(model, inputs):
    if os.environ.get("DECLINE") == "1":
        return None
    def samples():
        for i in range(int(os.environ["BUDGET"])):
            yield (torch.tensor([[1. + i, 2.]]),)
        raise RuntimeError("calibration requested beyond the recipe's count")
    return samples()
"""


@pytest.mark.parametrize(
    ("count", "budget", "decline", "observed", "source"),
    [
        (150, 150, "0", 150, "explicit_calibration_stream"),
        (None, 100, "0", 100, "explicit_calibration_stream"),
        (150, 150, "1", 1, "example_inputs"),
    ],
)
def test_the_capture_observes_the_recipes_count_of_the_loaders_stream(
    tmp_path, count, budget, decline, observed, source
):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text(STREAM_LOADER)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(_recipe(**({} if count is None else {"calibration_count": count}))))
    out = tmp_path / "capture"
    result = subprocess.run(
        [
            python,
            worker.__file__,
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe_path),
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "OMP_NUM_THREADS": "1", "BUDGET": str(budget), "DECLINE": decline},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    stats = json.loads((out / "meta.json").read_text())["quantization_stats"]
    assert stats["calibration_samples"] == observed
    assert stats["framework_capture_policy"]["calibration_source"] == source
    assert stats["framework_capture_policy"]["calibration_sample_limit"] == (count or 100)
