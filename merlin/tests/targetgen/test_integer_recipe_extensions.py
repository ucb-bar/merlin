"""Explicit numerical extensions compose through the ordinary PT2E quantizer."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import quant_recipe as Q


def _recipe():
    return json.loads((repo_root() / "examples/gemmini/protocol/recipes/resnet50.json").read_text())


POLICIES = {
    "accumulator_residual": {"schema": "integer_accumulator_residual_v1",
                             "coefficient_rounding": "align_weight_grid", "max_terms": 1,
                             "max_weight_elements": 4096, "max_mac_ratio": 2.0},
    "integer_mean": {"schema": "integer_mean_contraction_v1", "max_window_elements": 64,
                     "max_weight_elements": 64, "max_macs": 4096},
    "projection_input_reuse": {"schema": "shared_input_projection_v1"},
}


@pytest.mark.parametrize("key", POLICIES)
def test_closed_extension_is_content_bound_and_does_not_mutate_defaults(key):
    original = _recipe()
    snapshot = copy.deepcopy(original)
    selected = Q.with_integer_extension(original, key=key, policy=POLICIES[key])
    assert original == snapshot and key not in original
    assert selected["recipe_sha256"] == Q.digest(selected) != Q.digest(original)
    assert Q.with_integer_extension(selected, key=key, policy=POLICIES[key]) == selected
    for recipe in (original | {"epilogue": None}, original | {"accumulator_dtype": "fp32"}):
        with pytest.raises(ValueError):
            Q.with_integer_extension(recipe, key=key, policy=POLICIES[key])


def test_residual_policies_refuse_conflicting_ownership_in_both_selection_orders():
    original = _recipe()
    accumulator = Q.with_integer_extension(
        original, key="accumulator_residual", policy=POLICIES["accumulator_residual"]
    )
    with pytest.raises(ValueError, match="cannot own the same graph"):
        Q.with_shared_grid_operand_sums(accumulator)
    shared = Q.with_shared_grid_operand_sums(original)
    with pytest.raises(ValueError, match="cannot own the same graph"):
        Q.with_integer_extension(shared, key="accumulator_residual", policy=POLICIES["accumulator_residual"])


_SCRIPT = r"""
import copy,json,sys
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import torch,torchao
import _recipe_quantizer as RQ
from merlin.targetgen import quant_recipe as Q
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.capture import pt2e_integer_reference as IR
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(123)
recipe=json.loads(sys.argv[3]); policies=json.loads(sys.argv[4])
recipe['weight'].update(granularity='tensor',channels_per_scale=None)
recipe['recipe_sha256']=Q.digest(recipe)
recipe=Q.with_native_output_quantization(recipe,operators=('linear','conv2d'),reader_policy='compute')
for key in ('accumulator_residual','integer_mean'):
    recipe=Q.with_integer_extension(recipe,key=key,policy=policies[key])
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.stem=torch.nn.Conv2d(16,16,1);self.conv=torch.nn.Conv2d(16,16,1)
        self.head=torch.nn.Linear(16,4)
        self.pool=torch.nn.AdaptiveAvgPool2d((1,1))
        self.next=torch.nn.Conv2d(16,16,1)
    def forward(self,x):
        x=torch.relu(self.stem(x))
        if sys.argv[5]=='chain':
            y=self.conv(x);y.add_(x);y.relu_()
            z=self.next(y);z.add_(y);z.relu_();y=z
        else:
            y=torch.relu(self.conv(x)+x)
        y=self.pool(y).flatten(1) if sys.argv[5]=='adaptive' else y.mean((3,2) if sys.argv[5]=='reverse' else (2,3))
        return self.head(y)
x=torch.randn(1,16,2,2)
q=RQ.apply_recipe(Network().eval(),recipe,example_inputs=(x,),calibration_inputs=[(x,),(x*.5,)])
reference=IR.run_pt2e_integer_reference(q,(x,))
integer,receipt=integerize_pt2e(copy.deepcopy(q),(x,))
print(json.dumps({'stats':q._recipe_quantization_stats,'exact':bool(torch.equal(reference.output,integer(x))),
                  'output_shape':list(integer(x).shape),
                  'actual_mm':sum(n.target==torch.ops.aten._int_mm.default for n in integer.graph.nodes)}))
"""


@pytest.mark.parametrize("pool", ["mean", "reverse", "adaptive", "chain"])
def test_residual_and_pool_through_actual_quantizer_and_independent_reference(pool):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m:
        pytest.skip("selected TorchAO and model2MLIR interpreters are required")
    root = repo_root()
    result = subprocess.run(
        [python, "-B", "-c", _SCRIPT, str(root / "src/merlin/targetgen"), str(Path(m2m) / "src"),
         json.dumps(_recipe()), json.dumps(POLICIES), pool],
        env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr[-8000:]
    row = json.loads(result.stdout.strip().splitlines()[-1])
    assert row["exact"] and row["output_shape"] == [1, 4]
    stats = row["stats"]
    assert stats["logical_annotated_contractions"] == (4 if pool == "chain" else 3)
    assert stats["annotated_contractions"] == row["actual_mm"] == (5 if pool == "chain" else 4)
    assert stats["accumulator_residual"]["rewritten"] == (2 if pool == "chain" else 1)
    assert stats["integer_mean"]["rewritten"] == 1
