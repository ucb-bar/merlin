"""A late projection skip keeps one owner and exact selected integer math."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.tests.targetgen.test_integer_recipe_extensions import POLICIES, _recipe

SCRIPT = r"""
import copy,json,sys
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import torch,torchao
import _recipe_quantizer as RQ
from merlin.targetgen import quant_recipe as Q
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(31)
recipe=json.loads(sys.argv[3]);policies=json.loads(sys.argv[4])
recipe['weight'].update(granularity='tensor',channels_per_scale=None)
recipe['recipe_sha256']=Q.digest(recipe)
recipe=Q.with_native_output_quantization(recipe,operators=('linear','conv2d'),reader_policy='compute')
for key in ('accumulator_residual','integer_mean'):
    recipe=Q.with_integer_extension(recipe,key=key,policy=policies[key])
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.main=torch.nn.Conv2d(16,16,1)
        self.skip=torch.nn.Conv2d(16,16,1);self.head=torch.nn.Linear(16,4)
    def forward(self,x):
        main=self.main(x)
        skip=self.skip(x)
        return self.head(torch.relu(main+skip).mean((2,3)))
x=torch.randn(1,16,2,2)
q=RQ.apply_recipe(Network().eval(),recipe,example_inputs=(x,),calibration_inputs=[(x,),(x*.5,)])
q.graph.lint()
reference=run_pt2e_integer_reference(q,(x,))
integer,receipt=integerize_pt2e(copy.deepcopy(q),(x,))
assert torch.equal(reference.output,integer(x))
stats=q._recipe_quantization_stats
assert stats['accumulator_residual']['rewritten']==1,stats
assert stats['integer_mean']['rewritten']==1,stats
assert stats['logical_annotated_contractions']==3,stats
print(json.dumps({'exact':True,'stats':stats}))
"""


def test_actual_two_convolution_branches_have_one_residual_owner():
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("selected TorchAO interpreter and frontend required")
    root = repo_root()
    result = subprocess.run(
        [
            python,
            "-B",
            "-c",
            SCRIPT,
            str(root / "src/merlin/targetgen"),
            str(Path(m2m) / "src"),
            json.dumps(_recipe()),
            json.dumps(POLICIES),
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr[-8000:] + result.stdout
    assert json.loads(result.stdout.strip().splitlines()[-1])["exact"]
