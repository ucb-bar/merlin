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
large=sys.argv[5]=='coalesced'
if large:
    policies['accumulator_residual'].update(component_packing='coalesced',max_weight_elements=65536)
inputs,outputs=(64,256) if large else (16,16)
recipe['weight'].update(granularity='tensor',channels_per_scale=None)
recipe['recipe_sha256']=Q.digest(recipe)
recipe=Q.with_native_output_quantization(recipe,operators=('linear','conv2d'),reader_policy='compute')
for key in ('accumulator_residual','integer_mean'):
    recipe=Q.with_integer_extension(recipe,key=key,policy=policies[key])
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.main=torch.nn.Conv2d(inputs,outputs,1)
        self.skip=torch.nn.Conv2d(inputs,outputs,1);self.head=torch.nn.Linear(outputs,4)
    def forward(self,x):
        main=self.main(x)
        skip=self.skip(x)
        if sys.argv[5]=='inplace':
            main+=skip
            return self.head(torch.relu_(main).mean((2,3)))
        return self.head(torch.relu(main+skip).mean((2,3)))
x=torch.randn(1,inputs,2,2)
q=RQ.apply_recipe(Network().eval(),recipe,example_inputs=(x,),calibration_inputs=[(x,),(x*.5,)])
q.graph.lint()
reference=run_pt2e_integer_reference(q,(x,))
integer,receipt=integerize_pt2e(copy.deepcopy(q),(x,))
assert torch.equal(reference.output,integer(x))
expected=7 if large else 4
assert reference.contraction_count==expected,(reference.contraction_count,receipt)
assert receipt['integer_mm_emitted']==expected,receipt
stats=q._recipe_quantization_stats
assert stats['accumulator_residual']['rewritten']==1,stats
assert stats['integer_mean']['rewritten']==1,stats
assert stats['logical_annotated_contractions']==3,stats
if large:
    site=stats['accumulator_residual']['sites'][0]
    assert len(site['components'])==4 and site['component_columns_limit']==64,site
    assert site['mac_ratio']==2.0,site
print(json.dumps({'exact':True,'stats':stats}))
"""


@pytest.mark.parametrize("mode", ["functional", "inplace", "coalesced"])
def test_actual_two_convolution_branches_have_one_residual_owner(mode):
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
            mode,
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr[-8000:] + result.stdout
    assert json.loads(result.stdout.strip().splitlines()[-1])["exact"]
