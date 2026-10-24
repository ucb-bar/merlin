"""Prospective integer MAC admission cannot inherit the source floating spelling."""

import os
import subprocess

import pytest


@pytest.mark.parametrize("mode", ["no_mac", "allowed", "restricted_rank", "unreviewed"])
def test_bmm_quantization_uses_declared_prospective_mac(tmp_path, mode):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("capture interpreter unavailable")
    script = r"""
import sys,torch
from merlin.common.paths import repo_root
from merlin.targetgen import _recipe_quantizer as quant
from merlin.targetgen.software_spec import admit_operation,load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec
from merlin.targetgen.core_aten_w8a8_reference import run_pt2e_reference
from merlin.capture.integerization import integerize_static_pt2e
spec,_=resolve_spec(load_software_spec(repo_root()/'examples/gemmini/target/software-spec.yaml'),{})
source=dict(family='contraction',ordered_operand_dtypes=['f32','f32','f32'],
            ordered_result_dtypes=['f32'],accum_dtype='f32',rank=3)
assert admit_operation(spec,'aten.bmm.default',source,'host')['status']=='admitted'
assert admit_operation(spec,'aten.bmm.default',source,'accelerator')['status']=='unsupported'
mode=sys.argv[1];allowed=mode=='allowed'
if mode=='no_mac':spec['operations']=[row for row in spec['operations'] if row['id']!='contraction']
if mode in {'restricted_rank','unreviewed'}:
 spec['operations'].append(dict(id='restricted_bmm',ops=['aten.bmm.default'],placement='accelerator',
  status='unreviewed' if mode=='unreviewed' else 'reviewed',
  signature=dict(ordered_operand_dtypes=['i8','i8','i32'],ordered_result_dtypes=['i32'],
                 ranks=[2] if mode=='restricted_rank' else [3])))
recipe=dict(schema='quant_recipe_v1',status='derived',families=['contraction'],
 weight=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-127,quant_max=127),
 activation=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-128,quant_max=127,mode='static'),
 accumulator_dtype='i32',software_admission=spec)
class Bmm(torch.nn.Module):
 def forward(self,a,b):return torch.bmm(a,b)
inputs=(torch.linspace(-1,1,24).reshape(2,3,4),torch.linspace(-.7,.9,16).reshape(2,4,2))
try:converted=quant.apply_recipe(Bmm().eval(),recipe,example_inputs=inputs)
except quant.NoEligibleRecipeOperations:
 assert not allowed
else:
 assert allowed
 stats=converted._recipe_quantization_stats
 assert stats['annotated_contractions']==1
 rows=[row for row in stats['software_admission'] if row['source_operator']=='aten.bmm.default']
 assert all(row['prospective_signature']['numerical_family']=='F3' for row in rows)
 assert rows and all(row['prospective_operator']=='aten.bmm.default' and row['declaration']=='contraction'
                     and row['status']!='unsupported' for row in rows)
 assert all(row['ignored_disjoint_host_precision_declarations']==['host_f32_bmm'] for row in rows)
 reference=run_pt2e_reference(converted,inputs)
 integer,receipt=integerize_static_pt2e(converted,inputs,numerics="integer_readout")
 if 'integer_bmm_emitted' in receipt:
  assert receipt['integer_bmm_emitted']==1
 else:
  # Older v1 frontends lower each batch to a separate rank-2 integer matmul.
  matrices=[node for node in integer.graph.nodes if node.target==torch.ops.aten._int_mm.default]
  assert receipt['integer_mm_emitted']==len(matrices)==inputs[0].shape[0]
 assert torch.equal(integer(*inputs).view(torch.int32),reference.output.view(torch.int32))
"""
    result = subprocess.run([python, "-c", script, mode], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
