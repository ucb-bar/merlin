"""Only a statically indexed stored parameter keeps the recipe's weight policy."""

import os
import subprocess

import pytest


@pytest.mark.parametrize("source", ["parameter", "activation", "runtime_index"])
def test_selected_operand_uses_its_actual_weight_or_activation_policy(source):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("capture interpreter unavailable")
    script = r"""
import sys, torch
from pathlib import Path
from merlin.targetgen import _recipe_quantizer as quant
sys.path.append(str(Path(quant.__file__).parent))
source=sys.argv[1]
root=torch.nn.Module()
root.register_parameter('weights',torch.nn.Parameter(torch.ones(3,4,5)))
graph=torch.fx.Graph()
a=graph.placeholder('a');a.meta['val']=torch.ones(2,4)
if source=='activation':
    weights=graph.placeholder('b')
else:
    weights=graph.get_attr('weights')
weights.meta['val']=torch.ones(3,4,5)
index=graph.placeholder('index') if source=='runtime_index' else 1
selected=graph.call_function(torch.ops.aten.select.int,(weights,0,index))
selected.meta['val']=torch.ones(4,5)
mm=graph.call_function(torch.ops.aten.mm.default,(a,selected))
mm.meta.update(val=torch.ones(2,5),nn_module_stack={'root':('',torch.nn.Module)})
graph.output(mm)
module=torch.fx.GraphModule(root,graph)
recipe=dict(schema='quant_recipe_v1',status='derived',families=['contraction'],
 weight=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-127,quant_max=127),
 activation=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-128,quant_max=127,mode='static'))
quantizer=quant.build_quantizer(recipe,layer_plan={'layers':[]})
quantizer.annotate(module)
annotation=mm.meta['quantization_annotation']
assert annotation.input_qspec_map[a].quant_min==-128
assert annotation.input_qspec_map[selected].quant_min==(-127 if source=='parameter' else -128)
"""
    result = subprocess.run([python, "-c", script, source], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
