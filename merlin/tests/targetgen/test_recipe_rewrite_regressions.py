"""Exercise functional normalization in the selected Torch interpreter."""

import os
import subprocess

_COMMON = r"""
import torch
import sys
from merlin.targetgen import _recipe_quantizer
sys.path.insert(0, str(__import__("pathlib").Path(_recipe_quantizer.__file__).parent))
from merlin.targetgen._recipe_quantizer import build_quantizer
recipe = {
    "schema": "quant_recipe_v1", "status": "derived",
    "families": ["contraction", "operand_sum"],
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
               "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}
quantizer = build_quantizer(recipe, layer_plan={"layers": []})
"""


def run_torch(script):
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", _COMMON + script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_conv1d_normalization_refuses_with_reason():
    run_torch(r"""
class Model(torch.nn.Module):
    def forward(self, x, w):
        return torch.ops.aten.convolution.default(x, w, None, [1], [0], [1], False, [0], 1)
model = torch.export.export(Model(), (torch.ones(1, 2, 7), torch.ones(3, 2, 3))).module()
quantizer.transform_for_annotation(model)
quantizer.annotate(model)
node = next(n for n in model.graph.nodes if n.target == torch.ops.aten.convolution.default)
assert "rank-4" in node.meta["quantization_refusal"]
assert "quantization_annotation" not in node.meta
assert quantizer.annotated == 0
assert len(quantizer.refused_rewrites) == 1
assert not any(n.target == torch.ops.aten.conv2d.default for n in model.graph.nodes)
""")


def test_addmm_retains_its_boundary_and_is_not_an_operand_sum():
    run_torch(r"""
class Model(torch.nn.Module):
    def forward(self, bias, a, b, c):
        return torch.addmm(bias, a, b, alpha=2, beta=3) @ c
inputs = (torch.ones(4), torch.ones(2, 3), torch.ones(3, 4), torch.ones(4, 5))
model = torch.export.export(Model(), inputs).module()
expected = model(*inputs)
quantizer.transform_for_annotation(model)
assert torch.equal(model(*inputs).view(torch.uint8), expected.view(torch.uint8))
quantizer.annotate(model)
addmm = next(n for n in model.graph.nodes if n.target == torch.ops.aten.addmm.default)
assert addmm.kwargs == {"alpha": 2, "beta": 3}
assert tuple(addmm.args[0].meta["val"].shape) == (4,)
assert tuple(addmm.meta["val"].shape) == (2, 4)
assert not any(n.meta.get("merlin_contraction_epilogue") for n in model.graph.nodes)
assert not any(n.target == torch.ops.aten.add.Tensor for n in model.graph.nodes)
assert quantizer.annotated_sums == 0
assert quantizer.annotated == 2
""")
