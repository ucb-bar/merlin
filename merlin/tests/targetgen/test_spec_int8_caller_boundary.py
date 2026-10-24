"""Caller layout and mutation observations survive static int8 recapture."""

import os
import subprocess

import pytest

from merlin.targetgen.core_aten_capture import loader_source
from merlin.targetgen.core_aten_numeric import map_reference_boundary


@pytest.mark.parametrize("stride,offset", [([8, 2], 3), ([1, 2], 0)])
def test_recapture_loader_materializes_caller_storage(stride, offset):
    tensor = dict(
        kind="tensor",
        dtype="float32",
        shape=[2, 3],
        stride=stride,
        storage_offset=offset,
        requires_grad=False,
        values=[[1, 2, 3], [4, 5, 6]],
    )
    case = dict(overload="aten.mm.default", arguments=dict(args=[tensor, tensor], kwargs={}), expected=tensor)
    script = (
        loader_source(case)
        + f"""
model, inputs = get_model_and_inputs()
assert list(inputs[0].stride()) == {stride!r}
assert inputs[0].storage_offset() == {offset}
assert inputs[0].tolist() == [[1, 2, 3], [4, 5, 6]]
"""
    )
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_maps_mutated_inputs_in_exported_role_order():
    fresh = dict(output_indices=[1], semantic_boundary=dict(post_indices=[0, 2]))
    reference = dict(outputs=[[[7.0]]], post_inputs=[[[8.0]], [[9.0]]])
    assert map_reference_boundary(fresh, reference, 3) == [[[8.0]], [[7.0]], [[9.0]]]
    reference["post_inputs"].pop()
    with pytest.raises(ValueError):
        map_reference_boundary(fresh, reference, 3)


def test_independent_reference_observes_post_call_mutation():
    script = r"""
import torch
import torchao.quantization.pt2e
from merlin.targetgen.core_aten_w8a8_reference import selected_pt2e_reference

graph = torch.fx.Graph()
a, b = graph.placeholder("a"), graph.placeholder("b")
dequant = torch.ops.quantized_decomposed.dequantize_per_tensor.default
x = graph.call_function(dequant, (a, 1., 0, -128, 127, torch.int8))
y = graph.call_function(dequant, (b, 1., 0, -128, 127, torch.int8))
out = graph.call_function(torch.ops.aten.mm.default, (x, y))
graph.call_function(torch.ops.aten.add_.Tensor, (a, 1))
graph.output(out)
model = torch.fx.GraphModule({}, graph)
inputs = (torch.tensor([[3]], dtype=torch.int8), torch.tensor([[5]], dtype=torch.int8))
reference, _ = selected_pt2e_reference(model, inputs, declared_readout=True)
assert reference.output.item() == 15
assert reference.post_inputs[0].item() == 4
assert inputs[0].item() == 3
"""
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("stride,offset", [([8, 2], 3), ([0, 1], 0)])
def test_normalized_post_descriptor_recovers_caller_layout(tmp_path, stride, offset):
    import json

    from xdsl.parser import Parser

    from merlin.frontends.linalg_mlir import make_context
    from merlin.llvmlower.input_layout import refuse_overlapping_input_aliases
    from merlin.llvmlower.result_layout import ATTRIBUTE, module_attribute
    from merlin.targetgen.core_aten_numeric import restore_caller_layouts

    dense = dict(dtype="float32", shape=[2, 3], stride=[3, 1], storage_offset=0, requires_grad=False)
    source = (
        "builtin.module"
        + module_attribute(dict(inputs=[dense], results=[dense]))
        + """ {
      func.func @forward(%a: tensor<2x3xf32>) -> tensor<2x3xf32> {
        func.return %a : tensor<2x3xf32>
      }
    }"""
    )
    module = Parser(make_context(), source).parse_module()
    caller = {**dense, "kind": "tensor", "stride": stride, "storage_offset": offset}
    original = dict(case=dict(arguments=dict(args=[caller], kwargs={})))
    fresh = dict(output_indices=[], semantic_boundary=dict(post_indices=[0]))
    path = tmp_path / "semantic_io.json"
    path.write_text(json.dumps(dict(outputs=[{**dense, "role": "user_output"}])))
    restore_caller_layouts(module, original, fresh, path)
    layout = json.loads(module.attributes[ATTRIBUTE].data)
    assert layout["inputs"][0]["stride"] == stride
    assert layout["results"][0]["stride"] == stride
    assert layout["results"][0]["storage_offset"] == offset
    assert json.loads(path.read_text())["outputs"][0]["stride"] == stride
    refuse_overlapping_input_aliases(layout)
    if stride[0] == 0:
        assert layout["results"][0]["alias_inputs"] == [0]
        fresh["output_indices"] = [0]
        restore_caller_layouts(module, original, fresh, path)
        refuse_overlapping_input_aliases(json.loads(module.attributes[ATTRIBUTE].data))
        fresh["output_indices"] = []
        boundary = json.loads(path.read_text())
        boundary["outputs"][0]["role"] = "user_input_mutation"
        path.write_text(json.dumps(boundary))
        restore_caller_layouts(module, original, fresh, path)
        with pytest.raises(ValueError, match="overlapping logical elements"):
            refuse_overlapping_input_aliases(json.loads(module.attributes[ATTRIBUTE].data))
