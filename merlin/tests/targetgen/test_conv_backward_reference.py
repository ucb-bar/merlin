"""convolution_backward has an exact independent integer reference on the core-aten int8 lane.

WHY THIS EXISTS. model2mlir integerizes a quantized ``convolution_backward`` into two backward GEMMs
(grad_input columns and grad_weight) plus an int32 sum (grad_bias). The core-aten capture wrote its
independent reference only for programs whose contractions were selected before the capture or that
had none, and Merlin's declared-graph reference covered mm/bmm/linear/conv2d only, so the lane failed
with FileNotFoundError on spec-int8-reference.json. ``run_pt2e_reference`` now evaluates
convolution_backward over the int8 codes: one site per backward GEMM (its accumulator, ascending K),
each integer result checked against torch's exact float64 gradient of the codes, then read out with
one f32 multiply per scale. A GEMM whose accumulator reaches its readout through more integer work
states ``readout=None``; the lane then observes only that accumulator and compares the final outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

_SCRIPT = r"""
import json
import torch
from torch.ao.quantization.fx import _decomposed  # noqa: F401 -- registers quantized_decomposed
from merlin.targetgen.core_aten_w8a8_reference import run_pt2e_reference

q = torch.ops.quantized_decomposed.quantize_per_tensor.default
dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
report = {}
for name, stride, padding, dilation, groups, mask in (
    ("canonical", [1, 1], [0, 0], [1, 1], 1, [True, True, True]),
    ("strided_padded", [2, 1], [1, 2], [1, 2], 1, [True, True, False]),
    ("grouped", [1, 1], [1, 1], [1, 1], 2, [True, True, True]),
):
    torch.manual_seed(0)
    cin, cout, k = 2 * groups, 2 * groups, 3
    x = torch.randn(1, cin, 6, 7)
    w = torch.randn(cout, cin // groups, k, k)
    out = torch.nn.functional.conv2d(x, w, None, stride, padding, dilation, groups)
    g = torch.randn_like(out)
    scales = [float(t.abs().max()) / 127 for t in (g, x, w)]

    graph = torch.fx.Graph()
    values = []
    for tensor_name, scale in zip(("g", "x", "w"), scales):
        placeholder = graph.placeholder(tensor_name)
        codes = graph.call_function(q, (placeholder, scale, 0, -128, 127, torch.int8))
        values.append(graph.call_function(dq, (codes, scale, 0, -128, 127, torch.int8)))
    backward = graph.call_function(
        torch.ops.aten.convolution_backward.default,
        (*values, [cout], stride, padding, dilation, False, [0, 0], groups, mask),
    )
    graph.output(backward)
    model = torch.fx.GraphModule({}, graph)
    reference = run_pt2e_reference(model, (g, x, w))

    from merlin.targetgen.exact_arithmetic import quantize, exact_array, rational, round_binary
    import numpy as np
    codes = [torch.tensor(quantize(t,s),dtype=torch.float64) for t,s in zip((g,x,w),scales)]
    f32 = [torch.tensor(s, dtype=torch.float32) for s in scales]
    exact = [
        torch.nn.grad.conv2d_input(tuple(x.shape), codes[2], codes[0], stride, padding, dilation, groups),
        torch.nn.grad.conv2d_weight(codes[1], tuple(w.shape), codes[0], stride, padding, dilation, groups),
        codes[0].sum((0, 2, 3)),
    ]
    expected = [torch.tensor(np.vectorize(round_binary,otypes=[float])(exact_array(value)*factor),dtype=torch.float32)
                for value,factor in zip(exact,[rational(scales[0])*rational(scales[2]),
                                              rational(scales[0])*rational(scales[1]),rational(scales[0])])]
    report[name] = {
        "sites": [site["node"].split(".")[-1] for site in reference.sites],
        "readouts": [site["readout"] for site in reference.sites],
        "counts": [reference.conv2d_count, reference.contraction_count],
        "exact": [
            None if actual is None else bool(torch.equal(actual, wanted))
            for actual, wanted in zip(reference.output, expected)
        ],
        "accumulator_shapes": [list(torch.tensor(site["accumulator"]).shape) for site in reference.sites],
    }
    try:
        from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    except ImportError:
        continue
    frontend = run_pt2e_integer_reference(model, (g, x, w))
    report[name]["frontend_equal"] = [
        None if a is None else bool(torch.equal(a, b)) for a, b in zip(reference.output, frontend.output)
    ]
print(json.dumps(report))
"""


def test_convolution_backward_reads_out_its_exact_integer_gradients():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    root = repo_root()
    paths = [root / "src", *sorted((root / "packages").glob("*/src"))]
    if os.environ.get("MERLIN_M2M_DIR"):
        paths.append(Path(os.environ["MERLIN_M2M_DIR"]))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(path) for path in paths)}
    result = subprocess.run([python, "-c", _SCRIPT], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])

    canonical = report["canonical"]
    assert canonical["sites"] == ["grad_input0", "grad_weight0"]
    assert canonical["readouts"] == [None, None]
    assert canonical["counts"] == [1, 1]  # one source contraction, as the integerizer counts it
    assert canonical["exact"] == [True, True, True]
    # grad_input columns are W^T[Cg*K, F] x dY[F, N*O]; grad_weight is dY[F, N*O] x im2col(X)[N*O, K*Cg].
    assert canonical["accumulator_shapes"] == [[18, 20], [2, 18]]

    assert report["strided_padded"]["exact"] == [True, True, None]
    assert report["grouped"]["sites"] == ["grad_input0", "grad_input1", "grad_weight0", "grad_weight1"]
    assert report["grouped"]["exact"] == [True, True, True]
    for row in report.values():
        if "frontend_equal" in row:
            # Frontend float readout is diagnostic; the rational gradients above are the oracle.
            assert len(row["frontend_equal"]) == 3


def test_a_size_one_stride_difference_keeps_the_caller_spelling(tmp_path):
    from xdsl.parser import Parser

    from merlin.frontends.linalg_mlir import make_context
    from merlin.llvmlower.result_layout import ATTRIBUTE, module_attribute
    from merlin.targetgen.core_aten_numeric import restore_caller_layouts

    dense = dict(dtype="float32", shape=[2, 1, 3], stride=[3, 3, 1], storage_offset=0, requires_grad=False)
    owned = dict(dense, stride=[3, 1, 1], role="user_output", alias_inputs=[])
    moved = dict(dense, stride=[1, 3, 2], role="user_output", alias_inputs=[])
    source = (
        "builtin.module"
        + module_attribute(dict(inputs=[dense], results=[owned, moved, dense]))
        + """ {
      func.func @forward(%a: tensor<2x1x3xf32>) -> (tensor<2x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>) {
        func.return %a, %a, %a : tensor<2x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>
      }
    }"""
    )
    module = Parser(make_context(), source).parse_module()
    caller = {**dense, "kind": "tensor"}
    original = dict(case=dict(arguments=dict(args=[caller], kwargs={}), expected=[caller, caller]))
    fresh = dict(output_indices=[0, 1], semantic_boundary=dict(post_indices=[2]))
    path = tmp_path / "semantic_io.json"
    path.write_text(json.dumps(dict(outputs=[owned, moved, {**dense, "role": "user_input"}])))
    restore_caller_layouts(module, original, fresh, path)
    layout = json.loads(module.attributes[ATTRIBUTE].data)
    boundary = json.loads(path.read_text())
    # Only the size-1 dimension's stride differed: the same storage, in the caller's spelling.
    assert layout["results"][0]["stride"] == [3, 3, 1] and boundary["outputs"][0]["stride"] == [3, 3, 1]
    # A different owned layout is restored too. result_layout enforces these
    # strides with an allocation and copy, rather than relabelling existing storage.
    assert layout["results"][1]["stride"] == [3, 3, 1] and boundary["outputs"][1]["stride"] == [3, 3, 1]
