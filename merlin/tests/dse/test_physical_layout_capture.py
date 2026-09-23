"""Provenance layout capture is multi-model and omission-explicit."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("xdsl")

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, math, memref, scf, tensor
from xdsl.parser import Parser

from merlin.perf.physical_layout import PhysicalEncoding
from merlin.perf.physical_layout_capture import RegionLayoutPolicy, extract_region_layout_graph


def _parse(source: str):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, linalg.Linalg, tensor.Tensor, arith.Arith,
                    math.Math, scf.Scf, memref.MemRef):
        context.load_dialect(dialect)
    module = Parser(context, source).parse_module()
    module.verify()
    return module


def _encodings(_name: str, shape: tuple[int, ...], dtype: str):
    bits = 16 if dtype == "bf16" else int(dtype[1:])
    logical_bytes = 1
    for dim in shape:
        logical_bytes *= dim
    logical_bytes *= bits // 8
    return (
        PhysicalEncoding("channels_first", "channels_first", "external", logical_bytes),
        PhysicalEncoding("channels_last", "channels_last", "external", logical_bytes),
    )


def _policy() -> RegionLayoutPolicy:
    return RegionLayoutPolicy(
        "channels_first",
        "channels_last",
        frozenset({"conv2d", "convolution_im2col_matmul"}),
        frozenset({"relu", "quantize"}),
        frozenset({"add"}),
        frozenset({"view"}),
        frozenset({"pool"}),
        frozenset({"weight_transform"}),
        _encodings,
    )


def test_conv2d_semantic_spelling_is_classified_when_policy_declares_it() -> None:
    module = _parse('''builtin.module {
      func.func @forward(%x: tensor<1x4x2x2xi8>, %w: tensor<8x4x1x1xi8>)
          -> tensor<1x8x2x2xi8> {
        %y = "fixture.conv"(%x, %w) {prov.region_id = "conv_0", prov.op = "conv2d"}
          : (tensor<1x4x2x2xi8>, tensor<8x4x1x1xi8>) -> tensor<1x8x2x2xi8>
        func.return %y : tensor<1x8x2x2xi8>
      }
    }''')
    capture = extract_region_layout_graph(module, _policy())
    assert capture.applicable
    assert capture.graph is not None
    assert dict(capture.census)["semantic_region_counts"] == {"conv2d": 1}
    assert dict(capture.census)["included_op_counts"]["accelerator"] == 1
    conv = next(op for op in capture.graph.ops if op.kind == "accelerator")
    assert len(conv.inputs) == 1
    assert len(conv.outputs) == 1
    assert set(conv.ports) == {"arg0", "v0_0"}
    serialized = json.loads(json.dumps(capture.to_dict(), sort_keys=True))
    assert serialized["graph"]["schema"] == "target_neutral_physical_layout_graph_v1"


def test_absent_activation_rank_is_not_applicable_instead_of_an_exception() -> None:
    module = _parse('''builtin.module {
      func.func @forward(%x: tensor<2x4xi8>) -> tensor<2x4xi8> {
        %y = "fixture.matmul"(%x) {prov.region_id = "linear_0", prov.op = "matmul"}
          : (tensor<2x4xi8>) -> tensor<2x4xi8>
        func.return %y : tensor<2x4xi8>
      }
    }''')
    capture = extract_region_layout_graph(module, _policy())
    assert not capture.applicable
    assert capture.status == "not_applicable"
    assert capture.graph is None
    assert [item.code for item in capture.refusals] == ["no_activation_of_required_rank"]


def test_rank_equal_weights_do_not_define_batch_and_every_omission_has_a_reason() -> None:
    arguments = ["%x: tensor<1x4x2x2xi8>"]
    arguments.extend(f"%w{i}: tensor<{32 + i}x4x1x1xi8>" for i in range(11))
    operations = []
    previous = "%x"
    for index in range(11):
        current = f"%v{index}"
        operations.append(
            f'''    {current} = "fixture.conv"({previous}, %w{index}) '''
            f'''{{prov.region_id = "conv_{index}", '''
            f'''prov.op = "convolution_im2col_matmul"}} '''
            f''': (tensor<1x4x2x2xi8>, tensor<{32 + index}x4x1x1xi8>) '''
            f'''-> tensor<1x4x2x2xi8>''')
        previous = current
    operations.append(
        '''    %bad = "fixture.conv"(%v10) {prov.region_id = "conv_unproved", '''
        '''prov.op = "convolution_im2col_matmul"} '''
        ''': (tensor<1x4x2x2xi8>) -> tensor<2x4xi8>''')
    source = ("builtin.module {\n  func.func @forward(" + ", ".join(arguments)
              + ") -> tensor<1x4x2x2xi8> {\n" + "\n".join(operations)
              + "\n    func.return %v10 : tensor<1x4x2x2xi8>\n  }\n}")
    capture = extract_region_layout_graph(_parse(source), _policy())
    assert capture.graph is not None
    census = dict(capture.census)
    assert census["semantic_region_counts"]["convolution_im2col_matmul"] == 12
    assert census["included_op_counts"]["accelerator"] == 11
    assert census["omission_reason_counts"] == {"accelerator_activation_output_unproven": 1}
    assert [(item.region, item.reason) for item in capture.omissions] == [
        ("conv_unproved", "accelerator_activation_output_unproven")]
    assert capture.status == "captured_with_refusals"
    assert len(capture.refusals) == 1 and capture.refusals[0].blocks_plan
    convs = [op for op in capture.graph.ops if op.kind == "accelerator"]
    assert len(convs) == 11
    assert all(len(op.inputs) == 1 for op in convs)
