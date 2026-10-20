"""Only a source-proved complete contiguous result view can change the logical ABI shape."""

from types import SimpleNamespace

import pytest
from fake_quant_layer import Oracle
from fake_quant_layer import module as linear
from xdsl.dialects import tensor
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, TensorType, f32, i8, i64
from xdsl.ir import Block

from merlin.common import mlir_query as mq
from merlin.llvmlower.group_host_call import _contiguous_result_views, _device_result
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import group_command as GC


def _closed(*, column_vector=False):
    text = linear(weight_dequantize="per_tensor")
    rows, columns = (16, 1) if column_vector else (4, 16)
    if column_vector:
        for old, new in (
            ("tensor<4x8xi8>", "tensor<16x8xi8>"), ("tensor<4x8xf32>", "tensor<16x8xf32>"),
            ("tensor<8x16xi8>", "tensor<8x1xi8>"), ("tensor<8x16xf32>", "tensor<8x1xf32>"),
            ("tensor<4x16xf32>", "tensor<16x1xf32>"), ("tensor<4x16xi8>", "tensor<16x1xi8>"),
            ("tensor<16xf32>", "tensor<1xf32>"),
        ):
            text = text.replace(old, new)
    size, matrix = rows * columns, f"{rows}x{columns}"
    marker = '    %out = "quant_ext.quantize_per_tensor"'
    # The same complete mean->flatten view topology, with non-singleton matrix axes too.
    views = f"""
    %flat0 = tensor.collapse_shape %relu [[0, 1]] : tensor<{matrix}xf32> into tensor<{size}xf32>
    %spatial = tensor.expand_shape %flat0 [[0, 1, 2, 3]] output_shape [1, {rows}, {columns}, 1]
        : tensor<{size}xf32> into tensor<1x{matrix}x1xf32>
    %flat1 = tensor.collapse_shape %spatial [[0, 1, 2, 3]] : tensor<1x{matrix}x1xf32> into tensor<{size}xf32>
    %row = tensor.expand_shape %flat1 [[0, 1]] output_shape [1, {size}] : tensor<{size}xf32> into tensor<1x{size}xf32>
"""
    before, after = text.split(marker)
    after = after.replace("(%relu,", "(%row,").replace(f"tensor<{matrix}x", f"tensor<1x{size}x")
    text = (before + views + marker + after).replace(f") -> tensor<{matrix}xi8>", f") -> tensor<1x{size}xi8>", 1)
    module = mq.parse(text)
    (group,) = CG.form_groups(module, "synthetic", oracle=Oracle())
    assert group.stages[-1] == CG.QUANTIZE
    entry = GC.program(group, weight_args={1, 2, 3, 4}).entry
    return module, group, entry


def _binding(group, entry, **kwargs):
    return _device_result(group, group.members, entry, group.members[-1].results[0].type, **kwargs)


def test_complete_source_views_are_replayed_on_the_committed_element_type():
    module, group, entry = _closed()
    module.verify()
    relayout, result_type, why = _binding(group, entry)
    assert why == "" and result_type == TensorType(i8, [4, 16])
    value = Block(arg_types=[result_type]).args[0]
    operations = relayout(value)
    assert [op.name for op in operations] == [
        "tensor.collapse_shape", "tensor.expand_shape", "tensor.collapse_shape", "tensor.expand_shape",
    ]
    for op in operations:
        op.verify()
        assert op.operands[0] is value
        assert op.results[0].type.element_type == i8
        value = op.results[0]
    assert value.type == group.members[-1].results[0].type


def test_column_vector_mean_flatten_has_the_exact_reported_destination():
    module, group, entry = _closed(column_vector=True)
    module.verify()
    relayout, result_type, why = _binding(group, entry)
    assert why == "" and result_type == TensorType(i8, [16, 1])
    assert relayout(Block(arg_types=[result_type]).args[0])[-1].results[0].type == TensorType(i8, [1, 16])


def test_ordinary_group_rewrite_declares_the_device_shape_and_restores_the_host_shape(tmp_path, monkeypatch):
    import json
    import struct

    import numpy as np
    from logical_device_target import install

    from merlin.llvmlower import device_offload, device_shim

    install(monkeypatch)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: device_shim.KernelAbi("kernel", version=2))
    arrays = {"w": np.ones((8, 1), dtype=np.int8), "b": np.zeros(1, dtype=np.float32)}
    header, payload = {}, b""
    for name, value in arrays.items():
        raw = value.tobytes()
        header[name] = {
            "dtype": "I8" if value.dtype == np.int8 else "F32", "shape": list(value.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    encoded = json.dumps(header).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (tmp_path / "weights.manifest.json").write_text(json.dumps({
        "1": {"kind": "weight", "weight": "w"}, "4": {"kind": "weight", "weight": "b"},
    }))
    module, _, _ = _closed(column_vector=True)
    rewrite = device_offload.rewrite_groups_to_device(
        module, "synthetic", select=lambda _: True, oracle=Oracle(), weight_args={1, 4},
        capture=tmp_path, require_complete_groups=True,
    )
    assert rewrite.moved == 1 and not rewrite.skipped
    module.verify()
    calls = [op for op in module.walk() if op.name == "func.call"]
    assert len(calls) == 1 and calls[0].results[0].type == TensorType(i8, [16, 1])
    function = next(op for op in module.walk() if op.name == "func.func" and op.body.blocks)
    assert function.body.block.last_op.operands[0].type == TensorType(i8, [1, 16])


@pytest.mark.parametrize("mutation", [
    "permutation", "duplicate_axis", "wrong_extent", "dynamic", "extra_operand", "dtype", "attribute", "encoding",
])
def test_equal_footprint_does_not_replace_a_valid_source_view_proof(mutation):
    _, group, entry = _closed()
    view = next(op for op in group.members if isinstance(op, tensor.CollapseShapeOp))
    if mutation in ("permutation", "duplicate_axis"):
        axes = [1, 0] if mutation == "permutation" else [0, 0]
        view.properties["reassociation"] = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in axes])])
    elif mutation in ("wrong_extent", "dynamic", "dtype"):
        view.results[0]._type = TensorType(i8 if mutation == "dtype" else f32, [-1 if mutation == "dynamic" else 63])
    elif mutation == "encoding":
        from xdsl.dialects.builtin import StringAttr

        view.results[0]._type = TensorType(f32, [64], StringAttr("unproved_layout"))
    elif mutation == "extra_operand":
        view.operands = (*view.operands, view.operands[0])
    else:
        view.attributes["unknown_semantics"] = IntegerAttr(1, i64)
    assert _binding(group, entry)[2]


def test_transposed_device_orientation_is_not_a_contiguous_source_view():
    _, group, entry = _closed()
    assert _binding(group, entry, transposed=True)[2]
    assert _binding(group, {**entry, "M": 16, "N": 4})[2]


def test_pointwise_coordinate_permutation_cannot_be_hidden_by_a_later_reshape():
    from xdsl.dialects.builtin import AffineMapAttr
    from xdsl.ir.affine import AffineMap

    _, group, entry = _closed()
    relu = next(op for op in group.members if CG.classify(op).kind == CG.RELU)
    maps = list(relu.indexing_maps)
    maps[0] = AffineMapAttr(AffineMap.transpose_map())
    relu.properties["indexing_maps"] = ArrayAttr(maps)
    assert _binding(group, entry)[2]


def test_explicit_transpose_is_not_accepted_as_an_equal_footprint_view():
    from xdsl.dialects.builtin import DenseArrayBase
    from xdsl.dialects.linalg.ops import TransposeOp

    _, group, entry = _closed()
    view = next(op for op in group.members if isinstance(op, tensor.CollapseShapeOp))
    empty = tensor.EmptyOp((), TensorType(f32, [16, 4]))
    transpose = TransposeOp(view.operands[0], empty.results[0], DenseArrayBase.from_list(i64, [1, 0]))
    view.parent.insert_ops_before([empty, transpose], view)
    view.operands = (transpose.results[0],)
    group.members.insert(group.members.index(view), transpose)
    assert _binding(group, entry)[2]


def test_a_partial_slice_before_the_views_does_not_authorize_a_destination():
    _, group, entry = _closed()
    view = next(op for op in group.members if isinstance(op, tensor.CollapseShapeOp))
    sliced = tensor.ExtractSliceOp.from_static_parameters(view.operands[0], [0, 0], [4, 8])
    view.parent.insert_op_before(sliced, view)
    view.operands = (sliced.results[0],)
    group.members.insert(group.members.index(view), sliced)
    assert _binding(group, entry)[2]


def test_an_escaping_intermediate_and_an_unrelated_output_are_refused():
    _, group, entry = _closed()
    source = group.root.results[0]
    tensor.CollapseShapeOp(
        operands=[source], result_types=[TensorType(f32, [64])],
        properties={"reassociation": ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in (0, 1)])])},
    )
    assert _binding(group, entry)[2]
    other = SimpleNamespace(root=group.root, members=[group.root, group.members[-1]], window_mean=None)
    assert _binding(other, entry)[2]


def test_no_source_reshape_cannot_authorize_an_equal_size_destination():
    _, group, _ = _closed()
    assert _contiguous_result_views(group, [group.root], [4, 16]) is None


def test_unknown_body_effects_cannot_hide_in_an_index_preserving_stage():
    from xdsl.dialects.func import CallOp

    _, group, entry = _closed()
    relu = next(op for op in group.members if CG.classify(op).kind == CG.RELU)
    relu.body.block.insert_op_before(CallOp("effect", [], []), relu.body.block.last_op)
    assert _binding(group, entry)[2]
