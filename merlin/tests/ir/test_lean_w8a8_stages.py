"""Actual W8A8 tensor stages, source-declared axes and native requantization qualification."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported, lowered
from test_lean_w8a8_scalar import parts

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff


def check_stage(document, index, kind, left=(), right=()):
    def axes(dims):
        return ",".join(map(str, dims)) if dims else "-"

    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_stages.lean"),
            str(document),
            str(index),
            kind,
            axes(left),
            axes(right),
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "w8a8_tensor_stages_only"
    assert report["whole_pass_verified"] is False
    assert (proc.returncode == 0) == (report["refused"] == 0)
    return report["refused"] == 0


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 0, 2), (0, 3, 2)])
def test_all_actual_float_tensor_stages(tmp_path, lean_tools, shape):
    module, _, _ = lowered(*shape)
    document = exported(module, tmp_path, lean_tools)
    stages = parts(module)
    assert len(stages) == 9
    for position, (kind, index, _) in enumerate(stages):
        if kind in {"absmax", "quantize"}:
            left, right = ((0,), ()) if position < 4 else ((1,), ())
        elif kind == "requantize2":
            left, right = (0,), (1,)
        else:
            left, right = (), ()
        assert check_stage(document, index, kind, left, right), (kind, index)


@pytest.mark.parametrize("mutation", ["reduction_axis", "quantize_axis", "quantize_output", "scale_axis_order"])
def test_valid_mlir_layout_mutations_refused(tmp_path, lean_tools, mutation):
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    module, _, _ = lowered(2, 2, 2)
    kind = "absmax" if mutation == "reduction_axis" else "requantize2" if mutation == "scale_axis_order" else "quantize"
    _, index, generic = next(part for part in parts(module) if part[0] == kind)
    maps = [list(attr.data.results) for attr in generic.indexing_maps]
    if mutation == "reduction_axis":
        maps[-1] = [AffineDimExpr(1)]
        generic.properties["iterator_types"] = ArrayAttr([IteratorTypeAttr.reduction(), IteratorTypeAttr.parallel()])
    elif mutation == "quantize_axis":
        maps[1] = [AffineDimExpr(1)]
    elif mutation == "quantize_output":
        maps[-1] = [AffineDimExpr(1), AffineDimExpr(0)]
    else:
        maps[1], maps[2] = maps[2], maps[1]
    generic.properties["indexing_maps"] = ArrayAttr([AffineMapAttr(AffineMap(2, 0, tuple(exprs))) for exprs in maps])
    assert not check_stage(exported(module, tmp_path, lean_tools), index, kind, (0,), (1,))


@pytest.mark.parametrize("kind", ["scale", "reciprocal", "quantize", "requantize2"])
def test_tensor_checks_still_refuse_wrong_scalar_bodies(tmp_path, lean_tools, kind):
    from xdsl.dialects.builtin import FloatAttr, f32

    module, _, _ = lowered()
    _, index, generic = next(part for part in parts(module) if part[0] == kind)
    body = list(generic.body.block.ops)
    if kind == "scale":
        body[2].properties["value"] = FloatAttr(1e-4, f32)
    elif kind == "reciprocal":
        body[0].properties["value"] = FloatAttr(2.0, f32)
    elif kind == "quantize":
        body[1].name = "math.round"
    else:
        first, second = body[1:3]
        left, right = first.operands[1], second.operands[1]
        first.operands = [first.operands[0], right]
        second.operands = [second.operands[0], left]
    assert not check_stage(exported(module, tmp_path, lean_tools), index, kind, (0,), (1,))


def requant_module(shape, left_dims, right_dims, scale_count=2):
    """Retain the emitted operation order in rank/layout and scale-count component fixtures.

    Two scales come from the actual float/float pass output. Removing trailing scale
    operations exercises the zero/one-scale templates without claiming those fixtures
    were emitted by that production path.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, ModuleOp, TensorType, f32, i32
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    module, _, _ = lowered()
    _, _, actual = next(part for part in parts(module) if part[0] == "requantize2")
    assert scale_count in (0, 1, 2)
    scale_types = [
        TensorType(f32, [shape[d] for d in left_dims]),
        TensorType(f32, [shape[d] for d in right_dims]),
    ]
    types = [TensorType(i32, shape), *scale_types[:scale_count], TensorType(f32, shape)]
    block = Block(arg_types=types)
    body = actual.body.clone()
    operations = list(body.block.ops)
    body.block.last_op.operands = [operations[scale_count].results[0]]
    for operation in reversed(operations[scale_count + 1 : -1]):
        body.block.erase_op(operation)
    for arg in reversed(body.block.args[scale_count + 1 : 3]):
        body.block.erase_arg(arg)
    dims = [tuple(range(len(shape))), *[left_dims, right_dims][:scale_count], tuple(range(len(shape)))]
    operation = linalg.GenericOp(
        inputs=block.args[:-1],
        outputs=block.args[-1:],
        body=body,
        indexing_maps=ArrayAttr(
            [AffineMapAttr(AffineMap(len(shape), 0, tuple(AffineDimExpr(d) for d in ds))) for ds in dims]
        ),
        iterator_types=ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * len(shape)),
        result_types=[types[-1]],
    )
    block.add_ops([operation, func.ReturnOp(*operation.results)])
    return ModuleOp([func.FuncOp.from_region("forward", types, [types[-1]], Region(block))])


@pytest.mark.parametrize(
    "shape,left,right", [((), (), ()), ((3,), (), (0,)), ((2, 3), (0,), (1,)), ((2, 3, 4), (0, 1), (2,))]
)
@pytest.mark.parametrize("scale_count", [0, 1, 2])
def test_requantization_layouts_across_ranks(tmp_path, lean_tools, shape, left, right, scale_count):
    module = requant_module(shape, left, right, scale_count)
    assert check_stage(exported(module, tmp_path, lean_tools), 0, f"requantize{scale_count}", left, right)


def test_duplicate_and_out_of_range_contract_axes_refused(tmp_path, lean_tools):
    module, _, _ = lowered(2, 2, 2)
    _, index, _ = next(part for part in parts(module) if part[0] == "requantize2")
    document = exported(module, tmp_path, lean_tools)
    assert not check_stage(document, index, "requantize2", (0, 0), (1,))
    assert not check_stage(document, index, "requantize2", (2,), (1,))


@pytest.mark.parametrize("shape,left,right", [((), (), ()), ((2, 3), (0,), (1,))])
@pytest.mark.parametrize("scale_count", [0, 1, 2])
def test_requantization_tensor_execution_against_native_mlir(tmp_path, fidelity_tools, shape, left, right, scale_count):
    import itertools
    import math
    import struct

    import numpy as np

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    module = requant_module(shape, left, right, scale_count)
    count = math.prod(shape)
    scales_left = [np.float32(0.125 * (i + 1)) for i in range(math.prod(shape[d] for d in left))]
    scales_right = [np.float32(0.2 * (i + 1)) for i in range(math.prod(shape[d] for d in right))]
    samples, expected = [], []
    for acc in [[0] * count, [-2147483648, 2147483647, 16777217, -16777217, 3, -7][:count]]:
        scale_inputs = [
            [bits(x) for x in scales_left],
            [bits(x) for x in scales_right],
        ]
        sample = [[x % (1 << 32) for x in acc], *scale_inputs[:scale_count], [bits(-999.0)] * count]
        answer = []
        for value, point in zip(acc, itertools.product(*(range(d) for d in shape)), strict=True):
            li = np.ravel_multi_index(tuple(point[d] for d in left), tuple(shape[d] for d in left)) if left else 0
            ri = np.ravel_multi_index(tuple(point[d] for d in right), tuple(shape[d] for d in right)) if right else 0
            result = np.float32(value)
            for scale in [scales_left[li], scales_right[ri]][:scale_count]:
                result = np.float32(result * scale)
            answer.append(str(bits(result)))
        samples.append(sample)
        expected.append(answer)
    diff.compare(str(module), samples, tmp_path, fidelity_tools, function="forward")
    actual = json.loads((tmp_path / "native-results.json").read_text())["results"]
    assert actual == expected
