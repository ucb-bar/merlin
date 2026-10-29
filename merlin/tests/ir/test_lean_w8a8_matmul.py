"""Qualify the tensor MAC component on real Merlin output and native MLIR execution."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported, lowered

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff


def check_tensor(document, selection="inspect"):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_matmul.lean"),
            str(document),
            str(selection),
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "w8a8_integer_tensor_component_only"
    assert report["whole_pass_verified"] is False
    assert (proc.returncode == 0) == (report["refused"] == 0)
    return report


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (3, 17, 5), (1, 0, 2), (0, 3, 2)])
def test_actual_integer_tensor_geometry(tmp_path, lean_tools, shape):
    module, index, _ = lowered(*shape)
    report = check_tensor(exported(module, tmp_path, lean_tools), index)
    assert report["sites"] == [{"index": index, "matches": True}]


@pytest.mark.parametrize("k,accepted", [(131071, True), (131072, False)])
def test_bound_comes_from_actual_reduction_extent(tmp_path, lean_tools, k, accepted):
    # These are symbolic IR shapes, not allocated model tensors or a caller-supplied K hint.
    module, index, _ = lowered(1, k, 1)
    report = check_tensor(exported(module, tmp_path, lean_tools), index)
    assert report["sites"][0]["matches"] is accepted


@pytest.mark.parametrize("mutation", ["left_map", "right_map", "reduction_axis", "i16_accumulator"])
def test_valid_mlir_tensor_mutations_refused(tmp_path, lean_tools, mutation):
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerAttr, TensorType, i16, i32
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    module, index, generic = lowered(2, 2, 2)
    if mutation == "i16_accumulator":
        for op in list(module.walk()):
            value = op.properties.get("value")
            if isinstance(value, IntegerAttr) and value.type == i32:
                op.properties["value"] = IntegerAttr(value.value.data, i16)
            for region in op.regions:
                for block in region.blocks:
                    for arg in list(block.args):
                        if arg.type == i32:
                            Rewriter.replace_value_with_new_type(arg, i16)
            for result in list(op.results):
                if result.type == i32:
                    Rewriter.replace_value_with_new_type(result, i16)
                elif isinstance(result.type, TensorType) and result.type.element_type == i32:
                    Rewriter.replace_value_with_new_type(result, TensorType(i16, result.type.get_shape()))
    else:
        maps = [(0, 2), (2, 1), (0, 1)]
        if mutation == "left_map":
            maps[0] = (2, 0)
        elif mutation == "right_map":
            maps[1] = (1, 2)
        else:
            maps = [(0, 1), (1, 2), (0, 2)]
            generic.properties["iterator_types"] = ArrayAttr(
                [IteratorTypeAttr.parallel(), IteratorTypeAttr.reduction(), IteratorTypeAttr.parallel()]
            )
        generic.properties["indexing_maps"] = ArrayAttr(
            [AffineMapAttr(AffineMap(3, 0, tuple(AffineDimExpr(d) for d in dims))) for dims in maps]
        )
    report = check_tensor(exported(module, tmp_path, lean_tools), index)
    assert report["refused"] == 1


def isolated_integer_generic(generic):
    """Keep the actual emitted body/maps while exposing its byte inputs and accumulator."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    types = [value.type for value in generic.operands]
    block = Block(arg_types=types)
    operation = generic.clone(value_mapper=dict(zip(generic.operands, block.args, strict=True)))
    block.add_ops([operation, func.ReturnOp(*operation.results)])
    return ModuleOp([func.FuncOp.from_region("subject", types, [generic.results[0].type], Region(block))])


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (3, 7, 2)])
def test_actual_tensor_execution_against_native_mlir(tmp_path, fidelity_tools, shape):
    m, k, n = shape
    _, _, generic = lowered(m, k, n)
    source = str(isolated_integer_generic(generic))
    edges = [-128, 127, -1, 0, 1, -17, 63]
    samples, expected = [], []
    for offset, initial in [(0, 0), (1, 17), (0, 2147483647), (1, -2147483648)]:
        left = [edges[(offset + i) % len(edges)] for i in range(m * k)]
        right = [edges[(2 * offset + i) % len(edges)] for i in range(k * n)]
        accumulators = [initial] * (m * n)
        samples.append([[x % (1 << w) for x in values] for values, w in [(left, 8), (right, 8), (accumulators, 32)]])
        expected.append(
            [
                str((initial + sum(left[i * k + t] * right[t * n + j] for t in range(k))) % (1 << 32))
                for i in range(m)
                for j in range(n)
            ]
        )
    diff.compare(source, samples, tmp_path, fidelity_tools)
    actual = json.loads((tmp_path / "native-results.json").read_text())["results"]
    assert actual == expected
