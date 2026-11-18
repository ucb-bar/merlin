"""Unit-axis reduction is derived from standard mixed sizes, never a tag."""

import numpy as np
import pytest
from test_execution_model_correspondence import printed
from xdsl.dialects import arith, builtin, cf, func, scf, tensor
from xdsl.ir import Block, Region

from merlin.frontends.execution_source_coordinates import SourceCoordinates, _retained_axes
from merlin.llvmlower import toolchain
from merlin.llvmlower.abi import ScalarArg
from merlin.llvmlower.owned_results import compile_owned_results

DYNAMIC = builtin.DYNAMIC_INDEX


@pytest.mark.parametrize(
    "sizes,result,expected",
    (
        ((1, 1, 5), (1, 5), (1, 2)),
        ((DYNAMIC, 1, 5), (DYNAMIC, 5), (0, 2)),
        ((1, DYNAMIC, 1), (DYNAMIC,), (1,)),
        ((1, 1), (), ()),
        ((1, 1), (1,), (1,)),
    ),
)
def test_standard_reverse_axis_matching_preserves_dynamic_and_retained_unit_axes(sizes, result, expected):
    source = Block(arg_types=[builtin.TensorType(builtin.i64, [3] * len(sizes))]).args[0]
    dynamic = [arith.ConstantOp.from_int_and_width(1, builtin.IndexType()).result for size in sizes if size == DYNAMIC]
    extract = tensor.ExtractSliceOp.build(
        operands=[source, [], dynamic, []],
        result_types=[builtin.TensorType(builtin.i64, result)],
        properties={
            "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, [0] * len(sizes)),
            "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, sizes),
            "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, [1] * len(sizes)),
        },
    )
    assert tuple(_retained_axes(extract)) == expected


@pytest.mark.parametrize("sizes,result", (((2, 3), (3,)), ((DYNAMIC, 3), (3,)), ((1, 3), (2, 3))))
def test_nonunit_reductions_and_changed_dimensions_cannot_claim_a_coordinate_map(sizes, result):
    source = Block(arg_types=[builtin.TensorType(builtin.i64, [3] * len(sizes))]).args[0]
    dynamic = [arith.ConstantOp.from_int_and_width(1, builtin.IndexType()).result for size in sizes if size == DYNAMIC]
    extract = tensor.ExtractSliceOp.build(
        operands=[source, [], dynamic, []],
        result_types=[builtin.TensorType(builtin.i64, result)],
        properties={
            "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, [0] * len(sizes)),
            "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, sizes),
            "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, [1] * len(sizes)),
        },
    )
    with pytest.raises(ValueError):
        _retained_axes(extract)


@pytest.mark.skipif(not toolchain.available(), reason="existing native compiler required")
def test_actual_logical_stride_of_a_retained_unit_axis_survives_runtime_reduction(tmp_path):
    root_type = builtin.TensorType(builtin.i64, [1, DYNAMIC, DYNAMIC])
    target = builtin.TensorType(builtin.i64, [DYNAMIC, DYNAMIC])
    block = Block(arg_types=[root_type, *([builtin.i64] * 5)])

    def emit(op, role="source"):
        block.add_op(op)
        return op.results[0] if op.results else None

    zero, one, two = [emit(arith.ConstantOp.from_int_and_width(n, builtin.IndexType())) for n in range(3)]
    rows, cols = emit(tensor.DimOp(block.args[0], one)), emit(tensor.DimOp(block.args[0], two))
    emit(cf.AssertOp(emit(arith.CmpiOp(rows, two, "sge")), "source row selection needs two rows"))
    predicate = emit(arith.CmpiOp(cols, zero, "sgt"))
    extraction = tensor.ExtractSliceOp.build(
        operands=[block.args[0], [], [one, cols], []],
        result_types=[target],
        properties={
            "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, [0, 1, 0]),
            "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, [DYNAMIC, 1, DYNAMIC]),
            "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, [1, 1, 1]),
        },
    )
    empty = tensor.EmptyOp([one, cols], target)
    conditional = scf.IfOp(
        predicate,
        [target],
        Region(Block([extraction, scf.YieldOp(extraction.result)])),
        Region(Block([empty, scf.YieldOp(empty.tensor)])),
    )
    view = emit(conditional)
    record = {
        "view": {
            "binding": {
                "root": "state",
                "source_arg": 0,
                "shape": ["rows", "cols"],
                "strides": ["s0", "s1"],
                "offset": "origin",
            },
            "root_schema_operand": "owner",
            "root_type": str(root_type),
            "view_type": str(target),
            "layout": {
                "schema": "t2m.runtime-source-coordinates.v1",
                "root_order": "logical-dense-row-major",
                "steps": [
                    {
                        "op": "scf.if",
                        "result_type": str(target),
                        "offsets": [0, 1, 0],
                        "sizes": [DYNAMIC, 1, DYNAMIC],
                        "strides": [1, 1, 1],
                        "retained_axes": [0, 2],
                    }
                ],
            },
        }
    }
    SourceCoordinates(emit).check_views(
        record,
        {"state": "owner"},
        {"owner": block.args[0], "view": view},
        dict(zip(("rows", "cols", "s0", "s1", "origin"), block.args[1:], strict=True)),
        {"source_argument_binding": {"bindings": [{"kind": "pointer", "allocation": "state", "offset": "origin"}]}},
    )
    block.add_op(func.ReturnOp(view))
    entry = func.FuncOp("forward", ([root_type, *([builtin.i64] * 5)], [target]), Region(block))
    entry.properties["arg_attrs"] = builtin.ArrayAttr(
        [
            builtin.DictionaryAttr({"bufferization.writable": builtin.IntegerAttr(0, builtin.i1)}),
            *[builtin.DictionaryAttr({}) for _ in range(5)],
        ]
    )
    module = builtin.ModuleOp([entry])
    module.verify()
    program = compile_owned_results(printed(module), workdir=tmp_path / "native")
    for rows, cols in ((3, 5), (2, 0), (9, 17), (2, 1)):
        original = (np.arange(rows * cols, dtype=np.int64) + (1 << 56) + 1).reshape(1, rows, cols)
        coordinates = (1, cols, rows * cols, 1, cols)
        (actual,), _ = program.execute((original, *(ScalarArg(n, "i64") for n in coordinates)))
        wanted = original[:, 1, :]
        assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
