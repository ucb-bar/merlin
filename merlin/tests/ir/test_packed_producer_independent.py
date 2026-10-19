"""Independent scalar/coordinate checks for producer packing; no target claim."""

from itertools import product
from math import prod

import numpy as np
import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
from merlin.llvmlower.destination_check import verify_permutation_rewrite

CASES = [
    ((2, 3, 4, 5), ((0, 1), (2, 3)), (2, 0, 3, 1)),
    ((2, 1, 3, 1, 4, 5), ((0, 1, 2), (3, 4, 5)), (5, 2, 0, 3, 1, 4)),
]


def program(expanded, groups, permutation, bits):
    source = f"tensor<6x20xi{bits}>"
    broadcast = f"tensor<1x20xi{bits}>"
    row = f"tensor<6xi{bits}>"
    scalar = f"tensor<i{bits}>"
    unpacked = "tensor<6x20xi8>"
    split = "tensor<" + "x".join(map(str, expanded)) + "xi8>"
    packed = "tensor<" + "x".join(str(expanded[i]) for i in permutation) + "xi8>"
    reassociation = "[" + ",".join(str(list(group)) for group in groups) + "]"
    return f"""module {{
    func.func private @opaque({packed},{packed})->({packed},{packed})
    func.func @forward(%x:{source},%b:{broadcast},%r:{row},%s:{scalar},%observer:{packed})
      ->({packed},{packed},{source}) {{
      %zero=arith.constant 0:i{bits}
      %high=arith.constant 127:i{bits}
      %init=tensor.empty():{unpacked}
      %p=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
        affine_map<(i,j)->(0,j)>,affine_map<(i,j)->(i)>,affine_map<(i,j)->()>,
        affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
        ins(%x,%x,%b,%r,%s:{source},{source},{broadcast},{row},{scalar})
        outs(%init:{unpacked}) {{
        ^bb0(%a:i{bits},%alias:i{bits},%bias:i{bits},%row:i{bits},%scalar:i{bits},%unused:i8):
          %twice=arith.addi %a,%alias:i{bits}
          %difference=arith.subi %twice,%bias:i{bits}
          %product=arith.muli %difference,%row:i{bits}
          %shifted=arith.addi %product,%scalar:i{bits}
          %positive=arith.maxsi %shifted,%zero:i{bits}
          %bounded=arith.minsi %positive,%high:i{bits}
          %byte=arith.trunci %bounded:i{bits} to i8
          linalg.yield %byte:i8
      }} -> {unpacked}
      %split=tensor.expand_shape %p {reassociation} output_shape {list(expanded)}:{unpacked} into {split}
      %empty=tensor.empty():{packed}
      %t=linalg.transpose ins(%split:{split}) outs(%empty:{packed}) permutation={list(permutation)}
      %a,%b0=func.call @opaque(%t,%observer):({packed},{packed})->({packed},{packed})
      return %a,%b0,%x:{packed},{packed},{source}
    }} }}"""


def prepare(source):
    return prepare_automatic_destinations(source, serial=True, permuted_outputs=True)


def signed(value, bits):
    return (int(value) + 2 ** (bits - 1)) % 2**bits - 2 ** (bits - 1)


@pytest.mark.parametrize("expanded,groups,permutation", CASES)
@pytest.mark.parametrize("bits", [16, 32])
def test_both_split_axes_broadcasts_and_aliases_preserve_every_point(expanded, groups, permutation, bits):
    source = program(expanded, groups, permutation, bits)
    before = parse_mlir_text(source)
    selected, report = prepare(source)
    assert report["rewrites"] == 1
    after = parse_mlir_text(selected)
    writer = next(op for op in after.walk() if op.name == "linalg.generic")
    assert writer.inputs[0] is writer.inputs[1]
    assert not any(op.name in ("tensor.expand_shape", "linalg.transpose") for op in after.walk())
    x = np.resize([2 ** (bits - 1) - 1, -(2 ** (bits - 1)), 17, 1, -3, 23, 7], (6, 20))
    bias = np.arange(20, dtype=np.int64).reshape(1, 20) - 7
    rows = np.array([1, -1, 3, 11, 0, 2 ** (bits - 1) - 1], dtype=np.int64)
    scalar = np.array(-11, dtype=np.int64)
    expected = np.empty((6, 20), dtype=np.int8)
    for i, j in product(range(6), range(20)):
        twice = signed(int(x[i, j]) * 2, bits)
        difference = signed(twice - int(bias[0, j]), bits)
        value = signed(signed(difference * int(rows[i]), bits) + int(scalar), bits)
        expected[i, j] = min(127, max(0, value))
    # NumPy's byte/element reshape and permutation establish the origin table;
    # this does not use the rewrite/checker's coordinate-composition helper.
    origins = np.arange(120).reshape(expanded).transpose(permutation)
    expected = expected.reshape(expanded).transpose(permutation)
    actual = np.empty(expected.shape, dtype=np.int8)
    maps = [mapping.data for mapping in writer.indexing_maps]
    bounds = [None] * len(expanded)
    for extent, expression in zip(actual.shape, maps[-1].results, strict=True):
        assert isinstance(expression, AffineDimExpr)
        bounds[expression.position] = extent
    assert prod(bounds) == actual.size == 120
    constants = {op.results[0]: op.value.value.data for op in after.walk() if op.name == "arith.constant"}
    for point in product(*(range(bound) for bound in bounds)):
        destination = maps[-1].eval(point, ())
        i, j = divmod(int(origins[destination]), 20)
        coordinates = [mapping.eval(point, ()) for mapping in maps[:-1]]
        assert coordinates == [(i, j), (i, j), (0, j), (i,), ()]
        arguments = [
            array[coordinate] for array, coordinate in zip((x, x, bias, rows, scalar), coordinates, strict=True)
        ]
        # The output initializer is unread. External integer constants retain
        # their actual SSA binding when evaluating the cloned scalar body.
        block = writer.body.block
        external = {value: constants[value] for op in block.ops for value in op.operands if value in constants}
        values = dict(external)
        values.update(zip(block.args, [*arguments, None], strict=True))
        actual[destination] = evaluate(writer, values, bits)
    assert actual.tobytes() == expected.tobytes()
    call = next(op for op in after.walk() if isinstance(op, func.CallOp))
    forward = call.parent_op()
    returned = next(op for op in forward.walk() if isinstance(op, func.ReturnOp))
    assert tuple(call.arguments) == (writer.results[0], forward.args[-1])
    assert tuple(returned.operands) == (*call.results, forward.args[0])
    assert len(verify_permutation_rewrite(before, after, index_bits=64)) == 1


def evaluate(writer, values, bits):
    """Evaluate only this fixture's exact emitted integer SSA instructions."""
    for op in writer.body.block.ops:
        if op.name == "linalg.yield":
            return values[op.operands[0]]
        operands = [int(values[value]) for value in op.operands]
        if op.name == "arith.addi":
            value = signed(operands[0] + operands[1], bits)
        elif op.name == "arith.subi":
            value = signed(operands[0] - operands[1], bits)
        elif op.name == "arith.muli":
            value = signed(operands[0] * operands[1], bits)
        elif op.name == "arith.maxsi":
            value = max(operands)
        elif op.name == "arith.minsi":
            value = min(operands)
        else:
            assert op.name == "arith.trunci"
            value = signed(operands[0], op.results[0].type.width.data)
        values[op.results[0]] = value
    raise AssertionError("missing yield")


@pytest.mark.parametrize("mutation", ["broadcast", "observer"])
def test_checker_rejects_wrong_broadcast_origin_or_omitted_observer(mutation):
    source = program(*CASES[0], 16)
    selected, _ = prepare(source)
    after = parse_mlir_text(selected)
    if mutation == "broadcast":
        writer = next(op for op in after.walk() if op.name == "linalg.generic")
        maps = list(writer.indexing_maps)
        # Still in bounds for [1,20], but visits the wrong source columns.
        maps[2] = AffineMapAttr(AffineMap(4, 0, (AffineConstantExpr(0), AffineDimExpr(3) * 4 + AffineDimExpr(2))))
        writer.properties["indexing_maps"] = ArrayAttr(maps)
    else:
        returned = next(op for op in after.walk() if isinstance(op, func.ReturnOp))
        returned.operands = (returned.operands[0], returned.operands[0], returned.operands[2])
    after.verify()
    with pytest.raises(ValueError, match="coordinates" if mutation == "broadcast" else "unmodeled operations"):
        verify_permutation_rewrite(parse_mlir_text(source), after, index_bits=64)


def test_unknown_effect_between_producer_and_pack_retains_original():
    source = program(*CASES[0], 16)
    source = source.replace("module {", "module {func.func private @barrier()\n", 1)
    source = source.replace("%split=tensor.expand_shape", "func.call @barrier():()->()\n%split=tensor.expand_shape")
    selected, report = prepare(source)
    assert selected == source and report["rewrites"] == 0
