"""Exact pointwise producers write directly into split/permuted storage."""

from __future__ import annotations

from math import prod

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
from merlin.llvmlower.destination_check import verify_permutation_rewrite


def program(m=2, outer=3, inner=4, permutation=(1, 0, 2), bits=32):
    source = f"tensor<{m}x{outer * inner}xi{bits}>"
    narrow = f"tensor<{m}x{outer * inner}xi8>"
    expanded = f"tensor<{m}x{outer}x{inner}xi8>"
    shape = (m, outer, inner)
    packed = "tensor<" + "x".join(map(str, (shape[i] for i in permutation))) + "xi8>"
    return f"""module {{func.func @forward(%x:{source}, %state:{source}) -> ({packed},{source}) {{
      %zero=arith.constant 0:i{bits}
      %high=arith.constant 127:i{bits}
      %init=tensor.empty():{narrow}
      %p=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
        iterator_types=["parallel","parallel"]}}
        ins(%x,%state:{source},{source}) outs(%init:{narrow}) {{
        ^bb0(%a:i{bits},%s:i{bits},%unused:i8):
          %sum=arith.addi %a,%s:i{bits}
          %positive=arith.maxsi %sum,%zero:i{bits}
          %bounded=arith.minsi %positive,%high:i{bits}
          %byte=arith.trunci %bounded:i{bits} to i8
          linalg.yield %byte:i8
      }} -> {narrow}
      %split=tensor.expand_shape %p [[0],[1,2]] output_shape [{m},{outer},{inner}]:{narrow} into {expanded}
      %empty=tensor.empty():{packed}
      %t=linalg.transpose ins(%split:{expanded}) outs(%empty:{packed}) permutation=[{",".join(map(str, permutation))}]
      %ni=tensor.empty():{source}
      %next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
        iterator_types=["parallel","parallel"]}}
        ins(%state:{source}) outs(%ni:{source}) {{
        ^bb1(%old:i{bits},%unused_next:i{bits}):
          %one=arith.constant 1:i{bits}
          %new=arith.addi %old,%one:i{bits}
          linalg.yield %new:i{bits}
      }} -> {source}
      return %t,%next:{packed},{source}
    }}}}"""


def prepare(source, **controls):
    return prepare_automatic_destinations(source, serial=True, permuted_outputs=True, **controls)


@pytest.mark.parametrize("shape,permutation,bits", [((2, 3, 4), (1, 0, 2), 32), ((3, 2, 5), (2, 0, 1), 16)])
def test_split_dimension_producer_writes_packed_output_without_second_scan(shape, permutation, bits):
    source = program(*shape, permutation, bits)
    before = parse_mlir_text(source)
    selected, report = prepare(source)
    after = parse_mlir_text(selected)
    assert report["rewrites"] == 1
    assert not any(op.name in ("tensor.expand_shape", "linalg.transpose") for op in after.walk())
    original = next(op for op in before.walk() if op.name == "linalg.generic")
    actual = next(op for op in after.walk() if op.name == "linalg.generic")
    constants_before = [op.results[0] for op in before.walk() if op.name == "arith.constant"]
    constants_after = [op.results[0] for op in after.walk() if op.name == "arith.constant"]
    assert original.body.is_structurally_equivalent(
        actual.body, dict(zip(constants_before, constants_after, strict=True))
    )
    assert prod(actual.results[0].type.get_shape()) == prod(original.results[0].type.get_shape())
    assert len(verify_permutation_rewrite(before, after, index_bits=64)) == 1


def test_actual_wrong_split_coordinates_cannot_hide_behind_equal_shapes():
    import pytest
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    source = program(2, 3, 3)
    selected, _ = prepare(source)
    after = parse_mlir_text(selected)
    actual = next(op for op in after.walk() if op.name == "linalg.generic")
    maps = list(actual.indexing_maps)
    # A transposed within-group coordinate still fits exactly the same shapes.
    maps[0] = AffineMapAttr(AffineMap(3, 0, (AffineDimExpr(0), AffineDimExpr(2) * 3 + AffineDimExpr(1))))
    actual.properties["indexing_maps"] = ArrayAttr(maps)
    after.verify()
    with pytest.raises(ValueError, match="coordinates"):
        verify_permutation_rewrite(parse_mlir_text(source), after, index_bits=64)


def test_unknown_expansion_metadata_and_live_observers_keep_original_route():
    from xdsl.dialects.builtin import UnitAttr

    for reason in ("effect", "observer", "initialized"):
        original = parse_mlir_text(program())
        expansion = next(op for op in original.walk() if op.name == "tensor.expand_shape")
        if reason == "effect":
            expansion.attributes["unknown_execution"] = UnitAttr()
        elif reason == "observer":
            # A second tensor use is a live observer regardless of later DCE.
            from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, i64
            from xdsl.dialects.tensor import CollapseShapeOp

            collapse = CollapseShapeOp.create(
                operands=[expansion.result],
                result_types=[expansion.src.type],
                properties={
                    "reassociation": ArrayAttr(
                        [ArrayAttr([IntegerAttr(0, i64)]), ArrayAttr([IntegerAttr(1, i64), IntegerAttr(2, i64)])]
                    )
                },
            )
            expansion.parent.insert_op_after(collapse, expansion)
        else:
            producer = expansion.src.owner
            producer.body.block.ops.last.operands = (producer.body.block.args[-1],)
        original.verify()
        from merlin.transforms.certificates import ir_text

        source = ir_text(original)
        selected, report = prepare(source)
        assert report["rewrites"] == 0 and selected == source


@pytest.mark.parametrize("control", [{"max_added_operations": 1}, {"max_metadata_items": 1}, {"index_bits": 5}])
def test_packing_exhausted_growth_and_index_bounds_retain_original(control):
    source = program()
    assert prepare(source, **control)[0] == source


def test_opaque_consumer_keeps_exact_final_tensor_type_order_and_binding():
    from xdsl.dialects import func

    packed = "tensor<3x2x4xi8>"
    source = (
        program()
        .replace("module {", f"module {{func.func private @consume({packed})->{packed}\n", 1)
        .replace("return %t,%next:", f"%received=func.call @consume(%t):({packed})->{packed}\nreturn %received,%next:")
    )
    before = parse_mlir_text(source)
    selected, report = prepare(source)
    assert report["rewrites"] == 1
    after = parse_mlir_text(selected)
    original_call = next(op for op in before.walk() if isinstance(op, func.CallOp))
    actual_call = next(op for op in after.walk() if isinstance(op, func.CallOp))
    assert original_call.callee == actual_call.callee
    assert [v.type for v in original_call.arguments] == [v.type for v in actual_call.arguments]
    assert [v.type for v in original_call.results] == [v.type for v in actual_call.results]
    assert actual_call.arguments[0].owner.name == "linalg.generic"
    verify_permutation_rewrite(before, after, index_bits=64)
    # A valid same-type argument from an unwritten tensor is still wrong.
    actual_call.operands = (actual_call.arguments[0].owner.outputs[0],)
    after.verify()
    with pytest.raises(ValueError, match="coordinates"):
        verify_permutation_rewrite(before, after, index_bits=64)
