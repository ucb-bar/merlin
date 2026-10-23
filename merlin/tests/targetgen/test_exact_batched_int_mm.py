"""Exact integer contraction recognition preserves independent batch axes."""

from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.kernels.shapes import zero_initialised
from merlin.targetgen._m2m_capture_worker import _exported_integer_mm_count
from merlin.targetgen.application_inventory import exact_int_mm_generic_operation


def contraction(rank, *, initializer=0):
    batch = [2, 3][: rank - 2]
    shapes = [(*batch, 5, 7), (*batch, 7, 11), (*batch, 5, 11)]
    types = [
        "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">" for shape, dtype in zip(shapes, ["i8", "i8", "i32"])
    ]
    dims = [f"d{i}" for i in range(rank + 1)]
    prefix, m, n, k = dims[:-3], *dims[-3:]
    maps = [
        f"affine_map<({', '.join(dims)}) -> ({', '.join(indices)})>"
        for indices in [(*prefix, m, k), (*prefix, k, n), (*prefix, m, n)]
    ]
    iterators = ", ".join(['"parallel"'] * rank + ['"reduction"'])
    text = f"""module {{
      func.func @forward(%a: {types[0]}, %b: {types[1]}) -> {types[2]} {{
        %z = arith.constant {initializer} : i32
        %init = tensor.splat %z : {types[2]}
        %r = linalg.generic {{indexing_maps = [{", ".join(maps)}], iterator_types = [{iterators}]}}
          ins(%a, %b : {types[0]}, {types[1]}) outs(%init : {types[2]})
          attrs = {{prov.op = "int_matmul"}} {{
          ^bb0(%x: i8, %w: i8, %acc: i32):
            %xx = arith.extsi %x : i8 to i32
            %ww = arith.extsi %w : i8 to i32
            %p = arith.muli %xx, %ww : i32
            %s = arith.addi %acc, %p : i32
            linalg.yield %s : i32
          }} -> {types[2]}
        return %r : {types[2]}
      }}
    }}"""
    module = mq.parse(text)
    module.verify()
    return module, next(op for op in module.walk() if mq.op_name(op) == "linalg.generic")


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_exact_matrix_and_independent_batches(rank):
    module, op = contraction(rank)
    assert exact_int_mm_generic_operation(op)
    assert zero_initialised(op)
    assert _exported_integer_mm_count(module) == int(rank == 2)
    assert _exported_integer_mm_count(module, batched=True) == int(rank > 2)


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize(
    "damage", ["unsigned", "self_product", "drop_accumulator", "batch_broadcast", "mismatched_extent", "nonzero"]
)
def test_matcher_fails_closed(rank, damage):
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, TensorType
    from xdsl.ir.affine import AffineExpr, AffineMap

    _module, op = contraction(rank, initializer=int(damage == "nonzero"))
    if damage == "unsigned":
        text = str(_module).replace("arith.extsi", "arith.extui")
        op = next(op for op in mq.parse(text).walk() if mq.op_name(op) == "linalg.generic")
    elif damage == "self_product":
        ext_a, _ext_b, multiply, *_ = list(op.regions[0].blocks[0].ops)
        multiply.operands = [ext_a.results[0], ext_a.results[0]]
    elif damage == "drop_accumulator":
        _ext_a, _ext_b, multiply, add, _yield = list(op.regions[0].blocks[0].ops)
        add.operands = [multiply.results[0], multiply.results[0]]
    elif damage == "batch_broadcast":
        maps = list(op.properties["indexing_maps"])
        affine = maps[0].data
        maps[0] = AffineMapAttr(AffineMap(affine.num_dims, 0, (AffineExpr.constant(0), *affine.results[1:])))
        op.properties["indexing_maps"] = ArrayAttr(maps)
    elif damage == "mismatched_extent":
        value = op.operands[1]
        shape = list(value.type.get_shape())
        shape[-2] += 1
        value._type = TensorType(value.type.element_type, shape)
    assert not exact_int_mm_generic_operation(op)


@pytest.mark.parametrize("rank", [3, 4])
def test_batched_receipt_counts_are_separate_and_checked(rank):
    from test_pt2e_integerization_admission import _receipt

    from merlin.targetgen.capsule_source import M2MUnavailable, _require_pt2e_integerization_receipt

    module, _op = contraction(rank)
    receipt = _receipt()
    receipt.update(
        integer_mm_emitted=0, exported_integer_mm_count=0, integer_bmm_emitted=1, exported_integer_bmm_count=1
    )
    meta = dict(integerization_receipt=receipt, quantization_stats=dict(annotated_contractions=1))
    _require_pt2e_integerization_receipt(str(module), meta, agreement_tolerance=(0.03125, 0.02))
    for field in ("integer_bmm_emitted", "exported_integer_bmm_count"):
        receipt[field] = 2
        with pytest.raises(M2MUnavailable, match="batched export"):
            _require_pt2e_integerization_receipt(str(module), meta, agreement_tolerance=(0.03125, 0.02))
        receipt[field] = 1
