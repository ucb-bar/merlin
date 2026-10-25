"""OS admission uses literal source weights and every partial, including D."""

import pytest

from merlin.common.mlir_query import parse
from merlin.targetgen.native_mac_bound import require_value_mac_bound


def contraction(*, weight=1, k=4, seed=0, runtime=False):
    weights = "" if runtime else f"%w=arith.constant dense<{weight}>:tensor<{k}x3xi8>"
    argument = f",%w:tensor<{k}x3xi8>" if runtime else ""
    module = parse(f"""module {{func.func @f(%x:tensor<2x{k}xi8>{argument})->tensor<2x3xi32> {{
      {weights}
      %d=arith.constant dense<{seed}>:tensor<2x3xi32>
      %out=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
        affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
        iterator_types=["parallel","parallel","reduction"]}}
        ins(%x,%w:tensor<2x{k}xi8>,tensor<{k}x3xi8>) outs(%d:tensor<2x3xi32>) {{
        ^bb0(%a:i8,%b:i8,%c:i32):
          %aa=arith.extsi %a:i8 to i32
          %bb=arith.extsi %b:i8 to i32
          %p=arith.muli %aa,%bb:i32
          %v=arith.addi %p,%c:i32
          linalg.yield %v:i32
        }} ->tensor<2x3xi32>
      return %out:tensor<2x3xi32> }} }}""")
    module.verify()
    return module, next(op for op in module.walk() if op.name == "linalg.generic")


def test_constant_weights_admit_and_runtime_twin_refuses():
    _, root = contraction(weight=1)
    proof = require_value_mac_bound(root, mesh_signed_bits=11)
    assert proof["column_lower"] == [-512] * 3
    assert proof["column_upper"] == [508] * 3
    _, twin = contraction(runtime=True)
    with pytest.raises(ValueError, match="constant weights"):
        require_value_mac_bound(twin, mesh_signed_bits=11)


def test_inside_and_outside_width_boundary_and_initializer():
    _, inside = contraction(k=4, seed=3)
    assert require_value_mac_bound(inside, mesh_signed_bits=10)["column_upper"] == [511] * 3
    _, outside = contraction(k=4, seed=4)
    with pytest.raises(ValueError, match="wrap"):
        require_value_mac_bound(outside, mesh_signed_bits=10)
    _, negative = contraction(weight=-128, k=1)
    assert require_value_mac_bound(negative, mesh_signed_bits=16)["column_upper"] == [16384] * 3


def test_cancellation_does_not_prove_partial_safety():
    module, root = contraction(k=4)
    weights = root.operands[1].owner
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    weights.properties["value"] = DenseIntOrFPElementsAttr.from_list(
        weights.result.type, [127] * 3 + [-127] * 3 + [127] * 3 + [-127] * 3
    )
    with pytest.raises(ValueError, match="wrap"):
        require_value_mac_bound(root, mesh_signed_bits=16)


@pytest.mark.parametrize("bits", [None, 0, True])
def test_unknown_width_refuses(bits):
    _, root = contraction()
    with pytest.raises(ValueError, match="width"):
        require_value_mac_bound(root, mesh_signed_bits=bits)
