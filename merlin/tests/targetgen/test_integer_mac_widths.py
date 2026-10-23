"""The signed-byte MAC interval must not authenticate wider operands."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_spec_int8 import closed_integer_mac


@pytest.mark.parametrize(
    "operand,accumulator,admitted", [(8, 32, True), (16, 32, False), (8, 64, False), (32, 64, False)]
)
def test_closed_mac_checks_both_storage_and_scalar_widths(operand, accumulator, admitted):
    module = parse_mlir_text(f"""builtin.module {{
      func.func @forward(%a: tensor<1x1xi{operand}>, %b: tensor<1x1xi{operand}>,
                         %seed: tensor<1x1xi{accumulator}>) -> tensor<1x1xi{accumulator}> {{
        %r = linalg.generic {{indexing_maps = [affine_map<(i,j,k)->(i,k)>,
          affine_map<(i,j,k)->(k,j)>, affine_map<(i,j,k)->(i,j)>],
          iterator_types = ["parallel","parallel","reduction"]}}
          ins(%a,%b : tensor<1x1xi{operand}>,tensor<1x1xi{operand}>)
          outs(%seed : tensor<1x1xi{accumulator}>) {{
        ^bb0(%x: i{operand},%y: i{operand},%z: i{accumulator}):
          %ax = arith.extsi %x : i{operand} to i{accumulator}
          %bx = arith.extsi %y : i{operand} to i{accumulator}
          %p = arith.muli %ax,%bx : i{accumulator}
          %v = arith.addi %p,%z : i{accumulator}
          linalg.yield %v : i{accumulator}
        }} -> tensor<1x1xi{accumulator}>
        func.return %r : tensor<1x1xi{accumulator}>
      }}
    }}""")
    module.verify()
    op = next(op for op in module.walk() if op.name == "linalg.generic")
    assert closed_integer_mac(op) is admitted
