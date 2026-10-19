"""Actual typed arithmetic, prefixes, readout and complete ownership are checked."""

import numpy as np
import pytest
from merlin_experiments.phase0.connected_integer_bounds import check_arithmetic_rule, derive

from merlin.common import mlir_query as mq


def program(k=3):
    return f"""builtin.module {{
      func.func @forward(%x: tensor<2x{k}xi8>, %w: tensor<{k}x2xi8>) -> (tensor<2x2xi32>, tensor<2x2xi8>) {{
        %z = arith.constant 0 : i32
        %initial = tensor.splat %z : tensor<2x2xi32>
        %wide = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->(m,k)>,
          affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
          iterator_types = ["parallel", "parallel", "reduction"]}}
          ins(%x, %w : tensor<2x{k}xi8>, tensor<{k}x2xi8>) outs(%initial : tensor<2x2xi32>) {{
          ^bb0(%a: i8, %b: i8, %acc: i32):
            %a32 = arith.extsi %a : i8 to i32
            %b32 = arith.extsi %b : i8 to i32
            %p = arith.muli %a32, %b32 : i32
            %sum = arith.addi %acc, %p : i32
            linalg.yield %sum : i32
          }} -> tensor<2x2xi32>
        %empty = tensor.empty() : tensor<2x2xi8>
        %state = linalg.generic {{indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%wide : tensor<2x2xi32>) outs(%empty : tensor<2x2xi8>) {{
          ^bb1(%value: i32, %unused: i8):
            %half = arith.constant 4 : i32
            %divisor = arith.constant 8 : i32
            %low = arith.constant -128 : i32
            %high = arith.constant 127 : i32
            %rounded = arith.addi %value, %half : i32
            %shifted = arith.floordivsi %rounded, %divisor : i32
            %lower = arith.maxsi %shifted, %low : i32
            %clipped = arith.minsi %lower, %high : i32
            %out = arith.trunci %clipped : i32 to i8
            linalg.yield %out : i8
          }} -> tensor<2x2xi8>
        func.return %wide, %state : tensor<2x2xi32>, tensor<2x2xi8>
      }}
    }}"""


def test_complete_connected_prefixes_and_half_up_saturated_readout():
    proof = derive(mq.parse(program()), [(-3, 3), (-3, 3)])
    assert proof["status"] == "source_arithmetic_proven"
    assert proof["output_intervals"] == [[-27, 27], [-3, 3]]
    reduction = next(r for r in proof["operations"] if r.get("kind") == "multiply_accumulate")
    assert reduction["prefix_interval"] == [-27, 27]
    assert reduction["product_intervals"] == [[-9, 9]]
    assert reduction["operand_bits"] == [8, 8]
    assert proof["target_arithmetic_status"].startswith("UNVERIFIED")
    rng = np.random.default_rng(6029)
    for _ in range(30):
        x = rng.integers(-3, 4, (2, 3), dtype=np.int8)
        w = rng.integers(-3, 4, (3, 2), dtype=np.int8)
        wide = x.astype(np.int64) @ w.astype(np.int64)
        state = np.clip((wide + 4) // 8, -128, 127).astype(np.int8)
        for value, (low, high) in zip([wide, state], proof["output_intervals"], strict=True):
            assert np.all(value >= low) and np.all(value <= high)


def test_source_i32_is_not_silently_replaced_by_twenty_bit_full_k():
    proof = derive(mq.parse(program(k=1024)), [(-128, 127), (-128, 127)])
    reduction = next(r for r in proof["operations"] if r.get("kind") == "multiply_accumulate")
    assert reduction["result_bits"] == 32
    assert reduction["prefix_interval"][1] == 16777216
    assert "UNVERIFIED" in proof["target_arithmetic_status"]
    rule = dict(
        operand_bits=8,
        accumulator_bits=32,
        primitive_mac_bits=20,
        max_primitive_k=16,
        accumulation="exact_i32_before_readout",
    )
    checked = check_arithmetic_rule(proof, rule)
    assert checked[0]["primitive_prefix_interval"] == [-260096, 262144]
    assert checked[0]["full_prefix_interval"][1] == 16777216
    with pytest.raises(ValueError, match="overflow signed i20"):
        check_arithmetic_rule(proof, {**rule, "max_primitive_k": 1024})
    with pytest.raises(ValueError, match="overflow signed i32"):
        derive(mq.parse(program(k=200000)), [(-128, 127), (-128, 127)])


def test_unsupported_recurrence_undefined_storage_and_input_domain_refuse():
    with pytest.raises(ValueError, match="additive recurrence"):
        derive(mq.parse(program().replace("%sum = arith.addi %acc, %p", "%sum = arith.muli %acc, %p")), [(-3, 3)] * 2)
    with pytest.raises(ValueError, match="undefined storage"):
        derive(
            mq.parse(
                program().replace("%out = arith.trunci %clipped : i32 to i8", "%out = arith.addi %unused, %unused : i8")
            ),
            [(-3, 3)] * 2,
        )
    for domain in [[(3, -3), (-3, 3)], [(-129, 127), (-3, 3)], [(-3, 3)]]:
        with pytest.raises(ValueError, match="overflow|ownership"):
            derive(mq.parse(program()), domain)
