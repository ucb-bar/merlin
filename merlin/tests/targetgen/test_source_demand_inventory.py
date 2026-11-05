"""Routing census omissions cannot change complete source-operation indices."""

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen import coverage_certificate as CC
from merlin.targetgen.capsule_source import model_op_demands_checked
from merlin.targetgen.routing import route_plan_on
from merlin.targetgen.source_demand_inventory import indexed_demand_operations

SOURCE = '''"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<2xi32>) -> tensor<2xi32>}> ({
  ^bb0(%x: tensor<2xi32>):
    %bound = "arith.constant"() <{value = -128 : i32}>
      {prov.op = "fill", prov.region_id = "bound"} : () -> i32
    %bounds = "tensor.splat"(%bound) {prov.op = "fill", prov.region_id = "bound"}
      : (i32) -> tensor<2xi32>
    %zero = "arith.constant"() <{value = 0 : i32}>
      {prov.op = "seed", prov.region_id = "seed"} : () -> i32
    %out = "tensor.splat"(%zero) {prov.op = "seed", prov.region_id = "seed"}
      : (i32) -> tensor<2xi32>
    "func.return"(%out) : (tensor<2xi32>) -> ()
  }) : () -> ()
}) : () -> ()'''


def _ops(source):
    module = mq.parse(source)
    function = next(op for op in module.body.block.ops if mq.op_name(op) == "func.func")
    return [op for op in function.body.block.ops if mq.op_name(op) != "func.return"]


def test_fill_initializers_do_not_renumber_routing_demands_or_erase_precision():
    indexed = indexed_demand_operations(_ops(SOURCE))
    assert [index for index, _ in indexed] == [2, 3]
    demands = model_op_demands_checked(SOURCE, "int8")
    assert [demand.carrier_op for demand in demands] == ["arith.constant", "tensor.splat"]
    certificate = CC.build(route_plan_on(demands, []), {}, linalg_mlir=SOURCE)
    assert certificate["n_regions"] == 2
    assert certificate["n_unknown_capture_formats"] == 0
    assert certificate["n_precision_transform_obligations"] == 0
    assert certificate["precision_transform_verification"]["status"] == "not_required"
    assert certificate["regions"][1]["source_precision_witness"]["result_types"] == ["tensor<2xi32>"]


def test_fill_tag_cannot_hide_arbitrary_arithmetic_from_the_join():
    source = SOURCE.replace(
        '%bounds = "tensor.splat"(%bound)', '%bounds = "test.hidden_arithmetic"(%bound)'
    )
    with pytest.raises(ValueError, match="unsupported omitted fill carrier"):
        indexed_demand_operations(_ops(source))
    demands = model_op_demands_checked(source, "int8")
    certificate = CC.build(route_plan_on(demands, []), {}, linalg_mlir=source)
    assert certificate["n_unknown_capture_formats"] > 0


def test_untagged_operation_stays_visible_for_completeness_checks():
    source = SOURCE.replace('prov.op = "seed", ', "", 1)
    indexed = indexed_demand_operations(_ops(source))
    assert [index for index, _ in indexed] == [2, 3]
    assert len(model_op_demands_checked(source, "int8")) == 1
    certificate = CC.build(route_plan_on(model_op_demands_checked(source, "int8"), []), {}, linalg_mlir=source)
    assert certificate["n_unknown_capture_formats"] == 1
