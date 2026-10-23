"""Only uniform or identity behavior over the complete Boolean domain is admitted."""

import pytest
from mlir_fixtures import assert_unchanged, forward, generic, returned
from xdsl.dialects import arith, cf
from xdsl.dialects.builtin import IntegerAttr, UnregisteredOp, i1

from merlin.llvmlower.image_folds import fold_boolean_domain


def source(operator, constant, extension="extui"):
    body = f"%wide=arith.{extension} %a : i1 to i64\n%v=arith.{operator} %wide,%b : i64"
    computation = generic([("%x", "tensor<4xi1>"), ("%s", "tensor<4xi64>")], "tensor<4xi64>", body)
    prelude = f"%c=arith.constant {constant} : i64\n%s=tensor.splat %c : tensor<4xi64>\n"
    return forward("%x:tensor<4xi1>", "tensor<4xi64>", prelude + computation)


@pytest.mark.parametrize(
    "operator,constant,kind,expected",
    [
        ("andi", 3, "identity", None),
        ("andi", 0, "fill", 0),
        ("ori", 3, "fill", 3),
        ("ori", 0, "identity", None),
        ("xori", 0, "identity", None),
        ("andi", -1, "identity", None),
    ],
)
def test_literal_integer_operations_fold_to_identity_widening_or_uniform_fill(operator, constant, kind, expected):
    module = source(operator, constant)
    assert fold_boolean_domain(module) == 1
    module.verify()
    assert not any(op.name == "arith." + operator for op in module.walk())
    if kind == "identity":
        assert any(op.name == "arith.extui" for op in module.walk())
    else:
        assert returned(module).owner.name == "tensor.splat"
        assert returned(module).owner.operands[0].owner.value.value.data == expected
    assert fold_boolean_domain(module) == 0


# A literal zero divisor keeps its undefined operation.
@pytest.mark.parametrize(
    "operator,constant,extension",
    [("xori", 1, "extui"), ("andi", 3, "extsi"), ("ori", 2, "extui"), ("floordivsi", 0, "extui")],
)
def test_nonuniform_nonidentity_table_is_not_synthesized(operator, constant, extension):
    assert_unchanged(source(operator, constant, extension), fold_boolean_domain)


def test_runtime_self_product_or_self_or_cannot_hide_their_construction_class():
    for operator in ("muli", "ori", "andi", "xori"):
        module = source(operator, 3)
        generic = next(op for op in module.walk() if op.name == "linalg.generic")
        arithmetic = next(op for op in generic.body.block.ops if op.name == "arith." + operator)
        arithmetic.operands = [arithmetic.operands[0], arithmetic.operands[0]]
        assert_unchanged(module, fold_boolean_domain)


def test_unknown_input_carrier_cannot_be_erased_by_a_uniform_boolean_result():
    module = source("ori", 3)
    function = next(op for op in module.walk() if op.name == "func.func")
    original_input = function.body.block.args[0]
    unknown = UnregisteredOp.with_name("opaque.payload").create(
        operands=[original_input], result_types=[original_input.type]
    )
    function.body.block.insert_op_before(unknown, function.body.block.first_op)
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    generic.operands[0] = unknown.results[0]
    assert_unchanged(module, fold_boolean_domain)


@pytest.mark.parametrize("operator,divisor", [("floordivsi", 2), ("divsi", -2)])
def test_complete_boolean_integer_quotient_is_a_shape_only_zero_fill(operator, divisor):
    module = source(operator, divisor)
    assert fold_boolean_domain(module) == 1
    module.verify()


@pytest.mark.parametrize("condition,folded", [(0, False), (1, True)])
def test_assertions_are_removed_only_when_true_for_the_entire_domain(condition, folded):
    module = source("floordivsi", 2)
    block = next(op for op in module.walk() if op.name == "linalg.generic").body.block
    constant = arith.ConstantOp(IntegerAttr(condition, i1))
    assertion = cf.AssertOp(constant.result, "division defined")
    block.insert_ops_before([constant, assertion], block.first_op)
    original = str(module)
    assert bool(fold_boolean_domain(module)) == folded
    if not folded:
        assert str(module) == original
