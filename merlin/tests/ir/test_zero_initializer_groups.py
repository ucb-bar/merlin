"""Closed source seed grouping and refusals, with exact SSA custody."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i32
from xdsl.dialects.linalg.ops import GenericOp

from merlin.compile.scheduling.zero_initializer_groups import find_zero_initializer_groups
from merlin.frontends.linalg_mlir import parse_mlir_text


def source(m=17, k=16, n=32, *, zero=0, returned_init=False, shared_constant=False, gap=False):
    result = f"tensor<{m}x{n}xi32>"
    extra = f'%extra = "tensor.splat"(%zero) : (i32) -> {result}' if shared_constant else ""
    extra_name = "%init" if returned_init else "%extra" if shared_constant else None
    returns = f"({result}, {result})" if extra_name else result
    values = f"%r, {extra_name} : {result}, {result}" if extra_name else f"%r : {result}"
    separation = "%separation = arith.constant 19 : i32" if gap else ""
    return parse_mlir_text(f'''module {{
      func.func @forward(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> {returns} {{
        %zero = arith.constant {zero} : i32
        %init = "tensor.splat"(%zero) : (i32) -> {result}
        {separation}
        %r = linalg.generic {{
          indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>,
                           affine_map<(m,n,k)->(m,n)>],
          iterator_types = ["parallel", "parallel", "reduction"]}}
          ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%init : {result}) {{
        ^bb0(%x: i8, %y: i8, %s: i32):
          %xx = arith.extsi %x : i8 to i32
          %yy = arith.extsi %y : i8 to i32
          %p = arith.muli %xx, %yy : i32
          %v = arith.addi %s, %p : i32
          linalg.yield %v : i32
        }} -> {result}
        {extra}
        return {values}
      }}
    }}''')


@pytest.mark.parametrize("shape", [(2, 3, 5), (17, 16, 32), (1, 257, 17), (31, 17, 33)])
def test_exact_source_group_keeps_all_operations_and_original_uses(shape):
    module = source(*shape)
    before = str(module)
    uses = {value: tuple(value.uses) for op in module.walk() for value in (*op.operands, *op.results)}
    (group,) = find_zero_initializer_groups(module)
    assert group.source_op_indices == (0, 1, 2)
    assert group.virtual_result_positions == ((0, 0), (1, 0))
    assert group.geometry_m_k_n == shape
    assert group.initializer_elements == shape[0] * shape[2]
    assert group.initializer_bytes == shape[0] * shape[2] * 4
    group.verify_current(module)
    assert str(module) == before
    assert all(tuple(value.uses) == original for value, original in uses.items())
    assert "original final execution checks required" in group.record()["authority"]


@pytest.mark.parametrize("options", [{"zero": 1}, {"returned_init": True}, {"gap": True}])
def test_nonzero_live_or_noncontiguous_initializer_retains_original_route(options):
    module = source(**options)
    before = str(module)
    assert find_zero_initializer_groups(module) == ()
    assert str(module) == before


def test_shared_scalar_constant_stays_in_its_original_task():
    module = source(shared_constant=True)
    (group,) = find_zero_initializer_groups(module)
    assert group.source_op_indices == (1, 2)
    assert group.virtual_result_positions == ((1, 0),)
    group.verify_current(module)


@pytest.mark.parametrize("unknown_attribute", [False, True])
def test_frontend_generic_subclass_preserves_the_complete_standard_shell(unknown_attribute):
    class FrontendGeneric(GenericOp):
        pass

    module = source()
    operation = next(op for op in module.walk() if type(op) is GenericOp)
    operation.__class__ = FrontendGeneric
    if unknown_attribute:
        operation.attributes["unknown_effect"] = StringAttr("write")
    before = str(module)
    uses = {value: tuple(value.uses) for op in module.walk() for value in (*op.operands, *op.results)}
    assert bool(find_zero_initializer_groups(module)) is not unknown_attribute
    assert str(module) == before
    assert all(tuple(value.uses) == original for value, original in uses.items())


@pytest.mark.parametrize("change", ["unknown_splat_effect", "unknown_constant_effect", "unsigned_product"])
def test_unknown_effects_or_changed_signed_arithmetic_cannot_group(change):
    module = source()
    if change == "unknown_splat_effect":
        next(op for op in module.walk() if op.name == "tensor.splat").attributes["effect"] = StringAttr("write")
    elif change == "unknown_constant_effect":
        next(op for op in module.walk() if op.name == "arith.constant").attributes["effect"] = StringAttr("write")
    else:
        text = str(module).replace("arith.extsi", "arith.extui")
        module = parse_mlir_text(text)
    assert find_zero_initializer_groups(module) == ()


@pytest.mark.parametrize("change", ["constant", "metadata", "record"])
def test_changed_source_or_forged_group_requires_replanning(change):
    module = source()
    (group,) = find_zero_initializer_groups(module)
    if change == "constant":
        next(op for op in module.walk() if op.name == "arith.constant").properties["value"] = IntegerAttr(1, i32)
    elif change == "metadata":
        next(op for op in module.walk() if op.name == "linalg.generic").attributes["prov.region_id"] = StringAttr("new")
    else:
        group = replace(group, initializer_bytes=0)
    with pytest.raises(ValueError, match="complete current typed source"):
        group.verify_current(module)


def test_original_explicit_budgets_and_non_model_source_remain_bounded():
    module = source()
    with pytest.raises(ValueError, match="operation budget"):
        find_zero_initializer_groups(module, max_source_operations=2)
    with pytest.raises(ValueError, match="positive source budgets"):
        find_zero_initializer_groups(module, max_source_elements=0)
    assert find_zero_initializer_groups(module, max_source_elements=1) == ()
    assert find_zero_initializer_groups(parse_mlir_text("module {}")) == ()
