"""Preparation reuse is explicit, immutable, bounded, and effect-aware."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, TensorType, i8

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects.lowering.shared_preparation import (
    TensorPreparation,
    share_tensor_preparations,
)


def source(*, barrier="", second="%view", access="read"):
    return f'''builtin.module {{
      func.func private @effect()
      func.func private @consume(tensor<2x16xf32> {{bufferization.access="{access}"}},
        tensor<2x16xf32> {{bufferization.access="write"}}) -> tensor<2x16xf32>
      func.func @forward(%a:tensor<2x16xf32>,%b:tensor<2x16xf32>)
        -> (tensor<2x16xf32>,tensor<2x16xf32>) {{
        %e0=tensor.empty():tensor<2x16xf32>
        %r0=func.call @consume(%a,%e0):(tensor<2x16xf32>,tensor<2x16xf32>)->tensor<2x16xf32>
        {barrier}
        %flat=tensor.collapse_shape %a [[0,1]]:tensor<2x16xf32> into tensor<32xf32>
        %view=tensor.expand_shape %flat [[0,1]] output_shape [2,16]
          :tensor<32xf32> into tensor<2x16xf32>
        %e1=tensor.empty():tensor<2x16xf32>
        %r1=func.call @consume({second},%e1):(tensor<2x16xf32>,tensor<2x16xf32>)->tensor<2x16xf32>
        func.return %r0,%r1:tensor<2x16xf32>,tensor<2x16xf32>
      }}
    }}'''


def select(call):
    if call.callee.root_reference.data == "consume":
        return TensorPreparation(0, "pack", TensorType(i8, (48,)), "prepared_consume", (0, 1))
    return None


def parse(source_text, *, access="read"):
    module = parse_mlir_text(source_text)
    declaration = next(o for o in module.walk()
                       if o.name == "func.func" and o.sym_name.data == "consume")
    # The frontend's upstream syntax normalizer strips printed argument attrs.
    # Set the typed external contract, as device routing does after parsing.
    declaration.properties["arg_attrs"] = ArrayAttr([
        DictionaryAttr({"bufferization.access": StringAttr(access)}),
        DictionaryAttr({"bufferization.access": StringAttr("write")}),
    ])
    return module


def test_shared_views_keep_consumers_order_and_live_outputs():
    module = parse(source())
    before = str(module)
    result, groups = share_tensor_preparations(module, select, max_live_bytes=48)
    assert str(module) == before
    assert len(groups) == 1 and groups[0]["removed_preparations"] == 1
    calls = [o for o in result.walk() if o.name == "func.call"]
    assert [c.callee.root_reference.data for c in calls] == ["pack", "prepared_consume", "prepared_consume"]
    assert calls[1].arguments[0] is calls[2].arguments[0] is calls[0].results[0]
    result.verify()


@pytest.mark.parametrize("kwargs", [
    {"barrier": "func.call @effect():()->()"}, {"second": "%b"},
    {"second": "%r0"}, {"access": "write"},
])
def test_barriers_mutation_and_different_contents_refuse(kwargs):
    result, groups = share_tensor_preparations(parse(source(**kwargs), access=kwargs.get("access", "read")), select,
                                               max_live_bytes=4096)
    assert not groups
    assert sum(o.name == "func.call" and o.callee.root_reference.data == "consume"
               for o in result.walk()) == 2


def test_byte_budget_retains_original_execution():
    _, groups = share_tensor_preparations(parse(source()), select, max_live_bytes=47)
    assert not groups


@pytest.mark.parametrize("size", [0, -1, True])
def test_invalid_budget_refuses(size):
    with pytest.raises(ValueError, match="positive"):
        share_tensor_preparations(parse(source()), select, max_live_bytes=size)


def test_representation_is_part_of_group_identity():
    count = 0

    def different(call):
        nonlocal count
        count += 1
        spec = select(call)
        return TensorPreparation(0, "pack" + str(count), spec.prepared_type,
                                 spec.consumer_symbol, spec.arguments)

    _, groups = share_tensor_preparations(parse(source()), different, max_live_bytes=4096)
    assert not groups


def test_existing_symbol_collision_refuses_without_mutating_source():
    module = parse(source().replace("@effect()", "@pack()"))
    before = str(module)
    with pytest.raises(ValueError, match="collides"):
        share_tensor_preparations(module, select, max_live_bytes=4096)
    assert str(module) == before


@pytest.mark.parametrize("overlap,budget,expected", [(True, 48, 1), (True, 96, 2), (False, 48, 2)])
def test_budget_counts_overlapping_lifetimes(overlap, budget, expected):
    from xdsl.dialects import func, tensor

    module = parse(source())
    calls = [o for o in module.walk() if o.name == "func.call"]
    block = calls[0].parent
    typ = block.args[1].type
    for position in (calls[1] if overlap else block.last_op, block.last_op):
        empty = tensor.EmptyOp((), typ)
        call = func.CallOp("consume", [block.args[1], empty.results[0]], [typ])
        block.insert_ops_before([empty, call], position)
    result, groups = share_tensor_preparations(module, select, max_live_bytes=budget)
    assert len(groups) == expected
    result.verify()


def test_shared_destination_is_not_private():
    from xdsl.dialects import func

    module = parse(source())
    call = next(o for o in module.walk() if o.name == "func.call")
    block = call.parent
    block.insert_op_before(func.CallOp("consume", list(call.arguments), [call.results[0].type]), call)
    _, groups = share_tensor_preparations(module, select, max_live_bytes=4096)
    assert not groups
