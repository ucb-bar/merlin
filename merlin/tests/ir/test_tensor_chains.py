"""Private tensor grouping refuses observable intermediates and effect barriers."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects.lowering.tensor_chains import (
    destination_only_tensor_call,
    private_tensor_chain,
    private_tensor_pairs,
    pure_tensor_compute,
    replace_private_tensor_chain,
)


def source(*, barrier="", outputs="%r", result_type="tensor<137xf32>"):
    def multiply(result, a, b):
        return f'''%e{result} = tensor.empty() : tensor<137xf32>
        %{result} = linalg.generic {{indexing_maps = [affine_map<(i)->(i)>,
          affine_map<(i)->(i)>,affine_map<(i)->(i)>], iterator_types = ["parallel"]}}
          ins(%{a},%{b}:tensor<137xf32>,tensor<137xf32>) outs(%e{result}:tensor<137xf32>) {{
          ^bb0(%x:f32,%y:f32,%z:f32):
            %v=arith.mulf %x,%y:f32
            linalg.yield %v:f32
          }} -> tensor<137xf32>'''
    return f'''builtin.module {{
      func.func private @effect()
      func.func @forward(%a:tensor<137xf32>,%b:tensor<137xf32>,%c:tensor<137xf32>)
        -> {result_type} {{
        {multiply('m','a','b')}
        {barrier}
        {multiply('r','m','c')}
        func.return {outputs}: {result_type.removeprefix('(').removesuffix(')')}
      }}
    }}'''


def candidates(module):
    return [op for op in module.walk() if op.name == "linalg.generic"]


def test_pair_analysis_is_structural_and_does_not_mutate_input():
    module = parse_mlir_text(source(barrier="%unused=arith.constant 7:i32"))
    before = str(module)
    assert len(private_tensor_pairs(candidates(module))) == 1
    assert str(module) == before
    module.verify()


@pytest.mark.parametrize("barrier", ["func.call @effect():()->()",
                                    "%n=memref.alloc():memref<4xf32>"])
def test_unknown_effects_and_memory_operations_are_barriers(barrier):
    module = parse_mlir_text(source(barrier=barrier))
    assert not private_tensor_pairs(candidates(module))


def test_live_output_cannot_be_hidden():
    module = parse_mlir_text(source(outputs="%m,%r", result_type="(tensor<137xf32>,tensor<137xf32>)"))
    module.verify()
    assert not private_tensor_pairs(candidates(module))


def test_repeated_consumer_operand_is_fanout():
    module = parse_mlir_text(source())
    first, second = candidates(module)
    second.operands = [first.results[0], first.results[0], second.outputs[0]]
    module.verify()
    assert not private_tensor_pairs((first, second))


def test_unknown_call_is_not_a_destination_only_call():
    module = parse_mlir_text(source(barrier="func.call @effect():()->()"))
    call = next(op for op in module.walk() if op.name == "func.call")
    assert not destination_only_tensor_call(call)
    assert not pure_tensor_compute(call)


@pytest.mark.parametrize("access,shared,expected", [("read", False, 1),
                                                 ("write", False, 0),
                                                 ("read", True, 0)])
def test_crossing_call_requires_read_inputs_and_private_destination(access, shared, expected):
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Region

    module = parse_mlir_text(source())
    first, second = candidates(module)
    block = first.parent
    tensor_type = first.results[0].type
    empty = tensor.EmptyOp((), tensor_type)
    call = func.CallOp("projection", [block.args[2], empty.results[0]], [tensor_type])
    block.insert_ops_before([empty, call], second)
    attrs = ArrayAttr([DictionaryAttr({"bufferization.access": StringAttr(access)}),
                       DictionaryAttr({"bufferization.access": StringAttr("write")})])
    module.body.block.add_op(func.FuncOp("projection", ([tensor_type, tensor_type], [tensor_type]),
                                        Region(), visibility="private", arg_attrs=attrs))
    if shared:
        block.insert_op_before(func.CallOp("projection", [block.args[2], empty.results[0]], [tensor_type]), second)
    module.verify()
    assert not private_tensor_pairs((first, second))
    selected = private_tensor_pairs((first, second), can_cross=lambda op:
                                   pure_tensor_compute(op) or destination_only_tensor_call(op))
    assert len(selected) == expected


def three_operations(**kwargs):
    module = parse_mlir_text(source(**kwargs))
    first, second = candidates(module)
    last = second.clone()
    last.operands = [second.results[0], first.inputs[0], second.outputs[0]]
    result = second.next_op
    result.operands = [last.results[0]]
    second.parent.insert_op_before(last, result)
    module.verify()
    return module, (first, second, last)


def test_three_operation_endpoint_preserves_public_result_and_input_order():
    module, chain = three_operations()
    inputs = tuple(chain[0].parent.args)
    assert private_tensor_chain(chain)
    call = replace_private_tensor_chain(module, chain, symbol="connected", inputs=inputs[::-1])
    assert tuple(call.arguments[:-1]) == inputs[::-1]
    assert not candidates(module)
    assert call.next_op.arguments[0] is call.results[0]
    assert destination_only_tensor_call(call)
    module.verify()


@pytest.mark.parametrize("mutation", ["effect", "fanout", "initializer", "inputs", "collision"])
def test_three_operation_refusal_does_not_modify_the_module(mutation):
    from xdsl.dialects import func
    from xdsl.ir import Region

    module, chain = three_operations(barrier="func.call @effect():()->()" if mutation == "effect" else "")
    inputs = tuple(chain[0].parent.args)
    if mutation == "fanout":
        chain[-1].operands = [chain[1].results[0], chain[0].results[0], chain[-1].outputs[0]]
    elif mutation == "initializer":
        arithmetic = next(iter(chain[0].body.block.ops))
        arithmetic.operands = [chain[0].body.block.args[0], chain[0].body.block.args[-1]]
    elif mutation == "inputs":
        inputs = inputs[:-1]
    elif mutation == "collision":
        module.body.block.add_op(func.FuncOp("connected", ([], []), Region(), visibility="private"))
    module.verify()
    before = str(module)
    with pytest.raises(ValueError):
        replace_private_tensor_chain(module, chain, symbol="connected", inputs=inputs)
    assert str(module) == before


def test_chain_order_and_module_identity_are_checked():
    module, chain = three_operations()
    assert not private_tensor_chain(chain[::-1])
    assert not private_tensor_chain((chain[0], chain[0]))
    other, _ = three_operations()
    with pytest.raises(ValueError, match="different module"):
        replace_private_tensor_chain(other, chain, symbol="connected", inputs=tuple(chain[0].parent.args))
    module.verify()
