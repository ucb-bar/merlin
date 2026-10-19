"""Private tensor grouping refuses observable intermediates and effect barriers."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects.lowering.tensor_chains import (
    destination_only_tensor_call,
    private_tensor_pairs,
    pure_tensor_compute,
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
