"""Batch preparation reuse follows exact source coordinates, never sample equality."""

import numpy as np
import pytest
from xdsl.dialects import builtin, func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.repeated_batch_input import (
    RepeatedBatchConsumer,
    private_repeated_batch_inputs,
    prove_repeated_batch_input,
    rewrite_repeated_batch_consumers,
    validate_repeated_batch_input,
)


def fixture(groups=3, repeat=4, rows=5, cols=7, transpose=False, wrong_order=False, arithmetic=False):
    batch = groups * repeat
    source = f"tensor<1x{groups}x1x{rows}x{cols}xf32>"
    expanded = f"tensor<1x{groups}x{repeat}x{rows}x{cols}xf32>"
    out_rows, out_cols = (cols, rows) if transpose else (rows, cols)
    logical = f"tensor<{batch}x{out_rows}x{out_cols}xf32>"
    before = "%expanded"
    view = ""
    if transpose or wrong_order:
        permutation = "[0, 2, 1, 3, 4]" if wrong_order else "[0, 1, 2, 4, 3]"
        reshaped = (f"tensor<1x{repeat}x{groups}x{rows}x{cols}xf32>" if wrong_order else
                    f"tensor<1x{groups}x{repeat}x{cols}x{rows}xf32>")
        view = f"""%init = tensor.empty() : {reshaped}
        %permuted = linalg.transpose ins(%expanded : {expanded}) outs(%init : {reshaped}) permutation = {permutation}
        """
        before, expanded = "%permuted", reshaped
    body = "%sum = arith.addf %x, %x : f32\n linalg.yield %sum : f32" if arithmetic else "linalg.yield %x : f32"
    module = parse_mlir_text(f"""builtin.module {{
      func.func @ordinary(%input: {source}, %other: {logical}) -> {logical} {{
        %empty = tensor.empty() : tensor<1x{groups}x{repeat}x{rows}x{cols}xf32>
        %expanded = linalg.generic {{indexing_maps = [affine_map<(a,b,c,d,e)->(a,b,0,d,e)>,
          affine_map<(a,b,c,d,e)->(a,b,c,d,e)>],
          iterator_types = ["parallel","parallel","parallel","parallel","parallel"]}}
          ins(%input : {source}) outs(%empty : tensor<1x{groups}x{repeat}x{rows}x{cols}xf32>) {{
          ^bb0(%x: f32, %unused: f32):
          {body}
        }} -> tensor<1x{groups}x{repeat}x{rows}x{cols}xf32>
        {view}
        %flat = tensor.collapse_shape {before} [[0,1,2,3,4]] : {expanded} into tensor<{batch * rows * cols}xf32>
        %value = tensor.expand_shape %flat [[0,1,2]] output_shape [{batch},{out_rows},{out_cols}]
          : tensor<{batch * rows * cols}xf32> into {logical}
        %destination = tensor.empty() : {logical}
        %result = func.call @reader(%value, %destination) : ({logical}, {logical}) -> {logical}
        func.return %result : {logical}
      }}
      func.func private @reader({logical}, {logical}) -> {logical}
    }}""")
    declaration = next(op for op in module.body.block.ops if op.sym_name.data == "reader")
    declaration.properties["arg_attrs"] = builtin.ArrayAttr([
        builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr("read")}),
        builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr("write")}),
    ])
    module.verify()
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    return module, call


@pytest.mark.parametrize("groups,repeat,rows,cols", [(2, 3, 5, 7), (4, 8, 64, 160), (5, 2, 17, 3)])
@pytest.mark.parametrize("transpose", [False, True])
def test_complete_independent_coordinate_and_bit_repetition(groups, repeat, rows, cols, transpose):
    module, call = fixture(groups, repeat, rows, cols, transpose)
    proof = prove_repeated_batch_input(call.arguments[0])
    assert proof.repeat == repeat and proof.shape[0] == groups * repeat
    validate_repeated_batch_input(proof)
    # Independent bit arrays include signed-zero/NaN words. No FP arithmetic is used.
    words = (np.arange(groups * rows * cols, dtype=np.uint32) * np.uint32(2654435761)).reshape(1, groups, 1, rows, cols)
    words.reshape(-1)[:6] = [0, 0x80000000, 0x7FC12345, 0x7F812345, 0x7F800000, 0xFF800000]
    broadcast = np.broadcast_to(words, (1, groups, repeat, rows, cols))
    if transpose:
        broadcast = broadcast.transpose(0, 1, 2, 4, 3)
    matrices = broadcast.reshape(proof.shape)
    for batch in range(groups * repeat):
        assert matrices[batch].tobytes() == matrices[batch // repeat * repeat].tobytes()


@pytest.mark.parametrize("change", ["arithmetic", "batch_permutation", "unproved_root", "context", "yield", "library"])
def test_unsupported_or_mutated_sources_refuse(change):
    if change in {"arithmetic", "batch_permutation"}:
        module, call = fixture(arithmetic=change == "arithmetic", wrong_order=change == "batch_permutation")
        with pytest.raises(ValueError):
            prove_repeated_batch_input(call.arguments[0])
        return
    module, call = fixture()
    if change == "unproved_root":
        with pytest.raises(ValueError):
            prove_repeated_batch_input(call.parent.args[1])
        return
    proof = prove_repeated_batch_input(call.arguments[0])
    if change == "context":
        module.attributes["numeric.policy"] = builtin.StringAttr("changed")
    elif change == "yield":
        copy = next(op for op in module.walk() if op.name == "linalg.generic")
        copy.body.block.last_op.operands = (copy.body.block.args[1],)
    else:
        copy = next(op for op in module.walk() if op.name == "linalg.generic")
        copy.properties["library_call"] = builtin.StringAttr("opaque_copy")
    with pytest.raises(ValueError, match="changed"):
        validate_repeated_batch_input(proof)


def test_retarget_preserves_operands_results_and_numeric_attributes():
    module, call = fixture(transpose=True)
    original = next(op for op in module.body.block.ops if op.sym_name.data == "reader")
    original.attributes["numeric.policy"] = builtin.StringAttr("unchanged")
    operands, results, typ = tuple(call.arguments), tuple(call.results), original.function_type
    reports = rewrite_repeated_batch_consumers(module, [RepeatedBatchConsumer("reader", "retained_reader", 0, 4)])
    assert reports[0]["input_proofs"][0]["contiguous_batch_repeat"] == 4
    assert tuple(call.arguments) == operands and tuple(call.results) == results
    selected = next(op for op in module.body.block.ops if op.sym_name.data == "retained_reader")
    assert selected.function_type == typ
    assert selected.attributes["numeric.policy"] == original.attributes["numeric.policy"]
    assert selected.arg_attrs == original.arg_attrs and call.callee.root_reference.data == "retained_reader"


@pytest.mark.parametrize("change", ["repeat", "access", "collision", "missing", "argument"])
def test_invalid_selection_is_atomic(change):
    module, call = fixture()
    original = next(op for op in module.body.block.ops if op.sym_name.data == "reader")
    if change == "access":
        original.properties["arg_attrs"] = builtin.ArrayAttr([
            builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr("write")}),
            original.arg_attrs.data[1],
        ])
    contract = RepeatedBatchConsumer("missing" if change == "missing" else "reader",
                                     "reader" if change == "collision" else "retained_reader",
                                     2 if change == "argument" else 0, 3 if change == "repeat" else 4)
    before = str(module)
    with pytest.raises(ValueError):
        rewrite_repeated_batch_consumers(module, [contract])
    assert str(module) == before


def test_one_unsupported_call_prevents_all_selected_mutation():
    module, call = fixture()
    declaration = next(op for op in module.body.block.ops if op.sym_name.data == "reader")
    unsupported = declaration.clone()
    unsupported.properties["sym_name"] = builtin.StringAttr("unproved")
    module.body.block.add_op(unsupported)
    other = func.CallOp("unproved", [call.parent.args[1], call.arguments[1]], [call.results[0].type])
    call.parent.insert_op_before(other, call)
    before = str(module)
    contracts = [RepeatedBatchConsumer("reader", "retained", 0, 4),
                 RepeatedBatchConsumer("unproved", "unproved_retained", 0, 4)]
    with pytest.raises(ValueError):
        rewrite_repeated_batch_consumers(module, contracts)
    assert str(module) == before


def test_explicit_calls_preserve_unproved_users_of_a_shared_declaration():
    module, call = fixture()
    unknown = func.CallOp("reader", [call.parent.args[1], call.arguments[1]], [call.results[0].type])
    call.parent.insert_op_before(unknown, call)
    reports = rewrite_repeated_batch_consumers(module, [RepeatedBatchConsumer("reader", "retained", 0, 4, (call,))])
    assert reports[0]["calls"] == 1
    assert call.callee.root_reference.data == "retained" and unknown.callee.root_reference.data == "reader"


def test_overlapping_explicit_call_selections_refuse_atomically():
    module, call = fixture()
    before = str(module)
    with pytest.raises(ValueError, match="overlap"):
        rewrite_repeated_batch_consumers(module, [RepeatedBatchConsumer("reader", "first", 0, 4, (call,)),
                                                  RepeatedBatchConsumer("reader", "second", 0, 4, (call,))])
    assert str(module) == before


def test_shared_declaration_can_have_different_proved_repetition_groups():
    module, first = fixture(groups=3, repeat=4)
    other_module, _ = fixture(groups=4, repeat=3)
    original_block = first.parent
    other_function = next(op for op in other_module.body.block.ops if op.sym_name.data == "ordinary")
    other_block = other_function.body.block
    new_arg = original_block.insert_arg(other_block.args[0].type, len(original_block.args))
    function = original_block.parent_op()
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        [value.type for value in original_block.args], function.function_type.outputs)
    mapper = {other_block.args[0]: new_arg, other_block.args[1]: original_block.args[1]}
    for operation in other_block.ops:
        if isinstance(operation, func.ReturnOp):
            break
        cloned = operation.clone(value_mapper=mapper)
        original_block.insert_op_before(cloned, first)
        if isinstance(cloned, func.CallOp):
            second = cloned
    reports = rewrite_repeated_batch_consumers(module, [
        RepeatedBatchConsumer("reader", "repeat_four", 0, 4, (first,)),
        RepeatedBatchConsumer("reader", "repeat_three", 0, 3, (second,)),
    ])
    assert [row["input_proofs"][0]["contiguous_batch_repeat"] for row in reports] == [4, 3]
    assert first.callee.root_reference.data == "repeat_four" and second.callee.root_reference.data == "repeat_three"


@pytest.mark.parametrize("rows,cols", [(5, 7), (7, 5), (5, 5)])
@pytest.mark.parametrize("transpose", [False, True])
def test_compact_input_reconstructs_every_original_raw_word_and_preserves_other_users(rows, cols, transpose):
    from xdsl.dialects import tensor
    from xdsl.dialects.linalg.ops import TransposeOp

    groups, repeat = 3, 4
    module, call = fixture(groups=groups, repeat=repeat, rows=rows, cols=cols, transpose=transpose)
    original = call.arguments[0]
    other = call.clone()
    call.parent.insert_op_before(other, call)
    inputs = call.parent.args[0]
    words = np.arange(groups * rows * cols, dtype=np.uint32).reshape(inputs.type.get_shape())
    words.reshape(-1)[:6] = [0, 0x80000000, 0x7FC12345, 0x7F812345, 0x7F800000, 0xFF800000]

    def evaluate(value):
        if value is inputs:
            return words
        operation = value.owner
        if isinstance(operation, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
            return evaluate(operation.operands[0]).reshape(value.type.get_shape())
        assert isinstance(operation, TransposeOp)
        return evaluate(operation.inputs[0]).transpose(tuple(operation.permutation.get_values()))

    reports = rewrite_repeated_batch_consumers(module, [
        RepeatedBatchConsumer("reader", "compact_reader", 0, repeat, (call,), compact=True)])
    compact = evaluate(call.arguments[0])
    expected = np.broadcast_to(words, (1, groups, repeat, rows, cols))
    if transpose:
        expected = expected.transpose(0, 1, 2, 4, 3)
    assert np.repeat(compact, repeat, axis=0).tobytes() == expected.tobytes()
    assert compact.size == groups * rows * cols and call.arguments[0] is not original
    assert other.arguments[0] is original and other.callee.root_reference.data == "reader"
    assert reports[0]["compact_input"] and reports[0]["input_proofs"][0]["matrix_transposed"] == transpose
    assert tuple(call.results[0].type.get_shape()) == (groups * repeat, *compact.shape[-2:])
    module.verify()


def test_compact_binding_rejects_unproved_input_before_any_mutation():
    module, call = fixture()
    arguments = list(call.arguments)
    arguments[0] = call.parent.args[1]
    call.operands = arguments
    before = str(module)
    with pytest.raises(ValueError, match="no proved"):
        rewrite_repeated_batch_consumers(module, [
            RepeatedBatchConsumer("reader", "compact_reader", 0, 4, compact=True)])
    assert str(module) == before


def test_transpose_orientation_is_part_of_the_immutable_proof():
    from dataclasses import replace

    _, call = fixture(rows=5, cols=5, transpose=True)
    proof = prove_repeated_batch_input(call.arguments[0])
    assert proof.matrix_transposed
    with pytest.raises(ValueError, match="coordinates changed"):
        validate_repeated_batch_input(replace(proof, matrix_transposed=False))


@pytest.mark.parametrize("observer", ["none", "source", "expanded"])
def test_private_compaction_distinguishes_source_observation_from_live_expansion(observer):
    module, call = fixture(transpose=True)
    proof = prove_repeated_batch_input(call.arguments[0])
    if observer != "none":
        function = call.parent.parent_op()
        returned = call.parent.last_op
        value = proof.source if observer == "source" else proof.value
        returned.operands = [*returned.operands, value]
        function.properties["function_type"] = builtin.FunctionType.from_lists(
            function.function_type.inputs, [operand.type for operand in returned.operands])
    module.verify()
    proof = prove_repeated_batch_input(call.arguments[0])
    before = str(module)
    selected = private_repeated_batch_inputs({(call, 0): proof})
    assert bool(selected) == (observer != "expanded")
    assert str(module) == before


@pytest.mark.parametrize("both", [False, True])
def test_shared_expansion_requires_the_complete_selected_consumer_set(both):
    _, first = fixture()
    second = first.clone()
    first.parent.insert_op_before(second, first)
    proof = prove_repeated_batch_input(first.arguments[0])
    uses = {(first, 0): proof}
    if both:
        uses[(second, 0)] = proof
    assert private_repeated_batch_inputs(uses) == (frozenset(uses) if both else frozenset())


def test_private_compaction_rechecks_sibling_views_to_a_fixed_point():
    from xdsl.dialects import tensor

    module, first = fixture()
    logical = first.arguments[0]
    group = builtin.ArrayAttr([builtin.ArrayAttr([builtin.IntegerAttr(axis, builtin.i64) for axis in range(3)])])
    flat = tensor.CollapseShapeOp.create(
        operands=[logical], result_types=[builtin.TensorType(builtin.f32, [int(np.prod(logical.type.get_shape()))])],
        properties={"reassociation": group})
    first.parent.insert_op_before(flat, first)
    view = tensor.ExpandShapeOp(flat.results[0], (), group, logical.type.get_shape(), logical.type)
    first.parent.insert_op_before(view, first)
    second = func.CallOp("reader", [view.results[0], first.arguments[1]], [first.results[0].type])
    first.parent.insert_op_before(second, first)
    returned = first.parent.last_op
    returned.operands = [*returned.operands, view.results[0]]
    function = first.parent.parent_op()
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        function.function_type.inputs, [value.type for value in returned.operands])
    module.verify()
    uses = {(call, 0): prove_repeated_batch_input(call.arguments[0]) for call in [first, second]}
    assert private_repeated_batch_inputs(uses) == frozenset()


def test_private_compaction_refuses_changed_source_uses_and_wrong_argument_binding():
    _, call = fixture()
    proof = prove_repeated_batch_input(call.arguments[0])
    with pytest.raises(ValueError, match="original input use"):
        private_repeated_batch_inputs({(call, 1): proof})
    other = call.clone()
    call.parent.insert_op_before(other, call)
    with pytest.raises(ValueError, match="changed"):
        private_repeated_batch_inputs({(call, 0): proof})
