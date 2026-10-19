"""Batch preparation reuse follows exact source coordinates, never sample equality."""

import numpy as np
import pytest
from xdsl.dialects import builtin, func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.repeated_batch_input import (
    RepeatedBatchConsumer,
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
