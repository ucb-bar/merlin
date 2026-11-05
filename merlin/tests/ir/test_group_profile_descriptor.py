"""Descriptor source ABI observations never reconstruct checked effects."""

import json
from dataclasses import asdict

import pytest
from test_descriptor_result_identity import selection, source
from xdsl.dialects import arith, func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.descriptor_result_identity import rewrite_descriptor_result_identities
from merlin.model_check.chain_image_profile_descriptor import _functions, _source_binding, completed_descriptor_bindings
from merlin.runtime.whole_model_readback import ReadbackError, callees, render_group_profile


def observed_pair():
    original = parse_mlir_text(source())
    identity = selection(original)
    selected = original.clone()
    rewrite_descriptor_result_identities(selected, (identity,))
    row = dict(symbol=identity.symbol, borrowed_symbol=identity.borrowed_symbol,
               wrapper_symbol=identity.wrapper_symbol, result_argument=identity.result_argument,
               argument_types=list(map(str, identity.argument_types)),
               argument_access=list(identity.argument_access), borrowed_effects=asdict(identity.borrowed_effects))
    return row, _functions(original), _functions(selected)


def test_observed_descriptor_profile_uses_complete_void_pointer_abi_without_effect_owner():
    row, original, selected = observed_pair()
    observation, binding = _source_binding(row, original, selected)
    assert binding == dict(borrowed_symbol="borrowed", buffers=3, argument_ranks=[2, 2, 2],
                           wrapper_symbol="writer_tensor")
    assert not hasattr(observation, "borrowed_effects")
    actual = callees([dict(symbol="writer", group=1)], {"writer": [2, 7, 7]},
                     {"_mlir_ciface_borrowed"}, writer_bindings={"writer": binding})
    generated = render_group_profile(actual, [("g1_writer", "writer")])
    assert "void __wrap__mlir_ciface_borrowed(void *arg_0, void *arg_1, void *arg_2)" in generated
    assert "__real__mlir_ciface_borrowed(arg_0, arg_1, arg_2);" in generated
    assert "void *result" not in generated
    assert json.loads(json.dumps(binding)) == binding


@pytest.mark.parametrize("fault", ["result", "arity", "access", "layout", "symbol", "extra", "call", "external"])
def test_changed_descriptor_source_or_abi_cannot_select_a_profiling_prototype(fault):
    row, original, selected = observed_pair()
    wrapper = selected[row["wrapper_symbol"]]
    if fault == "result":
        row["result_argument"] = 0
    elif fault == "arity":
        row["argument_types"].pop()
    elif fault == "access":
        row["argument_access"][0] = "write"
    elif fault == "layout":
        row["argument_layouts"] = ["unexpected"] * 3
    elif fault == "symbol":
        row["borrowed_symbol"] = "borrowed); injected"
    elif fault == "extra":
        wrapper.body.block.insert_op_before(arith.ConstantOp.from_int_and_width(1, 32),
                                            wrapper.body.block.last_op)
    elif fault == "call":
        call = next(op for op in wrapper.walk() if isinstance(op, func.CallOp))
        call.operands = tuple(reversed(tuple(call.operands)))
    else:
        selected[row["borrowed_symbol"]].attributes.clear()
    with pytest.raises(ReadbackError):
        _source_binding(row, original, selected)


def test_unselected_completed_program_has_no_descriptor_profile_inputs():
    assert completed_descriptor_bindings({"receipt": {"output": {}}}, {}, set()) == ({}, [])
