"""Consumer-owned view reconstruction and independent numeric transport."""

from copy import deepcopy

import numpy as np
import pytest
from test_execution_model_correspondence import selection as selection
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.frontends.execution_call_contracts import CallContracts
from merlin.frontends.execution_model_correspondence import checked_execution_model
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import toolchain
from merlin.llvmlower.owned_results import compile_owned_results


def test_declared_views_preserve_the_complete_original_expansion(selection):
    parent, bound, trace, interfaces = selection
    trace["execution_view_binding_profile"] = "t2m.execution-checked-call-views.v1"
    trace["calls"][0]["tensor_views"] = "dense-row-major-v1"
    assert checked_execution_model(parent, bound, trace, interfaces)["calls"] == 1


@pytest.mark.parametrize("damage", ("missing-profile", "unknown-profile", "unknown-policy", "unaccounted-view"))
def test_view_profile_or_coordinate_declarations_cannot_replace_source_proof(selection, damage):
    parent, bound, trace, interfaces = selection
    trace = deepcopy(trace)
    trace["execution_view_binding_profile"] = "t2m.execution-checked-call-views.v1"
    trace["calls"][0]["tensor_views"] = "dense-row-major-v1"
    if damage == "missing-profile":
        trace.pop("execution_view_binding_profile")
    elif damage == "unknown-profile":
        trace["execution_view_binding_profile"] = "unchecked"
    elif damage == "unknown-policy":
        trace["calls"][0]["tensor_views"] = "physical-alias"
    else:
        trace["calls"][0]["source_tensor_views"] = {"invented": {}}
    with pytest.raises(ValueError):
        checked_execution_model(parent, bound, trace, interfaces)


@pytest.mark.skipif(not toolchain.available(), reason="existing native compiler required")
@pytest.mark.parametrize("shape,target", [((2, 3), (3, 2)), ((2, 0, 3), (0, 6)), ((), (1,))])
@pytest.mark.parametrize("dtype", ("i64", "f32"))
def test_independent_dense_adapters_keep_exact_words_and_empty_scalar_ranks(tmp_path, shape, target, dtype):
    from merlin.frontends.execution_call_views import reshape

    element = builtin.i64 if dtype == "i64" else builtin.f32
    before, after = (builtin.TensorType(element, dimensions) for dimensions in (shape, target))
    block = Block(arg_types=[before])

    def append(op, role):
        block.add_op(op)
        return op.results[0] if op.results else None

    changed = reshape(block.args[0], after, append, role="checked-view")
    block.add_op(func.ReturnOp(changed))
    module = builtin.ModuleOp([func.FuncOp("forward", ([before], [after]), Region(block))])
    module.verify()
    from io import StringIO

    from xdsl.printer import Printer

    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    parse_mlir_text(stream.getvalue()).verify()
    program = compile_owned_results(stream.getvalue(), workdir=tmp_path / "native", timeout=30)
    count = int(np.prod(shape))
    words = (
        np.asarray([2**53 + 1, -(2**53) - 1, (1 << 63) - 1], np.int64)
        if dtype == "i64"
        else np.asarray([0x7F812345, 0xFFC05678, 0x80000000, 1], np.uint32).view(np.float32)
    )
    x = np.resize(words, count).reshape(shape)
    snapshot = x.tobytes()
    (actual,), observation = program.execute((x,))
    assert (actual.shape, actual.dtype, actual.tobytes()) == (target, x.dtype, snapshot)
    assert x.tobytes() == snapshot and observation["source_calls"] == 1


def test_live_dimensions_do_not_gain_a_captured_dense_reshape():
    value = Block(arg_types=[builtin.TensorType(builtin.i32, [2, 3])]).args[0]
    contracts = CallContracts(lambda *args: pytest.fail("emitted before the missing-coordinate refusal"))
    with pytest.raises(ValueError, match="source-bound concrete"):
        contracts.view(
            value,
            builtin.TensorType(builtin.i32, [-1]),
            {"shape": ["n"]},
            {"tensor_views": "dense-row-major-v1", "runtime_scalar_arguments": {}},
            builtin.DictionaryAttr({}),
            role="caller-to-kernel-view",
        )
