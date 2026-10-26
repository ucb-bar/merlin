"""SDK-free preparation regressions: independently materialized actual after IR.

No test here invokes Lean or claims positive/negative kernel acceptance. Valid
wrong programs are retained as different obligations for the future real audit.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.verify.lean_graph_input_admission import InputAdmissionError
from merlin.verify.lean_graph_loops_shape_after import prepare_pair
from merlin.verify.lean_graph_loops_shape_after_projection import normalized_code
from merlin.verify.lean_graph_loops_shape_projection import Decoder


def parser(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


def render(module):
    output = io.StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue()


def paths(name):
    configured = os.environ.get("MERLIN_GRAPH_SHAPE_AFTER_CASES")
    if not configured:
        pytest.skip("select the retained actual producer cases explicitly")
    case = Path(configured) / name
    return case / "source.mlir", case / "shape/after.mlir"


def prepare(before, after, bits=64):
    return prepare_pair(before, after, parser=parser, index_bits=bits, max_nodes=5000)


def permuted(value, swaps):
    """Diagnostic term comparison only; this does not replace Lean validation."""
    if isinstance(value, list):
        return [permuted(row, swaps) for row in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"name", "kind", "known"}:
        name = value["name"]
        for left, right in reversed(swaps):
            name = right if name == left else left if name == right else name
        return dict(value, name=name)
    return {key: permuted(row, swaps) for key, row in value.items()}


def syntax_equal(pair):
    source = pair["before"]
    expected = dict(
        source,
        functions=[dict(function, code=normalized_code(function["code"], {})) for function in source["functions"]],
    )
    return pair["beforeFrame"] == pair["afterFrame"] and expected == permuted(pair["after"], pair["swaps"])


@pytest.mark.parametrize(
    "name,bits",
    (
        ("two_epochs", 64),
        ("tail_nonunit", 64),
        ("index32", 32),
        ("single_row", 64),
        ("element16", 64),
        ("large_symbolic", 64),
    ),
)
def test_actual_pair_has_two_separate_program_terms(name, bits, tmp_path):
    before, after = paths(name)
    result = prepare(before.read_text(), after.read_text(), bits)
    (tmp_path / "Obligation.lean").write_text(result["query"])
    (tmp_path / "PREPARATION.json").write_text(json.dumps({k: v for k, v in result.items() if k != "query"}, indent=2))
    assert "def actualBefore : Module :=" in result["query"]
    assert "def actualAfter : Module :=" in result["query"]
    assert "After.permuteModule actual.swaps actualAfter = Epoch.normalizedModule actualBefore" in result["query"]
    assert result["model"]["before"] != result["model"]["after"]
    assert syntax_equal(result["model"])
    assert not any(result[key] for key in ("lean_invoked", "lean_accepted", "qualified", "pipeline_receipt_issued"))


def test_old_full_tree_matcher_is_not_used(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("old source/target full-tree matcher was invoked")

    for name in ("project", "block", "operands"):
        monkeypatch.setattr(Decoder, name, forbidden)
    before, after = paths("two_epochs")
    result = prepare(before.read_text(), after.read_text())
    assert result["independent_after_term"] and not result["model"]["host_pair_match_used"]


MUTATIONS = ("dimension", "scalar", "return", "observer", "initializer", "control", "buffer")


def mutate(module, name):
    functions = [op for op in module.walk() if isinstance(op, func.FuncOp)]
    kernel, observer = functions[:2]
    block = kernel.body.block
    constants = [op for op in block.ops if isinstance(op, arith.ConstantOp)]
    if name == "dimension":
        constants[-1].properties["value"] = builtin.IntegerAttr.from_index_int_value(7)
    elif name == "scalar":
        operation = next(op for op in block.ops if isinstance(op, arith.AddiOp))
        operation.operands = (operation.operands[0], block.args[1])
    elif name == "return":
        operation = next(op for op in block.ops if isinstance(op, func.ReturnOp))
        operation.operands = (block.args[0], operation.operands[1])
    elif name == "observer":
        operation = next(op for op in observer.walk() if isinstance(op, arith.AddiOp))
        operation.operands = (operation.operands[0], operation.operands[0])
    elif name == "initializer":
        loop = next(op for op in block.ops if isinstance(op, scf.ForOp))
        store = next(op for op in loop.walk() if isinstance(op, memref.StoreOp))
        store.operands = (*store.operands[:2], constants[0].result, store.operands[-1])
    elif name == "control":
        loop = next(op for op in block.ops if isinstance(op, scf.ForOp))
        loop.operands = (loop.operands[0], constants[0].result, *loop.operands[2:])
    elif name == "buffer":
        allocations = [op for op in block.ops if isinstance(op, memref.AllocOp)]
        loads = [op for op in block.ops if isinstance(op, memref.LoadOp)]
        loads[-1].operands = (allocations[0].memref, *loads[-1].operands[1:])
    else:
        raise AssertionError(name)
    module.verify()


@pytest.mark.parametrize("name", MUTATIONS)
def test_valid_wrong_after_reaches_a_distinct_lean_obligation(name, tmp_path):
    before_path, after_path = paths("two_epochs")
    before, after = before_path.read_text(), after_path.read_text()
    good = prepare(before, after)
    module = parser(after)
    mutate(module, name)
    changed = render(module)
    bad = prepare(before, changed)
    (tmp_path / "wrong-after.mlir").write_text(changed)
    (tmp_path / "WrongObligation.lean").write_text(bad["query"])
    assert bad["model"]["before"] == good["model"]["before"]
    assert bad["model"]["after"] != good["model"]["after"]
    assert bad["query"] != good["query"]
    assert not syntax_equal(bad["model"])
    # Native checking is held. Do not mislabel a different query as a rejection.
    assert not bad["lean_invoked"] and not bad["lean_accepted"]


def test_materialized_after_decode_is_independent_of_source():
    before, after = paths("two_epochs")
    first = prepare(before.read_text(), after.read_text())
    source = parser(before.read_text())
    mutate(source, "scalar")
    second = prepare(render(source), after.read_text())
    assert first["model"]["before"] != second["model"]["before"]
    assert first["model"]["after"] == second["model"]["after"]


def test_resource_metadata_refused_before_parser():
    calls = []

    def forbidden(text):
        calls.append(text)
        raise AssertionError("resource-bearing original reached parsing")

    with pytest.raises(InputAdmissionError, match="file-metadata"):
        prepare_pair(
            "builtin.module {}\n{-# dialect_resources: {} #-}",
            "builtin.module {}",
            parser=forbidden,
            index_bits=64,
            max_nodes=50,
        )
    assert not calls


def test_function_value_route_is_not_admitted_as_opaque_action():
    source = """builtin.module {
      func.func @callee(%x: i32) -> i32 { func.return %x : i32 }
      func.func @caller(%x: i32) -> i32 {
        %f = "func.constant"() {value = @callee} : () -> ((i32) -> i32)
        %y = "func.call_indirect"(%f, %x) : ((i32) -> i32, i32) -> i32
        func.return %y : i32
      }
    }"""
    with pytest.raises(InputAdmissionError, match="function-value"):
        prepare(source, source)


def test_swaps_are_bijective_even_when_retained_source_ids_have_gaps():
    before, after = paths("two_epochs")
    result = prepare(before.read_text(), after.read_text())
    swaps = result["model"]["swaps"]
    assert swaps
    names = set(range(100))
    for left, right in reversed(swaps):
        names = {right if name == left else left if name == right else name for name in names}
    assert names == set(range(100))


def test_direct_call_table_is_in_each_actual_program():
    before, after = paths("two_epochs")
    result = prepare(before.read_text(), after.read_text())
    assert result["model"]["before"]["calls"] == result["model"]["after"]["calls"]
    assert any(target == "touch" for _, target in result["model"]["after"]["calls"])


def test_valid_wrong_direct_callee_is_emitted_and_fails_pair_syntax(tmp_path):
    before, after = paths("two_epochs")
    modules = [parser(path.read_text()) for path in (before, after)]
    for module in modules:
        observer = next(
            op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "touch"
        )
        alternative = observer.clone()
        alternative.properties["sym_name"] = builtin.StringAttr("touch_other")
        operation = next(op for op in alternative.walk() if isinstance(op, arith.AddiOp))
        operation.operands = (operation.operands[0], operation.operands[0])
        module.body.block.add_op(alternative)
        module.verify()
    good = prepare(*(render(module) for module in modules))
    assert syntax_equal(good["model"])
    call = next(op for op in modules[1].walk() if isinstance(op, func.CallOp))
    call.properties["callee"] = builtin.SymbolRefAttr("touch_other")
    modules[1].verify()
    bad = prepare(*(render(module) for module in modules))
    (tmp_path / "wrong-callee.mlir").write_text(render(modules[1]))
    (tmp_path / "WrongCallee.lean").write_text(bad["query"])
    assert not syntax_equal(bad["model"])
    assert bad["model"]["after"]["calls"] != good["model"]["after"]["calls"]
    assert not bad["lean_invoked"]


def test_changed_candidate_function_frame_is_a_kernel_obligation(tmp_path):
    before, after = paths("two_epochs")
    module = parser(after.read_text())
    function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    function.attributes["review.marker"] = builtin.StringAttr("changed")
    module.verify()
    result = prepare(before.read_text(), render(module))
    (tmp_path / "ChangedFrame.lean").write_text(result["query"])
    assert result["model"]["beforeFrame"] != result["model"]["afterFrame"]
    assert not syntax_equal(result["model"])
    assert "review.marker" in result["query"]


@pytest.mark.parametrize(
    "name",
    (
        "test_actual_whole_module_rank_0",
        "test_actual_whole_module_rank_1",
        "test_actual_whole_module_rank_2",
        "test_actual_whole_module_neste0",
        "test_actual_whole_module_zero_0",
        "test_actual_whole_module_calle0",
        "test_actual_whole_module_compa0",
    ),
)
def test_retained_actual_shape_families_have_independent_terms(name, tmp_path):
    selected = os.environ.get("MERLIN_GRAPH_SHAPE_AFTER_STANDALONE")
    if not selected:
        pytest.skip("select the retained actual standalone producer cases explicitly")
    case = Path(selected) / name / "proof"
    result = prepare((case / "before.mlir").read_text(), (case / "after.mlir").read_text())
    (tmp_path / "Obligation.lean").write_text(result["query"])
    assert syntax_equal(result["model"])
    assert result["model"]["before"] != result["model"]["after"]
    assert not result["lean_invoked"]


def test_declared_external_call_is_bound_in_both_terms():
    before = """builtin.module {
      func.func private @opaque(index) -> index
      func.func @entry(%buffer: memref<3xi32>) -> index {
        %zero = arith.constant 0 : index
        %size = memref.dim %buffer, %zero : memref<3xi32>
        %result = func.call @opaque(%size) : (index) -> index
        func.return %result : index
      }
    }"""
    after = before.replace("memref.dim %buffer, %zero : memref<3xi32>", "arith.constant 3 : index")
    result = prepare(before, after)
    assert syntax_equal(result["model"])
    assert len(result["model"]["after"]["functions"]) == 1
    assert result["model"]["after"]["calls"][0][1] == "opaque"


def test_valid_changed_region_yield_is_independently_bound(tmp_path):
    # A finite carried value checks the complete body, captures, yielded value
    # and loop result together. This pair is a hand-written regression, not a
    # new producer/qualification case.
    before = """builtin.module {
      func.func @entry(%buffer: memref<3xi32>, %x: i32, %y: i32) -> i32 {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %size = memref.dim %buffer, %zero : memref<3xi32>
        %result = scf.for %i = %zero to %size step %one iter_args(%carry = %x) -> (i32) {
          scf.yield %carry : i32
        }
        func.return %result : i32
      }
    }"""
    after = before.replace("memref.dim %buffer, %zero : memref<3xi32>", "arith.constant 3 : index")
    good = prepare(before, after)
    assert syntax_equal(good["model"])
    changed = after.replace("scf.yield %carry", "scf.yield %y")
    parser(changed).verify()
    bad = prepare(before, changed)
    (tmp_path / "wrong-yield.mlir").write_text(changed)
    (tmp_path / "WrongYield.lean").write_text(bad["query"])
    assert not syntax_equal(bad["model"])
    assert bad["model"]["after"] != good["model"]["after"]
