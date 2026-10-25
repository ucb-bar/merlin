"""Actual SCF segmentation, complete SSA/carry/control checks and valid mutations."""

from __future__ import annotations

import io
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.verify.lean_graph_loops import LeanSelection, _reference
from merlin.verify.lean_graph_loops_segment import SEGMENT_ROOTS, SegmentSelection, verify_segment_files


@pytest.fixture(scope="module")
def context():
    selected = os.environ.get("MERLIN_GRAPH_LOOP_SEGMENT_SELECTION")
    cases = os.environ.get("MERLIN_GRAPH_LOOP_SEGMENT_CASES")
    if not selected or not cases:
        pytest.skip("explicit SCF model/producer/case selection unavailable")
    data = json.loads(Path(selected).read_bytes())
    for key in ("executable", "library", "build_record"):
        data["proof"][key] = Path(data["proof"][key])
    data["proof"] = LeanSelection(**data["proof"])
    for key in ("source", "producer_manifest"):
        data[key] = Path(data[key])
    root = Path(cases)
    plan = {row["name"]: row for row in json.loads((root / "CASE_PLAN.json").read_bytes())}
    return SegmentSelection(**data), root, plan


def parse(text):
    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, arith.Arith, scf.Scf, memref.MemRef):
        ctx.load_dialect(dialect)
    module = Parser(ctx, text).parse_module()
    module.verify()
    return module


def text(module):
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def check(
    context, tmp_path, *, name="lower_stride_tail", before=None, after=None, selection=None, bits=None, max_nodes=1000
):
    selected, root, plan = context
    case = root / name
    source, target = case / "before.mlir", case / "after.mlir"
    if before is not None:
        source = tmp_path / "before.mlir"
        source.write_text(before)
    if after is not None:
        target = tmp_path / "after.mlir"
        target.write_text(after)
    return verify_segment_files(
        source,
        target,
        selection=selection or selected,
        output=tmp_path / "proof",
        tile=plan[name]["tile"],
        index_bits=bits or plan[name]["bits"],
        max_nodes=max_nodes,
    )


@pytest.mark.parametrize(
    "name", ["simple", "lower_stride_tail", "empty", "multiple", "effects", "no_carry_effects", "large"]
)
def test_independent_actual_producer_domains_are_kernel_checked(context, tmp_path, name):
    manifest = json.loads((context[1] / name / "CAPTURE.json").read_bytes())
    assert manifest["tile"] == context[2][name]["tile"] and manifest["index_bits"] == context[2][name]["bits"]
    result = check(context, tmp_path, name=name)
    assert result["lean_complete_relation_checked"]
    assert len(result["transitive_axioms"]) == len(SEGMENT_ROOTS)
    assert not result["untrusted_witness_used"] and not result["source_success_inferred"]
    assert result["body_and_context_interpretation_explicit"] and result["python_projection_trusted"]
    assert not any(
        result[key]
        for key in (
            "pipeline_receipt_issued",
            "numerical_gates_discharged",
            "SCF_countedKernel_bridge_proved",
            "renderer_or_hardware_proved",
        )
    )


def test_large_trip_count_uses_constant_structural_ir_and_parametric_proof(context, tmp_path):
    small = check(context, tmp_path / "small", name="lower_stride_tail")
    large = check(context, tmp_path / "large", name="large")
    assert small["lean_complete_relation_checked"] and large["lean_complete_relation_checked"]
    assert large["projection"]["bounds"]["upper"] == 1_000_000_000_003
    assert small["projection"]["structural_nodes"] == large["projection"]["structural_nodes"]
    assert abs(small["query"]["bytes"] - large["query"]["bytes"]) < 1024
    assert large["query"]["bytes"] < 20_000


def mutate(module, kind):
    loops = [op for op in module.walk() if isinstance(op, scf.ForOp)]
    outer, inner = loops
    add, minimum, _inner, outer_yield = tuple(outer.body.block.ops)
    if kind == "wrong_outer_stride":
        outer.operands = (*outer.operands[:2], inner.step, *outer.operands[3:])
    elif kind == "wrong_inner_stride":
        inner.operands = (*inner.operands[:2], outer.step, *inner.operands[3:])
    elif kind == "omit_tail":
        minimum.operands = (minimum.lhs, outer.step)
    elif kind == "repeat_tail":
        minimum.operands = (outer.ub, outer.ub)
    elif kind == "wrong_inner_lower":
        inner.operands = (outer.lb, *inner.operands[1:])
    elif kind == "swap_inner_initial":
        inner.operands = (*inner.operands[:3], inner.operands[4], inner.operands[3])
    elif kind == "swap_outer_yield":
        outer_yield.operands = tuple(reversed(outer_yield.operands))
    elif kind == "swap_inner_yield":
        last = inner.body.block.last_op
        last.operands = tuple(reversed(last.operands))
    elif kind == "swap_capture":
        first = inner.body.block.first_op
        function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
        first.operands = (first.operands[0], function.body.block.args[1])
    elif kind == "changed_suffix_observer":
        observer = next(
            op for op in module.walk() if isinstance(op, arith.AddiOp) and op.operands[0] is outer.results[0]
        )
        observer.operands = (outer.results[1], observer.operands[1])
    elif kind == "changed_untouched_function":
        other = [op for op in module.walk() if isinstance(op, func.FuncOp)][1]
        constant = next(op for op in other.walk() if isinstance(op, arith.ConstantOp))
        constant.properties["value"] = arith.ConstantOp.from_int_and_width(2, 32).value
    elif kind == "unknown_control_attribute":
        add.attributes["unmodeled"] = builtin.UnitAttr()
    else:
        raise AssertionError(kind)
    module.verify()


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_outer_stride",
        "wrong_inner_stride",
        "omit_tail",
        "repeat_tail",
        "wrong_inner_lower",
        "swap_inner_initial",
        "swap_outer_yield",
        "swap_inner_yield",
        "swap_capture",
        "changed_suffix_observer",
        "changed_untouched_function",
        "unknown_control_attribute",
    ],
)
def test_valid_ir_mutations_refuse_complete_actual_boundary(context, tmp_path, kind):
    module = parse((context[1] / "lower_stride_tail/after.mlir").read_text())
    mutate(module, kind)
    candidate = text(module)
    parse(candidate).verify()
    try:
        result = check(context, tmp_path, after=candidate)
    except ValueError:
        refusal = json.loads((tmp_path / "proof/REFUSAL.json").read_bytes())
        assert not refusal["lean_complete_relation_checked"]
    else:
        assert result["returncode"] != 0 and not result["lean_complete_relation_checked"]


def test_changed_ssa_names_preserve_actual_bindings(context, tmp_path):
    module = parse((context[1] / "lower_stride_tail/after.mlir").read_text())
    count = 0
    for op in module.walk():
        for result in op.results:
            result.name_hint = "renamed" + str(count)
            count += 1
        for region in op.regions:
            for block in region.blocks:
                for argument in block.args:
                    argument.name_hint = "argument" + str(count)
                    count += 1
    module.verify()
    assert check(context, tmp_path, after=text(module))["lean_complete_relation_checked"]


def test_untrusted_witness_hint_cannot_replace_the_actual_ir(context, tmp_path):
    (tmp_path / "witnesses.json").write_text(json.dumps([{"lower": 9999, "tile_iterations": 1, "verified": True}]))
    result = check(context, tmp_path)
    assert result["lean_complete_relation_checked"] and not result["untrusted_witness_used"]


def test_complete_node_budget_cannot_accept_a_good_prefix(context, tmp_path):
    with pytest.raises(ValueError, match="no prefix accepted"):
        check(context, tmp_path, max_nodes=1)


def test_producer_and_kernel_both_refuse_signed_index_overflow(context, tmp_path):
    records = json.loads((context[1] / "PRODUCTION.json").read_bytes())
    refused = next(row for row in records if row["case"]["name"] == "overflow")
    assert refused["expected_source_overflow_refused"] and refused["returncode"] != 0
    case = context[1] / "lower_stride_tail"
    before, after = parse((case / "before.mlir").read_text()), parse((case / "after.mlir").read_text())
    for module in (before, after):
        loop = next(op for op in module.walk() if isinstance(op, scf.ForOp))
        loop.ub.owner.properties["value"] = arith.ConstantOp.from_int_and_width(2**31 - 2, builtin.IndexType()).value
        module.verify()
    result = check(context, tmp_path, before=text(before), after=text(after), bits=32)
    assert result["returncode"] != 0 and not result["lean_complete_relation_checked"]


@pytest.mark.parametrize("changed", ["manifest", "tool", "proof", "revision"])
def test_stale_source_parser_model_or_tool_identity_refuses(context, tmp_path, changed):
    selected = context[0]
    if changed == "manifest":
        selected = replace(selected, producer_manifest_sha256="0" * 64)
    elif changed == "tool":
        selected = replace(selected, proof=replace(selected.proof, sha256="0" * 64))
    elif changed == "proof":
        selected = replace(selected, proof=replace(selected.proof, build_sha256="0" * 64))
    else:
        selected = replace(selected, revision="0" * 40)
    with pytest.raises(ValueError):
        check(context, tmp_path, selection=selected)


@pytest.mark.parametrize("product", ["Obligation.lean", "actual-trees.json", "model.json", "before.mlir"])
def test_intervening_actual_projection_model_or_query_writer_refuses(context, tmp_path, monkeypatch, product):
    from merlin.verify import lean_graph_loops_segment as implementation

    original = implementation.subprocess.run
    query = tmp_path / "proof/Obligation.lean"

    def write_after_kernel(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]) == str(query.resolve()):
            path = tmp_path / "proof" / product
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", write_after_kernel)
    result = check(context, tmp_path)
    assert result["returncode"] == 0 and not result["lean_complete_relation_checked"]
    assert "changed during checking" in result["refusal"]


def test_compiler_invalid_ir_is_separate_from_a_valid_mutation_refusal(context, tmp_path):
    module = parse((context[1] / "lower_stride_tail/after.mlir").read_text())
    inner = [op for op in module.walk() if isinstance(op, scf.ForOp)][1]
    last = inner.body.block.last_op
    last.operands = ()
    with pytest.raises(Exception):
        module.verify()
    with pytest.raises(ValueError, match="parsing/SSA export refused"):
        check(context, tmp_path, after=text(module))


def _wrap32(value):
    return (value + 2**31) % 2**32 - 2**31


def _execute_with_trace(module, inputs, bits, memory):
    from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
    from xdsl.interpreters.arith import ArithFunctions
    from xdsl.interpreters.func import FuncFunctions
    from xdsl.interpreters.memref import MemRefFunctions
    from xdsl.interpreters.scf import ScfFunctions
    from xdsl.interpreters.shaped_array import ShapedArray

    @register_impls
    class BoundFunctions(InterpreterFunctions):
        @impl(arith.MinUIOp)
        def minimum(self, interpreter, op, args):
            assert all(type(value) is int and value >= 0 for value in args)
            return (min(args),)

    loops = [op for op in module.walk() if isinstance(op, scf.ForOp)]
    body = loops[-1].body.block
    tracked = set(body.ops)

    def value(item):
        if isinstance(item, ShapedArray):
            return ("memory", tuple(item.shape), tuple(item.data))
        assert type(item) is int
        return item

    class Listener(Interpreter.Listener):
        def __init__(self):
            self.trace = []
            self.entries = []
            self.interpreter = None

        def will_interpret_op(self, op, args):
            if op in tracked:
                if op is body.first_op:
                    self.entries.append(tuple(self.interpreter.get_values(body.args)))
                self.trace.append(("before", op.name, tuple(value(arg) for arg in args)))

        def did_interpret_op(self, op, results):
            if op in tracked:
                self.trace.append(
                    (
                        "after",
                        op.name,
                        tuple(value(result) for result in results),
                        None if memory is None else tuple(memory.data),
                    )
                )

    listener = Listener()
    interpreter = Interpreter(module, index_bitwidth=bits, listeners=(listener,))
    listener.interpreter = interpreter
    for implementation in (ArithFunctions(), ScfFunctions(), MemRefFunctions(), FuncFunctions(), BoundFunctions()):
        interpreter.register_implementations(implementation)
    result = interpreter.call_op("kernel", inputs)
    untouched = interpreter.call_op("untouched", (2**31 - 1,))
    assert untouched == (-(2**31),)
    return result, listener.trace, listener.entries


@pytest.mark.parametrize("name", ["simple", "lower_stride_tail", "empty", "multiple", "effects", "no_carry_effects"])
def test_finite_actual_execution_preserves_all_carry_body_events_and_memory(context, name, tmp_path):
    from xdsl.interpreters.memref import xtype_for_el_type
    from xdsl.interpreters.shaped_array import ShapedArray
    from xdsl.interpreters.utils.ptr import TypedPtr

    _selected, root, plans = context
    row = plans[name]
    before = parse((root / name / "before.mlir").read_text())
    after = parse((root / name / "after.mlir").read_text())
    records = []
    for seed, captured in ((0, -3), (2**31 - 1, 5), (-(2**31), -1)):
        arguments = tuple(_wrap32(seed + index) for index in range(row["carries"]))
        initial_memory = [-999] * (row["upper"] + 3)
        memories = [None, None]
        if row["effects"]:
            memories = [
                ShapedArray(
                    TypedPtr.new(initial_memory, xtype=xtype_for_el_type(builtin.i32, row["bits"])),
                    [len(initial_memory)],
                )
                for _ in range(2)
            ]
        reference_memory = list(initial_memory)
        for _repeat in range(2):
            carry = list(arguments)
            expected_entries = []
            for index in range(row["lower"], row["upper"], row["step"]):
                expected_entries.append((index, *carry))
                previous = captured
                updated = []
                for incoming in carry:
                    previous = _wrap32(incoming + previous)
                    updated.append(previous)
                if row["effects"]:
                    reference_memory[index] = updated[0] if updated else _wrap32(captured + captured)
                carry = updated
            expected_result = (_wrap32(carry[0] + captured), *carry[1:]) if carry else (captured,)
            actual = []
            for module, memory in zip((before, after), memories, strict=True):
                inputs = (*arguments, captured) + ((memory,) if memory is not None else ())
                actual.append(_execute_with_trace(module, inputs, row["bits"], memory))
                if memory is not None:
                    assert memory.data == reference_memory
            assert actual[0] == actual[1]
            assert actual[0][0] == expected_result and actual[0][2] == expected_entries
            records.append(
                {
                    "seed": seed,
                    "capture": captured,
                    "repeat": _repeat,
                    "source": actual[0],
                    "target": actual[1],
                    "expected_result": expected_result,
                    "expected_entries": expected_entries,
                    "memory": None if memories[0] is None else list(memories[0].data),
                    "expected_memory": list(reference_memory) if row["effects"] else None,
                }
            )
    (tmp_path / "RUNTIME.json").write_text(
        json.dumps(
            {
                "schema": "merlin.graph.segmentation.finite_xdsl_trace_evidence.v1",
                "case": row,
                "before": _reference(root / name / "before.mlir"),
                "after": _reference(root / name / "after.mlir"),
                "records": records,
                "native_MLIR_or_hardware_execution": False,
                "formal_bridge_proved": False,
            },
            indent=2,
        )
        + "\n"
    )


def test_intervening_original_module_writer_refuses_even_identical_parsed_meaning(context, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops_segment as implementation

    original = implementation.subprocess.run
    query = tmp_path / "proof/Obligation.lean"
    original_target = tmp_path / "after.mlir"
    candidate = (context[1] / "lower_stride_tail/after.mlir").read_text()

    def change_original(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]) == str(query.resolve()):
            original_target.write_bytes(original_target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", change_original)
    result = check(context, tmp_path, after=candidate)
    assert result["returncode"] == 0 and not result["lean_complete_relation_checked"]
    assert "changed during checking" in result["refusal"]
