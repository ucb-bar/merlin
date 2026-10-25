"""Actual ordered unrolling: complete proofs, valid mutations and finite witnesses."""

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
from merlin.verify.lean_graph_loops_segment import SegmentSelection
from merlin.verify.lean_graph_loops_unroll import UNROLL_ROOTS, UnrollSelection, verify_unroll_files
from merlin.verify.lean_graph_loops_unroll_cases import CASES
from merlin.verify.lean_graph_loops_unroll_composition import verify_composition_files


def selection(path, kind):
    data = json.loads(Path(path).read_bytes())
    for key in ("executable", "library", "build_record"):
        data["proof"][key] = Path(data["proof"][key])
    data["proof"] = LeanSelection(**data["proof"])
    for key in ("source", "producer_manifest"):
        data[key] = Path(data[key])
    return kind(**data)


@pytest.fixture(scope="module")
def context():
    selected, cases = (os.environ.get(key) for key in ("MERLIN_GRAPH_UNROLL_SELECTION", "MERLIN_GRAPH_UNROLL_CASES"))
    if not selected or not cases:
        pytest.skip("explicit ordered-unroll proof/producer/context selection unavailable")
    return selection(selected, UnrollSelection), Path(cases), {row.name: row.record() for row in CASES}


def parse(text):
    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, scf.Scf, func.Func, memref.MemRef):
        ctx.load_dialect(dialect)
    module = Parser(ctx, text).parse_module()
    module.verify()
    return module


def text(module):
    output = io.StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue()


def check(context, tmp_path, *, name="bulk_tail", before=None, after=None, selected=None, max_nodes=1000):
    chosen, root, plan = context
    first, second = root / name / "before.mlir", root / name / "after.mlir"
    tmp_path.mkdir(parents=True, exist_ok=True)
    if before is not None:
        first = tmp_path / "before.mlir"
        first.write_text(before)
    if after is not None:
        second = tmp_path / "after.mlir"
        second.write_text(after)
    return verify_unroll_files(
        first,
        second,
        selection=selected or chosen,
        output=tmp_path / "proof",
        factor=plan[name]["factor"],
        index_bits=plan[name]["bits"],
        max_nodes=max_nodes,
    )


@pytest.mark.parametrize("name", [row.name for row in CASES])
def test_actual_complete_changed_program(context, tmp_path, name):
    result = check(context, tmp_path, name=name)
    assert result["lean_complete_relation_checked"] and result["returncode"] == 0
    assert set(result["transitive_axioms"]) == set(UNROLL_ROOTS)
    assert result["python_projection_trusted"] and not result["untrusted_witness_used"]
    assert not any(
        result[key]
        for key in (
            "pipeline_receipt_issued",
            "numerical_gates_discharged",
            "renderer_or_hardware_proved",
            "source_success_inferred",
            "SCF_countedKernel_bridge_proved",
        )
    )


def test_large_tripcount_has_no_proof_unrolling(context, tmp_path):
    small = check(context, tmp_path / "small", name="large_shape")
    large = check(context, tmp_path / "large", name="large")
    assert large["lean_complete_relation_checked"] and small["lean_complete_relation_checked"]
    assert large["projection"]["structural_nodes"] == small["projection"]["structural_nodes"]
    assert abs(large["query"]["bytes"] - small["query"]["bytes"]) < 1024
    assert large["projection"]["bounds"]["count"] > 100_000_000_000


def mutate(module, kind):
    loops = [op for op in module.walk() if isinstance(op, scf.ForOp)]
    bulk, tail = loops
    ops = list(bulk.body.block.ops)
    stores = [op for op in ops if isinstance(op, memref.StoreOp)]
    adds = [op for op in ops if isinstance(op, arith.AddiOp) and op.result.type == builtin.i32]
    offsets = [op for op in ops if isinstance(op, arith.ConstantOp)]
    if kind == "wrong_stride":
        bulk.operands = (*bulk.operands[:2], tail.step, *bulk.operands[3:])
    elif kind == "wrong_bulk_bound":
        bulk.ub.owner.properties["value"] = arith.ConstantOp.from_int_and_width(23, builtin.IndexType()).value
    elif kind == "wrong_offset":
        offsets[0].properties["value"] = arith.ConstantOp.from_int_and_width(4, builtin.IndexType()).value
    elif kind == "wrong_lane_iv":
        add = next(op for op in ops if isinstance(op, arith.AddiOp) and isinstance(op.result.type, builtin.IndexType))
        add.operands = (bulk.lb, add.operands[1])
    elif kind == "dropped_effect":
        bulk.body.block.erase_op(stores[0])
    elif kind == "duplicated_effect":
        bulk.body.block.insert_op_after(stores[0].clone(), stores[0])
    elif kind == "reordered_effect":
        bulk.body.block.detach_op(stores[0])
        bulk.body.block.insert_op_after(stores[0], stores[1])
    elif kind == "carry_chain":
        adds[2].operands = (bulk.body.block.args[1], adds[2].operands[1])
    elif kind == "source_capture":
        function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
        adds[0].operands = (adds[0].operands[0], function.body.block.args[1])
    elif kind == "bulk_initial":
        bulk.operands = (*bulk.operands[:3], bulk.operands[4], bulk.operands[3])
    elif kind == "bulk_yield":
        last = bulk.body.block.last_op
        last.operands = tuple(reversed(last.operands))
    elif kind == "tail_initial":
        tail.operands = (*tail.operands[:3], tail.operands[4], tail.operands[3])
    elif kind == "tail_yield":
        last = tail.body.block.last_op
        last.operands = tuple(reversed(last.operands))
    elif kind == "tail_step":
        tail.operands = (*tail.operands[:2], bulk.step, *tail.operands[3:])
    elif kind == "tail_lower":
        tail.operands = (bulk.lb, *tail.operands[1:])
    elif kind == "missing_tail":
        for old, new in zip(tail.results, bulk.results, strict=True):
            old.replace_all_uses_with(new)
        tail.parent.erase_op(tail)
    elif kind == "observer":
        observer = next(
            op for op in module.walk() if isinstance(op, arith.AddiOp) and op.operands[0] is tail.results[0]
        )
        observer.operands = (tail.results[1], observer.operands[1])
    elif kind == "untouched":
        other = [op for op in module.walk() if isinstance(op, func.FuncOp)][1]
        constant = next(op for op in other.walk() if isinstance(op, arith.ConstantOp))
        constant.properties["value"] = arith.ConstantOp.from_int_and_width(2, 32).value
    elif kind == "unknown_control":
        offsets[0].attributes["unmodeled"] = builtin.UnitAttr()
    else:
        raise AssertionError(kind)
    module.verify()


@pytest.mark.parametrize(
    "kind",
    (
        "wrong_stride",
        "wrong_bulk_bound",
        "wrong_offset",
        "wrong_lane_iv",
        "dropped_effect",
        "duplicated_effect",
        "reordered_effect",
        "carry_chain",
        "source_capture",
        "bulk_initial",
        "bulk_yield",
        "tail_initial",
        "tail_yield",
        "tail_step",
        "tail_lower",
        "missing_tail",
        "observer",
        "untouched",
        "unknown_control",
    ),
)
def test_type_valid_actual_mutations_refused(context, tmp_path, kind):
    module = parse((context[1] / "bulk_tail/after.mlir").read_text())
    mutate(module, kind)
    candidate = text(module)
    parse(candidate).verify()
    try:
        result = check(context, tmp_path, after=candidate)
    except ValueError:
        assert not json.loads((tmp_path / "proof/REFUSAL.json").read_bytes())["lean_complete_relation_checked"]
    else:
        assert not result["lean_complete_relation_checked"]


def test_empty_forwarding_alias_observer_is_checked(context, tmp_path):
    module = parse((context[1] / "inline_empty/after.mlir").read_text())
    observer = next(op for op in module.walk() if isinstance(op, arith.AddiOp))
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    observer.operands = (function.body.block.args[1], observer.operands[1])
    module.verify()
    with pytest.raises(ValueError):
        check(context, tmp_path, name="inline_empty", after=text(module))


def test_compiler_invalid_is_separate(context, tmp_path):
    module = parse((context[1] / "bulk_tail/after.mlir").read_text())
    bulk = next(op for op in module.walk() if isinstance(op, scf.ForOp))
    bulk.body.block.last_op.operands = ()
    with pytest.raises(Exception):
        module.verify()
    with pytest.raises(ValueError, match="parsing/SSA export refused"):
        check(context, tmp_path, after=text(module))


@pytest.mark.parametrize("changed", ("manifest", "tool", "proof", "revision"))
def test_stale_selected_identity(context, tmp_path, changed):
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
        check(context, tmp_path, selected=selected)


@pytest.mark.parametrize("product", ("Obligation.lean", "actual-trees.json", "model.json", "before.mlir"))
def test_intervening_writer_refuses(context, tmp_path, monkeypatch, product):
    from merlin.verify import lean_graph_loops_unroll as implementation

    original = implementation.subprocess.run
    query = (tmp_path / "proof/Obligation.lean").resolve()

    def writer(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]) == str(query):
            path = tmp_path / "proof" / product
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", writer)
    result = check(context, tmp_path)
    assert result["returncode"] == 0 and not result["lean_complete_relation_checked"]


def test_complete_budget_and_witness_authority(context, tmp_path):
    (tmp_path / "witnesses.json").write_text('{"verified":true,"factor":999}')
    with pytest.raises(ValueError, match="no prefix accepted"):
        check(context, tmp_path, max_nodes=1)


def test_real_two_boundary_same_scf_composition(context, tmp_path):
    segment_path = os.environ.get("MERLIN_GRAPH_LOOP_SEGMENT_SELECTION")
    after = os.environ.get("MERLIN_GRAPH_UNROLL_COMPOSITION_AFTER")
    assert segment_path and after
    root = context[1] / "bulk_exact"
    result = verify_composition_files(
        root / "before.mlir",
        root / "after.mlir",
        Path(after),
        unroll_selection=context[0],
        segment_selection=selection(segment_path, SegmentSelection),
        output=tmp_path / "join",
        factor=3,
        tile=3,
        index_bits=64,
        max_nodes=1000,
    )
    assert result["lean_composition_checked"] and result["group_body_interpretation_explicit"]
    assert result["middle"] == _reference(root / "after.mlir")
    assert not result["SCF_countedKernel_bridge_proved"]


def test_unchanged_domains_and_actual_producer_refusals(context):
    controls = Path(os.environ["MERLIN_GRAPH_UNROLL_CONTROLS"])
    records = json.loads((controls / "PRODUCTION.json").read_bytes())
    assert len(records) == 5 and all((row["returncode"] != 0) == row["expected_refusal"] for row in records)
    for name in ("factor_one", "disabled", "metadata"):
        capture = json.loads((controls / name / "CAPTURE.json").read_bytes())
        assert not capture["changed"] and not capture["lean_checked"]
        assert (controls / name / "before-generic.mlir").read_bytes() == (controls / name / "after.mlir").read_bytes()


def wrap32(value):
    return (value + 2**31) % 2**32 - 2**31


def execute(module, inputs, bits, memory):
    from xdsl.interpreter import Interpreter
    from xdsl.interpreters.arith import ArithFunctions
    from xdsl.interpreters.func import FuncFunctions
    from xdsl.interpreters.memref import MemRefFunctions
    from xdsl.interpreters.scf import ScfFunctions
    from xdsl.interpreters.shaped_array import ShapedArray

    # Original body labels survive cloning. Generated index control and context
    # observers are excluded; every original arithmetic/effect op is included.
    tracked = {
        op
        for op in module.walk()
        if isinstance(op, memref.StoreOp)
        or (
            isinstance(op, arith.AddiOp)
            and op.result.type == builtin.i32
            and (op.result.name_hint or "").startswith(("next", "stored"))
        )
    }

    def snapshot(value):
        if isinstance(value, ShapedArray):
            return ("memory", tuple(value.shape), tuple(value.data))
        assert type(value) is int
        return value

    class Listener(Interpreter.Listener):
        def __init__(self):
            self.trace = []

        def will_interpret_op(self, op, args):
            if op in tracked:
                self.trace.append(("before", op.name, tuple(snapshot(arg) for arg in args)))

        def did_interpret_op(self, op, results):
            if op in tracked:
                self.trace.append(
                    (
                        "after",
                        op.name,
                        tuple(snapshot(value) for value in results),
                        None if memory is None else tuple(memory.data),
                    )
                )

    listener = Listener()
    interpreter = Interpreter(module, index_bitwidth=bits, listeners=(listener,))
    for implementation in (ArithFunctions(), ScfFunctions(), MemRefFunctions(), FuncFunctions()):
        interpreter.register_implementations(implementation)
    result = interpreter.call_op("kernel", inputs)
    assert interpreter.call_op("untouched", (2**31 - 1,)) == (-(2**31),)
    return result, listener.trace


@pytest.mark.parametrize("name", [row.name for row in CASES if row.name != "large"])
def test_initialized_finite_actual_body_effects_and_carries(context, tmp_path, name):
    from xdsl.interpreters.memref import xtype_for_el_type
    from xdsl.interpreters.shaped_array import ShapedArray
    from xdsl.interpreters.utils.ptr import TypedPtr

    row, root = context[2][name], context[1] / name
    records = []
    for seed, captured in ((0, -3), (2**31 - 1, 5), (-(2**31), -1)):
        arguments = tuple(wrap32(seed + index) for index in range(row["carries"]))
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
        modules = [parse((root / file).read_text()) for file in ("before.mlir", "after.mlir")]
        expected_memory = list(initial_memory)
        for repeat in range(2):
            carry = list(arguments)
            for index in range(row["lower"], row["upper"], row["step"]):
                updated, previous = [], captured
                for incoming in carry:
                    previous = wrap32(incoming + previous)
                    updated.append(previous)
                if row["effects"]:
                    expected_memory[index] = updated[0] if updated else wrap32(captured + captured)
                if row["yield_mode"] == "updated":
                    carry = updated
                elif row["yield_mode"] == "capture":
                    carry = [captured] * row["carries"]
            expected = (wrap32(carry[0] + captured), *carry[1:]) if carry else (captured,)
            executions = [
                execute(module, (*arguments, captured) + ((memory,) if memory is not None else ()), row["bits"], memory)
                for module, memory in zip(modules, memories, strict=True)
            ]
            assert executions[0] == executions[1] and executions[0][0] == expected
            trips = len(range(row["lower"], row["upper"], row["step"]))
            arithmetic = 0 if row["yield_mode"] == "identity" else row["carries"] or int(row["effects"])
            assert len(executions[0][1]) == 2 * trips * (arithmetic + int(row["effects"]))
            if row["effects"]:
                assert all(memory.data == expected_memory for memory in memories)
                stores = [event[2][-1] for event in executions[0][1] if event[:2] == ("before", "memref.store")]
                assert stores == list(range(row["lower"], row["upper"], row["step"]))
            records.append(
                dict(
                    seed=seed,
                    captured=captured,
                    repeat=repeat,
                    source=executions[0],
                    target=executions[1],
                    expected_result=expected,
                    expected_memory=expected_memory if row["effects"] else None,
                )
            )
    (tmp_path / "RUNTIME.json").write_text(
        json.dumps(
            dict(
                schema="merlin.graph.unroll.finite_initialized_xdsl_evidence.v1",
                case=row,
                before=_reference(root / "before.mlir"),
                after=_reference(root / "after.mlir"),
                records=records,
                all_initial_values_explicit=True,
                native_MLIR_or_hardware_execution=False,
                formal_bridge_proved=False,
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("change", ("count", "partition", "signed_width", "clone_index"))
def test_kernel_refuses_corrupt_projection_arithmetic(context, tmp_path, monkeypatch, change):
    from merlin.verify import lean_graph_loops_unroll as implementation

    decode = implementation.project_trees

    def corrupt(*args, **kwargs):
        result = decode(*args, **kwargs)
        if change == "count":
            result["bounds"]["count"] += 1
        elif change == "partition":
            result["bounds"]["groups"] += 1
        elif change == "signed_width":
            result["bounds"]["bits"] = 32
            result["bounds"]["upper"] += 2**32
        else:
            result["clones"][1]["index"] += 1
        return result

    monkeypatch.setattr(implementation, "project_trees", corrupt)
    result = check(context, tmp_path)
    assert result["returncode"] != 0 and not result["lean_complete_relation_checked"]


def test_changed_private_compiled_object_refuses(context, tmp_path):
    selected = context[0]
    build = json.loads(selected.proof.build_record.read_bytes())
    library = tmp_path / "lib"
    for row in build["attempts"]:
        target = library / Path(*row["module"].split(".")).with_suffix(".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(row["object"]["path"]).read_bytes())
        row["object"] = _reference(target)
    record = tmp_path / "build.json"
    record.write_text(json.dumps(build))
    altered = replace(
        selected,
        proof=replace(selected.proof, library=library, build_record=record, build_sha256=_reference(record)["sha256"]),
    )
    altered.dependencies()
    last = Path(build["attempts"][-1]["object"]["path"])
    last.write_bytes(last.read_bytes() + b" ")
    with pytest.raises(ValueError, match="compiled proof dependency changed"):
        check(context, tmp_path / "case", selected=altered)


def test_unapproved_or_missing_audit_report_refuses(context, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops_unroll as implementation

    original = implementation.subprocess.run

    def corrupt_stdout(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]).endswith("Obligation.lean"):
            result.stdout = result.stdout.replace("propext", "unapproved_test_axiom")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", corrupt_stdout)
    with pytest.raises(ValueError, match="unapproved axiom"):
        check(context, tmp_path)
    assert not json.loads((tmp_path / "proof/REFUSAL.json").read_bytes())["lean_complete_relation_checked"]


def test_original_input_writer_refuses(context, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops_unroll as implementation

    original = implementation.subprocess.run

    def writer(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]).endswith("Obligation.lean"):
            path = tmp_path / "after.mlir"
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", writer)
    result = check(context, tmp_path, after=(context[1] / "bulk_tail/after.mlir").read_text())
    assert result["returncode"] == 0 and not result["lean_complete_relation_checked"]
