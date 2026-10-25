"""Complete actual buffer-reuse bindings, live memory and necessary premises."""

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
from xdsl.rewriter import Rewriter

from merlin.verify.lean_graph_loops import LeanSelection, _reference
from merlin.verify.lean_graph_loops_reuse import ALLOCATOR_MODE, REUSE_ROOTS, ReuseSelection, verify_reuse_files


@pytest.fixture(scope="module")
def context():
    path, cases = (os.environ.get(name) for name in ("MERLIN_GRAPH_REUSE_SELECTION", "MERLIN_GRAPH_REUSE_CASES"))
    if not path or not cases:
        pytest.skip("explicit reuse memory/proof/producer/allocator context unavailable")
    data = json.loads(Path(path).read_bytes())
    for key in ("executable", "library", "build_record"):
        data["proof"][key] = Path(data["proof"][key])
    data["proof"] = LeanSelection(**data["proof"])
    for key in ("source", "producer_manifest", "contract"):
        data[key] = Path(data[key])
    root = Path(cases)
    plan = {row["name"]: row for row in json.loads((root / "CASE_PLAN.json").read_bytes())}
    return ReuseSelection(**data), root, plan


def parse(text):
    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, scf.Scf, memref.MemRef, func.Func):
        ctx.load_dialect(dialect)
    module = Parser(ctx, text).parse_module()
    module.verify()
    return module


def text(module):
    output = io.StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue()


def check(
    context,
    tmp_path,
    *,
    name="two_epochs",
    before=None,
    after=None,
    selected=None,
    max_nodes=2000,
    allocator=ALLOCATOR_MODE,
):
    selection, root, plans = context
    tmp_path.mkdir(parents=True, exist_ok=True)
    first, second = root / name / "before.mlir", root / name / "after.mlir"
    if before is not None:
        first = tmp_path / "before.mlir"
        first.write_text(before)
    if after is not None:
        second = tmp_path / "after.mlir"
        second.write_text(after)
    return verify_reuse_files(
        first,
        second,
        selection=selected or selection,
        output=tmp_path / "proof",
        index_bits=plans[name]["index_bits"],
        max_nodes=max_nodes,
        allocator_model=allocator,
    )


@pytest.mark.parametrize(
    "name",
    ("two_epochs", "three_epochs", "one_dim", "three_dim", "live_other_slot", "large", "large_shape", "resource_gap"),
)
def test_actual_complete_live_memory_observations(context, tmp_path, name):
    result = check(context, tmp_path, name=name)
    assert result["lean_memory_relation_checked"] and result["returncode"] == 0
    assert set(result["transitive_axioms"]) == set(REUSE_ROOTS)
    assert result["allocator_policy_is_admitted_premise"] and result["initializer_semantics_explicit"]
    assert not any(
        result[key]
        for key in (
            "allocator_failure_or_pointer_identity_proved",
            "raw_SCF_bridge_proved",
            "pipeline_receipt_issued",
            "numerical_gates_discharged",
            "renderer_or_hardware_proved",
            "source_success_inferred",
            "untrusted_witness_or_plan_used",
        )
    )


def test_large_shape_coverage_is_compact(context, tmp_path):
    small = check(context, tmp_path / "small", name="large_shape")
    large = check(context, tmp_path / "large", name="large")
    assert small["lean_memory_relation_checked"] and large["lean_memory_relation_checked"]
    assert large["projection"]["structural_nodes"] == small["projection"]["structural_nodes"]
    assert abs(large["query"]["bytes"] - small["query"]["bytes"]) < 8192
    assert large["projection"]["objects"][0]["shape"] == [1_000_000_000, 1024]


def mutate(module, kind):
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    block = function.body.block
    allocs = [op for op in block.ops if isinstance(op, memref.AllocOp)]
    roots = [op for op in block.ops if isinstance(op, scf.ForOp)]
    zeros = next(op for op in block.ops if isinstance(op, arith.ConstantOp) and op.value.value.data == 0)
    ones = next(op for op in block.ops if isinstance(op, arith.ConstantOp) and op.value.value.data == 1)
    initializer = roots[1]
    inner = [op for op in initializer.walk() if isinstance(op, scf.ForOp)][-1]
    initial_store = next(op for op in initializer.walk() if isinstance(op, memref.StoreOp))
    frees = [op for op in block.ops if isinstance(op, memref.DeallocOp)]
    if kind == "incomplete_initializer":
        inner.operands = (inner.lb, ones.result, inner.step)
    elif kind == "wrong_initializer_index":
        initial_store.operands = (*initial_store.operands[:2], zeros.result, *initial_store.operands[3:])
    elif kind == "wrong_initializer_value":
        initial_store.operands = (block.args[0], *initial_store.operands[1:])
    elif kind == "missing_initializer":
        block.erase_op(initializer)
    elif kind == "read_before_reinitialize":
        load = next(op for op in block.ops if isinstance(op, memref.LoadOp) and op.prev_op is initializer)
        block.detach_op(load)
        block.insert_op_before(load, initializer)
    elif kind == "same_typed_wrong_buffer":
        load = next(op for op in block.ops if isinstance(op, memref.LoadOp) and op.res.name_hint == "late")
        load.operands = (allocs[1].memref, *load.operands[1:])
    elif kind == "overlapping_slot_alias":
        later = allocs[1]
        later.memref.replace_all_uses_with(allocs[0].memref)
        block.erase_op(later)
    elif kind == "added_view_alias":
        block.insert_op_after(memref.CastOp.get(allocs[0].memref, allocs[0].memref.type), allocs[0])
    elif kind == "missing_final_free":
        block.erase_op(frees[-1])
    elif kind == "early_free_use_after_free":
        last = frees[-1]
        block.detach_op(last)
        block.insert_op_before(last, initializer)
    elif kind == "double_free":
        block.insert_op_after(frees[-1].clone(), frees[-1])
    elif kind == "wrong_final_free":
        frees[0].operands = (allocs[0].memref,)
    elif kind == "wrong_alignment":
        allocs[0].properties["alignment"] = builtin.IntegerAttr(32, builtin.i64)
    elif kind in ("wrong_type_shape", "unknown_layout"):
        original = allocs[0]
        replacement = memref.AllocOp.get(
            builtin.i32,
            alignment=16,
            shape=(3, 2) if kind == "wrong_type_shape" else (2, 3),
            layout=builtin.StridedLayoutAttr((3, 1)) if kind == "unknown_layout" else builtin.NoneAttr(),
        )
        Rewriter.replace_op(original, replacement, [replacement.memref])
    elif kind == "retained_calls_reordered":
        calls = [op for op in block.ops if isinstance(op, func.CallOp)]
        block.detach_op(calls[0])
        block.insert_op_after(calls[0], calls[-1])
    elif kind == "caller_observation":
        callee = [op for op in module.walk() if isinstance(op, func.FuncOp)][1]
        store = next(op for op in callee.walk() if isinstance(op, memref.StoreOp))
        load = next(op for op in callee.walk() if isinstance(op, memref.LoadOp))
        store.operands = (load.res, *store.operands[1:])
    elif kind == "untouched_function":
        other = [op for op in module.walk() if isinstance(op, func.FuncOp)][2]
        constant = next(op for op in other.walk() if isinstance(op, arith.ConstantOp))
        constant.properties["value"] = arith.ConstantOp.from_int_and_width(2, 32).value
    else:
        raise AssertionError(kind)
    module.verify()


FAULTS = (
    "incomplete_initializer",
    "wrong_initializer_index",
    "wrong_initializer_value",
    "missing_initializer",
    "read_before_reinitialize",
    "same_typed_wrong_buffer",
    "overlapping_slot_alias",
    "added_view_alias",
    "missing_final_free",
    "early_free_use_after_free",
    "double_free",
    "wrong_final_free",
    "wrong_alignment",
    "wrong_type_shape",
    "unknown_layout",
    "retained_calls_reordered",
    "caller_observation",
    "untouched_function",
)


@pytest.mark.parametrize("kind", FAULTS)
def test_valid_ir_mutations_and_unsupported_aliases_refuse(context, tmp_path, kind):
    name = (
        "live_other_slot"
        if kind in ("same_typed_wrong_buffer", "overlapping_slot_alias", "wrong_final_free")
        else "two_epochs"
    )
    module = parse((context[1] / name / "after.mlir").read_text())
    mutate(module, kind)
    candidate = text(module)
    parse(candidate).verify()
    try:
        result = check(context, tmp_path, name=name, after=candidate)
    except ValueError:
        assert not json.loads((tmp_path / "proof/REFUSAL.json").read_bytes())["lean_memory_relation_checked"]
    else:
        assert not result["lean_memory_relation_checked"]


def test_invalid_ir_is_separate(context, tmp_path):
    module = parse((context[1] / "two_epochs/after.mlir").read_text())
    load = next(op for op in module.walk() if isinstance(op, memref.LoadOp))
    load.operands = (load.operands[0],)
    with pytest.raises(Exception):
        module.verify()
    with pytest.raises(ValueError, match="parsing/SSA export refused"):
        check(context, tmp_path, after=text(module))


@pytest.mark.parametrize("mode", ("observable_allocator", "may_fail", "finite_capacity", "pointer_identity"))
def test_unsupported_allocator_models_refuse(context, tmp_path, mode):
    with pytest.raises(ValueError, match="explicit total/private/unobservable"):
        check(context, tmp_path, allocator=mode)


@pytest.mark.parametrize("changed", ("contract", "manifest", "revision", "tool", "proof", "import"))
def test_changed_selection_refuses_under_original_contract(context, tmp_path, changed):
    selected = context[0]
    if changed == "contract":
        selected = replace(selected, contract_sha256="0" * 64)
    elif changed == "manifest":
        selected = replace(selected, producer_manifest_sha256="0" * 64)
    elif changed == "revision":
        selected = replace(selected, revision="0" * 40)
    elif changed == "tool":
        selected = replace(selected, proof=replace(selected.proof, sha256="0" * 64))
    elif changed == "proof":
        selected = replace(selected, proof=replace(selected.proof, build_sha256="0" * 64))
    else:
        selected = replace(selected, proof=replace(selected.proof, library=tmp_path / "foreign-imports"))
    with pytest.raises(ValueError):
        check(context, tmp_path, selected=selected)


@pytest.mark.parametrize(
    "product", ("Obligation.lean", "model.json", "actual-exports.json", "before.mlir", "after.mlir")
)
def test_intervening_writers_refuse(context, tmp_path, monkeypatch, product):
    from merlin.verify import lean_graph_loops_reuse as implementation

    original = implementation.subprocess.run

    def writer(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]).endswith("Obligation.lean"):
            path = tmp_path / "proof" / product
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", writer)
    result = check(context, tmp_path)
    assert result["returncode"] == 0 and not result["lean_memory_relation_checked"]


def test_no_prefix_and_no_witness_authority(context, tmp_path):
    (tmp_path / "StoragePlan.json").write_text('{"verified":true,"slot":0}')
    with pytest.raises(ValueError, match="no prefix accepted"):
        check(context, tmp_path, max_nodes=1)


def wrap(value, bits):
    return (value + 2 ** (bits - 1)) % 2**bits - 2 ** (bits - 1)


@pytest.mark.parametrize(
    "name", ("two_epochs", "three_epochs", "one_dim", "three_dim", "live_other_slot", "large_shape", "resource_gap")
)
def test_finite_nonzero_initialized_memory_and_caller_observations(context, tmp_path, name):
    from xdsl.interpreters.memref import xtype_for_el_type
    from xdsl.interpreters.shaped_array import ShapedArray
    from xdsl.interpreters.utils.ptr import TypedPtr

    from merlin.verify.lean_graph_loops_reuse_runtime import execute

    row = context[2][name]
    root = context[1] / name
    records = []
    for base, initial in ((3, 11), (-7, -9), (2 ** (row["bits"] - 1) - 1, 5)):
        inputs = tuple(wrap(base + 2 * index, row["bits"]) for index in range(row["epochs"]))
        caller_value = initial
        observed = []
        for index, value in enumerate(inputs):
            updated = wrap(2 * value, row["bits"])
            caller_value = wrap(caller_value + updated, row["bits"])
            observed.append(caller_value)
            if row["overlap"] and index == 1:
                caller_value = wrap(2 * inputs[0], row["bits"])
        expected = (wrap(observed[0] + observed[-1], row["bits"]), row["shape"][0])
        runs = []
        for file in ("before.mlir", "after.mlir"):
            caller = ShapedArray(
                TypedPtr.new(
                    [initial, 23, -7], xtype=xtype_for_el_type(builtin.IntegerType(row["bits"]), row["index_bits"])
                ),
                [3],
            )
            runs.append(execute(parse((root / file).read_text()), (*inputs, caller), index_bits=row["index_bits"]))
        assert runs[0]["result"] == runs[1]["result"] == expected
        assert runs[0]["trace"] == runs[1]["trace"]
        assert runs[0]["caller"] == runs[1]["caller"] == [caller_value, 23, -7]
        assert runs[0]["allocation_count"] == row["epochs"] and runs[1]["allocation_count"] < row["epochs"]
        assert all(run["all_private_reads_initialized"] and run["all_private_frees_valid"] for run in runs)
        if name == "resource_gap":
            assert runs[0]["peak_live_bytes"] == 8 and runs[1]["peak_live_bytes"] == 16
        records.append(
            dict(inputs=inputs, source=runs[0], target=runs[1], expected=expected, expected_caller=caller_value)
        )
    (tmp_path / "RUNTIME.json").write_text(
        json.dumps(
            dict(
                schema="merlin.graph.reuse.finite_nonzero_witness.v1",
                case=row,
                source=_reference(root / "before.mlir"),
                target=_reference(root / "after.mlir"),
                records=records,
                allocation_observations_excluded_only_under_explicit_contract=True,
                native_or_formal_SCF_bridge=False,
            ),
            indent=2,
        )
        + "\n"
    )


def test_missing_reinitializer_exposes_nonzero_stale_value(context, tmp_path):
    from xdsl.interpreters.memref import xtype_for_el_type
    from xdsl.interpreters.shaped_array import ShapedArray
    from xdsl.interpreters.utils.ptr import TypedPtr

    from merlin.verify.lean_graph_loops_reuse_runtime import execute

    root = context[1] / "two_epochs"
    candidate = parse((root / "after.mlir").read_text())
    mutate(candidate, "missing_initializer")
    results = []
    for module in (parse((root / "before.mlir").read_text()), candidate):
        caller = ShapedArray(TypedPtr.new([11, 23, -7], xtype=xtype_for_el_type(builtin.i32, 64)), [3])
        results.append(execute(module, (17, 29, caller), index_bits=64))
    assert results[0]["result"] != results[1]["result"]
    (tmp_path / "STALE_COUNTEREXAMPLE.json").write_text(
        json.dumps(
            dict(
                source=results[0],
                target=results[1],
                source_initializers_complete=True,
                target_second_initializer_removed=True,
                native_or_formal_SCF_bridge=False,
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "changed", ("count_context", "initializer_bounds", "initializer_indices", "live_slot", "signed_bytes")
)
def test_kernel_refuses_corrupt_model_facts(context, tmp_path, monkeypatch, changed):
    from merlin.verify import lean_graph_loops_reuse as implementation

    original = implementation.project_exports

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if changed == "count_context":
            result["targetContext"] += "altered"
        elif changed == "initializer_bounds":
            result["epochs"][0]["initializer"]["controls"][0][1] -= 1
        elif changed == "initializer_indices":
            result["epochs"][0]["initializer"]["indices"][0] = 0
        elif changed == "live_slot":
            for row in result["objects"]:
                row["slot"] = 0
        else:
            result["bits"] = 32
        return result

    monkeypatch.setattr(implementation, "project_exports", corrupt)
    result = check(context, tmp_path, name="large" if changed == "signed_bytes" else "live_other_slot")
    assert result["returncode"] != 0 and not result["lean_memory_relation_checked"]


def test_coherently_rehashed_proof_or_model_selection_is_not_original_contract(context, tmp_path):
    selected = context[0]
    build = json.loads(selected.proof.build_record.read_bytes())
    build["native_reflection"] = True
    record = tmp_path / "substitute-build.json"
    record.write_text(json.dumps(build))
    forged = replace(
        selected, proof=replace(selected.proof, build_record=record, build_sha256=_reference(record)["sha256"])
    )
    with pytest.raises(ValueError, match="differs from the frozen reuse contract"):
        check(context, tmp_path, selected=forged)


def test_private_object_corruption_refuses_without_global_writes(context, tmp_path):
    selected = context[0]
    build = json.loads(selected.proof.build_record.read_bytes())
    library = tmp_path / "objects"
    for row in build["attempts"]:
        path = library / Path(*row["module"].split(".")).with_suffix(".olean")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(row["object"]["path"]).read_bytes())
        row["object"] = _reference(path)
    record = tmp_path / "proof.json"
    record.write_text(json.dumps(build))
    proof = replace(selected.proof, library=library, build_record=record, build_sha256=_reference(record)["sha256"])
    contract = json.loads(selected.contract.read_bytes())
    contract["selected_context"]["proof"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(proof).items()
    }
    path = tmp_path / "private-contract.json"
    path.write_text(json.dumps(contract))
    private = replace(selected, proof=proof, contract=path, contract_sha256=_reference(path)["sha256"])
    private.dependencies()
    target = Path(build["attempts"][-1]["object"]["path"])
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="compiled proof dependency changed"):
        check(context, tmp_path / "case", selected=private)


def test_unapproved_audit_cannot_issue_acceptance(context, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops_reuse as implementation

    original = implementation.subprocess.run

    def corrupt(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]).endswith("Obligation.lean"):
            result.stdout = result.stdout.replace("propext", "unapproved_test_axiom")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", corrupt)
    with pytest.raises(ValueError, match="unapproved axiom"):
        check(context, tmp_path)


def test_original_input_writer_refuses(context, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops_reuse as implementation

    original = implementation.subprocess.run

    def writer(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]).endswith("Obligation.lean"):
            path = tmp_path / "after.mlir"
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", writer)
    result = check(context, tmp_path, after=(context[1] / "two_epochs/after.mlir").read_text())
    assert result["returncode"] == 0 and not result["lean_memory_relation_checked"]


def test_actual_declined_inputs_are_separate_from_changed_checks(context):
    root = Path(os.environ["MERLIN_GRAPH_REUSE_CONTROLS"])
    records = json.loads((root / "PRODUCTION.json").read_bytes())
    assert len(records) == 7
    for row in records:
        assert row["returncode"] == 0 and not row["expected_changed"]
        capture = json.loads((root / row["name"] / "CAPTURE.json").read_bytes())
        assert not capture["changed"] and not capture["lean_checked"]
        assert (root / row["name"] / "before-generic.mlir").read_bytes() == (
            root / row["name"] / "after.mlir"
        ).read_bytes()


def test_real_neighbor_join_remains_explicit_refusal(context):
    path = Path(os.environ["MERLIN_GRAPH_REUSE_NEIGHBOR_GAP"])
    gap = json.loads(path.read_bytes())
    assert not gap["composition_proved"] and "separate admitted semantics" in gap["refusal"]
    assert gap["middle"] == _reference(context[1] / "two_epochs/after.mlir")
    assert gap["standalone_memory_proof_not_replaced"]
