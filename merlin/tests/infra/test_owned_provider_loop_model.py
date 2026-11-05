"""Fresh loop-carried storage through ordinary mixed-family model execution.

The model returns the actual carried row and integer buffers, not views read
back from history. This exercises upstream's false-entry/true-backedge ownership
form. The existing mixed-family harness independently checks all six outputs,
native qualifications, actual linked programs and both family tuple orders.
"""

import copy
import importlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import test_mixed_source_model as mixed
from test_owned_buffer_host_model import generic_snapshot
from test_row_softmax import exp_object as exp_object  # noqa: F401
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.rewriter import InsertPoint, Rewriter

from merlin.common import compile_trace
from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.llvmlower.pipeline import dealloc_placement_violations
from merlin.model_check import full_build as FB
from merlin.transforms.certificates import ir_text


def carried_model_text():
    """Restore the independently equivalent original tensor recurrence."""
    source = _HISTORY_MODEL_TEXT()
    first = source.index(" %history,%integer_history=scf.for")
    last = source.index(" %prob=func.call @row_placeholder", first)
    source = (
        source[:first]
        + """
 %ie=tensor.empty():tensor<2x19xi8>
 %iz=linalg.fill ins(%izero:i8) outs(%ie:tensor<2x19xi8>)->tensor<2x19xi8>
 %last,%integer_last,%history,%integer_history=scf.for %step=%c0 to %c3 step %c1
 iter_args(%current=%scores,%previous_integer=%iz,%fh=%fz,%ih=%hz)
 ->(tensor<2x19xf32>,tensor<2x19xi8>,tensor<3x2x19xf32>,tensor<3x2x19xi8>) {
"""
        + source[last:]
    )
    old = "scf.yield %fh1,%ih1:tensor<3x2x19xf32>,tensor<3x2x19xi8>"
    new = "scf.yield %prob,%integer,%fh1,%ih1:tensor<2x19xf32>,tensor<2x19xi8>,tensor<3x2x19xf32>,tensor<3x2x19xi8>"
    assert source.count(old) == 1
    source = source.replace(old, new)
    first = source.index(' %last="tensor.extract_slice"')
    last = source.index(" return %history", first)
    return source[:first] + source[last:]


_HISTORY_MODEL_TEXT = mixed.model_text


def owned_loop(module):
    loops = [
        op
        for op in module.walk()
        if isinstance(op, scf.ForOp) and len(op.iter_args) == 3 and str(op.iter_args[-1].type) == "i1"
    ]
    assert len(loops) == 1
    loop = loops[0]
    assert all(isinstance(value.owner, memref.AllocOp) for value in loop.iter_args[:2])
    assert isinstance(loop.iter_args[-1].owner, arith.ConstantOp)
    assert loop.iter_args[-1].owner.value.value.data == 0
    terminator = loop.body.block.last_op
    assert isinstance(terminator, scf.YieldOp) and len(terminator.operands) == 3
    assert all(
        isinstance(value.owner, memref.AllocOp) and value.owner.parent is loop.body.block
        for value in terminator.operands[:2]
    )
    assert isinstance(terminator.operands[-1].owner, arith.ConstantOp)
    assert terminator.operands[-1].owner.value.value.data == -1
    guarded = [
        op
        for op in loop.body.block.ops
        if isinstance(op, scf.IfOp) and tuple(op.operands) == (loop.body.block.args[-1],)
    ]
    assert len(guarded) == 1
    frees = [op for op in guarded[0].walk() if isinstance(op, memref.DeallocOp)]
    assert len(frees) == 2
    for free, previous in zip(frees, loop.body.block.args[1:3], strict=True):
        base = free.operands[0].owner
        assert base.name == "memref.extract_strided_metadata" and tuple(base.operands) == (previous,)
    return loop


def select_reviewed_effects(monkeypatch):
    from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects

    # These are ORIGINAL caller premises from review of the test implementations:
    # the ranked wrappers hold no pointers globally, never free descriptors, and
    # complete the synchronous interpreter before returning. Primitive effect
    # flags and successful numeric qualification do not establish these facts.
    for provider in (mixed.RowProvider, mixed.Provider):
        original = provider.selection

        def selection(self, trace, original=original):
            return replace(
                original(self, trace),
                borrowed_effects=BorrowedBufferEffects(
                    "reviewed-neutral-ranked-ABI-lifetime-v1",
                    True,
                    True,
                    True,
                ),
            )

        monkeypatch.setattr(provider, "selection", selection)


def assert_call_barrier(messages):
    assert messages and any("call" in message for message in messages), messages


def mutate_checkpoint(text, directory, calls):
    """Break actual valid native IR; missing original premises must never heal it."""
    module = mq.parse(text)
    loop = owned_loop(module)
    known = next(op for op in loop.body.block.ops if isinstance(op, func.CallOp))
    unknown = func.FuncOp.external("uncontracted_borrower", [value.type for value in known.operands], [])
    unknown.properties["arg_attrs"] = builtin.ArrayAttr(
        [builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr("read")}) for _ in known.operands]
    )
    module.body.block.add_op(unknown)
    call = func.CallOp("uncontracted_borrower", known.operands, [])
    Rewriter.insert_op(call, InsertPoint.before(known))
    module.verify()
    unknown_path = directory / "unknown-call.mlir"
    unknown_path.write_text(ir_text(module))
    assert_call_barrier(dealloc_placement_violations(unknown_path.read_text(), borrowed_calls=calls))

    module = mq.parse(text)
    symbols = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    changed = False
    for contract in calls:
        wrapper = symbols[contract.wrapper_symbol]
        nested = next(op for op in wrapper.walk() if isinstance(op, func.CallOp))
        if nested.operands[0].type != wrapper.body.block.args[-1].type:
            continue
        operands = list(nested.operands)
        operands[0] = wrapper.body.block.args[-1]
        nested.operands = tuple(operands)
        changed = True
        break
    assert changed
    module.verify()
    wrong_wrapper = directory / "wrong-wrapper.mlir"
    wrong_wrapper.write_text(ir_text(module))
    assert_call_barrier(dealloc_placement_violations(wrong_wrapper.read_text(), borrowed_calls=calls))
    return dict(unknown_call_sha256=sha256_file(unknown_path), wrong_wrapper_sha256=sha256_file(wrong_wrapper))


def mutate_retained_observations(selection, source, directory):
    """Positive evidence is restored exactly after each completed-file mutation."""
    from merlin.llvmlower import borrowed_buffer_calls as borrowed

    path = directory / borrowed.REPORT
    original = path.read_bytes()
    report = json.loads(original)
    report["accepted_symbols"] = []
    variant = directory / "mutation-accepted-symbols.json"
    variant.write_text(json.dumps(report))
    try:
        path.write_bytes(variant.read_bytes())
        with pytest.raises(ValueError, match="original|consumption|checkpoint|changed|observ"):
            borrowed.verify_native(selection, source, directory)
    finally:
        path.write_bytes(original)
    selection.revalidate(source)
    assert path.read_bytes() == original

    path = directory / borrowed.CHECKPOINT
    original = path.read_bytes()
    checkpoint_variant = directory / "mutation-checkpoint.mlir"
    checkpoint_variant.write_bytes(original + b"\n")
    try:
        path.write_bytes(checkpoint_variant.read_bytes())
        with pytest.raises(ValueError, match="changed|original|observ"):
            selection.revalidate(source)
    finally:
        path.write_bytes(original)
    selection.revalidate(source)
    assert path.read_bytes() == original
    borrowed.verify_native(selection, source, directory)
    return dict(report_variant_sha256=sha256_file(variant), checkpoint_variant_sha256=sha256_file(checkpoint_variant))


def test_ordinary_provider_loop_uses_original_borrowed_effects(tmp_path, record_property, exp_object, monkeypatch):
    from merlin.llvmlower import borrowed_buffer_calls as borrowed

    monkeypatch.setattr(mixed, "model_text", carried_model_text)
    select_reviewed_effects(monkeypatch)
    lower = importlib.import_module("merlin.llvmlower.lower")
    original_lower = lower.lower_to_llvm_ir
    captured = []

    def observe(source, *args, **kwargs):
        selection = kwargs.get("borrowed_buffer_calls")
        if selection is not None:
            captured.append((source, selection, Path(kwargs["workdir"])))
        return original_lower(source, *args, **kwargs)

    monkeypatch.setattr(lower, "lower_to_llvm_ir", observe)
    trace_root = tmp_path / "lifetime_trace"
    request = compile_trace.Request(str(trace_root), dump_after=("mlir:optimize-allocation-liveness",))
    with compile_trace.session(request):
        mixed.test_ordinary_mixed_source_families_share_one_model_transaction(tmp_path, record_property, exp_object)
    index = json.loads((trace_root / "trace.json").read_text())
    assert index["outcome"] == compile_trace.COMPLETED
    snapshots = []
    assert len(captured) == 2
    for source, selection, directory in captured:
        selection.revalidate(source)
        roster_path, checkpoint = directory / borrowed.ROSTER, directory / borrowed.CHECKPOINT
        roster = json.loads(roster_path.read_text())
        calls = borrowed.read_borrowed_calls(roster)
        assert len(calls) == 2 and calls == selection.calls
        assert roster == selection.record()
        consumption = borrowed.verify_native(selection, source, directory)
        generic = directory / "complete-post-sink.generic.mlir"
        text = generic_snapshot(checkpoint, generic)
        module = mq.parse(text)
        loop = owned_loop(module)
        assert dealloc_placement_violations(checkpoint.read_text(), borrowed_calls=calls) == []
        # Even the complete native declarations/attributes do not supply effects.
        assert_call_barrier(dealloc_placement_violations(checkpoint.read_text()))
        assert_call_barrier(dealloc_placement_violations(checkpoint.read_text(), borrowed_calls=calls[:1]))
        entry = loop.iter_args[-1].owner
        entry.properties["value"] = loop.body.block.last_op.operands[-1].owner.properties["value"]
        module.verify()
        wrong_owner = directory / "wrong-initial-owner.mlir"
        wrong_owner.write_text(ir_text(module))
        assert dealloc_placement_violations(wrong_owner.read_text(), borrowed_calls=calls)
        mutations = mutate_checkpoint(text, directory, calls)
        mutations.update(mutate_retained_observations(selection, source, directory))

        changed = mq.parse(source)
        constant = next(
            op
            for op in changed.walk()
            if isinstance(op, arith.ConstantOp)
            and isinstance(op.value, builtin.FloatAttr)
            and op.value.value.data == 128.0
        )
        constant.properties["value"] = builtin.FloatAttr(64.0, builtin.f32)
        changed.verify()
        with pytest.raises(ValueError, match="stale.*source"):
            selection.revalidate(ir_text(changed))

        # Valid typed DATA still cannot replace the original retained roster.
        packet = copy.deepcopy(roster)
        packet["calls"][0]["effects"]["qualification_identity"] += "-changed"
        assert borrowed.read_borrowed_calls(packet)
        changed_directory = directory / "changed-roster"
        changed_directory.mkdir()
        for filename in (borrowed.CHECKPOINT, borrowed.REPORT):
            shutil.copy2(directory / filename, changed_directory / filename)
        (changed_directory / borrowed.ROSTER).write_text(json.dumps(packet))
        with pytest.raises(ValueError, match="original|consumption|checkpoint|changed|observ"):
            borrowed.verify_native(selection, source, changed_directory)
        recipe = json.loads((directory / "lowering_recipe.json").read_text())
        assert recipe["sources"]["borrowed_buffer_calls"]["sha256"] == sha256_file(roster_path)
        implementation = recipe["sources"]["borrowed_call_contracts"]
        assert Path(implementation["path"]) == Path(borrowed.__file__).resolve()
        assert implementation["sha256"] == sha256_file(Path(implementation["path"]))
        snapshots.append(
            dict(
                path=str(checkpoint),
                sha256=sha256_file(checkpoint),
                generic_sha256=sha256_file(generic),
                roster=roster,
                roster_sha256=sha256_file(roster_path),
                consumption=consumption,
                consumption_sha256=sha256_file(directory / borrowed.REPORT),
                wrong_initial_owner_sha256=sha256_file(wrong_owner),
                mutations=mutations,
            )
        )
    assert len(snapshots) == 2  # Both ordinary family tuple orders reached the checker.
    assert captured[0][1].calls == captured[1][1].calls
    report = dict(
        schema="merlin.test_owned_loop_model.v1",
        snapshots=snapshots,
        trace_sha256=sha256_file(trace_root / "trace.json"),
        complete_outputs=6,
        recurrence_steps=3,
        original_calls_identical=(
            [asdict(call) for call in captured[0][1].calls] == [asdict(call) for call in captured[1][1].calls]
        ),
        scope="ordinary fresh carried allocation lifetime and complete execution; no arena/peak-memory or timing claim",
    )
    (tmp_path / "lifetime_model.json").write_text(json.dumps(report, indent=2))
    record_property("owned_loop_model", json.dumps(report, sort_keys=True))


def test_provider_callback_cannot_change_its_checked_lifetime_checkpoint(
    tmp_path, record_property, exp_object, monkeypatch
):
    from merlin.llvmlower import borrowed_buffer_calls as borrowed

    monkeypatch.setattr(mixed, "model_text", carried_model_text)
    select_reviewed_effects(monkeypatch)
    original = mixed.Provider.build
    mutations = []

    def changed_build(self, bindings, context):
        checkpoint = context.model_llvm_path.parent / borrowed.CHECKPOINT
        assert checkpoint.is_file()
        before = sha256_file(checkpoint)
        checkpoint.write_bytes(checkpoint.read_bytes() + b"\n")
        marker = self.work / "changed-own-checkpoint.json"
        marker.write_text(
            json.dumps(
                dict(
                    path=str(checkpoint),
                    before_sha256=before,
                    changed_sha256=sha256_file(checkpoint),
                )
            )
        )
        mutations.append(marker)
        return original(self, bindings, context)

    def no_simulator(*args, **kwargs):
        pytest.fail("changed original lifetime observation reached model simulation")

    monkeypatch.setattr(mixed.Provider, "build", changed_build)
    monkeypatch.setattr(FB, "run_host", no_simulator)
    with pytest.raises(ValueError, match="changed|original|observ"):
        mixed.test_ordinary_mixed_source_families_share_one_model_transaction(tmp_path, record_property, exp_object)
    assert len(mutations) == 1
    assert not (tmp_path / "row_pointwise" / "model.elf").exists()
    mutation = json.loads(mutations[0].read_text())
    assert mutation["before_sha256"] != mutation["changed_sha256"] == sha256_file(Path(mutation["path"]))
