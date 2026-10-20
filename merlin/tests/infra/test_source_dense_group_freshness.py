"""Original source/callback identity must precede group discovery callbacks."""

from dataclasses import replace

import pytest
from test_dense_pointer_contracts import pointer_inputs
from test_dense_source import contraction

from merlin.compile.scheduling.compact_dense_group_check import DenseLHSReuseLaw
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.source_dense import SourceDenseLowering
from merlin.llvmlower.source_dense_group import (
    SourceDenseGrouping,
    discover_dense_groups,
    schedule_source_dense_group,
)
from merlin.transforms.certificates import ir_text


def fixture(tmp_path):
    def capacity(facts, software, dense):
        facts["capacities"] = {name: size * 2 for name, size in facts["capacities"].items()}

    selected, problem, storage, adapter = pointer_inputs(tmp_path, shape=(2, 3, 4), mutate=capacity)
    second = replace(
        storage,
        rhs=replace(storage.rhs, name="rhs_second", base=storage.rhs.base + storage.rhs.capacity),
        accumulator=replace(
            storage.accumulator,
            name="acc_second",
            base=storage.accumulator.base + storage.accumulator.capacity,
        ),
    )
    reached = []

    def forbidden(*args, **kwargs):
        reached.append("provider")
        pytest.fail("a stale source/callback must refuse before instruction or provider callbacks")

    law = DenseLHSReuseLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
        "original-neutral-source-group-premises",
        lhs_unchanged_by_member_operations=True,
        independent_member_interleaving=True,
        suppressed_loads_no_faults=True,
        suppressed_loads_no_extra_observations=True,
    )
    config = SourceDenseLowering(
        selected,
        storage,
        adapter,
        "host",
        64,
        160,
        forbidden,
        forbidden,
        private_ownership=True,
        external_buffers_disjoint=True,
        borrowed_effects=BorrowedBufferEffects("original-neutral-borrowed-effects", True, True, True),
        lhs_grouping=SourceDenseGrouping((storage, second), forbidden, forbidden, law),
    )
    dtypes = selected.dense.dtypes
    first = contraction((problem.m, problem.k, problem.n), dtypes, rhs="b", output="first")
    second = contraction((problem.m, problem.k, problem.n), dtypes, rhs="c", output="second")
    # Keep both initializers before the adjacent real contractions; neither
    # result depends on the other, and both have an independent live observer.
    first_prefix, first_body = first.split(" %first =", 1)
    second_prefix, second_body = second.split(" %second =", 1)
    module = parse_mlir_text(
        f"""module {{ func.func @forward(%a:tensor<2x3xi8>,%b:tensor<3x4xi8>,%c:tensor<3x4xi8>)
        ->(tensor<2x4xi32>,tensor<2x4xi32>) {{
        {first_prefix}{second_prefix}
        %first ={first_body}
        %second ={second_body}
        func.return %first,%second:tensor<2x4xi32>,tensor<2x4xi32>
        }} }}"""
    )
    module.verify()
    operations = tuple(op for op in module.walk() if op.name == "linalg.generic")
    assert len(operations) == 2 and operations[0].next_op is operations[1]
    assert operations[0].inputs[0] is operations[1].inputs[0]
    assert operations[0].inputs[1].type == operations[1].inputs[1].type
    return config, module, operations, reached


def validator(tmp_path, hook):
    path = tmp_path / "original_group_validator.py"
    text = "def validate():\n    hook()\n"
    path.write_text(text)
    namespace = {"hook": hook}
    exec(compile(text, str(path), "exec"), namespace)
    return namespace["validate"], path


@pytest.mark.parametrize("entry", ["schedule", "discover"])
@pytest.mark.parametrize("mutation", ["rhs", "callback_code", "callback_file"])
def test_first_original_validation_cannot_rebind_source_or_callbacks(tmp_path, entry, mutation):
    config, module, operations, reached = fixture(tmp_path)
    before = ir_text(module)
    calls = []

    def hook():
        calls.append("validate")
        if len(calls) != 1:
            return
        if mutation == "rhs":
            left, right = operations
            old_left, old_right = left.inputs[1], right.inputs[1]
            left.operands = (left.inputs[0], old_right, left.outputs[0])
            right.operands = (right.inputs[0], old_left, right.outputs[0])
            module.verify()
            assert ir_text(module) != before
        elif mutation == "callback_code":
            validate.__code__ = replacement.__code__
        else:
            callback_path.write_bytes(callback_path.read_bytes() + b"# changed original implementation\n")

    validate, callback_path = validator(tmp_path, hook)
    original_code = validate.__code__

    def replacement():
        raise AssertionError("replacement callback must never execute")

    try:
        with pytest.raises(ValueError, match="changed|replan"):
            if entry == "schedule":
                schedule_source_dense_group(module, operations, config, validate)
            else:
                discover_dense_groups(module, config, validate)
    finally:
        validate.__code__ = original_code
    assert calls == ["validate"]
    assert reached == []


@pytest.mark.parametrize("entry", ["schedule", "discover"])
def test_unchanged_original_validation_is_actually_reached(tmp_path, entry):
    config, module, operations, reached = fixture(tmp_path)

    class OriginalStop(Exception):
        pass

    def stop():
        raise OriginalStop

    validate, _ = validator(tmp_path, stop)
    with pytest.raises(OriginalStop):
        if entry == "schedule":
            schedule_source_dense_group(module, operations, config, validate)
        else:
            discover_dense_groups(module, config, validate)
    assert reached == []
