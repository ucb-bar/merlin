"""Ordinary preparation realizes exact readouts or preserves the original epilogue."""

from dataclasses import replace

import pytest
from test_dense_integer_readout import QUALIFICATION, readout_inputs, readout_source
from test_source_dense_lowering import run
from test_source_logical_abi import declaration
from xdsl.rewriter import InsertPoint, Rewriter

from merlin.common import mlir_query as mq
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.logical_borrowed_abi import LogicalBorrowedABI, dense_buffers, scratch_bytes
from merlin.llvmlower.source_dense import SourceDenseLowering


def selection(tmp_path, *, logical=False, format_identity=None, prepare=None, **controls):
    def declare(facts, software, dense):
        if logical:
            dense["logical_pointer_abi"] = declaration()

    selected, problem, storage, adapter = readout_inputs(tmp_path, shape=(3, 7, 5), mutate=declare)
    effects = BorrowedBufferEffects("neutral-reviewed-readout-ABI-v1", True, True, True)
    abi = (
        LogicalBorrowedABI(
            selected.dense.logical_pointer_abi,
            effects,
            "original-wide-format",
            4096,
            readout_format_qualification_identity=format_identity,
        )
        if logical
        else None
    )
    seen = []

    def implement(bindings, directory):
        seen.extend(bindings)
        if prepare is not None:
            prepare(bindings, directory)

    def build(bindings, context):
        pytest.fail("preparation test must not build or execute provider code")

    config = SourceDenseLowering(
        selected,
        storage,
        adapter,
        "host",
        64,
        80,
        implement,
        build,
        private_ownership=True,
        external_buffers_disjoint=True,
        verification_timeout_ms=20_000,
        readout_qualification_identity=QUALIFICATION,
        borrowed_effects=effects if logical else None,
        logical_abi=abi,
    )
    return replace(config, **controls), problem, seen


@pytest.mark.parametrize("external_constants", [False, True])
def test_ordinary_preparation_consumes_exact_epilogue_and_publishes_narrow_type(tmp_path, external_constants):
    config, problem, seen = selection(tmp_path)
    module, dense, output = readout_source((3, 7, 5), config.selected_contracts.dense.dtypes)
    if external_constants:
        for constant in tuple(op for op in output.body.block.ops if op.name == "arith.constant"):
            output.body.block.detach_op(constant)
            Rewriter.insert_op(constant, InsertPoint.before(dense))
    result, _ = run(tmp_path, config, module)
    compilation = seen[0].compilation
    assert len(seen) == 1 and len(compilation.operations) == 2
    assert compilation.output_dtype == "i8" and problem.accumulator_bytes == 4
    assert not any(op.name in ("arith.trunci", "arith.maxsi", "arith.minsi") for op in mq.parse(result).walk())
    compilation.revalidate()


@pytest.mark.parametrize("reason", ["disabled", "unqualified", "extra_use", "different_arithmetic"])
def test_unavailable_fusion_preserves_dense_route_and_cpu_epilogue(tmp_path, reason):
    config, _, seen = selection(tmp_path)
    if reason == "disabled":
        config = replace(config, fuse_output_readout=False)
    elif reason == "unqualified":
        config = replace(config, readout_qualification_identity=None)
    module, _, _ = readout_source(
        (3, 7, 5),
        config.selected_contracts.dense.dtypes,
        extra_use=reason == "extra_use",
        altered=reason == "different_arithmetic",
    )
    result, _ = run(tmp_path, config, module)
    assert len(seen) == 1 and seen[0].compilation.output_dtype == "i32"
    assert seen[0].compilation.output_operation is None
    assert any(op.name == "arith.trunci" for op in mq.parse(result).walk())


def test_failed_fused_planning_retries_raw_dense_once(tmp_path, monkeypatch):
    from merlin.compile.scheduling import dense_source

    config, _, seen = selection(tmp_path)
    module, _, _ = readout_source((3, 7, 5), config.selected_contracts.dense.dtypes)
    actual, attempts = dense_source.schedule_source_dense, []

    def bounded(*args, **kwargs):
        attempts.append(kwargs["output_operation"] is not None)
        return None if attempts[-1] else actual(*args, **kwargs)

    monkeypatch.setattr(dense_source, "schedule_source_dense", bounded)
    result, _ = run(tmp_path, config, module)
    assert attempts == [True, False]
    assert seen[0].compilation.output_dtype == "i32"
    assert any(op.name == "arith.trunci" for op in mq.parse(result).walk())


@pytest.mark.parametrize("identity", [None, "qualified-wide-only", QUALIFICATION])
def test_logical_abi_needs_separate_matching_narrow_format_qualification(tmp_path, identity):
    config, problem, seen = selection(tmp_path, logical=True, format_identity=identity)
    module, _, _ = readout_source((3, 7, 5), config.selected_contracts.dense.dtypes)
    run(tmp_path, config, module)
    compilation = seen[0].compilation
    dtype = "i8" if identity == QUALIFICATION else "i32"
    assert compilation.output_dtype == dtype
    bridge = seen[0].logical_abi_source.read_text()
    assert f"span(d2,{1 if dtype == 'i8' else 4}u" in bridge
    narrow = scratch_bytes(
        dense_buffers(problem, config.selected_contracts.dense.dtypes, output_dtype="i8"), config.logical_abi
    )
    wide = scratch_bytes(dense_buffers(problem, config.selected_contracts.dense.dtypes), config.logical_abi)
    # The declared output alignment remains in force for either width, so
    # scratch padding can consume a few bytes of the logical payload saving.
    assert 0 < wide - narrow <= 3 * problem.m * problem.n


@pytest.mark.parametrize("mutation", ["readout_qualification_identity", "fuse_output_readout", "format"])
def test_callback_cannot_change_readout_or_logical_format_admission(tmp_path, mutation):
    def change(bindings, directory):
        if mutation == "format":
            object.__setattr__(config.logical_abi, "readout_format_qualification_identity", "changed")
        else:
            object.__setattr__(config, mutation, "changed" if mutation.endswith("identity") else False)

    config, _, _ = selection(tmp_path, logical=True, format_identity=QUALIFICATION, prepare=change)
    module, _, _ = readout_source((3, 7, 5), config.selected_contracts.dense.dtypes)
    with pytest.raises(ValueError, match="changed; replan"):
        run(tmp_path, config, module)
