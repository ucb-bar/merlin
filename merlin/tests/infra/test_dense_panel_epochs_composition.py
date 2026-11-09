"""Generic checked-owner composition: actual retained storage, effects and emission."""

from dataclasses import asdict, dataclass, replace

import numpy as np
import pytest
from test_compact_dense import ADAPTER, fixture
from test_dense_panel_epochs import law, ordered_execution

from merlin.compiler.schedule.compact_dense import _digest
from merlin.compiler.schedule.dense_panel_epochs import STREAM_FLAGS, schedule_dense_panel_epochs
from merlin.compiler.schedule.dense_panel_pipeline import (
    DRAIN,
    DensePanelOverlapAdmission,
    DenseRetainedReductionAdmission,
    _serial_projection,
    schedule_dense_panel_pipeline,
    verify_dense_panel_pipeline,
)
from merlin.compiler.schedule.dense_segmented_pipeline import (
    DenseSegmentedAdmission,
    build_dense_segmented_pipeline,
    schedule_dense_segmented_pipeline,
)
from merlin.compiler.schedule.schedule_ir import dense_schedule_text, make_dense_schedule_plan
from merlin.sched.ir import Call, Loop, instances
from merlin.transforms.dense_planning import dense_blocks


@dataclass(frozen=True)
class CheckedWrapper:
    original: object
    checked: object
    kernel: object
    realization: dict

    def revalidate(self):
        self.original.revalidate()
        self.checked.revalidate()
        if self.kernel != self.checked.kernel:
            raise ValueError("wrapper actual kernel changed")


def retained(shape=(32, 64, 80), *, segment=1, legacy=False):
    p, r, s = fixture(shape=shape, tile=(16, 16, 16))
    # Shared operand backing, distinct byte intervals, nonzero bases; ACC is separate.
    s = replace(
        s,
        lhs=replace(s.lhs, backing="operands", base=16, capacity=512),
        rhs=replace(s.rhs, backing="operands", base=528, capacity=512),
        accumulator=replace(s.accumulator, base=8, capacity=128),
    )
    r = replace(
        r, operand_bytes=1024 * 16, accumulator_bytes=128 * 64, max_m_blocks=4, max_k_blocks=segment, max_n_blocks=2
    )
    bm = min(4, p.tiles[0])
    block = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (bm, segment, 2))
    kw = dict(
        max_issued_readouts=bm * 2, pending_readouts=1, readout_configuration="ordered_snapshot_before_store_issue"
    )
    a = DensePanelOverlapAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "synthetic-source-conjunction",
        bm * 2 * p.tiles[1],
        (bm + 2) * p.tiles[1],
        2,
        2,
        **kw,
    )
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="c" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(block.m, block.k, block.n),
        pipeline_kind="retained_lhs_segmented_k",
        pipeline_prefetch=True,
    )
    if not legacy:
        reduction = DenseRetainedReductionAdmission(
            ADAPTER.configuration_identity,
            ADAPTER.numerical_policy_identity,
            "synthetic-combined-source-conjunction",
            a.max_issued_compute_pairs,
            a.max_issued_loads,
            2,
            2,
            **kw,
        )
        admission = DenseSegmentedAdmission(a, reduction, "synthetic-independent-COMBINED-law")
        result = schedule_dense_segmented_pipeline(
            p,
            r,
            s,
            ADAPTER,
            admission,
            name="composition",
            dtypes=("i8", "i8", "i32"),
            schedule_plan=plan,
            source_identity="c" * 64,
            external_buffers_disjoint=True,
            timeout_ms=15000,
        )
        return p, r, s, block, admission, result, None
    assert segment == 1
    oldplan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="c" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(block.m, 1, block.n),
        pipeline_kind="two_window",
        pipeline_prefetch=True,
    )
    incumbent = schedule_dense_panel_pipeline(
        p,
        r,
        s,
        ADAPTER,
        a,
        name="composition",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=oldplan,
        source_identity="c" * 64,
        external_buffers_disjoint=True,
        timeout_ms=15000,
    )
    kernel = build_dense_segmented_pipeline(p, block, r, s, ADAPTER, name="composition", dtypes=("i8", "i8", "i32"))
    report = verify_dense_panel_pipeline(kernel, p, block, r, s, ADAPTER, a, static_budget=128, timeout_ms=15000)
    report["schedule_ir"] = dict(text=dense_schedule_text(plan), pipeline_kind="retained_lhs_segmented_k")
    checked = replace(
        incumbent,
        kernel=kernel,
        realization=report,
        checked_kernel_sha256=kernel.digest(),
        checked_admission_sha256=_digest(asdict(a)),
    )
    wrapped = CheckedWrapper(incumbent, checked, kernel, report)
    return p, r, s, block, a, wrapped, checked


def run(values, **kwargs):
    p, r, s, b, a, original, source = values
    return schedule_dense_panel_epochs(
        original,
        p,
        b,
        r,
        s,
        ADAPTER,
        a,
        kwargs.pop("selected_law", law(accumulator_result_visibility=False)),
        checked_source=source,
        timeout_ms=15000,
        **kwargs,
    )


@pytest.mark.parametrize("m", [16, 24, 32, 48, 64])
@pytest.mark.parametrize("legacy,segment", [(True, 1), (False, 1), (False, 2)])
def test_retained_owner_and_combined_segmented_large_rows_preserve_actual_effects_and_numerics(m, legacy, segment):
    values = retained((m, 64, 80), legacy=legacy, segment=segment)
    p, r, s, b, a, original, source = values
    result = run(values)
    assert _serial_projection(result.kernel) == _serial_projection(original.kernel)
    assert result.realization["panel_epochs"]["removed_dynamic_drains"] > 0
    assert result.realization["panel_epochs"]["same_output_block_proofs"] > 0
    assert result.admission == a
    assert result._context[4] == s  # Original assigned view; no larger allocation.
    result.revalidate()
    result.materialize_scf().verify()
    actual, expected = ordered_execution(result.kernel, ADAPTER, p, s, visibility=False)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(17, 65, 81), (33, 63, 79), (49, 67, 83)])
def test_ragged_domains_keep_exact_source_and_real_completion_callback(shape):
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp

    from merlin.compiler.schedule.llvm_program import FnBuilderIndexAdapter
    from merlin.llvmlower.host_builder import FnBuilder

    values = retained(shape, segment=2)
    result = run(values)
    actual, expected = ordered_execution(result.kernel, ADAPTER, values[0], values[2], visibility=False)
    np.testing.assert_array_equal(actual, expected)
    builder, effects, completions = FnBuilder([]), [], []

    def issue(tag, operands):
        effects.append(tag)
        builder.add(llvm.InlineAsmOp("", "~{memory}", [], has_side_effects=True))

    def complete():
        completions.append(True)
        builder.add(llvm.InlineAsmOp("fence", "~{memory}", [], has_side_effects=True))

    receipt = result.emit_index_program(FnBuilderIndexAdapter(builder), issue, complete_epoch=complete)
    module = ModuleOp([llvm.FuncOp("entry", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())])
    module.verify()
    assert len(completions) == receipt["completion_mapping"]["emitted_static_drains"]
    assert {"preload", "compute", "lhs_load", "weight_load", "store"} <= set(effects)
    # Actual MLIR contains the real callback; roundtrip verification is above.
    assert str(module).count('"fence"') == len(completions)


@pytest.mark.parametrize("flag", STREAM_FLAGS)
def test_unknown_premise_keeps_every_original_boundary(flag):
    values = retained(segment=2)
    before = (values[5].kernel.digest(), _digest(values[5].realization))
    result = run(values, selected_law=law(**{flag: False}))
    assert result.kernel == values[5].kernel
    assert flag in result.realization["unknown_premises"]
    assert before == (values[5].kernel.digest(), _digest(values[5].realization))


def test_wrapper_binding_and_context_mutations_refuse():
    values = retained(legacy=True)
    p, r, s, b, a, original, source = values
    before = (original.kernel.digest(), _digest(original.realization))
    with pytest.raises(ValueError, match="explicit supported checked source"):
        schedule_dense_panel_epochs(original, p, b, r, s, ADAPTER, a, law())
    changed = retained((48, 64, 80), legacy=True)[-1]
    with pytest.raises(ValueError, match="differs"):
        schedule_dense_panel_epochs(original, p, b, r, s, ADAPTER, a, law(), checked_source=changed)
    with pytest.raises(ValueError, match="admission"):
        schedule_dense_panel_epochs(
            original,
            p,
            b,
            r,
            s,
            ADAPTER,
            replace(a, max_issued_loads=a.max_issued_loads + 1),
            law(),
            checked_source=source,
        )
    overlap = replace(s, rhs=replace(s.rhs, base=s.lhs.base))
    with pytest.raises(ValueError):
        schedule_dense_panel_epochs(original, p, b, r, overlap, ADAPTER, a, law(), checked_source=source)
    with pytest.raises(ValueError):
        schedule_dense_panel_epochs(original, replace(p, n=p.n + 1), b, r, s, ADAPTER, a, law(), checked_source=source)
    result = run(values)
    with pytest.raises(ValueError):
        replace(result, _context=(source, p, b, r, overlap, ADAPTER, a)).revalidate()
    with pytest.raises(ValueError):
        replace(result, kernel=original.kernel).revalidate()
    with pytest.raises(ValueError, match="source/context owner changed"):
        replace(result, _source_result=replace(source, source_identity="e" * 64)).revalidate()
    assert before == (original.kernel.digest(), _digest(original.realization))


def test_combined_admission_uses_both_limits_and_missing_final_completion_refuses():
    values = retained(segment=2)
    p, r, s, b, a, original, source = values
    changed = replace(a, reduction=replace(a.reduction, max_issued_loads=a.reduction.max_issued_loads + 1))
    with pytest.raises(ValueError, match="admission"):
        schedule_dense_panel_epochs(original, p, b, r, s, ADAPTER, changed, law())

    def drop(body):
        last = body[-1]
        return (*body[:-1], replace(last, body=drop(last.body))) if type(last) is Loop else body[:-1]

    broken = replace(original, kernel=replace(original.kernel, body=drop(original.kernel.body)))
    with pytest.raises(ValueError):
        schedule_dense_panel_epochs(broken, p, b, r, s, ADAPTER, a, law())


def test_explicit_cross_output_block_stream_is_rejected_without_mutating_source():
    import time

    from merlin.compiler.schedule.blocks import Compute, Preload
    from merlin.compiler.schedule.dense_panel_epochs import _prove_output_block

    values = retained()
    b, original = values[3], values[5]
    before = (original.kernel.digest(), _digest(original.realization))
    calls = []
    for m in (0, b.m):
        calls.extend(
            (ADAPTER.encode(Preload(0, (0, 0), 0, 0, 16, 16, 16)), ADAPTER.encode(Compute(0, (m, 0), 16, 16, True)))
        )
    with pytest.raises(ValueError, match="cross an original output-block"):
        _prove_output_block(dict(elements=tuple(calls), ancestry=()), ADAPTER, b, time.monotonic() + 5)
    assert before == (original.kernel.digest(), _digest(original.realization))


def test_direct_retained_full_k_owner_preserves_source_emission_and_unknown_fallback():
    p, r, s, small, _, _, _ = retained()
    r = replace(r, max_k_blocks=p.tiles[1])
    block = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (small.m, p.tiles[1], small.n))
    admission = DenseRetainedReductionAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "synthetic-full-reduction-law",
        block.m * block.k * block.n,
        (block.m + block.n) * block.k,
        2,
        2,
        max_issued_readouts=block.m * block.n,
        pending_readouts=1,
        readout_configuration="ordered_snapshot_before_store_issue",
    )
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="f" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(block.m, block.k, block.n),
        pipeline_kind="retained_full_k",
        pipeline_prefetch=False,
    )
    original = schedule_dense_panel_pipeline(
        p,
        r,
        s,
        ADAPTER,
        admission,
        name="direct_retained",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=plan,
        source_identity="f" * 64,
        external_buffers_disjoint=True,
        timeout_ms=15000,
    )
    result = schedule_dense_panel_epochs(
        original, p, block, r, s, ADAPTER, admission, law(accumulator_result_visibility=False), timeout_ms=15000
    )
    assert _serial_projection(result.kernel) == _serial_projection(original.kernel)
    assert result.admission == admission
    result.materialize_scf().verify()
    result.revalidate()
    actual, expected = ordered_execution(result.kernel, ADAPTER, p, s, visibility=False)
    np.testing.assert_array_equal(actual, expected)
    unchanged = schedule_dense_panel_epochs(
        original, p, block, r, s, ADAPTER, admission, law(local_write_after_read=False), timeout_ms=15000
    )
    assert unchanged.kernel == original.kernel
