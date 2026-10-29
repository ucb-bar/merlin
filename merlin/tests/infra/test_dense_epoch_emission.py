"""Checked completion epochs reach a distinct real-completion emitter."""

from dataclasses import replace

import pytest
from test_dense_panel_readout import checked
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp

from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
from merlin.llvmlower.host_builder import FnBuilder


def test_checked_batched_effects_and_full_drains_have_separate_emitters():
    p, r, s, b, adapter_admission, original = checked()
    from test_compact_dense import ADAPTER

    admission = replace(
        adapter_admission, max_issued_readouts=4, readout_configuration="ordered_snapshot_before_store_issue"
    )
    result = batch_dense_panel_readouts(original, p, b, r, s, ADAPTER, admission)
    builder = FnBuilder([])
    effects, drains = [], []

    def issue(tag, values):
        assert tag != DRAIN
        effects.append(tag)
        builder.add(llvm.InlineAsmOp("", "~{memory}", [], has_side_effects=True))

    def complete():
        drains.append(True)
        builder.add(llvm.InlineAsmOp("fence", "~{memory}", [], has_side_effects=True))

    receipt = result.emit_index_program(FnBuilderIndexAdapter(builder), issue, complete_epoch=complete)
    module = ModuleOp([llvm.FuncOp("entry", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())])
    module.verify()
    assert {"preload", "compute", "store", "lhs_load", "weight_load"} <= set(effects)
    assert len(drains) == receipt["completion_mapping"]["emitted_static_drains"] > 0
    assert result.realization["readout_batching"]["removed_dynamic_drains"] > 0


def test_missing_completion_or_changed_checked_program_refuses_before_emission():
    *_, result = checked()
    calls = []
    with pytest.raises(ValueError, match="full epoch completion callbacks"):
        result.emit_index_program(None, lambda *_: calls.append("issue"), complete_epoch=None)
    corrupted = replace(result, kernel=replace(result.kernel, body=()))
    with pytest.raises(ValueError, match="checked emitted program"):
        corrupted.emit_index_program(
            None, lambda *_: calls.append("issue"), complete_epoch=lambda: calls.append("drain")
        )
    assert not calls
