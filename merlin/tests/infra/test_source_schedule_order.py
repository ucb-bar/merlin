"""Source policy reaches the actual inner scheduler; final checks stay fixed."""

from dataclasses import replace

import pytest
from test_dense_source import compile_source, source
from test_pointwise_source import case, source_program
from test_selected_dense_contracts import inputs

from merlin.compile.scheduling.pointwise_source import compile_source_pointwise


def test_dense_source_policy_reaches_inner_passes_and_keeps_actual_realization(tmp_path):
    model = inputs(tmp_path)
    module, operation = source((5, 14, 13), model[0].dense.dtypes)
    order = ("admission", "blocking", "emission", "realization")
    original = compile_source(model, module, operation)
    selected = compile_source(model, module, operation, schedule_pass_order=order)
    assert selected.compiled.kernel.digest() == original.compiled.kernel.digest()
    assert [row["name"] for row in selected.compiled.realization["pipeline"]["passes"]] == list(order)
    assert selected.source_stamp.context.resources != original.source_stamp.context.resources
    with pytest.raises(ValueError, match="ordering|prerequisites"):
        compile_source(
            model, module, operation, schedule_pass_order=("blocking", "admission", "emission", "realization")
        )


def test_pointwise_inner_order_cannot_move_final_checks_or_omit_them(tmp_path):
    module, operation = source_program()
    options = case(tmp_path)
    order = ("semantic_admission", "fusion", "storage", "completion", "emission", "final_program")
    original = compile_source_pointwise(module, operation, **options)
    selected = compile_source_pointwise(module, operation, schedule_pass_order=order, **options)
    assert selected.images() == original.images()
    assert [row["name"] for row in selected.compiled.report["passes"]] == list(order)
    assert selected.source_passes[0].before.context.resources != original.source_passes[0].before.context.resources
    emitted = []
    options["encoding"] = replace(options["encoding"], encode=lambda kernel: emitted.append(kernel))
    for invalid in (order[:-1], (*order[:3], "emission", "completion", "final_program")):
        with pytest.raises(ValueError, match="order|prerequisites"):
            compile_source_pointwise(module, operation, schedule_pass_order=invalid, **options)
    assert emitted == []


@pytest.mark.parametrize("invalid", [[], None, "blocking", {"pass": "emission"}])
def test_inner_orders_are_explicit_typed_policy_inputs(tmp_path, invalid):
    (tmp_path / "dense").mkdir()
    (tmp_path / "pointwise").mkdir()
    model = inputs(tmp_path / "dense")
    module, operation = source((5, 14, 13), model[0].dense.dtypes)
    with pytest.raises(ValueError, match="explicit tuple"):
        compile_source(model, module, operation, schedule_pass_order=invalid)
    module, operation = source_program()
    options = case(tmp_path / "pointwise")
    with pytest.raises(ValueError, match="explicit tuple"):
        compile_source_pointwise(module, operation, schedule_pass_order=invalid, **options)
