"""Source grouping is selected once and remains bound through materialization."""

from __future__ import annotations

import pytest
from logical_device_target import install
from test_whole_model_group_route import _TARGET, _WEIGHT_ARGS, _bundle_with_manifest, _compiled, _two_layers
from xdsl.dialects.builtin import StringAttr

from merlin.common import mlir_query as mq
from merlin.compiler.planning.groups import plan_groups
from merlin.compiler.planning.reports import group_reports
from merlin.llvmlower import group_offload
from merlin.xdsl_dialects.lowering import compute_groups

pytestmark = pytest.mark.target("gemmini")


@pytest.fixture(autouse=True)
def capabilities(monkeypatch):
    install(monkeypatch)


def test_materialization_and_reports_reuse_selected_groups(monkeypatch):
    module = mq.parse(_two_layers())
    grouping = plan_groups(module, _TARGET, weight_args=_WEIGHT_ARGS)

    def replan(*args, **kwargs):
        pytest.fail("a consumer formed new groups instead of using the selected plan")

    monkeypatch.setattr(compute_groups, "form_groups", replan)
    offload = group_offload.plan(module, _TARGET, grouping=grouping, weight_args=_WEIGHT_ARGS)
    assert len(offload.calls) == 2
    reports = group_reports(grouping, offload)
    assert not any("error" in report for report in reports.values())
    assert reports["compute_groups"]["summary"]["accelerator_groups"] == 2
    assert reports["group_programs"]["stated"] == 2
    assert reports["stream_plan"]["stored_argument_binding"]["argument_indices"] == sorted(_WEIGHT_ARGS)


def test_changed_source_or_an_equal_text_clone_cannot_reuse_group_plan():
    module = mq.parse(_two_layers())
    grouping = plan_groups(module, _TARGET, weight_args=_WEIGHT_ARGS)
    with pytest.raises(ValueError, match="another source"):
        group_offload.plan(mq.parse(_two_layers()), _TARGET, grouping=grouping, weight_args=_WEIGHT_ARGS)
    module.attributes["changed"] = StringAttr("after planning")
    with pytest.raises(ValueError, match="source changed"):
        group_offload.plan(module, _TARGET, grouping=grouping, weight_args=_WEIGHT_ARGS)


def test_group_decisions_and_stored_arguments_cannot_change_after_selection():
    module = mq.parse(_two_layers())
    grouping = plan_groups(module, _TARGET, weight_args=_WEIGHT_ARGS)
    with pytest.raises(ValueError, match="stored-argument binding"):
        group_offload.plan(module, _TARGET, grouping=grouping, weight_args={1})
    grouping.groups[0].placement = "changed-placement"
    with pytest.raises(ValueError, match="decisions changed"):
        group_offload.plan(module, _TARGET, grouping=grouping, weight_args=_WEIGHT_ARGS)


def test_selection_callback_cannot_publish_materialization_after_mutating_source():
    module = mq.parse(_two_layers())
    grouping = plan_groups(module, _TARGET, weight_args=_WEIGHT_ARGS)

    def select(operation):
        operation.attributes["changed"] = StringAttr("during materialization")
        return True

    with pytest.raises(ValueError, match="source changed"):
        group_offload.plan(module, _TARGET, grouping=grouping, weight_args=_WEIGHT_ARGS, select=select)


def test_equal_text_replacement_during_planning_is_refused(monkeypatch):
    module = mq.parse(_two_layers())
    original = compute_groups.form_groups

    def replace_source(module, *args, **kwargs):
        block = module.body.block
        function = block.first_op
        replacement = function.clone()
        block.erase_op(function)
        block.add_op(replacement)
        return original(module, *args, **kwargs)

    monkeypatch.setattr(compute_groups, "form_groups", replace_source)
    with pytest.raises(ValueError, match="planning changed the source"):
        plan_groups(module, _TARGET, weight_args=_WEIGHT_ARGS)


@pytest.mark.parametrize("change", ["another_plan", "dispatch", "module"])
def test_stream_report_refuses_unrelated_or_changed_materialization(change):
    module = mq.parse(_two_layers())
    grouping = plan_groups(module, _TARGET, weight_args=_WEIGHT_ARGS)
    offload = group_offload.plan(module, _TARGET, grouping=grouping, weight_args=_WEIGHT_ARGS)
    if change == "another_plan":
        grouping = plan_groups(module, _TARGET, weight_args=_WEIGHT_ARGS)
    elif change == "dispatch":
        offload.dispatches[0].symbol = "changed"
    else:
        offload.module.attributes["changed"] = StringAttr("changed materialization")
    assert "error" in group_reports(grouping, offload)["stream_plan"]


def test_compile_describes_groups_before_build_and_reuses_the_report(tmp_path, monkeypatch):
    from merlin.compiler.planning import reports

    order = []
    original = reports.group_reports
    observed = []

    def record(*args, **kwargs):
        order.append("group-reports")
        result = original(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(reports, "group_reports", record)
    result = _compiled(capture=_bundle_with_manifest(tmp_path), order=order)
    assert order[:4] == ["route", "group-reports", "build", "device"]
    assert len(observed) == 1
    for key, value in observed[0].items():
        assert result[key] == value


def test_unavailable_grouping_keeps_named_report_failures(tmp_path, monkeypatch):
    from merlin.compiler.planning import groups

    def unavailable(*args, **kwargs):
        raise ValueError("selected source form is unavailable")

    monkeypatch.setattr(groups, "plan_groups", unavailable)
    result = _compiled(capture=_bundle_with_manifest(tmp_path))
    for key in ("compute_groups", "stream_plan", "group_programs"):
        assert "selected source form is unavailable" in result[key]["error"]
