"""Public request grammar, not admission, measurement or qualification."""

import json

import pytest
from merlin_experiments.phase2 import capsule_ordinary_scope as C
from merlin_experiments.phase2 import component_workflow as W
from merlin_experiments.phase2 import ordinary_fast_feedback as F
from merlin_experiments.phase2 import ordinary_measurement_selection as M
from merlin_experiments.phase2 import queued_search_actions as Q
from merlin_experiments.phase2 import stage_inputs as I
from merlin_experiments.phase2.broker import BrokerAction
from merlin_experiments.phase2.contracts import StageGateError


class PublicRoster:
    def validate_search_inputs(self):
        return {("family_a", "capsule_a"), ("family_b", "capsule_b")}


def test_selected_public_help_matches_original_member_parser():
    roster = PublicRoster()
    text = "family_a/capsule_a,family_b/capsule_b"
    assert M.parse_members(roster, text) == (("family_a", "capsule_a"), ("family_b", "capsule_b"))
    for invalid in ("all", "capsule_a", json.dumps(["capsule_a", "capsule_b"]), "old_catalog/capsule_a"):
        with pytest.raises(StageGateError):
            M.parse_members(roster, invalid)
    for actions in (Q.actions(), C.actions()):
        action = next(a for a in actions if a.name == F.SELECTED_ACTION)
        tool = next(t for t in I._broker_tools(actions) if t.name == action.name)
        assert "members=FAMILY/CAPSULE[,FAMILY/CAPSULE...]" in tool.command
        assert "members=PATH" not in tool.command
        for required in ("comma-separated", "FAMILY/CAPSULE", "search-status", "frozen public roster",
                         "not the inherited Phase 1 catalog", "No all sentinel or JSON list"):
            assert required in tool.purpose
        assert action.argv_template == ("__host_corpus_search_pool__", F.SELECTED_ACTION)
        assert action.placeholders == ("revision_id", "members") and not action.required


def test_diagnostic_generic_help_names_exact_kinds_without_advertising_rtl():
    actions = C.actions()
    tools = {tool.name: tool for tool in I._broker_tools(actions)}
    purpose = tools[W.EVALUATION_SUBMIT_ACTION].purpose
    assert set(C.allowed_kinds()) == {F.KIND, F.SELECTED_KIND}
    assert f"kind={F.KIND}" in purpose and f"kind={F.SELECTED_KIND}" in purpose
    assert F.SELECTED_ACTION in purpose and "search-status" in purpose
    assert "kind=KIND" in tools[W.EVALUATION_SUBMIT_ACTION].command
    assert "kind=PATH" not in tools[W.EVALUATION_SUBMIT_ACTION].command
    assert "corpus-rtl-feedback" not in purpose and "measures certified RTL" not in purpose
    assert "no RTL/gSIM or promotion" in purpose
    normal = next(a for a in Q.actions() if a.name == W.EVALUATION_SUBMIT_ACTION)
    assert "measures certified RTL" in normal.purpose  # Original formal route remains distinct.


def test_public_usage_distinguishes_ids_text_and_actual_path_bindings():
    tools = {tool.name: tool for tool in I._broker_tools(Q.actions())}
    freeze = tools[W.REVISION_FREEZE_ACTION].command
    assert "parent_revision=REVISION_ID_OR_none" in freeze
    assert "hypothesis=TEXT" in freeze and "edit_family=TEXT" in freeze
    assert "job_id=JOB_ID" in tools[W.EVALUATION_STATUS_ACTION].command
    assert "revision_id=REVISION_ID" in tools[F.SELECTED_ACTION].command
    path_action = BrokerAction("path-control", ("tool", "{input_mlir}"), ("input_mlir",), "neutral", False)
    assert "input_mlir=PATH" in I._broker_tools((path_action,))[0].command
