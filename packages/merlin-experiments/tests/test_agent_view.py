"""Installed presentation owner filters authority without mutating host evidence."""

import copy

from merlin_experiments.phase2 import agent_view as AV


def test_digest_keeps_only_exact_host_authority_and_does_not_mutate():
    allowed = {"id": "surface", "path": "compiler.py", "symbol": "compile", "effects": ["schedule"]}
    wrong = {**allowed, "path": "other.py"}
    record = {
        "portfolio": {"members": [{"identity": {"capsule": "fixture"}}]},
        "analysis": {"optimization_brief": {"ranked_actions": [{"edit_surfaces": [allowed, wrong]}]}},
    }
    before = copy.deepcopy(record)
    digest = AV.portfolio_action_digest(
        record,
        complete_evidence="evidence.json",
        edit_contract={"existing_symbols": [{"surface_id": "surface", "path": "compiler.py", "symbol": "compile"}]},
    )
    surfaces = digest["members"][0]["top_ranked_actions"][0]["authorized_edit_surfaces"]
    assert len(surfaces) == 1
    assert surfaces[0]["authority"] == "exact_host_frozen_existing_symbol"
    surfaces[0]["effects"].append("changed")
    assert record == before
    assert digest["timing_status"] == "UNMEASURED_FULL_MODEL"


def test_installed_provider_is_not_admitted_context_evidence():
    capability = AV.controlled_context_capability({}, provider_installed=True)
    assert capability["current_candidate_status"] == "UNKNOWN"
    assert not capability["available"]
    assert not capability["admission_verified"]


def test_prompt_exposes_explicit_declaration_and_refusal_without_discovery():
    prompt = AV.declared_instruction_prompt(
        {
            "status": "derived",
            "declared_count": 1,
            "custom_opcode": 7,
            "instructions": [{"name": "load", "roles": ["movement"]}],
        }
    )
    assert "load [movement]" in prompt
    assert "OPPORTUNITY, not a defect" in prompt
    assert "UNKNOWN" in AV.declared_instruction_prompt({"status": "UNKNOWN", "reason": "absent"})
    notice = AV.unavailable_action_notice({"probe": "provider absent"})
    assert "probe (provider absent)" in notice
    assert "no retry will make it available" in notice
    assert "installed" in AV.unavailable_action_notice({})
