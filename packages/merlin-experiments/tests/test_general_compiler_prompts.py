"""Shared requirements reach every EL without changing task grants or source facts."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.component_origin import render_fresh_phase1_prompt
from merlin_experiments.phase1.levels import LEVELS

from merlin.targetgen import generate_prompt as GP
from merlin.targetgen.generalization_prompt import (
    GENERAL_COMPILER_CONTRACT_V1,
    GENERAL_COMPILER_CONTRACT_V2,
    append_general_compiler_contract,
)


@pytest.mark.parametrize("level", LEVELS, ids=[level["id"] for level in LEVELS])
@pytest.mark.parametrize("target", ["independent_alpha", "independent_beta"])
@pytest.mark.parametrize("mode", ["full", "realistic"])
def test_rendered_tasks_serve_identical_requirements_for_every_level(monkeypatch, level, target, mode):
    # Isolate public slot data, not grading or execution authority. The actual
    # renderer still chooses the declared arm's ordinary workflow/tool blocks.
    slots = {
        "target": target,
        "corpus_rel": "public/corpus/",
        "corpus_families": [],
        "sim_tiers": {"L1": "independent_screen"},
        "endpoint_kind": "command_buffer",
        "screen_tier": "L1",
        "screen_sim": "independent_screen",
        "tool_stem": target + "-opt",
        "kernel_symbol": target + "_kernel",
        "endpoint_desc": "declared runtime artifact",
        "emit_framing": "declared command schema",
        "emit_symbol_note": "",
        "grading_model": "original numerical gate",
        "isa_facts": "independently selected public facts",
        "isa_spec": "",
        "dram_contract": "",
        "termination_contract": "",
    }
    monkeypatch.setattr(GP, "prompt_slots", lambda *_args: slots)
    grants = frozenset()
    task = GP.render_prompt(
        SimpleNamespace(target=target),
        SimpleNamespace(endpoint_kind="command_buffer"),
        mode,
        level["bundle_arm"],
        granted_tools=grants,
    )
    assert task.count(GENERAL_COMPILER_CONTRACT_V2) == 1
    assert target in task and target not in GENERAL_COMPILER_CONTRACT_V2
    assert "original numerical gate" in task
    assert grants == frozenset()
    assert "cannot need a smaller program" not in task
    assert "--capsules <changed-capsule-or-subset>" in task
    assert "--candidate-diagnostics" in task
    assert "--dynamic-census" in task
    assert "counts are descriptive" in task


def test_explicit_tasks_keep_their_bytes_and_contract_is_not_duplicated():
    original = "operator-selected domain and tools\n"
    appended = append_general_compiler_contract(original)
    assert appended.startswith(original)
    assert appended.count(GENERAL_COMPILER_CONTRACT_V2) == 1
    assert append_general_compiler_contract(appended) == appended


def test_actual_fresh_author_prompt_retains_shared_requirements_and_private_exclusions():
    prompt = render_fresh_phase1_prompt()
    assert prompt.count(GENERAL_COMPILER_CONTRACT_V2) == 1
    assert "initial driver has no lowering" in prompt
    assert "Obey the admitted instruction policy" in prompt
    assert "No prior compiler" in prompt
    assert "under /component-inputs" in prompt


def test_new_scale_acceptance_is_versioned_and_archived_explicit_contract_is_preserved():
    assert "runtime loop bounds" in GENERAL_COMPILER_CONTRACT_V2
    assert "complete eligible accelerator placement" in GENERAL_COMPILER_CONTRACT_V2
    assert "required structural/trace checks" in GENERAL_COMPILER_CONTRACT_V2
    assert "preserve all work and outputs" in GENERAL_COMPILER_CONTRACT_V2
    original = "archived explicit task\n" + GENERAL_COMPILER_CONTRACT_V1
    assert append_general_compiler_contract(original) == original
