"""Author-facing prose never states a helper or CPU count of its own.

The run's actual helper ceiling and CPU rosters come from the rendered resource
blocks. fresh10 prompts said "up to 16 helpers" and "All 20 CPUs" beside a resource
block that selected 8 helpers and 15 CPUs.
"""

from merlin_experiments.phase1 import joint_control, joint_pass_guidance
from merlin_experiments.phase1.fresh10_disclosure import author_prompt
from merlin_experiments.phase1.providers import authoring_prompt as prompts

STALE = ("16 helpers", "16 independent helpers", "20 CPUs", "20 selected CPUs")


def _assert_no_counts(text):
    for stale in STALE:
        assert stale not in text


def test_full_compiler_prompts_defer_to_the_selected_limits(monkeypatch):
    full = prompts._full_compiler_prompt("")
    _assert_no_counts(full)
    assert "Use the selected helper ceiling when independent work is available." in full
    assert "All selected CPUs form one dynamically shared pool." in full

    monkeypatch.delenv("MERLIN_AUTHOR_PROFILE", raising=False)
    monkeypatch.setattr(joint_control, "selected", lambda: {"policy": {}})
    monkeypatch.setattr(joint_control, "selfcheck_routing_guidance", lambda: "")
    joint = prompts.authoring_prompt()
    _assert_no_counts(joint)
    assert "up to the selected helper ceiling, with the selected CPUs dynamically shared" in joint


def test_shared_guidance_and_joint_block_defer_to_the_selected_limits(monkeypatch):
    guidance = joint_pass_guidance.prompt()
    _assert_no_counts(guidance)
    assert "up to the selected helper ceiling." in guidance
    monkeypatch.setattr(joint_control, "selfcheck_routing_guidance", lambda: "")
    block = joint_control._full_compiler_prompt_block({"duration_s": 7200, "drain_s": 900})
    _assert_no_counts(block)
    assert "Use the selected CPUs as one shared pool." in block


def test_fresh10_prompt_keeps_its_previous_wording():
    """The fresh10 filter used to rewrite these sentences; the source now says them directly."""
    text = author_prompt(prompts._full_compiler_prompt(""))
    _assert_no_counts(text)
    assert "Use the selected helper ceiling when independent work is available." in text
    assert "All selected CPUs form one dynamically shared pool." in text
