"""The RTL-checks arm's treatment is half prompt and half feedback — the agent must be told about both.

Arm 4 differs from arm 3 in exactly one way: its round verdict carries an advisory ``rtl_checks`` block.
The launcher's own docstring says the bundle adds "a TASK addendum describing the rtl_checks feedback", and
no such text existed anywhere: the arm's prompt told the agent to "run the CIRCT RTL checks" (which it
cannot — they run grader-side) and never mentioned that findings arrive in the verdict. An unexplained JSON
block in a verdict is noise, so the treatment could not act even once the block was populated.

These tests pin the two properties that make the treatment legible, and neither names a target:

1. the generator's arm-4 block tells the agent the block exists, that it is advisory, and — the part that
   the whole `checks-that-skip-and-report-success` class turns on — that a DROPPED check means *not
   inspected*, never *clean*;
2. any committed arm-4 starter prompt that mentions the arm at all carries that same item, so a frozen
   bundle cannot drift away from the generator that is supposed to define it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

_TARGETS = merlin_dir() / "experiments" / "capsule_bench" / "targets"
_ARM4_BUNDLE_PREFIX = "merlin_assisted_rtlchecks"
#: The anchor phrase of the arm-4 identity item, and of the feedback item that was missing.
_ARM_ITEM = "RTL-checks arm:"
_FEEDBACK_ITEM = "`rtl_checks` block"


def _arm4_block(target: str) -> str:
    """The arm-4 prompt as the GENERATOR renders it for a target."""
    import os

    from merlin.targetgen.generate_prompt import render_prompt
    from merlin.targetgen.target_experiment import load_capability_manifest, load_target_experiment

    desc = _TARGETS / target / "target_experiment.yaml"
    os.environ["MERLIN_TARGET_EXPERIMENT"] = str(desc)
    return render_prompt(load_target_experiment(desc), load_capability_manifest(target),
                         "full", "merlin_assisted_rtlchecks")


def _targets_with_a_descriptor() -> list[str]:
    return sorted(p.parent.name for p in _TARGETS.glob("*/target_experiment.yaml"))


def _committed_arm4_prompts() -> list[Path]:
    return sorted(_TARGETS.glob(f"*/input_bundles/{_ARM4_BUNDLE_PREFIX}*/STARTER_PROMPT.md"))


@pytest.mark.parametrize("target", _targets_with_a_descriptor())
def test_the_arm4_prompt_names_the_feedback_the_arm_actually_delivers(target):
    try:
        block = _arm4_block(target)
    except Exception as e:  # noqa: BLE001 — a target whose contract needs an absent toolchain
        pytest.skip(f"{target}: no renderable prompt in this environment ({type(e).__name__}: {e})")
    if _ARM_ITEM not in block:
        pytest.skip(f"{target}: this descriptor renders no RTL-checks arm block")
    assert _FEEDBACK_ITEM in block, (
        "the arm-4 prompt must tell the agent its verdict carries the rtl_checks block; without that the "
        "arm's only treatment arrives as an unexplained JSON blob")
    # The honesty rule has to reach the agent too, not just the harness.
    assert "checks_dropped" in block and "DID NOT RUN" in block, (
        "the prompt must say that a dropped check means NOT INSPECTED — an agent that reads an empty "
        "findings list as a clean bill makes the same mistake the harness used to make")
    assert "does NOT gate pass/fail" in block or "not gate pass/fail" in block


@pytest.mark.parametrize("prompt", _committed_arm4_prompts(), ids=lambda p: f"{p.parts[-4]}:{p.parts[-2]}")
def test_a_committed_arm4_prompt_does_not_drift_from_the_generator(prompt):
    """A committed bundle prompt is written only when ABSENT (hand-authored bundles stay frozen), so it can
    silently fall behind. Any committed arm-4 prompt that claims the arm must also carry its feedback item."""
    text = prompt.read_text()
    if _ARM_ITEM not in text:
        pytest.skip(f"{prompt.parts[-3]}: committed prompt states no RTL-checks arm item")
    assert _FEEDBACK_ITEM in text, (
        f"{prompt}: names the RTL-checks arm but not the rtl_checks feedback block the arm delivers")
    assert "DID NOT RUN" in text


def test_the_prompt_no_longer_asks_the_agent_to_run_a_grader_side_check():
    """The old wording told the agent to "run the CIRCT RTL checks on your lowering". It cannot: the checks
    run in the grading harness, and the runner module is not in the arm's grant set. Asking for it wasted
    rounds and made the real feedback path invisible."""
    for prompt in _committed_arm4_prompts():
        text = prompt.read_text()
        if _ARM_ITEM in text:
            assert "run the CIRCT RTL checks on your lowering" not in text, (
                f"{prompt}: asks the agent to run a check that only the grader can run")
