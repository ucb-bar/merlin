"""The agent's task brief must RENDER, for every arm, before a run is launched.

MEASURED, and it cost a launch. The brief is built with ``str.format`` over a module-level template,
so a literal ``{`` in the prose is read as a placeholder. Adding a JSON example to the wait
instructions -- ``{"status": "in_progress", ...}`` -- made ``render_prompt`` raise
``KeyError: '"status"'``.

Nothing caught it. The template is only rendered inside ``run_baseline_qa_loop._build_task``, which
runs after the launcher has already started, so the failure surfaced as a crashed run rather than as a
failing check. Every test in the suite passed and the gates were clean.

A prompt that cannot be rendered is a run that cannot start, which is the cheapest possible thing to
test and was the most expensive one to discover.
"""

from __future__ import annotations

import string

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.generate_prompt import _screen_tier, render_prompt  # noqa: PLC2701
from merlin.targetgen.target_experiment import load_capability_manifest, load_target_experiment

#: The arms a gemmini run is launched with. `merlin_assisted` is the one the CIRCT bundle uses.
ARMS = ("baseline", "merlin_assisted")


def _gemmini():
    te = load_target_experiment(repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    return te, load_capability_manifest("gemmini")


@pytest.mark.parametrize("arm", ARMS)
def test_the_brief_renders_for_every_arm(arm):
    te, man = _gemmini()
    text = render_prompt(te, man, "full", arm)
    assert text.strip(), f"arm {arm!r} rendered an empty brief"


def test_no_literal_brace_is_read_as_a_placeholder():
    """Every placeholder the template declares must be a plain identifier.

    This is the check that fails on prose braces: a JSON example leaves a 'placeholder' whose name is
    `"status"` -- quotes, colon and all -- which is not an identifier and is never a real field.
    """
    from merlin.targetgen.generate_prompt import _TEMPLATE  # noqa: PLC2701 -- the thing under test

    fields = {f for _, f, _, _ in string.Formatter().parse(_TEMPLATE) if f}
    bad = sorted(f for f in fields if not f.replace("_", "").isalnum())
    assert not bad, (
        f"these look like prose braces read as format placeholders: {bad}. Double them ({{{{ and }}}}) "
        "so str.format emits them literally -- an unescaped one makes render_prompt raise and the run "
        "dies after launch"
    )


def test_the_wait_contract_reaches_the_agent():
    """The resume contract is only useful if it is actually in the brief the agent reads."""
    te, man = _gemmini()
    text = render_prompt(te, man, "full", "merlin_assisted")
    assert "--attach" in text, "the brief no longer teaches how to resume an interrupted self-check wait"
    assert "600s" in text, "the brief no longer states the shell ceiling the resume contract exists for"


def test_the_per_edit_command_names_the_cheap_screen():
    """The brief's CONCRETE command must be the cheap one; prose alone did not hold.

    The surrounding text already said "Iterate FAST -- smallest scope, cheapest checks first" and the
    tier ladder already showed the screen. The recommended invocation named no tier, so it inherited
    the certifying default, and an agent following the brief literally ran every self-check on the
    slowest engine: measured turnaround 6 min, 7 min, 75 min, 96 min across five requests on one run.
    An instruction the command contradicts is an instruction the command wins.
    """
    te, man = _gemmini()
    tier, sim = _screen_tier(man)
    assert tier and sim, "this target declares a cheap screen tier; the assertion below needs one"
    for arm in ARMS:
        text = render_prompt(te, man, "full", arm)
        loop = [ln for ln in text.splitlines() if "agent_selfcheck.py --submission submission" in ln]
        assert loop, f"arm {arm!r}: the brief recommends no concrete self-check command"
        assert any(f"--tiers {tier}" in ln for ln in loop), (
            f"arm {arm!r}: the per-edit command does not name the cheap screen tier {tier!r}; "
            f"without it the agent inherits the certifying engine on every edit. Got: {loop}"
        )
        assert sim in text, f"arm {arm!r}: the brief never says which simulator the cheap tier {tier!r} runs"
        # ... and the certifying run is shown as the SEPARATE, less frequent step, not as the default
        # thing to do after every edit.
        assert "no `--tiers`" in text, f"arm {arm!r}: the brief no longer says how to run the certifying tier"
