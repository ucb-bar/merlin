"""The delegate-to-helpers skill uses the drivers' real helper vocabulary and informs without quotas."""

from __future__ import annotations

from merlin_experiments.phase1 import author_subagents as AS
from merlin_experiments.phase1 import skill_staging as SK
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import codex_context as CTX

SKILL = SK.SOURCE / "delegate-to-helpers"


def test_helper_vocabulary_matches_what_the_lead_is_told():
    text = (SKILL / SK.SKILL_FILE).read_text()
    guidance = AS.round_guidance(4, None)
    for tool in ("spawn_agent", "wait_agent", "followup_task"):
        assert tool in guidance and tool in text
    note = CTX.round_note(CTX.selection(parse_options(["--run-id", "t", "--driver", "codex", "--sandbox", "bwrap"])), 4)
    for phrase in ('fork_turns "none"', 'fork_turns` omitted or `"all"'):
        assert phrase.replace("`", "") in note.replace("`", "") and phrase in text
    assert "variants/NAME" in text and "--submission variants/NAME" in text


def test_brief_elements_and_referenced_skills_and_scripts_exist():
    text = (SKILL / SK.SKILL_FILE).read_text()
    for element in ("**Goal:**", "**Owned files:**", "**Cases and commands:**", "**Skills to open, by name:**",
                    "**Report:**"):
        assert element in text
    skills = SK.collect()
    assert "fix-failure-cluster" in skills and SK.FEEDBACK_SKILL in skills
    assert "scripts/clusters.py" in skills["fix-failure-cluster"] and "clusters.py --brief N" in " ".join(text.split())
    for root in SK.ROOTS.values():
        assert f"{root.as_posix()}/NAME/SKILL.md" in text


def test_information_not_quotas():
    body = SK.frontmatter((SKILL / SK.SKILL_FILE).read_text())[1].lower()
    for word in ("must ", "always use", "at least", "never spawn", "required to"):
        assert word not in body
