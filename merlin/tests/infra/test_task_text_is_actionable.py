"""Instructions handed to an agent must be followable as written.

Two defects in the arm-4 task text cost real runs. Its addendum told the agent to invoke its tool with
`/path/to/merlin/.venv/bin/python` -- a scrubbed placeholder that resolves to nothing, and advice that is
wrong anyway, since the runner prepends the right interpreter for a `language: python` tool by itself. And
nothing in the task stated the manifest's actual shape, while the prose called the four operations "the 4
entrypoints", inviting the reader to nest them under `entrypoints`. A live Nemotron round did exactly that
and its package was REJECTED at the contract gate before a single capsule graded -- a run that measured
nothing about its compiler. The winning codex run avoided it only by going and reading the schema itself.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir

_TARGETS = merlin_dir() / "experiments/capsule_bench/targets"


def _agent_facing_text():
    """Every markdown file an agent is handed: bundle prompts/addenda and the per-target task text."""
    for pat in ("*/input_bundles/*/*.md", "*/task/*.md"):
        yield from sorted(_TARGETS.glob(pat))


def test_no_unresolved_path_placeholder_reaches_an_agent():
    """A `/path/to/...` literal is a scrub artefact: the agent probes it, finds nothing, and burns turns."""
    bad = [f"{p.parent.name}/{p.name}" for p in _agent_facing_text() if "/path/to/" in p.read_text()]
    assert not bad, ("agent-facing text still contains an unresolved /path/to/ placeholder: "
                     + ", ".join(bad))


def test_the_arm_task_does_not_tell_the_agent_to_bake_an_interpreter_into_argv():
    """`oot_runner._needs_interpreter` supplies it; instructing otherwise produces a manifest that fails."""
    # NO PRE-GUARD. This used to run the assertion only for files mentioning BOTH "argv" and
    # ".venv/bin/python"; no file in the corpus matched both, so the assertion executed zero times and
    # the test was green by construction. The forbidden instruction is forbidden everywhere -- and a
    # file that phrases the interpreter differently is exactly the one the guard would have skipped.
    bad = [f"{p.parent.name}/{p.name}" for p in _agent_facing_text()
           if "invoke your tool with **that interpreter**" in p.read_text()]
    assert not bad, ("agent-facing text tells the agent to put an interpreter in argv; the runner "
                     "already does that for a `language: python` tool: " + ", ".join(bad))


@pytest.mark.parametrize("bundle", ["merlin_assisted_rtlchecks_hwbringup_v0"])
def test_the_manifest_shape_is_stated_where_the_agent_will_read_it(bundle):
    """`entrypoints` takes ONE key. Leaving that implicit cost a live round its entire grade."""
    d = _TARGETS / "gemmini" / "input_bundles" / bundle
    if not d.is_dir():
        pytest.skip(f"{bundle} not present")
    text = "\n".join(p.read_text() for p in d.glob("*.md"))
    assert "entrypoints" in text and "tool:" in text, "the manifest's entrypoints shape is never shown"
    assert "manifest.schema.json" in text, (
        "the task never points at the schema that decides whether the package is accepted")
