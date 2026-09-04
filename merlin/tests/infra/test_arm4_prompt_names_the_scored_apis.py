"""The arm-4 task text must NAME the APIs arm-4 is scored on.

`conformance.py` scores this arm on whether four specific calls actually happened -- `load_facts`,
`derived_levers`, a scaffold generator, and a read of the verdict's `rtl_checks` block. Measured on the
live round merlincirct_atlasp1arm4c: the generated TASK.md contained NONE of those names, said only
"derive the ISA / mesh / datapath from the granted RTL-extracted facts", and the agent did the sensible
thing -- `sed -n '1,320p' rtl_facts/facts.json`. It got the facts and scored zero on all four checks.

That is the recorded `atlas-isa-grounding-aw6` failure on a new surface: the agent faked an ISA because
the prompt never named the shipped ISA files. An arm graded on calls it was never told to make measures
prompt quality, not the treatment -- so this is a gate, not a comment. If a check in `conformance.py`
starts scoring a new API, this test is where the prompt is made to name it.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.generate_prompt import _enforced_workflow

# Each entry: the check in conformance.py, and the token the prompt must name for it. The scaffold row
# lists alternatives because the check accepts any one of several generators.
SCORED = [
    ("rtl_facts_used", ["load_facts"]),
    ("rtl_derived_levers_used", ["derived_levers"]),
    ("scaffold_generators_used", ["mlir_scaffold.generate", "llvm_plan.generate",
                                  "target_repo.generate_skeleton", "generate_skeleton"]),
    ("rtl_checks_read", ["rtl_checks"]),
]

ARM4 = "merlin_assisted_rtlchecks"


def _arm4_text(target="t"):
    return _enforced_workflow(ARM4, "external_backend", None, target)


@pytest.mark.parametrize("check,tokens", SCORED, ids=[c for c, _ in SCORED])
def test_the_arm4_workflow_names_the_api_the_check_scores(check, tokens):
    text = _arm4_text()
    assert any(t in text for t in tokens), (
        f"conformance.{check} scores a call the arm-4 task text never names ({tokens}); an agent "
        f"cannot be graded on an API it was not told about")


def test_the_verdict_readback_surface_is_named_not_just_the_block():
    """`rtl_checks_read` credits only a read that addresses `qa/verdict.json`. Naming the block without
    naming where it lives leaves the agent to find it."""
    assert "qa/verdict.json" in _arm4_text()


def test_the_names_are_arm4_only():
    """These are arm-4's scored APIs. Naming them to every arm would hand the lower rungs the treatment
    the ladder exists to withhold, and the A/B would measure nothing."""
    for arm in ("raw_baseline", "cpp_merlininfra", "merlin_assisted"):
        text = _enforced_workflow(arm, "external_backend", None, "t")
        assert "load_facts" not in text and "derived_levers" not in text, arm


def test_the_arm4_block_stays_target_agnostic():
    """The API names are fixed; the argument is the target, threaded from the descriptor. Two targets'
    arm-4 blocks must differ only where the target name is substituted."""
    a, b = _arm4_text("alpha"), _arm4_text("beta")
    assert a != b
    assert a.replace("alpha", "<t>") == b.replace("beta", "<t>")
