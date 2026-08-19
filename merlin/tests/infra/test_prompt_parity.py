"""Arms of one campaign must differ by the MODEL, not by their instructions.

The claude driver pipes TASK.md and nothing else. The codex driver adds a kickoff line and deliberately
authors no instruction file -- its module docstring says an arm that silently gets extra instructions is
not the same arm. The opencode driver used to layer ~2 KB of strategy on top of TASK.md, and every piece
of it pushed toward the failure mode the open models exhibited: invented turn scarcity, an instruction to
write early rather than investigate, and a prohibition on the RTL-facts generators that this arm's own
bundle manifest explicitly grants.

These guard the boundary: the driver may say which of ITS tools performs an action; it may not tell the
model how to approach the problem, how to pace itself, or which granted tooling to avoid.
"""
from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


@pytest.fixture(scope="module")
def prompt():
    src = _HARNESS / "opencode_agent.py"
    if not src.is_file():
        pytest.skip(f"{src} not present")
    import sys
    sys.path.insert(0, str(_HARNESS))
    try:
        spec = importlib.util.spec_from_file_location("opencode_agent_prompt_test", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    class _TE:
        target = "sometarget"
    return mod._system_prompt(_TE())


@pytest.mark.parametrize("banned, why", [
    ("do not over-explore", "the 20/20 run spent 33 of 45 actions reading before its first edit"),
    ("LIMITED number of tool turns", "no such cap exists in this driver; the scarcity was invented"),
    ("as early as possible", "pacing pressure is strategy, and the other arms never receive it"),
    ("Do NOT run the RTL-facts GENERATORS", "the arm's bundle manifest grants those generators"),
])
def test_the_driver_prompt_carries_no_strategy(prompt, banned, why):
    assert banned.lower() not in prompt.lower(), f"driver prompt steers the model: {why}"


def test_the_driver_prompt_defers_to_the_shared_task_file(prompt):
    """TASK.md is what every arm reads; the driver prompt must point at it, not replace it."""
    assert "TASK.md" in prompt
    assert len(prompt) < 1200, (
        f"driver prompt is {len(prompt)} chars — long enough to be carrying task content that belongs in "
        f"TASK.md, where every driver reads it")


def test_no_driver_forbids_tooling_the_bundle_grants(prompt):
    """The arm-4 bundle grants the RTL-facts generators; no driver may countermand its own grant.

    Checks the prompt the model RECEIVES, not the module source -- the source legitimately quotes the old
    wording to record why it was removed."""
    manifest = (merlin_dir() / "experiments/capsule_bench/targets/gemmini/input_bundles"
                / "merlin_assisted_rtlchecks_hwbringup_v0" / "input_bundle_manifest.yaml")
    if not manifest.is_file():
        pytest.skip("gemmini arm-4 bundle not present")
    if "gen_isa_module" not in manifest.read_text():
        pytest.skip("this bundle does not grant the generators")
    lowered = prompt.lower()
    for phrase in ("do not run the rtl-facts generators", "gen_isa_module", "gen_numeric_facts"):
        assert phrase not in lowered, \
            "the bundle grants the generators; the driver prompt must not mention or restrict them"
