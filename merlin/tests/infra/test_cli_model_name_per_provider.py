"""The claude CLI's ``--model`` name depends on the PROVIDER, and getting it wrong kills the agent turn.

`model_tiers.MODELS` maps an alias to a Bedrock inference profile. That is the right answer only when the
CLI is pointed at Bedrock; under a subscription the CLI talks to Anthropic directly and resolves its own
aliases, so handing it a Bedrock id names a model it cannot serve.

Measured 2026-09-01: `--model opus --provider subscription` reached the CLI as
`us.anthropic.claude-opus-4-6-v1` and the round ended immediately with "There's an issue with the selected
model ... It may not exist or you may not have access to it." The agent turn took 0 ms and cost nothing,
and the round then graded an unchanged submission and reported `NOT CONFORMANT -- failing: isa_tools_used,
cca_used, ...` -- so a provider/name mismatch presented itself as an agent that had not done its work.
That is the failure this test exists to prevent, and it is the second time in this bench that a harness
limitation has been reported as an agent defect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"))

agent_bridge = pytest.importorskip("agent_bridge")
model_tiers = pytest.importorskip("model_tiers")


ANTHROPIC_ALIASES = ("opus", "sonnet", "haiku")


@pytest.mark.parametrize("alias", ANTHROPIC_ALIASES)
def test_a_subscription_run_never_gets_a_bedrock_id(alias):
    got = agent_bridge.claude_model_name(alias, provider="subscription")
    assert got == alias, f"{alias!r} should reach a subscription CLI unchanged, got {got!r}"
    assert not got.startswith("us."), f"{got!r} is a Bedrock inference profile, not a CLI model name"
    assert "anthropic.claude" not in got, f"{got!r} is Bedrock-shaped"


@pytest.mark.parametrize("alias", ANTHROPIC_ALIASES)
def test_a_bedrock_run_still_gets_the_bedrock_id(alias):
    """The fix must not change the Bedrock path, which is what every existing arm used."""
    got = agent_bridge.claude_model_name(alias, provider="bedrock")
    assert got == model_tiers.MODELS[alias], got


def test_the_default_provider_preserves_the_previous_behaviour():
    """Callers that do not pass a provider must behave exactly as before the parameter existed."""
    assert agent_bridge.claude_model_name("opus") == model_tiers.MODELS["opus"]


def test_a_raw_cli_model_name_passes_through_under_both_providers():
    for provider in ("subscription", "bedrock"):
        assert agent_bridge.claude_model_name("claude-opus-5", provider=provider) == "claude-opus-5"


def test_the_loop_threads_the_provider_into_the_cli_invocation():
    """The parameter is useless unless the caller passes it; pin that it does."""
    src = (repo_root() / "merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py").read_text()
    assert "claude_model_name(model, provider=_PROVIDER)" in src, \
        "the claudecode branch must resolve its CLI model name against the run's provider"
    assert "_PROVIDER = a.provider" in src, "main() must record the provider it was invoked with"
