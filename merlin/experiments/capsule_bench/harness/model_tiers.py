"""Model registry + tier presets for the capsule-bench agent drivers.

Ported (NOT imported — the harness venv has no chia/Ray dependency) from ``chia/models/bedrock_config.py``
so the harness can resolve model aliases and tier mixes without pulling in chia. Kept deliberately small: a
``{alias: bedrock-id}`` map + the two default tier presets + a resolver.

A "tier" mirrors Claude Code's REAL delegation model — there is NO per-turn auto-routing. A strong ``primary``
orchestrates; a cheaper ``subagent`` handles delegated sub-tasks (the Task tool, or a bounded delegate loop on
the Converse/OpenCode drivers); a small/fast ``background`` model handles chores (titles/summaries). Anthropic
tiers drive the ``claude`` CLI on Bedrock (``CLAUDE_CODE_SUBAGENT_MODEL`` / ``ANTHROPIC_SMALL_FAST_MODEL``);
non-Anthropic tiers drive the Converse / OpenCode loops.
"""
from __future__ import annotations

from dataclasses import dataclass

# alias -> Bedrock inference-profile / model id. Verified invocable on the experiments account (us-east-1,
# bearer auth) per chia's bedrock_config MODELS registry; mirrored here so the harness resolves aliases
# without importing chia. Keep this in sync with bedrock_agent._CONVERSE_MODELS (the Converse driver's copy).
MODELS = {
    # Anthropic (claude CLI on Bedrock). opus-4-8 / sonnet-5 / opus-5 are listed but NOT invocable on the
    # account, so the tiers below use the invocable 4-6 / 4-5 profiles.
    "opus": "us.anthropic.claude-opus-4-6-v1",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    # Non-Anthropic (Converse / OpenCode).
    "glm5": "zai.glm-5", "glm4.7": "zai.glm-4.7",
    "deepseek": "deepseek.v3.2",              # tools-capable (Sonnet-class reasoning)
    "deepseek-r1": "deepseek.r1-v1:0",        # NO tools (Converse rejects its toolConfig) — not agentic
    "nemotron": "nvidia.nemotron-super-3-120b",
    "kimi": "moonshotai.kimi-k2.5",
    "qwen-coder": "qwen.qwen3-coder-next",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    "nova-lite": "us.amazon.nova-lite-v1:0",  # Haiku-tier: cheap/fast, the mechanical/background pick
}

# Aliases that cannot drive an agentic tool loop (the model rejects toolConfig).
NO_TOOLS = frozenset({"deepseek-r1"})


def resolve(model: str) -> str:
    """Alias -> concrete Bedrock id. A raw id (anything not in :data:`MODELS`) passes through unchanged."""
    return MODELS.get(model, model)


@dataclass(frozen=True)
class Tier:
    """A (primary, subagent, background) model mix. ``None`` tiers stay unset (the driver keeps its default)."""
    primary: str
    subagent: str | None = None
    background: str | None = None

    def resolved(self) -> "Tier":
        return Tier(resolve(self.primary),
                    resolve(self.subagent) if self.subagent else None,
                    resolve(self.background) if self.background else None)


# The two default tier mixes (mirror chia's ANTHROPIC_TIER / NON_ANTHROPIC_TIER). Among the non-Anthropic
# set, glm5 / deepseek are the Sonnet-class reasoners; nova-lite is the Haiku-tier (cheap/mechanical).
ANTHROPIC_TIER = Tier(primary="opus", subagent="sonnet", background="haiku")
NON_ANTHROPIC_TIER = Tier(primary="glm5", subagent="qwen-coder", background="nova-lite")
