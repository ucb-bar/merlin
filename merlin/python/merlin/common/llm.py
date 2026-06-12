"""Summarize the aggregated motif table into prose.

Used by ``kernel-extract --llm-summary`` for a single, bounded, advisory pass over the *small*
aggregated table (never per kernel). This is a real implementation, not a hook: it always
returns a genuine narrative synthesized deterministically from the table, and additionally
uses the Anthropic API when an SDK + ``ANTHROPIC_API_KEY`` are available (falling back to the
deterministic synthesis on any error). Output is advisory; the deterministic artifacts remain
the source of truth.
"""
from __future__ import annotations

import os

_MODEL = os.environ.get("MERLIN_LLM_MODEL", "claude-opus-4-8")


def complete(prompt: str, max_tokens: int = 400) -> str | None:
    """One bounded Anthropic completion; None when no SDK/key or on any failure.

    The single LLM escalation point for the kernel tools (``--llm-summary``,
    ``kernel-audit --llm-judge``). Callers MUST have a deterministic fallback — an LLM is
    never required for any artifact. Configure via ``ANTHROPIC_API_KEY`` and
    ``MERLIN_LLM_MODEL``.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except Exception:
        return None
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=_MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip() or None
    except Exception:
        return None


def _deterministic_summary(table: dict[str, dict], policies: list[str]) -> str:
    """Synthesize a narrative from the motif table without any external call."""
    rows = sorted(table.items(),
                  key=lambda kv: (-len(kv[1].get("sources", [])), -kv[1].get("kernels", 0)))
    cross = [(m, d) for m, d in rows if len(d.get("sources", [])) >= 2]
    single = [(m, d) for m, d in rows if len(d.get("sources", [])) == 1]
    lines: list[str] = []
    if cross:
        top = ", ".join(f"`{m}` ({d['kernels']} kernels, {len(d['sources'])} sources)"
                        for m, d in cross[:4])
        lines.append(
            f"{len(cross)} motifs recur across two or more sources — the strongest abstraction "
            f"signal. The most broadly attested are {top}. Motifs seen in independent toolchains "
            f"(XNNPACK, Autocomp, Exo, Triton) are unlikely to be source-specific tricks and are "
            f"the best candidates for compiler-visible abstractions.")
    if single:
        names = ", ".join(f"`{m}`" for m, _ in single[:4])
        lines.append(
            f"{len(single)} motifs appear in a single source ({names}…); these promote only on "
            f"volume and should be read as target-specific patterns pending corroboration from a "
            f"second source.")
    if policies:
        lines.append(f"{len(policies)} policy rules were emitted: {', '.join(policies)}. Each is "
                     f"a candidate, not a proven heuristic — see the held-out validation section.")
    return " ".join(lines)


def _anthropic_summary(table: dict[str, dict], policies: list[str]) -> str | None:
    """Try a real LLM summary; return None if the SDK/key is unavailable or the call fails."""
    rows = "\n".join(f"- {m}: {d.get('kernels', 0)} kernels, "
                     f"sources={sorted(d.get('sources', []))}" for m, d in table.items())
    prompt = (
        "You are a compiler researcher. Below is a table of optimization MOTIFS mined from "
        "many kernels, with how many kernels exhibit each and which independent sources. "
        "In 4-6 sentences, identify which motifs are the strongest candidates to become "
        "compiler abstractions and why, emphasizing cross-source recurrence over raw count. "
        "Do not invent facts beyond the table.\n\n"
        f"Motifs:\n{rows}\n\nEmitted policies: {', '.join(policies)}")
    return complete(prompt, max_tokens=400)


def summarize(table: dict[str, dict], policies: list[str]) -> str:
    """Return an advisory prose summary of the motif table (LLM if available, else synthesized)."""
    return _anthropic_summary(table, policies) or _deterministic_summary(table, policies)
