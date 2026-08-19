"""Token / cost / tool-call extraction from a `claude --output-format stream-json` transcript.

Self-contained adaptation of abc-testing's `bench/tokens.py` prefill/decode mandate (kept here so the
standalone harness has no cross-repo import). Parses one run's JSONL transcript and writes
`cost_time_toolcalls.yaml`. Honest: if the transcript has no usage metadata, records
`available: false` with a reason rather than inventing numbers.

Anthropic apples-to-apples mapping (dedupe assistant events by message.id):
    tok_in     = input_tokens + cache_creation_input_tokens     (PREFILL)
    tok_cached = cache_read_input_tokens                        (KV cache hit)
    tok_out    = output_tokens                                  (DECODE, incl. thinking)
Per-model $ projection prices each model at its own rate (the CLI may route some turns to a cheaper
model); cache-write 1.25x, cache-read 0.10x surcharges live only in the dollar projection.

Billing mode is a PARAMETER, not an inference from the model id. A seat-billed run (a ChatGPT/Codex
subscription) is not charged per token, so its dollar figure is reported as ``subscription_notional_usd``
with ``estimated_cost_usd: null`` — money spent and money the same traffic WOULD have cost metered are
different quantities and must never share a field. An unknown model yields no dollar figure at all.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

# USD per token (micro-rates → per-token). Adjust if provider pricing changes.
_RATES = {  # (in_per_tok, out_per_tok)
    "opus": (15e-6, 75e-6), "claude-opus-4-8": (15e-6, 75e-6),
    "sonnet": (3e-6, 15e-6), "claude-sonnet-4-6": (3e-6, 15e-6),
    "haiku": (0.8e-6, 4e-6), "claude-haiku-4-5-20251001": (0.8e-6, 4e-6),
    # Non-Anthropic Bedrock models (APPROXIMATE published rates; keyed on a substring of the
    # inference-profile id so an unknown-to-claude model is not mis-priced at opus rates). Refine if
    # exact Bedrock pricing is confirmed.
    "glm": (0.6e-6, 2.2e-6), "qwen": (0.3e-6, 1.2e-6),
    "nemotron": (0.6e-6, 2.4e-6), "nvidia": (0.6e-6, 2.4e-6),
    "deepseek": (0.5e-6, 1.5e-6), "kimi": (0.6e-6, 2.5e-6),
    "nova-lite": (0.06e-6, 0.24e-6), "nova-pro": (0.8e-6, 3.2e-6),
}
_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.10

# The two billing modes a transcript can come from. METERED = charged per token (an API key); the
# notional mode = a subscription seat, where per-token dollars are a projection and never a spend.
METERED = "metered"
SUBSCRIPTION_NOTIONAL = "subscription_notional"


def _load_price_overrides() -> dict[str, tuple[float, float]]:
    """Overlay rates from ``AET_PRICE_TABLE`` (the shared bedrock_prices file aet's PriceTable also reads),
    so this estimate and aet's agree on every model — ONE source of truth, not two divergent tables. The
    file gives USD per MILLION tokens ([input, output, ...] or a dict); converted to per-token here.
    Absent/malformed → no overlay (built-in ``_RATES`` only), never a crash."""
    try:  # process env wins, then the repo .env (same source the rest of the toolchain reads)
        from merlin.common.paths import _dotenv
        path = (os.environ.get("AET_PRICE_TABLE") or _dotenv().get("AET_PRICE_TABLE") or "").strip()
    except Exception:  # noqa: BLE001
        path = os.environ.get("AET_PRICE_TABLE", "").strip()
    if not path or not Path(path).is_file():
        return {}
    try:
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — malformed price file must never break token accounting
        return {}
    out: dict[str, tuple[float, float]] = {}
    for k, v in (raw.items() if isinstance(raw, dict) else []):
        try:
            i, o = (float(v["input"]), float(v["output"])) if isinstance(v, dict) \
                else (float(v[0]), float(v[1]))
            out[str(k).lower()] = (i / 1e6, o / 1e6)
        except Exception:  # noqa: BLE001 — skip a malformed entry, keep the rest
            continue
    return out


_OVERRIDES: dict[str, tuple[float, float]] | None = None


def _rate(model: str) -> tuple[float, float] | None:
    """USD-per-token ``(in, out)`` for ``model``, or ``None`` when this repo has no price for it.

    Returning ``None`` rather than a default is the fail-closed rule. The previous "conservative"
    fallback priced ANY unrecognized id at opus rates, which turned one ``gpt-5.6-sol`` run into a
    $17.21 "estimate" — a number with no provider behind it. An unpriced model now produces no dollar
    figure and a stated reason.
    """
    global _OVERRIDES
    if _OVERRIDES is None:
        _OVERRIDES = _load_price_overrides()
    m = (model or "")
    for k, v in _OVERRIDES.items():          # shared override (AET_PRICE_TABLE) wins
        if k in m.lower():
            return v
    for k, v in _RATES.items():
        if k in m:
            return v
    return None


def parse_transcript(path: str | Path, *, billing_mode: str = METERED,
                     trust_cli_cost: bool = True) -> dict:
    """Parse a stream-json JSONL transcript → token/cost/tool-call summary (honest if absent).

    ``billing_mode`` comes from the DRIVER that produced the transcript (see the harness's
    ``_billing_mode``), because how a run is billed is a property of the account/CLI, not of the
    model string.
    """
    p = Path(path)
    if not p.is_file():
        return {"available": False, "reason": f"transcript not found: {p}"}
    seen: set[str] = set()
    stream_by_model: dict[str, dict] = defaultdict(
        lambda: {"input": 0, "cache_create": 0, "cache_read": 0, "output": 0, "reasoning": 0,
                 "messages": 0})
    tool_calls = 0
    thinking_blocks = 0
    any_usage = False
    n_events = 0
    result_usage: dict | None = None      # authoritative, SUBAGENT-INCLUSIVE per-model usage (result event)
    # The CLI's own total_cost_usd is authoritative ONLY when the CLI is billing the model it thinks it
    # is running. Drive a foreign model through it (an agentic harness pointed at a proxy) and the figure
    # is priced against the CLI's own catalogue: a nemotron round on Bedrock, whose real cost is cents,
    # was reported by the claude CLI as $21.68 -- enough to trip a campaign's spend cap and terminate a
    # healthy run for a reason that never happened. Callers driving a bridged model pass
    # trust_cli_cost=False so the dollars come from tokens x the real model's rate instead.
    result_cost = None                    # the CLI's true total_cost_usd
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue
        n_events += 1
        if evt.get("type") == "result":
            # The terminal `result` event carries the AUTHORITATIVE per-model totals — the orchestrator PLUS
            # every delegated sub-agent / background tier — and the true cost. Streamed `assistant` usage is
            # a partial streaming artifact (its output_tokens is far too low) and NEVER includes the subagent
            # tiers, so prefer this when present. (The fallback below covers a truncated run with no result.)
            mu = evt.get("modelUsage")
            if isinstance(mu, dict) and mu:
                result_usage = mu
            if evt.get("total_cost_usd") is not None:
                result_cost = evt.get("total_cost_usd")
            continue
        if evt.get("type") != "assistant":
            continue
        msg = evt.get("message") or {}
        for block in msg.get("content", []) or []:
            bt = block.get("type") if isinstance(block, dict) else None
            if bt == "tool_use":
                tool_calls += 1
            elif bt == "thinking":
                thinking_blocks += 1
        mid = msg.get("id") or ""
        if not mid or mid in seen:
            continue
        seen.add(mid)
        u = msg.get("usage") or {}
        if u:
            any_usage = True
        m = stream_by_model[msg.get("model") or "unknown"]
        m["input"] += int(u.get("input_tokens", 0) or 0)
        m["cache_create"] += int(u.get("cache_creation_input_tokens", 0) or 0)
        m["cache_read"] += int(u.get("cache_read_input_tokens", 0) or 0)
        m["output"] += int(u.get("output_tokens", 0) or 0)
        # Reasoning tokens are a SUBSET of output (the provider bills them inside it); carried
        # alongside so effort is visible, never added into any total.
        m["reasoning"] += int(u.get("reasoning_output_tokens", 0) or 0)
        m["messages"] += 1

    # Token/cost accounting: PREFER the subagent-inclusive result-event usage; fall back to the streamed
    # per-assistant aggregation only when the run produced no result event (killed/truncated mid-turn).
    if result_usage:
        usage_source = "result_event"
        by_model: dict[str, dict] = defaultdict(
            lambda: {"input": 0, "cache_create": 0, "cache_read": 0, "output": 0, "reasoning": 0,
                     "messages": 0})
        for model, mu in result_usage.items():
            if not isinstance(mu, dict):
                continue
            m = by_model[model]
            m["input"] += int(mu.get("inputTokens", 0) or 0)
            m["cache_create"] += int(mu.get("cacheCreationInputTokens", 0) or 0)
            m["cache_read"] += int(mu.get("cacheReadInputTokens", 0) or 0)
            m["output"] += int(mu.get("outputTokens", 0) or 0)
        any_usage = True
    else:
        usage_source = "assistant_stream"
        by_model = stream_by_model

    if not any_usage:
        return {"available": False, "reason": "no usage metadata in transcript",
                "n_events": n_events, "tool_calls": tool_calls,
                "thinking_blocks": thinking_blocks}

    tok_in = sum(m["input"] + m["cache_create"] for m in by_model.values())
    tok_cached = sum(m["cache_read"] for m in by_model.values())
    tok_out = sum(m["output"] for m in by_model.values())
    # Reasoning is only ever reported on the streamed assistant events, so it is summed from there even
    # when the authoritative totals come from the result event. It is a SUBSET of output, never added in.
    tok_reason = sum(m["reasoning"] for m in stream_by_model.values())
    unpriced: list[str] = []
    if usage_source == "result_event" and result_cost is not None and trust_cli_cost:
        cost = float(result_cost)                 # the CLI's authoritative, subagent-inclusive total
    else:
        cost = 0.0
        for model, m in by_model.items():
            rate = _rate(model)
            if rate is None:
                unpriced.append(model)            # fail closed: no rate -> no dollar figure at all
                continue
            rin, rout = rate
            cost += m["input"] * rin
            cost += m["cache_create"] * rin * _CACHE_WRITE_MULT
            cost += m["cache_read"] * rin * _CACHE_READ_MULT
            cost += m["output"] * rout
    rec = {
        "available": True,
        "usage_source": usage_source,   # 'result_event' (subagent-inclusive) | 'assistant_stream' (fallback)
        "tokens_input": tok_in, "tokens_cached": tok_cached, "tokens_output": tok_out,
        "tokens_total": tok_in + tok_cached + tok_out,
        # tool_calls/thinking are TOP-LEVEL agent only — subagent tool_use never appears in the top-level
        # stream and the result event carries no per-subagent tool count, so this cannot include them.
        "tool_calls": tool_calls, "thinking_blocks": thinking_blocks,
        "subagent_tool_calls_tracked": False,
        "unique_messages": len(seen),
        "billing_mode": billing_mode,
        "tokens_native_by_model": {k: dict(v) for k, v in by_model.items()},
        "n_events": n_events,
    }
    if tok_reason:
        rec["tokens_reasoning"] = tok_reason      # subset of tokens_output; not added to any total
    if billing_mode != METERED:
        # Checked BEFORE the missing-price case: for a seat run both are usually true (a subscription
        # model has no metered rate), and "no price entry" is the wrong reason to print -- it reads as a
        # gap in the price table someone should fill, when the real answer is that this traffic is not
        # billed per token at all. A seat's dollars are notional; the spend field stays empty so no
        # aggregator can sum them into a budget.
        rec["estimated_cost_usd"] = None
        rec["cost_unavailable_reason"] = (
            f"{billing_mode}: a subscription seat is not billed per token; any dollar figure is what the "
            "same traffic would have cost metered, not money spent")
        if not unpriced:
            rec["subscription_notional_usd"] = round(cost, 4)
        else:
            rec["cost_unavailable_reason"] += ("; no metered rate for "
                                               + ", ".join(sorted(unpriced)) + " either")
    elif unpriced:
        # No rate for a model that actually ran ⇒ no dollar figure, with the gap named.
        rec["estimated_cost_usd"] = None
        rec["cost_unavailable_reason"] = ("no price entry for model(s): "
                                          + ", ".join(sorted(unpriced)))
    else:
        rec["estimated_cost_usd"] = round(cost, 4)
    return rec


def write_cost_yaml(summary: dict, out: str | Path, *, wall_time_seconds=None,
                    model=None, exit_code=None) -> None:
    import yaml
    rec = {"model": model, "wall_time_seconds": wall_time_seconds, "exit_code": exit_code, **summary}
    Path(out).write_text(yaml.safe_dump(rec, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Extract tokens/cost/tool-calls from a transcript")
    ap.add_argument("transcript")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wall", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--billing-mode", default=METERED,
                    choices=[METERED, SUBSCRIPTION_NOTIONAL],
                    help="how the run is billed (a subscription seat reports notional dollars only)")
    a = ap.parse_args(argv)
    s = parse_transcript(a.transcript, billing_mode=a.billing_mode)
    if a.out:
        write_cost_yaml(s, a.out, wall_time_seconds=a.wall, model=a.model)
        print(f"wrote {a.out}: available={s.get('available')}")
    else:
        print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
