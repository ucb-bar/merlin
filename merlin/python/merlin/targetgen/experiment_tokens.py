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
    result_usage: dict = {}               # authoritative, SUBAGENT-INCLUSIVE per-model usage, SUMMED
                                          # across every result event in the file (one per round)
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
            # ACCUMULATE, never assign. A multi-round loop concatenates one round transcript per round
            # into a single file, so a finished run carries ONE result event PER ROUND. Overwriting kept
            # only the last round's usage while tool_calls kept accumulating below, which reported a
            # 6-round claude run as 7.6 M tokens / $4.61 when its rounds summed to 50.1 M / $30.38, and
            # made every derived per-action metric meaningless (2 output tokens per action). Codex is
            # unaffected -- it has no result event and uses the streamed fallback.
            mu = evt.get("modelUsage")
            if isinstance(mu, dict) and mu:
                for model, one in mu.items():
                    if not isinstance(one, dict):
                        continue
                    acc = result_usage.setdefault(model, {})
                    for k, v in one.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            acc[k] = acc.get(k, 0) + v
                        else:
                            acc.setdefault(k, v)
            if evt.get("total_cost_usd") is not None:
                result_cost = (result_cost or 0.0) + evt["total_cost_usd"]
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


def _codex_usage(events: list[dict]) -> dict | None:
    """Totals from a `codex exec --json` transcript, or None if it reported none.

    Codex publishes usage ONLY on `turn.completed`. A turn that failed or was cut off carries no
    usage at all, and that is UNPRICED -- not zero. Reporting it as zero would make a timed-out round
    look free, and failed rounds are most of the cost in a repair loop.

    `input_tokens` ALREADY CONTAINS `cached_input_tokens` + `cache_write_input_tokens`, so fresh input
    is obtained by SUBTRACTION. Adding them would double-count the cached prefix -- on a measured
    round here that is 188928 of 221310 input tokens, an 85% overstatement.

    `reasoning_output_tokens` is a SUBSET of `output_tokens`; it is recorded beside the total and
    never added to it.
    """
    tot = {"input": 0, "cached": 0, "cache_write": 0, "output": 0, "reasoning": 0}
    turns = completed = 0
    for e in events:
        if e.get("type") == "turn.started":
            turns += 1
        if e.get("type") != "turn.completed":
            continue
        u = e.get("usage") or {}
        if not u:
            continue
        completed += 1
        tot["input"] += int(u.get("input_tokens", 0) or 0)
        tot["cached"] += int(u.get("cached_input_tokens", 0) or 0)
        tot["cache_write"] += int(u.get("cache_write_input_tokens", 0) or 0)
        tot["output"] += int(u.get("output_tokens", 0) or 0)
        tot["reasoning"] += int(u.get("reasoning_output_tokens", 0) or 0)
    if not completed:
        return None
    fresh = tot["input"] - tot["cached"] - tot["cache_write"]
    return {
        "tokens_input": max(fresh, 0),
        "tokens_cached": tot["cached"],
        "tokens_cache_write": tot["cache_write"],
        "tokens_output": tot["output"],
        "tokens_reasoning": tot["reasoning"],
        "turns_started": turns,
        "turns_completed": completed,
        # False when a turn produced no usage event: its tokens are unknown, not zero.
        "usage_complete": completed >= max(turns, 1),
        "input_included_cached": True,
        # Codex documents reasoning as part of output_tokens, and the measured data agrees
        # (reasoning <= output on every round). Recorded beside the total, never added to it.
        "reasoning_is_subset_of_output": True,
    }


def _opencode_usage(events: list[dict]) -> dict | None:
    """Totals from an `opencode run --format json` transcript, or None if it reported none.

    opencode reports usage PER STEP on `step-finish` parts, so the run total is a sum over steps
    rather than one terminal event. It also carries its own per-step `cost`, which is used only when
    the CLI is billing the model it is actually running.
    """
    tot = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    cost = 0.0
    steps = 0
    have_cost = False
    for e in events:
        part = e.get("part") or {}
        tk = part.get("tokens")
        if not isinstance(tk, dict):
            continue
        steps += 1
        tot["input"] += int(tk.get("input", 0) or 0)
        tot["output"] += int(tk.get("output", 0) or 0)
        tot["reasoning"] += int(tk.get("reasoning", 0) or 0)
        cache = tk.get("cache") or {}
        tot["cache_read"] += int(cache.get("read", 0) or 0)
        tot["cache_write"] += int(cache.get("write", 0) or 0)
        if isinstance(part.get("cost"), (int, float)):
            cost += float(part["cost"])
            have_cost = True
    if not steps:
        return None
    return {
        "tokens_input": tot["input"],
        "tokens_cached": tot["cache_read"],
        "tokens_cache_write": tot["cache_write"],
        "tokens_output": tot["output"],
        "tokens_reasoning": tot["reasoning"],
        "steps": steps,
        # opencode reports reasoning as its OWN counter, not a slice of output: a measured gemini run
        # shows reasoning 47377 against output 22608, which cannot be a subset. So it is additive
        # here where codex's is not, and the flag says which -- one field with two meanings would
        # either double-count codex or lose a third of gemini's generated tokens.
        "reasoning_is_subset_of_output": tot["reasoning"] <= tot["output"],
        "usage_complete": True,
        "cli_cost_usd": round(cost, 6) if have_cost else None,
        # Unlike codex, opencode does not document whether `input` nets out the cached prefix. On the
        # arms measured here cache_read is 0 (Bedrock allows cachePoint only for Anthropic and Nova),
        # so the two readings coincide and nothing is at stake -- but the flag stays UNKNOWN rather
        # than assuming, because a cached arm would silently over- or under-count by the prefix.
        "input_included_cached": None,
    }


#: Transcript readers by driver. Adding a driver is adding an entry, not editing a parser.
_AGENT_READERS = {"codex": _codex_usage, "opencode": _opencode_usage}


def parse_agent_transcript(path: str | Path, *, driver: str, model: str = "",
                           billing_mode: str = METERED) -> dict:
    """Token/cost totals from an agent CLI transcript that is NOT claude stream-json.

    `parse_transcript` reads the claude CLI's `result`/`assistant` event shape. Point it at a
    `codex exec --json` or `opencode run --format json` file and it finds no event it recognises and
    returns `available: False, "no usage metadata in transcript"` -- which is indistinguishable from
    a genuinely usage-free run, and would silently zero a study's entire cost axis while every
    transcript on disk carried complete usage. Both formats are read here instead.

    Returns the same shape as `parse_transcript` so the two are interchangeable downstream.
    """
    p = Path(path)
    if not p.is_file():
        return {"available": False, "reason": f"transcript not found: {p}"}
    reader = _AGENT_READERS.get(driver)
    if reader is None:
        return {"available": False,
                "reason": f"no transcript reader for driver {driver!r}; "
                          f"known: {sorted(_AGENT_READERS)}"}

    events, n_lines = [], 0
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        n_lines += 1
        try:
            events.append(json.loads(line))
        except ValueError:
            continue

    u = reader(events)
    if u is None:
        return {"available": False,
                "reason": f"{driver} transcript carries no usage (turn failed or was cut off); "
                          f"UNPRICED, not zero",
                "n_events": n_lines, "usage_complete": False, "billing_mode": billing_mode}

    tok_in = u["tokens_input"]
    tok_cached = u["tokens_cached"]
    tok_out = u["tokens_output"]
    rec = {
        "available": True,
        "usage_source": f"{driver}_transcript",
        "driver": driver,
        "model": model,
        "tokens_input": tok_in,
        "tokens_cached": tok_cached,
        "tokens_output": tok_out,
        # Cache WRITES are fresh input the provider charged for, so they belong in the total; cached
        # READS are counted separately because they are the cheap half and must never be summed into
        # the fresh-token axis a cost curve is plotted against.
        "tokens_total": tok_in + u.get("tokens_cache_write", 0) + tok_cached + tok_out
                        + (0 if u.get("reasoning_is_subset_of_output", True)
                           else u.get("tokens_reasoning", 0)),
        "billing_mode": billing_mode,
        "n_events": n_lines,
    }
    for k in ("tokens_cache_write", "tokens_reasoning", "reasoning_is_subset_of_output",
              "turns_started", "turns_completed", "steps", "usage_complete",
              "input_included_cached"):
        if k in u:
            rec[k] = u[k]

    rate = _rate(model) if model else None
    cost = None
    if rate is not None:
        rin, rout = rate
        billable_out = tok_out + (0 if u.get("reasoning_is_subset_of_output", True)
                                  else u.get("tokens_reasoning", 0))
        cost = (tok_in * rin + u.get("tokens_cache_write", 0) * rin * _CACHE_WRITE_MULT
                + tok_cached * rin * _CACHE_READ_MULT + billable_out * rout)

    if billing_mode != METERED:
        # A seat is not billed per token. The spend field stays empty so no aggregator can sum a
        # notional figure into a real budget.
        rec["estimated_cost_usd"] = None
        rec["cost_unavailable_reason"] = (
            f"{billing_mode}: a subscription seat is not billed per token; any dollar figure is what "
            "the same traffic would have cost metered, not money spent")
        if cost is not None:
            rec["subscription_notional_usd"] = round(cost, 4)
        else:
            rec["cost_unavailable_reason"] += f"; and no metered rate for {model!r} either"
    elif u.get("cli_cost_usd") is not None:
        # The CLI priced its own call against the provider it actually used.
        rec["estimated_cost_usd"] = u["cli_cost_usd"]
        rec["cost_source"] = f"{driver}_reported"
    elif cost is not None:
        rec["estimated_cost_usd"] = round(cost, 4)
        rec["cost_source"] = "price_table"
    else:
        rec["estimated_cost_usd"] = None
        rec["cost_unavailable_reason"] = f"no price entry for model {model!r}"
    return rec


# --- repricing an already-parsed run --------------------------------------------------------------
# A cost record keeps the per-model TOKEN COUNTS, so its dollar fields are a pure function of those
# counts and the price table. When a rate is added or corrected, the honest move is to RE-DERIVE from
# the stored tokens rather than hand-edit the number -- and to re-derive every copy of it, because the
# run manifest holds a snapshot of the same fields. Two hand-edited copies is exactly how a run ended
# up reporting `no price entry for model(s): gpt-5.6-sol` in its manifest while its own cost file
# carried the priced figure.

#: the money/provenance fields a reprice owns; everything else in a record is left untouched.
MONEY_FIELDS = ("estimated_cost_usd", "subscription_notional_usd", "cost_unavailable_reason",
                "cost_source")

#: the process-block keys a run manifest mirrors from the cost record (see the harness's grader).
_MANIFEST_PROCESS_KEYS = ("wall_time_seconds", "tokens_total", "tokens_input", "tokens_output",
                          "tokens_reasoning", "estimated_cost_usd", "billing_mode",
                          "subscription_notional_usd", "cost_unavailable_reason", "tool_calls")


def _cost_from_tokens(rec: dict) -> tuple[float | None, list[str]]:
    """Price a stored record from its own token counts. Returns ``(cost, unpriced_models)``.

    ``cost`` is ``None`` when the record carries no usable token breakdown at all -- which is NOT the
    same as a zero-cost run, and must never be reported as one.
    """
    by_model = rec.get("tokens_native_by_model")
    if isinstance(by_model, dict) and by_model:
        cost, unpriced = 0.0, []
        for model, m in by_model.items():
            if not isinstance(m, dict):
                continue
            rate = _rate(model)
            if rate is None:
                unpriced.append(model)
                continue
            rin, rout = rate
            cost += int(m.get("input", 0) or 0) * rin
            cost += int(m.get("cache_create", 0) or 0) * rin * _CACHE_WRITE_MULT
            cost += int(m.get("cache_read", 0) or 0) * rin * _CACHE_READ_MULT
            cost += int(m.get("output", 0) or 0) * rout
        return (None if unpriced and cost == 0.0 else cost), unpriced
    # Flat shape (the codex/`turn.completed` path): fresh input is already net of the cached prefix.
    model = rec.get("model")
    if not model:
        return None, []
    rate = _rate(str(model))
    if rate is None:
        return None, [str(model)]
    rin, rout = rate
    billable_out = int(rec.get("tokens_output", 0) or 0)
    if not rec.get("reasoning_is_subset_of_output", True):
        billable_out += int(rec.get("tokens_reasoning", 0) or 0)
    cost = (int(rec.get("tokens_input", 0) or 0) * rin
            + int(rec.get("tokens_cache_write", 0) or 0) * rin * _CACHE_WRITE_MULT
            + int(rec.get("tokens_cached", 0) or 0) * rin * _CACHE_READ_MULT
            + billable_out * rout)
    return cost, []


def price_fields(rec: dict) -> dict:
    """The money fields ``rec`` SHOULD carry, re-derived from its stored tokens + the current table.

    A metered figure the provider's own CLI reported is authoritative and is never overwritten -- our
    list-price table is an estimate and the CLI priced the call it actually made. Such a record is
    returned unchanged, so a reprice can only ever fill in or correct OUR OWN arithmetic.
    """
    billing_mode = rec.get("billing_mode") or METERED
    cli_priced = (billing_mode == METERED and rec.get("estimated_cost_usd") is not None
                  and rec.get("cost_source") not in ("price_table",))
    if cli_priced:
        return {k: rec[k] for k in MONEY_FIELDS if k in rec}

    cost, unpriced = _cost_from_tokens(rec)
    out: dict = {}
    if billing_mode != METERED:
        out["estimated_cost_usd"] = None
        reason = (f"{billing_mode}: a subscription seat is not billed per token; any dollar figure is "
                  "what the same traffic would have cost metered, not money spent")
        if cost is not None and not unpriced:
            out["subscription_notional_usd"] = round(cost, 4)
        else:
            named = ", ".join(sorted(unpriced)) if unpriced else repr(rec.get("model"))
            reason += f"; no metered rate for {named} either"
        out["cost_unavailable_reason"] = reason
    elif cost is None or unpriced:
        out["estimated_cost_usd"] = None
        named = ", ".join(sorted(unpriced)) if unpriced else repr(rec.get("model"))
        out["cost_unavailable_reason"] = f"no price entry for model(s): {named}"
    else:
        out["estimated_cost_usd"] = round(cost, 4)
        out["cost_source"] = "price_table"
    return out


def reprice_run(run_dir: str | Path, *, write: bool = True) -> dict:
    """Re-derive one run's dollar fields and keep its manifest copy in step.

    Returns ``{"path", "changed": {field: (was, now)}, "manifest_changed": {...}}``. With
    ``write=False`` nothing is touched, which is the drift CHECK: a non-empty ``changed`` means the
    file on disk disagrees with what its own tokens price out to.
    """
    import yaml

    d = Path(run_dir)
    cost_path = d if d.is_file() else d / "cost_time_toolcalls.yaml"
    if not cost_path.is_file():
        return {"path": str(cost_path), "changed": {}, "manifest_changed": {},
                "skipped": "no cost_time_toolcalls.yaml"}
    rec = yaml.safe_load(cost_path.read_text(encoding="utf-8")) or {}
    if not rec.get("available", False):
        return {"path": str(cost_path), "changed": {}, "manifest_changed": {},
                "skipped": "run recorded no usage metadata"}

    want = price_fields(rec)
    changed = {k: (rec.get(k), want.get(k)) for k in MONEY_FIELDS
               if (k in want) != (k in rec) or rec.get(k) != want.get(k)}
    if changed and write:
        for k in MONEY_FIELDS:
            rec.pop(k, None)
        rec.update(want)
        cost_path.write_text(yaml.safe_dump(rec, sort_keys=False), encoding="utf-8")

    # The manifest mirrors these fields; re-derive its copy from the SAME record so the two files can
    # never state different money for one run.
    man_changed: dict = {}
    man_path = cost_path.parent / "run_manifest.yaml"
    if man_path.is_file():
        man = yaml.safe_load(man_path.read_text(encoding="utf-8")) or {}
        proc = man.get("process")
        if isinstance(proc, dict):
            merged = {**rec, **want}
            for k in _MANIFEST_PROCESS_KEYS:
                if k in proc and proc.get(k) != merged.get(k):
                    man_changed[k] = (proc.get(k), merged.get(k))
            if man_changed and write:
                for k, (_was, now) in man_changed.items():
                    proc[k] = now
                man_path.write_text(yaml.safe_dump(man, sort_keys=False), encoding="utf-8")
    return {"path": str(cost_path), "changed": changed, "manifest_changed": man_changed}


def write_cost_yaml(summary: dict, out: str | Path, *, wall_time_seconds=None,
                    model=None, exit_code=None) -> None:
    import yaml
    rec = {"model": model, "wall_time_seconds": wall_time_seconds, "exit_code": exit_code, **summary}
    Path(out).write_text(yaml.safe_dump(rec, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Extract tokens/cost/tool-calls from a transcript")
    ap.add_argument("transcript", nargs="?",
                    help="transcript to parse (omit when using --reprice)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wall", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--billing-mode", default=METERED,
                    choices=[METERED, SUBSCRIPTION_NOTIONAL],
                    help="how the run is billed (a subscription seat reports notional dollars only)")
    ap.add_argument("--reprice", nargs="+", metavar="RUN_DIR",
                    help="re-derive the dollar fields of already-parsed runs from their STORED token "
                         "counts and the current price table, and re-sync each run manifest's copy. "
                         "Accepts run dirs or cost_time_toolcalls.yaml paths.")
    ap.add_argument("--check", action="store_true",
                    help="with --reprice: report drift and exit non-zero, writing nothing")
    a = ap.parse_args(argv)

    if a.reprice:
        drifted = 0
        for target in a.reprice:
            r = reprice_run(target, write=not a.check)
            if r.get("skipped"):
                print(f"skip  {r['path']}: {r['skipped']}")
                continue
            if not r["changed"] and not r["manifest_changed"]:
                print(f"ok    {r['path']}")
                continue
            drifted += 1
            verb = "DRIFT" if a.check else "fixed"
            print(f"{verb} {r['path']}")
            for k, (was, now) in r["changed"].items():
                print(f"        cost_time_toolcalls.yaml  {k}: {was!r} -> {now!r}")
            for k, (was, now) in r["manifest_changed"].items():
                print(f"        run_manifest.yaml         {k}: {was!r} -> {now!r}")
        return 1 if (a.check and drifted) else 0

    if not a.transcript:
        ap.error("a transcript is required unless --reprice is given")
    s = parse_transcript(a.transcript, billing_mode=a.billing_mode)
    if a.out:
        write_cost_yaml(s, a.out, wall_time_seconds=a.wall, model=a.model)
        print(f"wrote {a.out}: available={s.get('available')}")
    else:
        print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
