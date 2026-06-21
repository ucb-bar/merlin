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
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# USD per token (micro-rates → per-token). Adjust if provider pricing changes.
_RATES = {  # (in_per_tok, out_per_tok)
    "opus": (15e-6, 75e-6), "claude-opus-4-8": (15e-6, 75e-6),
    "sonnet": (3e-6, 15e-6), "claude-sonnet-4-6": (3e-6, 15e-6),
    "haiku": (0.8e-6, 4e-6), "claude-haiku-4-5-20251001": (0.8e-6, 4e-6),
}
_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.10


def _rate(model: str) -> tuple[float, float]:
    for k, v in _RATES.items():
        if k in (model or ""):
            return v
    return _RATES["opus"]  # conservative default


def parse_transcript(path: str | Path) -> dict:
    """Parse a stream-json JSONL transcript → token/cost/tool-call summary (honest if absent)."""
    p = Path(path)
    if not p.is_file():
        return {"available": False, "reason": f"transcript not found: {p}"}
    seen: set[str] = set()
    by_model: dict[str, dict] = defaultdict(
        lambda: {"input": 0, "cache_create": 0, "cache_read": 0, "output": 0, "messages": 0})
    tool_calls = 0
    thinking_blocks = 0
    any_usage = False
    n_events = 0
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue
        n_events += 1
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
        m = by_model[msg.get("model") or "unknown"]
        m["input"] += int(u.get("input_tokens", 0) or 0)
        m["cache_create"] += int(u.get("cache_creation_input_tokens", 0) or 0)
        m["cache_read"] += int(u.get("cache_read_input_tokens", 0) or 0)
        m["output"] += int(u.get("output_tokens", 0) or 0)
        m["messages"] += 1

    if not any_usage:
        return {"available": False, "reason": "no usage metadata in transcript",
                "n_events": n_events, "tool_calls": tool_calls,
                "thinking_blocks": thinking_blocks}

    tok_in = sum(m["input"] + m["cache_create"] for m in by_model.values())
    tok_cached = sum(m["cache_read"] for m in by_model.values())
    tok_out = sum(m["output"] for m in by_model.values())
    cost = 0.0
    for model, m in by_model.items():
        rin, rout = _rate(model)
        cost += m["input"] * rin
        cost += m["cache_create"] * rin * _CACHE_WRITE_MULT
        cost += m["cache_read"] * rin * _CACHE_READ_MULT
        cost += m["output"] * rout
    return {
        "available": True,
        "tokens_input": tok_in, "tokens_cached": tok_cached, "tokens_output": tok_out,
        "tokens_total": tok_in + tok_cached + tok_out,
        "tool_calls": tool_calls, "thinking_blocks": thinking_blocks,
        "unique_messages": sum(m["messages"] for m in by_model.values()),
        "estimated_cost_usd": round(cost, 4),
        "tokens_native_by_model": {k: dict(v) for k, v in by_model.items()},
        "n_events": n_events,
    }


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
    a = ap.parse_args(argv)
    s = parse_transcript(a.transcript)
    if a.out:
        write_cost_yaml(s, a.out, wall_time_seconds=a.wall, model=a.model)
        print(f"wrote {a.out}: available={s.get('available')}")
    else:
        print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
