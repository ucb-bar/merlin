"""Structured output from an LLM/agent turn — constrain + parse, never regex-scrape prose.

The old pattern was: run a free-text turn, then regex a JSON object/array out of the reply
(``re.search(r"\\{.*\\}")`` and friends). That is brittle (greedy, fence-blind, prose-sensitive)
and unprincipled. This module replaces it two ways:

  * :func:`parse_json` / :func:`extract_code_block` — regex-free *extraction* from a reply that may
    be fenced or wrapped in prose. A leading ```` ``` ```` fence is stripped structurally; a JSON
    value embedded in prose is found by a string-aware balanced-delimiter scan (the first complete
    ``{...}``/``[...]``), not a greedy ``.*``.
  * :func:`structured_agent_call` — the *proper* path when we control the turn: instruct the model
    to answer with ONLY a JSON value matching a JSON Schema, parse it, validate it, and retry once
    with the error fed back. Callers keep a deterministic fallback: on exhaustion it raises
    :class:`StructuredOutputError` rather than returning a silently-wrong value.

No ``re`` here by construction — this is part of the de-regex sweep.
"""
from __future__ import annotations

import json
from typing import Any, Callable

_RAISE = object()  # sentinel: no tolerant default -> raise on failure


class StructuredOutputError(RuntimeError):
    """The agent was unavailable, or never produced parseable/schema-valid output."""


def strip_code_fence(text: str) -> str:
    """Return the body of the first ```` ``` ```` fenced block, else the stripped text (regex-free).

    Handles ```` ```json ```` / ```` ``` ```` openers; if there is no closing fence, returns
    everything after the opener line."""
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    nl = s.find("\n")
    if nl == -1:
        return s
    body = s[nl + 1:]
    end = body.rfind("```")
    return (body[:end] if end != -1 else body).strip()


def locate_json(s: str) -> str | None:
    """The first complete balanced ``{...}`` or ``[...]`` substring, or None (string-aware, regex-free).

    Skips braces inside JSON string literals (honoring ``\\`` escapes), so it stops at the matching
    close of the first structure rather than the last one a greedy ``.*`` would grab."""
    start = open_close = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[start:j + 1]
    return None


def parse_json(text: str | None, *, default: Any = _RAISE, schema: dict | None = None) -> Any:
    """Extract a JSON value from a (possibly fenced / prose-wrapped) agent reply, regex-free.

    Strategy: strip a leading code fence; try ``json.loads`` on the whole; else locate the first
    balanced JSON structure and parse that. With ``schema`` and ``jsonschema`` installed, the value
    is validated. On any failure return ``default`` when one was given (tolerant), otherwise raise
    :class:`StructuredOutputError`."""
    def _fail(msg: str):
        if default is _RAISE:
            raise StructuredOutputError(msg)
        return default

    if not text:
        return _fail("empty agent reply")
    blob = strip_code_fence(text)
    value = _RAISE
    try:
        value = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        located = locate_json(blob)
        if located is not None:
            try:
                value = json.loads(located)
            except (json.JSONDecodeError, TypeError):
                value = _RAISE
    if value is _RAISE:
        return _fail(f"reply was not valid JSON; head={blob[:200]!r}")
    if schema is not None:
        errs = _schema_errors(value, schema)
        if errs:
            return _fail("schema validation failed: " + "; ".join(errs))
    return value


def extract_code_block(text: str, lang: str | None = None, *, default: Any = _RAISE) -> str:
    """Body of the first ```` ``` ```` fenced code block, preferring one tagged ``lang`` (regex-free).

    Falls back to the first block of any language. On no block, returns ``default`` if given, else
    raises :class:`StructuredOutputError`."""
    blocks = _fenced_blocks(text or "")
    if not blocks:
        if default is _RAISE:
            raise StructuredOutputError("agent reply contained no fenced code block")
        return default
    if lang is not None:
        for info, body in blocks:
            if info.strip().lower() == lang.lower():
                return body
    return blocks[0][1]


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """All ```` ``` ```` blocks as (info-string, body). Regex-free line scan."""
    out: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("```"):
            info = stripped[3:].strip()
            body_lines = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                body_lines.append(lines[i])
                i += 1
            out.append((info, "\n".join(body_lines)))
            i += 1  # skip the closing fence
        else:
            i += 1
    return out


def _schema_errors(value: Any, schema: dict) -> list[str]:
    """JSON-Schema validation errors (empty if valid or jsonschema is unavailable)."""
    try:
        import jsonschema
    except Exception:
        return []
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(value)]


def _instruction(schema: dict | None) -> str:
    base = ("Respond with ONLY a single JSON value and nothing else — no prose, no explanation, "
            "no markdown code fences.")
    if schema is not None:
        return base + " It MUST conform to this JSON Schema:\n" + json.dumps(schema, indent=2)
    return base


Runner = Callable[[str], "str | None"]  # prompt -> reply text (None when the agent is unavailable)


def structured_agent_call(runner: Runner, prompt: str, schema: dict | None = None, *,
                          retries: int = 1, instruction: str | None = None) -> Any:
    """Run ``runner(prompt)`` demanding a JSON reply matching ``schema``; parse, validate, retry.

    The turn is constrained (the schema/instruction is appended to ``prompt``) instead of the reply
    being regex-scraped afterward. Returns the parsed — and, when a schema is supplied and
    ``jsonschema`` is installed, validated — value. Retries up to ``retries`` times, feeding the
    parse/validation error back into the prompt. Raises :class:`StructuredOutputError` when the
    agent is unavailable or never returns valid output (callers keep a deterministic fallback)."""
    instr = instruction if instruction is not None else _instruction(schema)
    last_err = "no attempt made"
    for attempt in range(retries + 1):
        suffix = instr if attempt == 0 else (
            f"{instr}\n\nYour previous reply was rejected: {last_err}. Reply with JSON only.")
        reply = runner(f"{prompt}\n\n{suffix}")
        if not reply:
            raise StructuredOutputError("agent returned no text (unavailable?)")
        try:
            return parse_json(reply, schema=schema)
        except StructuredOutputError as e:
            last_err = str(e)
    raise StructuredOutputError(f"no schema-valid reply after {retries + 1} attempt(s): {last_err}")
