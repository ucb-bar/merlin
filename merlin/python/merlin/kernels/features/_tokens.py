"""Regex-free lexical helpers for feature extraction.

The feature extractors count *decision tokens* — accumulator registers, accelerator
dispatch opcodes, loop keywords — in heterogeneous kernel source (C / Triton-Py /
Exo-Py). These helpers tokenize the source into maximal word-character runs
*structurally* (the repo's no-regex principle: facts are derived from structure, not
scraped with patterns). A "word char" is ``[A-Za-z0-9_]`` — exactly the class a regex
``\\b`` boundary isolates — so a token here is what ``\\bTOKEN\\b`` used to match.
"""
from __future__ import annotations

from typing import Iterable, Iterator

_WS = " \t\r\n\f\v"


def _is_word_char(c: str) -> bool:
    return c == "_" or c.isalnum()


def iter_word_spans(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield ``(token, start, end)`` for each maximal run of word characters."""
    start: int | None = None
    for i, c in enumerate(text):
        if _is_word_char(c):
            if start is None:
                start = i
        elif start is not None:
            yield text[start:i], start, i
            start = None
    if start is not None:
        yield text[start:], start, len(text)


def identifier_tokens(text: str) -> list[str]:
    """All word-character tokens in ``text``, in source order, with repetition."""
    return [tok for tok, _s, _e in iter_word_spans(text)]


def distinct_registers(text: str, prefix: str) -> int:
    """Count DISTINCT ``<prefix><digits>`` register identifiers (e.g. ``vacc0..vacc3`` -> 4).

    Structural replacement for the historical ``\\b<prefix>(\\d+)\\b`` set-count."""
    return len({t for t in identifier_tokens(text)
                if t.startswith(prefix) and t[len(prefix):].isdigit()})


def count_opcode_uses(text: str, opcodes: Iterable[str]) -> int:
    """Count identifier OCCURRENCES that equal an opcode OR end with ``_<opcode>``.

    The suffix arm captures wrapper-macro names (e.g. ``gemmini_extended_mvin2`` for the
    ``mvin2`` opcode) — the target-agnostic generalization of the historical
    ``\\bmvin[23]\\b|gemmini_extended\\d*_mvin[23]`` alternation."""
    exact = set(opcodes)
    suffixes = tuple(f"_{op}" for op in exact)
    return sum(1 for t in identifier_tokens(text)
               if t in exact or t.endswith(suffixes))


def match_opcodes(text: str, opcodes: Iterable[str]) -> list[str]:
    """The list of identifier occurrences that EXACTLY equal one of ``opcodes`` (order preserved).

    Exact word match only — the historical dispatch regex used ``\\b`` boundaries, so a wrapped
    name like ``gemmini_extended_mvin2`` (no boundary before ``mvin2``) was NOT counted."""
    wanted = set(opcodes)
    return [t for t in identifier_tokens(text) if t in wanted]


def _is_all_ws(text: str, a: int, b: int) -> bool:
    return b > a and all(ch in _WS for ch in text[a:b])


def _next_nonspace(text: str, i: int) -> str:
    n = len(text)
    while i < n and text[i] in _WS:
        i += 1
    return text[i] if i < n else ""


def count_loops(text: str) -> int:
    """Count loop constructs across C and Python sources: C ``for (`` and ``do {``, plus
    Python ``for <name> in`` headers. Reproduces the historical ``\\bfor\\s*\\(``,
    ``\\bdo\\s*\\{`` and ``\\bfor\\s+\\w+\\s+in\\s`` counts, summed."""
    spans = list(iter_word_spans(text))
    n = len(spans)
    total = 0
    for idx in range(n):
        tok, _s, e = spans[idx]
        if tok == "do":
            if _next_nonspace(text, e) == "{":
                total += 1
        elif tok == "for":
            if _next_nonspace(text, e) == "(":
                total += 1
            elif idx + 2 < n:
                _name, name_s, name_e = spans[idx + 1]
                in_tok, in_s, in_e = spans[idx + 2]
                if (_is_all_ws(text, e, name_s)
                        and _is_all_ws(text, name_e, in_s)
                        and in_tok == "in"
                        and in_e < len(text) and text[in_e] in _WS):
                    total += 1
    return total
