"""Make a multi-result `linalg` op ingestible: strip the parentheses MLIR prints around its results.

WHY THIS EXISTS. MLIR prints a structured op's result types PARENTHESISED when there is more than
one -- `} -> (tensor<1xi64>, tensor<1xi64>)` -- and xDSL 0.68.0's `linalg.generic` parser reads that
list with `Delimiter.NONE`, which accepts the BARE form `} -> tensor<1xi64>, tensor<1xi64>` and
nothing else. So a perfectly valid module fails to parse, with an error that names neither the cause
nor the op: `Expected '->'`, pointing at the NEXT operation.

MEASURED consequence: exactly ONE such op -- an `aten.min.dim` lowering, which returns values AND
indices -- made a whole SmolVLA graph uncompilable, and the failure read as a frontend limitation
about `tensor.expand_shape` because that is the op the parser reached before giving up. Any argmin,
argmax, min.dim, max.dim, sort or topk lowering emits the same shape, so this is a class of models,
not one file.

WHAT IT DOES, AND WHAT IT REFUSES TO TOUCH. It removes one pair of parentheses, only where a region
close is followed by an arrow and an open parenthesis (`}` `->` `(`). Two other things in the same
text also spell `->` followed by `(` and MUST survive untouched:

* every `affine_map<(d0) -> (d0)>` -- its arrow follows `)`, never `}`;
* `func.func @f(...) -> (T1, T2)` -- likewise, and xDSL parses that form correctly already.

Scanned STRUCTURALLY with an explicit cursor and a parenthesis-depth count, not by pattern matching:
a pattern narrow enough to be safe here would be wide enough to eat an affine map somewhere else,
and this tree has been bitten by exactly that. The count is returned so a caller can report what it
changed rather than silently rewriting a module.
"""
from __future__ import annotations

__all__ = ["normalize_multi_result_linalg"]


def _skip_space(text: str, at: int) -> int:
    while at < len(text) and text[at] in " \t\r\n":
        at += 1
    return at


def _matching_paren(text: str, opened_at: int) -> int | None:
    """Index of the `)` matching the `(` at ``opened_at``, or None when unbalanced.

    Result types are tensor types, which carry no parentheses of their own, so a depth count over
    `(`/`)` is sufficient and needs no awareness of `<`/`>`.
    """
    depth = 0
    at = opened_at
    while at < len(text):
        char = text[at]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return at
        at += 1
    return None


def normalize_multi_result_linalg(text: str) -> tuple[str, int]:
    """``(text, n)`` with ``n`` parenthesised result lists un-parenthesised.

    Only a `}` `->` `(` sequence is rewritten. Returns the input unchanged with ``n == 0`` when there
    is nothing of that shape, so a caller can tell "nothing to do" from "rewrote something".
    """
    out: list[str] = []
    at = 0
    changed = 0
    while True:
        brace = text.find("}", at)
        if brace == -1:
            out.append(text[at:])
            return "".join(out), changed
        cursor = _skip_space(text, brace + 1)
        if not text.startswith("->", cursor):
            out.append(text[at:brace + 1])
            at = brace + 1
            continue
        cursor = _skip_space(text, cursor + 2)
        if cursor >= len(text) or text[cursor] != "(":
            out.append(text[at:brace + 1])
            at = brace + 1
            continue
        close = _matching_paren(text, cursor)
        if close is None:                      # unbalanced: leave it for the parser to report
            out.append(text[at:brace + 1])
            at = brace + 1
            continue
        out.append(text[at:cursor])            # everything up to and including the arrow
        out.append(text[cursor + 1:close])     # the type list, without its parentheses
        at = close + 1
        changed += 1
