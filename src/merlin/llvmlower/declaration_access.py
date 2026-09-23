"""Repair access attributes on printer-emitted private tensor declarations.

Callers provide exact symbols and positional access policy. This module knows no
accelerator, dtype, rewrite record or target ABI. It retains the existing simple
printed-declaration format; it is not a general MLIR parser or ABI verifier.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def patch_declaration_arg_attrs(text: str, symbols: Iterable[str], *, argument_access: Sequence[str]) -> str:
    """Put the ``bufferization.access`` attributes back into the printed declarations.

    **This repairs a printer limitation, it is not a design choice.** xDSL stores ``arg_attrs`` on a
    ``func.FuncOp`` correctly and prints them when the function has a body — but for a bodyless
    DECLARATION it prints only the types, so the attributes silently never reach the text mlir-opt
    parses. The consequence is not cosmetic: without them one-shot-bufferize defensively copies the
    weight operand of every routed contraction.

    Done with ``str`` operations rather than a pattern, and anchored on the exact symbols this pass
    emitted rather than on a shape of MLIR — the same discipline ``llvmlower/op_profile.instrument``
    uses. ``passes_xdsl`` already repairs ``tensor.extract_slice`` after the same round-trip, so a
    post-print fixup is the established way to handle the printer here.

    A declaration that cannot be found is left alone and reported by :func:`unpatched_declarations`
    rather than silently skipped.
    """
    out = text
    for sym in symbols:
        head = f"func.func private @{sym}("
        at = out.find(head)
        if at < 0:
            continue
        open_paren = at + len(head) - 1
        close = out.find(")", open_paren)
        if close < 0:
            continue
        inner = out[open_paren + 1 : close]
        if "bufferization.access" in inner:
            continue  # already annotated; re-splitting would double every attribute
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != len(argument_access):
            continue
        annotated = ", ".join(
            f'{p} {{bufferization.access = "{acc}"}}' for p, acc in zip(parts, argument_access, strict=True)
        )
        out = out[: open_paren + 1] + annotated + out[close:]
    return out


def unpatched_declarations(text: str, symbols: Iterable[str]) -> tuple[str, ...]:
    """Symbols whose declaration in ``text`` still carries no access attributes.

    Non-empty means the weight operands of those callees will be copied by bufferization, so a caller
    that cares about it can fail rather than ship the copies.
    """
    missing = []
    for sym in symbols:
        at = text.find(f"func.func private @{sym}(")
        if at < 0:
            missing.append(sym)
            continue
        line_end = text.find("\n", at)
        line = text[at : line_end if line_end > 0 else len(text)]
        if "bufferization.access" not in line:
            missing.append(sym)
    return tuple(missing)
