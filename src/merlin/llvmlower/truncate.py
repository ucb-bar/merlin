"""Truncate a linalg-on-tensors func to return an intermediate value.

The NaN-bisection tool (and the seed of the real outliner): given the model MLIR text,
produce a variant whose `@forward` returns the SSA value defined by the op at a chosen
*line index*, with everything after it dropped. Lowering + running the truncated module
and checking whether the output is finite localizes the first op that produces NaN —
no torch per-op golden needed (NaN is its own oracle).

Pure-text (operates before the xDSL/iree pipeline), mirroring passes_xdsl's textual
approach, so it composes with `lower_model(..., textual=True)`.

Everything here addresses the text by LINE, because the bisection's unit *is* the line:
the answer it reports is "the op on line N is where the numerics first go wrong". Within a
line the parsing is structural — an SSA-name scan, a balanced-angle-bracket type scan, and
literal `->` anchors — never pattern matching.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

_IDENT_CHARS = frozenset(string.ascii_letters + string.digits + "_")
_TENSOR = "tensor<"
_ARROW = "->"


@dataclass
class TensorDef:
    line_idx: int
    result: str
    type: str


def _skip_space(text: str, at: int) -> int:
    while at < len(text) and text[at].isspace():
        at += 1
    return at


def _ssa_result(line: str) -> str | None:
    """``    %123 = linalg.matmul …`` -> ``"%123"``; ``None`` if the line is not a def.

    A def opens (after indentation) with ``%`` + identifier characters + `` = ``. Anything
    else — a block label, a terminator, a closing brace — is not a single-result op.
    """
    body = line.lstrip()
    if not body.startswith("%"):
        return None
    end = 1
    while end < len(body) and body[end] in _IDENT_CHARS:
        end += 1
    if end == 1 or not body.startswith(" = ", end):
        return None
    return body[:end]


def _tensor_type_end(line: str, start: int) -> int | None:
    """End index (exclusive) of the ``tensor<…>`` type starting at ``start``, else ``None``.

    Angle brackets are counted, so a type carrying a nested one (``tensor<4x!q<i8:f32>>``,
    an affine-map encoding) is taken WHOLE rather than cut at its first ``>``. A ``>`` that
    closes an arrow (``->``, as in an inner ``affine_map<(d0) -> (d0)>``) is not a bracket
    and does not close anything. An unbalanced type yields ``None`` — the caller then treats
    the line as carrying no tensor type rather than inventing a truncated one.
    """
    if not line.startswith(_TENSOR, start):
        return None
    depth = 0
    at = start + len(_TENSOR) - 1          # sits on the opening `<`
    while at < len(line):
        ch = line[at]
        if ch == "<":
            depth += 1
        elif ch == ">" and not (at > 0 and line[at - 1] == "-"):
            depth -= 1
            if depth == 0:
                return at + 1
        at += 1
    return None


def _tensor_types(line: str) -> list[str]:
    """Every top-level ``tensor<…>`` type on the line, left to right."""
    found: list[str] = []
    at = 0
    while True:
        start = line.find(_TENSOR, at)
        if start < 0:
            return found
        end = _tensor_type_end(line, start)
        if end is None:
            return found                   # unterminated — stop, do not guess
        found.append(line[start:end])
        at = end


def tensor_defs(mlir_text: str) -> list[TensorDef]:
    """All single-tensor-result top-level defs inside the function body, in order.

    Covers every op syntax (linalg/tensor/arith), not just `-> tensor<...>` — so
    reshapes, transposes, concats and slices are bisectable too. The result tensor type is
    the LAST ``tensor<...>`` on the line (outs type for transpose, dst for extract_slice,
    the `into` type for reshapes, etc.) — robust across op syntaxes.
    """
    defs: list[TensorDef] = []
    for i, line in enumerate(mlir_text.splitlines()):
        if "^bb" in line or "linalg.yield" in line or "func.return" in line:
            continue
        res = _ssa_result(line)
        if res is None or ", %" in line.split("=")[0]:   # skip multi-result defs
            continue
        tys = _tensor_types(line)
        if not tys:
            continue
        defs.append(TensorDef(i, res, tys[-1]))
    return defs


def _rewrite_result_arrow(line: str, new_results: str) -> str:
    """First ``-> tensor<…>`` on the line becomes ``-> {new_results}``.

    Repairs the ``func.func @forward`` signature after the body has been cut short, e.g.
    ``… ) -> tensor<1x8x256000xf32> {`` becomes ``… ) -> tensor<1x8x2048xf32> {``. Returns
    the line untouched when there is no such arrow (the caller has already established that
    this is the signature line, so an untouched line means a spelling this code does not
    understand — it reaches the MLIR parser and fails there, not silently here).
    """
    at = 0
    while True:
        arrow = line.find(_ARROW, at)
        if arrow < 0:
            return line
        after = _skip_space(line, arrow + len(_ARROW))
        end = _tensor_type_end(line, after)
        if end is not None:
            return f"{line[:arrow]}-> {new_results}{line[end:]}"
        at = arrow + 1


def _paren_result_start(line: str, arrow: int) -> int | None:
    """Index of the ``(`` of a ``-> (tensor<…`` result list at ``arrow``, else ``None``."""
    after = _skip_space(line, arrow + len(_ARROW))
    return after if line.startswith("(" + _TENSOR, after) else None


def _rewrite_paren_result_arrow(line: str, new_results: str) -> str:
    """First ``-> (tensor<…)`` becomes ``-> {new_results}`` (already-tupled signature).

    The closing ``)`` is the first one after the list opens; a tensor type does not contain
    a parenthesis, so this is the list's own terminator.
    """
    head = "(" + _TENSOR
    at = 0
    while True:
        arrow = line.find(_ARROW, at)
        if arrow < 0:
            return line
        after = _paren_result_start(line, arrow)
        if after is not None:
            close = line.find(")", after + len(head))
            if close > after + len(head):       # the list must hold at least one character
                return f"{line[:arrow]}-> {new_results}{line[close + 1:]}"
        at = arrow + 1


def _retuple_result_arrow(line: str, new_results: str) -> str:
    """``-> (tensor<…) attributes`` / ``-> (tensor<…) {`` becomes ``-> ({new_results}) …``.

    Used when the signature is ALREADY a tuple, so the closing ``)`` cannot be found by
    scanning for the first one — a tuple's inner types are separated by ``, `` and the list
    ends at the last ``)`` still followed by the function's ``attributes`` dict or its body
    ``{``. Scanning right to left finds exactly that one.
    """
    head = "(" + _TENSOR
    at = 0
    while True:
        arrow = line.find(_ARROW, at)
        if arrow < 0:
            return line
        after = _paren_result_start(line, arrow)
        if after is not None:
            close = line.rfind(")")
            while close >= after + len(head):
                tail = _skip_space(line, close + 1)
                if line.startswith("attributes", tail) or line.startswith("{", tail):
                    return f"{line[:arrow]}-> ({new_results}) {line[tail:]}"
                close = line.rfind(")", 0, close)
        at = arrow + 1


def _is_forward_signature(line: str) -> bool:
    return line.lstrip().startswith("func.func @forward") and _ARROW in line


def truncate_to(mlir_text: str, target: TensorDef) -> str:
    """Return module text whose @forward returns ``target`` (drops later body lines).

    Keeps every line up to and including the target def, replaces the function's
    `func.return` with one returning the target, and rewrites the function result type.
    Relies on MLIR/clang dead-code elimination to prune now-unused weight args' uses;
    the arg list is unchanged (extra args are harmless).
    """
    lines = mlir_text.splitlines()
    # Rebuild: header..target, then return target, then close func + module.
    head = lines[: target.line_idx + 1]
    # function signature line -> patch result type
    new_head = []
    for line in head:
        if _is_forward_signature(line):
            line = _rewrite_result_arrow(line, target.type)
            line = _rewrite_paren_result_arrow(line, target.type)
        new_head.append(line)
    body_indent = "    "
    return "\n".join(new_head) + "\n" + \
        f"{body_indent}func.return {target.result} : {target.type}\n" + \
        "  }\n}\n"


def multi_return(mlir_text: str, targets: list[TensorDef]) -> str:
    """Rewrite @forward to return ALL of ``targets`` (a tuple), keeping the body up to
    the last one. One lower+run then yields every checkpoint's value at once — far
    cheaper than one run per checkpoint, and finiteness here is the true per-op signal.
    """
    targets = sorted(targets, key=lambda d: d.line_idx)
    last = targets[-1]
    lines = mlir_text.splitlines()
    head = lines[: last.line_idx + 1]
    res_types = ", ".join(t.type for t in targets)
    res_vals = ", ".join(t.result for t in targets)
    new_head = []
    for line in head:
        if _is_forward_signature(line):
            line = _rewrite_result_arrow(line, f"({res_types})")
            line = _retuple_result_arrow(line, res_types)
        new_head.append(line)
    return "\n".join(new_head) + "\n" + \
        f"    func.return {res_vals} : {res_types}\n  }}\n}}\n"
