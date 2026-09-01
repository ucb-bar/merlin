"""Structural (regex-free) location + rewriting of routable ``linalg.matmul`` ops in MLIR text.

Shared by the three board matmul-routing backends (:mod:`~merlin.runtime.backends.ours_board`,
:mod:`~merlin.runtime.backends.xnnpack_board`, :mod:`~merlin.runtime.backends.openblas_board`).
All three used to carry a byte-identical copy of the same compiled pattern and the same
``re.sub`` rewrite; three copies of a parser is three chances to drift, so the matcher lives here
once and each backend supplies only its symbol base. The leading ``_`` keeps this out of
``base._ensure_discovered``'s submodule sweep — it is a helper, not a backend.

WHY IT IS NOT A REGEX
---------------------
The retired pattern was::

    (?P<res>%[\\w$.]+)\\s*=\\s*linalg\\.matmul\\b(?P<attrs>\\s*\\{[^}]*\\})?\\s*
    ins\\(\\s*(?P<a>%[\\w$.]+)\\s*,\\s*(?P<b>%[\\w$.]+)\\s*:\\s*
    tensor<(?P<at>[^>]+)>\\s*,\\s*tensor<(?P<bt>[^>]+)>\\s*\\)\\s*
    outs\\(\\s*(?P<c>%[\\w$.]+)\\s*:\\s*tensor<(?P<ct>[^>]+)>\\s*\\)\\s*
    ->\\s*tensor<(?P<rt>[^>]+)>

which is exactly the failure mode the no-regex rule exists for: a valid-but-differently-spelled
``linalg.matmul`` does not match, the op is silently left alone, and the symptom downstream is
"the kernel backend routed nothing" rather than an error. Concretely, the pattern dropped

  * an attribute dictionary containing a NESTED ``{...}`` (``[^}]*`` stops at the first ``}``);
  * an operand or result type with a nested ``<...>`` (``tensor<[^>]+>`` cannot cross a ``>``,
    so ``tensor<4x4xcomplex<f32>>`` fails the whole match);
  * the ``indexing_maps = [...]`` / ``cast = #linalg.type_fn<...>`` spellings MLIR prints for a
    non-default ``linalg.matmul`` (the pattern allows only a ``{...}`` dict between the op name
    and ``ins``);
  * an SSA name containing ``-`` (``%[\\w$.]+`` truncates it, then the following ``\\s*=`` fails).

This module walks the text structurally instead: a string/comment-aware character scan finds the
``linalg.matmul`` op token, then a bracket-balanced reader parses the result binding, the
``ins``/``outs`` operand lists and the result type. It accepts everything the pattern accepted
plus all four cases above.

FAIL-CLOSED
-----------
Every ``linalg.matmul`` token is *accounted for*. A site is one of:

  * an op we parsed -> a routing candidate (routable or not, per :func:`is_routable`);
  * a REPORTED non-candidate with a reason (:attr:`MatmulSite.reason`) — the memref /
    destination-passing form with no result to bind, or non-tensor operand types. These are
    genuinely not routable through a tensor-typed ``func.call``, and the reason says so;
  * a hard parse failure -> :class:`MatmulRoutingError`. Never a silent "no match".

A full xDSL parse was considered and rejected for this seam: the rewrite must leave every byte
outside the matched ops untouched (the surrounding model text is handed straight to the compile
pipeline, and several dialects in a whole-model lowering do not survive an xDSL parse/print
round-trip — see the ``llvmlower/passes_xdsl.py`` text-repair note in the regex allowlist). A
structural scanner gives the parse's precision with the text pass's fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass

_OP_NAME = "linalg.matmul"

# MLIR bare-identifier characters (also the SSA-name character set). The retired pattern spelled
# an SSA name `%[\w$.]+` -- letters, digits, `_`, `$`, `.`; MLIR also allows `-` in a value name,
# which that pattern would have truncated, so we accept it too.
_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$.-")
# `\w` -- what a regex `\b` word boundary is defined against, mirrored for the op-name boundary.
_WORD_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_WS = frozenset(" \t\r\n\f\v")
_OPENERS = "([{"
_CLOSERS = ")]}"


class MatmulRoutingError(ValueError):
    """A ``linalg.matmul`` the structural reader cannot understand.

    Raised instead of silently leaving the op unrouted: an unroutable-because-unparsed matmul is
    indistinguishable, downstream, from a backend that produced nothing."""


@dataclass(frozen=True)
class MatmulOp:
    """One parsed tensor-form ``linalg.matmul``. ``start``/``end`` bound the text to replace."""
    start: int
    end: int
    res: str
    a: str
    b: str
    c: str
    at: str
    bt: str
    ct: str
    rt: str


@dataclass(frozen=True)
class MatmulSite:
    """A ``linalg.matmul`` token in the text. Either ``op`` is set (a routing candidate) or
    ``reason`` explains why the op is outside the rewrite domain — never both empty."""
    offset: int
    line: int
    op: MatmulOp | None
    reason: str


# --------------------------------------------------------------------------- lexical scaffolding

def _code_mask(text: str) -> bytearray:
    """1 for every character that is CODE, 0 inside a ``"..."`` string literal or a ``//`` comment.

    MLIR string attributes routinely carry arbitrary text (``prov.fqn = "...matmul..."``), and a
    rewrite that fired inside one would corrupt the module. The retired pattern had no such
    guard; it only escaped the issue because its shape happened not to occur inside a string."""
    n = len(text)
    mask = bytearray(n)
    i = 0
    in_str = False
    in_comment = False
    while i < n:
        c = text[i]
        if in_comment:
            if c == "\n":
                in_comment = False
                mask[i] = 1
            i += 1
        elif in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            in_comment = True
            i += 2
        else:
            mask[i] = 1
            i += 1
    return mask


def _is_token(text: str, i: int, token: str) -> bool:
    """``text`` has ``token`` at ``i`` delimited like a regex ``\\b``-bounded literal would be."""
    if not text.startswith(token, i):
        return False
    if i > 0 and text[i - 1] in _NAME_CHARS:
        return False
    j = i + len(token)
    return j >= len(text) or text[j] not in _WORD_CHARS


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in _WS:
        i += 1
    return i


def _rskip_ws(text: str, i: int) -> int:
    while i >= 0 and text[i] in _WS:
        i -= 1
    return i


def _match_bracket(text: str, mask: bytearray, i: int) -> int:
    """``text[i]`` is an opener; return the index one past its matching closer."""
    depth = 0
    start = i
    n = len(text)
    while i < n:
        if mask[i]:
            c = text[i]
            if c in _OPENERS:
                depth += 1
            elif c in _CLOSERS:
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    raise MatmulRoutingError(f"unbalanced bracket opened at offset {start}")


def _find_call_token(text: str, mask: bytearray, start: int, token: str) -> int:
    """Index of the next ``token`` immediately followed by ``(``, at bracket depth 0 relative to
    ``start``. Returns -1 if the enclosing region closes first (depth would go negative)."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        if mask[i]:
            c = text[i]
            if c in _OPENERS:
                depth += 1
            elif c in _CLOSERS:
                if depth == 0:
                    return -1           # left the op without finding it
                depth -= 1
            elif depth == 0 and _is_token(text, i, token):
                k = _skip_ws(text, i + len(token))
                if k < n and text[k] == "(":
                    return i
        i += 1
    return -1


def _split_top_level(s: str, seps: str) -> list[str]:
    """Split ``s`` on ``seps`` at bracket depth 0, honoring ``()[]{}``, ``<>`` and strings.

    The ``<>`` depth is why this is not a ``str.split``: ``memref<..., strided<[?], offset: ?>>``
    carries a ``:`` and ``affine_map<(d0, d1) -> (d0)>`` carries a ``,`` inside the type. A ``>``
    preceded by ``-`` is the arrow of a function/affine-map type, not a closer."""
    mask = _code_mask(s)
    parts: list[str] = []
    buf: list[str] = []
    depth = angle = 0
    for i, c in enumerate(s):
        if mask[i]:
            if c in _OPENERS:
                depth += 1
            elif c in _CLOSERS:
                depth -= 1
            elif c == "<":
                angle += 1
            elif c == ">" and (i == 0 or s[i - 1] != "-"):
                angle -= 1
            elif c in seps and depth == 0 and angle == 0:
                parts.append("".join(buf))
                buf = []
                continue
        buf.append(c)
    parts.append("".join(buf))
    return [p.strip() for p in parts]


def _read_type(text: str, mask: bytearray, i: int) -> tuple[str, int]:
    """Read one MLIR type starting at ``i``; return ``(type_text, end)``. Bracket/angle aware, so
    ``tensor<4x4xcomplex<f32>>`` and ``!my.type<a, b>`` come back whole (the retired
    ``tensor<[^>]+>`` could not cross the inner ``>`` and dropped the op entirely)."""
    n = len(text)
    depth = angle = 0
    j = i
    while j < n:
        if mask[j]:
            c = text[j]
            if c in _OPENERS:
                depth += 1
            elif c in _CLOSERS:
                if depth == 0:
                    break
                depth -= 1
            elif c == "<":
                angle += 1
            elif c == ">" and (j == 0 or text[j - 1] != "-"):
                angle -= 1
            elif depth == 0 and angle == 0 and (c in _WS or c == ","):
                break
        j += 1
    return text[i:j], j


def _read_ssa_backwards(text: str, end: int) -> tuple[str, int] | None:
    """Read a ``%name`` ending at index ``end`` (inclusive); return ``(name, start)``."""
    e = end
    while e >= 0 and text[e] in _NAME_CHARS:
        e -= 1
    if e < 0 or e == end or text[e] != "%":
        return None
    return text[e:end + 1], e


def _tensor_element(t: str) -> str | None:
    """``tensor<INNER>`` -> ``INNER``; ``None`` for any other type."""
    if t.startswith("tensor<") and t.endswith(">"):
        return t[len("tensor<"):-1]
    return None


# ------------------------------------------------------------------------------- the matmul scan

def _parse_at(text: str, mask: bytearray, start: int) -> tuple[MatmulOp | None, str]:
    """Parse the ``linalg.matmul`` whose op name begins at ``start``.

    Returns ``(op, "")`` for a parsed tensor-form matmul, or ``(None, reason)`` for a form that is
    outside the rewrite domain by construction. Raises :class:`MatmulRoutingError` for a spelling
    the reader does not understand — never returns a silent "no match"."""
    def _hard(msg: str) -> None:
        line = text.count("\n", 0, start) + 1
        excerpt = text[start:start + 160].split("\n", 1)[0]
        raise MatmulRoutingError(
            f"line {line}: cannot parse `linalg.matmul` ({msg}); refusing to leave it silently "
            f"unrouted. Offending text: {excerpt!r}")

    # --- result binding: `%res = ` immediately before the op name.
    j = _rskip_ws(text, start - 1)
    if j < 0 or text[j] != "=":
        # No result to bind. This is the memref / pure destination-passing form
        # (`linalg.matmul ins(...) outs(%c : memref<...>)`), which has no tensor value to hand to
        # a `func.call`. The retired pattern also skipped it -- but silently.
        return None, "no result binding (memref/destination-passing form)"
    k = _rskip_ws(text, j - 1)
    read = _read_ssa_backwards(text, k)
    if read is None:
        _hard("the result is not a single `%ssa` value")
    res, op_start = read              # type: ignore[misc]

    # --- everything between the op name and `ins(` is the op's leading attributes. Scanning for
    # the `ins(` token at bracket depth 0 accepts a NESTED attribute dict, `indexing_maps = [...]`
    # and `cast = #linalg.type_fn<...>` -- all of which the `\{[^}]*\}` alternative rejected.
    after = start + len(_OP_NAME)
    ins_i = _find_call_token(text, mask, after, "ins")
    if ins_i < 0:
        _hard("no `ins(...)` operand list")
    ins_open = _skip_ws(text, ins_i + 3)
    ins_end = _match_bracket(text, mask, ins_open)
    ins_ops, ins_tys = _operands_and_types(text[ins_open + 1:ins_end - 1], "ins", _hard)
    if len(ins_ops) != 2:
        _hard(f"`ins(...)` binds {len(ins_ops)} operands; `linalg.matmul` takes 2")

    outs_i = _find_call_token(text, mask, ins_end, "outs")
    if outs_i < 0:
        _hard("no `outs(...)` init operand")
    outs_open = _skip_ws(text, outs_i + 4)
    outs_end = _match_bracket(text, mask, outs_open)
    outs_ops, outs_tys = _operands_and_types(text[outs_open + 1:outs_end - 1], "outs", _hard)
    if len(outs_ops) != 1:
        _hard(f"`outs(...)` binds {len(outs_ops)} operands; `linalg.matmul` takes 1")

    # --- `-> <result type>`
    q = _skip_ws(text, outs_end)
    if not text.startswith("->", q):
        _hard("a result is bound but the op has no `-> <type>` result type")
    r = _skip_ws(text, q + 2)
    rtype, rend = _read_type(text, mask, r)
    if rtype.startswith("("):          # `-> (t)` -- a 1-element result list is the same op
        inner = _split_top_level(rtype[1:-1], ",")
        if len(inner) != 1:
            return None, f"{len(inner)} results; a routed call returns exactly 1"
        rtype = inner[0]

    at, bt = (_tensor_element(t) for t in ins_tys)
    ct = _tensor_element(outs_tys[0])
    rt = _tensor_element(rtype)
    if at is None or bt is None or ct is None or rt is None:
        # e.g. memref-typed operands with a tensor-shaped spelling elsewhere: recognised, but a
        # tensor-typed `func.call` cannot stand in for it. Reported, not dropped.
        return None, "operand/result types are not all `tensor<...>`"

    return MatmulOp(start=op_start, end=rend, res=res,
                    a=ins_ops[0], b=ins_ops[1], c=outs_ops[0],
                    at=at, bt=bt, ct=ct, rt=rt), ""


def _operands_and_types(inner: str, what: str, hard) -> tuple[list[str], list[str]]:
    """Split an ``ins``/``outs`` body ``%a, %b : ta, tb`` into its operand and type lists."""
    halves = _split_top_level(inner, ":")
    if len(halves) != 2:
        hard(f"`{what}(...)` is not `<operands> : <types>`")
    ops = _split_top_level(halves[0], ",")
    tys = _split_top_level(halves[1], ",")
    if len(ops) != len(tys):
        hard(f"`{what}(...)` binds {len(ops)} operands but {len(tys)} types")
    for o in ops:
        if not o.startswith("%"):
            hard(f"`{what}(...)` operand {o!r} is not an `%ssa` value")
    return ops, tys


def scan_matmuls(mlir_text: str) -> list[MatmulSite]:
    """Every ``linalg.matmul`` op token in ``mlir_text``, parsed or explained. Ordered by offset.

    Raises :class:`MatmulRoutingError` on a spelling the reader cannot parse."""
    mask = _code_mask(mlir_text)
    sites: list[MatmulSite] = []
    i = mlir_text.find(_OP_NAME)
    while i != -1:
        if mask[i] and _is_token(mlir_text, i, _OP_NAME):
            op, reason = _parse_at(mlir_text, mask, i)
            sites.append(MatmulSite(offset=i, line=mlir_text.count("\n", 0, i) + 1,
                                    op=op, reason=reason))
        i = mlir_text.find(_OP_NAME, i + 1)
    return sites


def is_routable(at: str, bt: str, ct: str, rt: str) -> bool:
    """Plain 2-D f32 matmul — the faithful set all three board backends route (and the same set
    the host classifier ``xnnpack_host.classify_matmul_kernel`` calls faithful). Unchanged from
    the per-backend copies this module replaces."""
    for t in (at, bt, ct, rt):
        dims = t.split("x")
        if dims[-1] != "f32":
            return False
        if len(dims) != 3:          # 2 shape dims + the element type -> rank 2
            return False
        if any(d.startswith("?") for d in dims[:-1]):   # static only
            return False
    return True


def routing_coverage(mlir_text: str) -> tuple[int, int]:
    """``(n_candidates, n_eligible)`` for the exact rewrite domain.

    A candidate is a parsed tensor-form ``linalg.matmul``; an eligible candidate additionally
    satisfies :func:`is_routable`. Recording both keeps a paper kernel-swap cell explicit about
    its denominator instead of proving only that *some* operation reached the expert kernel."""
    ops = [s.op for s in scan_matmuls(mlir_text) if s.op is not None]
    return len(ops), sum(is_routable(o.at, o.bt, o.ct, o.rt) for o in ops)


# ------------------------------------------------------------------------------------- rewriting

def _first_func_offset(text: str) -> int | None:
    """Offset of the start of the line holding the first top-level ``func.func``.

    The decls must land INSIDE the module body — not after ``builtin.module attributes {...} {``,
    where they would land in the attribute dictionary. The retired anchor was
    ``re.search(r"\\n(\\s*)func\\.func @", body)``, which additionally required the literal
    ``func.func @``: a module whose first function is ``func.func private @…`` was skipped over,
    and a module with no other function got NO decl block at all (silently emitting calls to
    undeclared symbols). We anchor on the ``func.func`` op token itself, so ``private`` and a
    first-line function both work."""
    mask = _code_mask(text)
    i = text.find("func.func")
    while i != -1:
        if mask[i] and _is_token(text, i, "func.func"):
            bol = text.rfind("\n", 0, i) + 1
            if not text[bol:i].strip():        # only indentation before it -> top-level statement
                return bol
        i = text.find("func.func", i + 1)
    return None


def rewrite_matmuls(mlir_text: str, sym_base: str) -> tuple[str, int]:
    """Replace every routable 2-D f32 ``linalg.matmul`` with ``call @<sym_base>_<i>``.

    Returns ``(rewritten_text, n_routed)``. MLIR func types are monomorphic, so one numbered decl
    ``@<sym_base>_<i>`` is emitted per distinct (A,B,C,R) type signature; each links to a thin C
    alias of the single signature-agnostic shim entry ``<sym_base>`` (which reads M/N/K from the
    memref descriptors). The operands are annotated read/read/write so one-shot-bufferize does NOT
    defensively copy the weight. Default-off: with no routable matmul the input is returned
    BYTE-IDENTICAL, and every byte outside a rewritten op is preserved."""
    sites = scan_matmuls(mlir_text)
    sigs: dict[tuple[str, str, str, str], str] = {}
    pieces: list[str] = []
    last = 0
    n = 0
    for site in sites:
        op = site.op
        if op is None or not is_routable(op.at, op.bt, op.ct, op.rt):
            continue
        key = (op.at, op.bt, op.ct, op.rt)
        sym = sigs.get(key)
        if sym is None:
            sym = f"{sym_base}_{len(sigs)}"
            sigs[key] = sym
        pieces.append(mlir_text[last:op.start])
        pieces.append(f"{op.res} = call @{sym}({op.a}, {op.b}, {op.c}) : "
                      f"(tensor<{op.at}>, tensor<{op.bt}>, tensor<{op.ct}>) -> tensor<{op.rt}>")
        last = op.end
        n += 1
    if n == 0:
        return mlir_text, 0
    pieces.append(mlir_text[last:])
    body = "".join(pieces)

    decls = [
        f'func.func private @{sym}('
        f'%a: tensor<{at}> {{bufferization.access = "read"}}, '
        f'%b: tensor<{bt}> {{bufferization.access = "read"}}, '
        f'%c: tensor<{ct}> {{bufferization.access = "write"}}) -> tensor<{rt}>'
        for (at, bt, ct, rt), sym in sigs.items()
    ]
    decl_block = "  " + "\n  ".join(decls) + "\n"
    pos = _first_func_offset(body)
    if pos is None:
        # FAIL CLOSED: the calls are already emitted; without the decls the module references
        # undeclared symbols. The retired code returned that broken text.
        raise MatmulRoutingError(
            f"routed {n} matmul(s) to @{sym_base}_* but found no top-level `func.func` to anchor "
            "the private declarations on — the rewritten module would call undeclared symbols")
    return body[:pos] + decl_block + body[pos:], n
