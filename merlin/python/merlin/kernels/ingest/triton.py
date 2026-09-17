"""Ingest Triton GPU kernels (``@triton.jit``).

Triton kernels are Python functions decorated with ``@triton.jit`` whose body names the
optimization decisions directly via ``tl.*`` intrinsics (``tl.dot`` accumulation, block-pointer
staging, masked tails, ``num_stages`` pipelining). We extract one record per jit-decorated
function — no Triton/CUDA install required, so it scales to any kernel corpus passed by path or
``MERLIN_TRITON_REPO``.

The corpus is Python, so the functions are found with ``ast``: the decorator list of each
``FunctionDef`` is inspected for ``triton.jit``. That is strictly better than matching decorator
text — a decorator or a signature spanning several lines is found, and ``@triton.jit`` written in a
comment or in prose ("can only be used in @triton.jit'd functions") no longer manufactures a phantom
kernel out of whatever ``def`` happens to follow it. The BODY handed downstream is still the source
SLICE (motif extraction reads raw text), cut at the same boundaries as before: the next jit-decorated
function, or the next column-0 ``def``.

Kernels also ship EMBEDDED in string literals — codegen templates, docstring examples — and those are
real kernels, so a string carrying the decorator is parsed as Python too and its functions ingested.
Nothing here is dropped in silence: a file that does not parse, an embedded kernel whose template is
not valid Python (placeholders in the signature), and a file whose decorator text yielded no function
at all are each reported in ``diagnostics`` (and warned about when no dict is passed).
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel, normalize_dtype

#: The decorator that marks a kernel, as ``<module>.<attr>``.
_JIT_MODULE, _JIT_ATTR = "triton", "jit"
_JIT_TEXT = f"@{_JIT_MODULE}.{_JIT_ATTR}"

_TL_PREFIX = "tl."
#: ``tl.*`` dtype spellings. ``float8`` is open-ended — the variants carry a suffix
#: (``float8e4nv``, ``float8_e5m2``) — so its trailing word run is taken whole.
_TL_DTYPES = ("float32", "float16", "bfloat16", "int8", "int32")
_TL_DTYPE_OPEN = "float8"

_OP_KEYWORDS = (
    ("matmul", "matmul"), ("gemm", "gemm"), ("mm_kernel", "matmul"), ("_mm", "matmul"),
    ("attention", "attention"), ("flash", "attention"), ("softmax", "softmax"),
    ("layernorm", "layernorm"), ("layer_norm", "layernorm"), ("rmsnorm", "rmsnorm"),
    ("conv", "conv"), ("dropout", "dropout"), ("gelu", "gelu"), ("add", "vadd"),
)


def _guess_op(name: str, body: str) -> str:
    n = name.lower()
    for kw, op in _OP_KEYWORDS:
        if kw in n:
            return op
    if "tl.dot" in body:
        return "matmul"
    return "unknown"


def _word_run(text: str, start: int) -> int:
    """End of the identifier run at ``start`` (``str.isalnum()`` plus ``_`` — Python's ``\\w``)."""
    i = start
    while i < len(text) and (text[i] == "_" or text[i].isalnum()):
        i += 1
    return i


def _space_run(text: str, start: int) -> int:
    """End of the whitespace run at ``start``; whitespace spans newlines, as ``\\s*`` did."""
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _guess_dtype(body: str) -> str:
    """The first ``tl.<dtype>`` spelling in ``body``, else ``unknown``.

    Substring-anchored, exactly as before: there is no identifier boundary before ``tl.``, and a
    longer name (``tl.int8_t``) still reads as its known prefix.
    """
    at = body.find(_TL_PREFIX)
    while at != -1:
        rest = body[at + len(_TL_PREFIX):]
        for name in _TL_DTYPES:
            if rest.startswith(name):
                return normalize_dtype(name)
        if rest.startswith(_TL_DTYPE_OPEN):
            return normalize_dtype(rest[:_word_run(rest, len(_TL_DTYPE_OPEN))])
        at = body.find(_TL_PREFIX, at + 1)
    return "unknown"


def _is_jit(node: ast.expr) -> bool:
    """``@triton.jit`` or ``@triton.jit(...)`` — the two spellings a kernel is declared with."""
    if isinstance(node, ast.Call):
        node = node.func
    return (isinstance(node, ast.Attribute) and node.attr == _JIT_ATTR
            and isinstance(node.value, ast.Name) and node.value.id == _JIT_MODULE)


def _line_starts(text: str) -> list[int]:
    """Character offset of each line start, for turning ``ast`` (line, col) into an offset."""
    starts, at = [0], text.find("\n")
    while at != -1:
        starts.append(at + 1)
        at = text.find("\n", at + 1)
    return starts


def _offset(text: str, starts: list[int], lineno: int, col: int) -> int:
    """``ast``'s (1-based line, UTF-8 byte column) as a character offset into ``text``."""
    begin = starts[lineno - 1]
    line = text[begin:starts[lineno]] if lineno < len(starts) else text[begin:]
    return begin + len(line.encode("utf-8")[:col].decode("utf-8", "replace"))


def _decorator_start(text: str, at: int) -> int:
    """The ``@`` opening the decorator whose expression begins at ``at`` (the old match start)."""
    i = at - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return i if i >= 0 and text[i] == "@" else at


def _next_toplevel_def(text: str, start: int) -> int | None:
    """Offset of the next column-0 ``def <name> (`` at or after ``start``, else ``None``."""
    at = text.find("def", start)
    while at != -1:
        if at == 0 or text[at - 1] == "\n":
            i = _space_run(text, at + 3)
            if i > at + 3:
                j = _word_run(text, i)
                if j > i and text.startswith("(", _space_run(text, j)):
                    return at
        at = text.find("def", at + 1)
    return None


#: How far to follow kernels embedded in string literals (a template inside a template).
_MAX_EMBED_DEPTH = 2


def _functions(text: str) -> list[tuple[str, str]]:
    """Return (name, body) for each ``@triton.jit`` function in ``text``."""
    return scan_functions(text)[0]


def scan_functions(text: str, depth: int = 0) -> tuple[list[tuple[str, str]], list[str]]:
    """``(functions, unparsed_embeds)`` for ``text``.

    ``unparsed_embeds`` names the string literals that declare a jit function but are not valid
    Python — codegen templates whose signatures hold placeholders. They cannot be parsed by anything,
    so they are reported instead of being dropped without trace. Raises ``SyntaxError`` when ``text``
    itself is not parseable Python; the caller reports that.
    """
    tree = ast.parse(text)
    starts = _line_starts(text)
    found: list[tuple[int, int, str]] = []      # (decorator offset, def-keyword offset, name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        jit = next((d for d in node.decorator_list if _is_jit(d)), None)
        if jit is None:
            continue
        at = _decorator_start(text, _offset(text, starts, jit.lineno, jit.col_offset))
        found.append((at, _offset(text, starts, node.lineno, node.col_offset), node.name))
    found.sort()

    ranked: list[tuple[int, int, str, str]] = []   # (outer offset, order, name, body)
    for i, (start, def_at, name) in enumerate(found):
        # body runs to the next jit-decorated function, or the next column-0 def, or EOF
        end = found[i + 1][0] if i + 1 < len(found) else len(text)
        nxt = _next_toplevel_def(text, def_at + 1)
        if nxt is not None and nxt < end:
            end = nxt
        ranked.append((start, 0, name, text[start:end]))

    unparsed: list[str] = []
    if depth < _MAX_EMBED_DEPTH:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and _JIT_TEXT in node.value):
                continue
            at = _offset(text, starts, node.lineno, node.col_offset)
            try:
                inner, inner_unparsed = scan_functions(node.value, depth + 1)
            except SyntaxError as exc:
                unparsed.append(f"line {node.lineno}: {exc}")
                continue
            # The embedded source is the honest body here: slicing the OUTER file would drag the
            # closing quotes and the prose after them into the kernel's raw text.
            ranked.extend((at, j + 1, name, body) for j, (name, body) in enumerate(inner))
            unparsed.extend(f"line {node.lineno}: {u}" for u in inner_unparsed)
    ranked.sort(key=lambda r: (r[0], r[1]))
    return [(name, body) for _at, _j, name, body in ranked], unparsed


def _record(diagnostics: dict | None, unparsed: dict, embedded: dict, decorator_only: list) -> None:
    """Publish what the scan could not read. Everything lands in ``diagnostics`` file by file; with
    no dict to put it in, the same facts go out as ONE summary warning — per-file warnings would run
    to hundreds on a library tree (docstring examples are the common case) and drown themselves."""
    if diagnostics is not None:
        diagnostics["unparsed"] = unparsed
        diagnostics["unparsed_embedded"] = embedded
        diagnostics["decorator_without_function"] = list(decorator_only)
        return
    for rel, exc in unparsed.items():        # a file that does not parse is rare and alarming
        warnings.warn(f"triton ingest: {rel} carries {_JIT_TEXT} but does not parse: {exc}",
                      stacklevel=3)
    if embedded or decorator_only:
        warnings.warn(
            f"triton ingest: {len(embedded)} file(s) embed {_JIT_TEXT} in a string that is not valid "
            f"Python (a codegen template or a doc example), and {len(decorator_only)} file(s) name "
            f"{_JIT_TEXT} without declaring a function; pass diagnostics= to list them. "
            f"First: {sorted(embedded)[:3] or sorted(decorator_only)[:3]}", stacklevel=3)


# Default subtrees to mine: real kernels (tutorials, shipped kernel library), not test files.
_DEFAULT_SUBDIRS = ("python/tutorials", "python/triton_kernels")


def _roots(repo: Path, subdirs: tuple[str, ...] | list[str] | None) -> list[Path]:
    chosen = [repo / s for s in (subdirs if subdirs is not None else _DEFAULT_SUBDIRS)]
    chosen = [p for p in chosen if p.is_dir()]
    return chosen or [repo]  # corpus dirs without the canonical layout: mine everything


def ingest_triton(repo: str, target: str = "triton", limit: int | None = None,
                  source: str = "triton",
                  subdirs: list[str] | None = None,
                  diagnostics: dict | None = None) -> Iterator[NormalizedKernel]:
    """Yield NormalizedKernels for each ``@triton.jit`` function under ``repo``.

    Mines ``python/tutorials`` + ``python/triton_kernels`` by default (falling back to the
    whole tree when absent) and extracts every jit-decorated function. Helper jit functions
    are included (they still carry real optimization markers). ``source`` lets sibling
    corpora (e.g. triton-cpu) be indexed as distinct evidence sources. Files that carry the
    decorator but do not parse are listed in ``diagnostics['unparsed']`` (and warned about when no
    dict is passed) rather than dropped without trace.
    """
    root = Path(repo)
    count = 0
    unparsed: dict[str, str] = {}
    embedded: dict[str, list[str]] = {}
    decorator_only: list[str] = []
    paths = sorted({p for r in _roots(root, subdirs) for p in r.rglob("*.py")})
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _JIT_TEXT not in text:
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        try:
            functions, unparsed_embeds = scan_functions(text)
        except SyntaxError as exc:
            unparsed[rel] = str(exc)
            continue
        if unparsed_embeds:
            embedded[rel] = unparsed_embeds
        if not functions:
            # The decorator text is present but declares nothing: prose, a comment, an error message.
            decorator_only.append(rel)
        for name, body in functions:
            yield NormalizedKernel(
                source=source, target=target, path=f"{rel}::{name}",
                op=_guess_op(name, body), dtype=_guess_dtype(body),
                raw_text=body, meta={"function": name},
            )
            count += 1
            if limit is not None and count >= limit:
                _record(diagnostics, unparsed, embedded, decorator_only)
                return
    _record(diagnostics, unparsed, embedded, decorator_only)
