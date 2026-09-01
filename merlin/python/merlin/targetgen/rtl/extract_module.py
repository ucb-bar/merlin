"""Extract a self-contained hw.module subtree (transitive closure) from a large CIRCT HW-dialect file.

For the arcilator middle-tier (#143): the elaborated SoC HW MLIR has 731 modules; to compile just one
accelerator we need its root module plus every module it transitively instantiates (and the externs it
references), emitted as a standalone `module { ... }`. This avoids the SoC-level blackboxes (e.g. the
EICG clock gate, which lives only in core/uncore) and yields a far smaller JIT.

Pure text/graph pass (no MLIR parse dependency): scan `hw.module[.extern|.generated] @NAME` definitions +
`hw.instance ... @NAME` references, BFS the instance graph from the root, emit the needed module blocks.
The blocks are emitted as the ORIGINAL LINES — this tool must not reformat the input, both because a
downstream `circt-opt`/arcilator diff should show only the slice and because the SoC file is routinely
larger than the CIRCT build available for a round-trip. That is also why the structured replacement for
the retired regexes is a real tokenizer over each line rather than a `circt-opt --mlir-print-op-generic`
instance graph (`rtl/mlc_bridge.py` has that route for the *analysis* seams, where reprinting is fine).

The retired patterns were::

    _MOD_RE  = ^\\s*hw\\.module(?:\\.extern|\\.generated)?\\s+(?:private\\s+)?@([A-Za-z0-9_]+)\\b
    _INST_RE = hw\\.instance\\s+\\S+\\s+@([A-Za-z0-9_]+)\\b

Both were silent-drop hazards, and a drop here does not look like an error — it looks like a smaller
accelerator: a module missing from the span table has its body swallowed into the PRECEDING module's
span, and a missed `hw.instance` target is simply never pulled into the closure, so the emitted slice
is quietly incomplete. Specifically:

  * `_INST_RE` required `hw.instance <one non-space token> @Name`, so it missed CIRCT's inner-symbol
    form `hw.instance "n" sym @s @Name(...)` and any instance name containing a space
    (`hw.instance "my inst" @Name`) — both legal, both dropped without a word;
  * both patterns read a symbol as `[A-Za-z0-9_]+`, so a symbol containing `$`, `.` or `-`
    (all legal in an MLIR bare symbol) was TRUNCATED — a definition and a reference to the same
    module could then disagree and the module would be reported as an unresolved extern;
  * a `hw.module` line whose symbol the pattern could not read was skipped entirely rather than
    flagged.

The replacement parses each line structurally (indent, op-name token, optional `.extern`/`.generated`
suffix, optional `private`, MLIR symbol) and FAILS CLOSED: a `hw.module` or `hw.instance` line it
cannot read raises :class:`ExtractModuleError` naming the line, instead of silently shrinking the slice.

CLI: python -m merlin.targetgen.rtl.extract_module <soc.hw.mlir> --root Top --out top.hw.mlir
"""
from __future__ import annotations

import argparse
from pathlib import Path

_MOD_OP = "hw.module"
_INST_OP = "hw.instance"
# The suffixed forms of `hw.module`, longest first so `.generated` is not read as a bare `hw.module`.
_MOD_SUFFIXES = (".extern", ".generated")
# MLIR bare-identifier characters, the symbol-name alphabet. Wider than the retired
# `[A-Za-z0-9_]+`, which truncated any symbol carrying `$`, `.` or `-`.
_SYM_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$.-")
_WS = " \t"
# Top-level `hw.module` indentation as CIRCT emits it: column 0, or 2 inside `module { ... }`.
# (Kept from the retired implementation, which required `ln.startswith("hw.module")` or
# `ln.startswith("  hw.module")` — nested `hw.module`s do not exist, but instance bodies are deeper.)
_TOP_LEVEL_INDENTS = (0, 2)


class ExtractModuleError(ValueError):
    """A `hw.module`/`hw.instance` line the reader cannot parse.

    Raised rather than skipped: an unparsed declaration silently corrupts the module span table, and
    an unparsed instance silently drops a subtree from the extracted closure."""


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in _WS:
        i += 1
    return i


def _read_symbol(s: str, i: int) -> tuple[str, int] | None:
    """Read the MLIR symbol at ``s[i]`` (which must be ``@``); return ``(name, end)``.

    Handles both spellings: a bare identifier (``@Gemmini``) and a quoted one (``@"odd name"``)."""
    if i >= len(s) or s[i] != "@":
        return None
    i += 1
    if i < len(s) and s[i] == '"':
        j = i + 1
        while j < len(s):
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == '"':
                return s[i + 1:j], j + 1
            j += 1
        return None
    j = i
    while j < len(s) and s[j] in _SYM_CHARS:
        j += 1
    if j == i:
        return None
    return s[i:j], j


def _read_quoted_or_bare(s: str, i: int) -> int:
    """Skip one instance-name token: a quoted string (spaces allowed) or a run of non-whitespace."""
    if i < len(s) and s[i] == '"':
        j = i + 1
        while j < len(s):
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == '"':
                return j + 1
            j += 1
        return j
    j = i
    while j < len(s) and s[j] not in _WS:
        j += 1
    return j


def _parse_module_decl(line: str) -> tuple[str, bool] | None:
    """Parse a top-level ``hw.module[.extern|.generated] [private] @Name`` line.

    Returns ``(name, is_extern)``, or ``None`` when the line is not a top-level module declaration.
    Raises :class:`ExtractModuleError` when it *is* one but the symbol cannot be read."""
    indent = len(line) - len(line.lstrip(" "))
    if indent not in _TOP_LEVEL_INDENTS:
        return None
    i = indent
    if not line.startswith(_MOD_OP, i):
        return None
    i += len(_MOD_OP)
    is_extern = False
    for suffix in _MOD_SUFFIXES:
        if line.startswith(suffix, i):
            is_extern = True                   # a body-less module: nothing to recurse into
            i += len(suffix)
            break
    else:
        # A bare `hw.module` must be followed by whitespace; otherwise this is a different op
        # (e.g. a hypothetical `hw.moduleFoo`), not a module declaration.
        if i < len(line) and line[i] not in _WS:
            return None
    j = _skip_ws(line, i)
    if j == i:
        raise ExtractModuleError(f"`{_MOD_OP}` not followed by whitespace: {line.rstrip()!r}")
    i = j
    if line.startswith("private", i) and _skip_ws(line, i + 7) > i + 7:
        i = _skip_ws(line, i + 7)
    read = _read_symbol(line, i)
    if read is None:
        raise ExtractModuleError(
            f"`{_MOD_OP}` declaration with no readable `@symbol`: {line.rstrip()!r}")
    return read[0], is_extern


def _instances_in_line(line: str) -> set[str]:
    """Every module symbol instantiated on ``line``.

    Grammar (CIRCT): ``hw.instance <instance-name> [sym @<inner-sym>] @<Module> ( ... )``. The
    instance name may be a quoted string containing spaces, and the optional inner symbol sits
    between it and the module symbol — the retired ``\\s+\\S+\\s+@`` shape handled neither."""
    refs: set[str] = set()
    i = 0
    n = len(line)
    while True:
        k = line.find(_INST_OP, i)
        if k < 0:
            return refs
        after = k + len(_INST_OP)
        # Token boundaries, mirroring the retired pattern's implicit literal match.
        if (k > 0 and line[k - 1] in _SYM_CHARS) or (after < n and line[after] not in _WS):
            i = k + 1
            continue
        p = _skip_ws(line, after)
        p = _read_quoted_or_bare(line, p)       # instance name
        p = _skip_ws(line, p)
        if line.startswith("sym", p) and _skip_ws(line, p + 3) > p + 3:
            p = _skip_ws(line, p + 3)
            inner = _read_symbol(line, p)       # optional inner symbol
            if inner is None:
                raise ExtractModuleError(
                    f"`{_INST_OP} ... sym` with no readable inner `@symbol`: {line.rstrip()!r}")
            p = _skip_ws(line, inner[1])
        read = _read_symbol(line, p)
        if read is None:
            # FAIL CLOSED: this instance would vanish from the closure and the slice would be
            # quietly missing a subtree.
            raise ExtractModuleError(
                f"`{_INST_OP}` with no readable target `@symbol`: {line.rstrip()!r}")
        refs.add(read[0])
        i = read[1]


def _module_spans(text: str) -> dict[str, tuple[int, int, bool]]:
    """name -> (start_line_idx, end_line_idx_exclusive, is_extern). Top-level modules only."""
    lines = text.splitlines()
    starts = []
    for i, ln in enumerate(lines):
        decl = _parse_module_decl(ln)
        if decl is not None:
            starts.append((i, decl[0], decl[1]))
    spans = {}
    for k, (i, name, ext) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        spans[name] = (i, end, ext)
    return spans


def _instances_in(lines: list[str], span: tuple[int, int, bool]) -> set[str]:
    s, e, _ = span
    refs: set[str] = set()
    for ln in lines[s:e]:
        refs.update(_instances_in_line(ln))
    return refs


def extract(text: str, root: str) -> tuple[str, list[str], list[str]]:
    """Return (standalone_module_text, included_modules, missing_refs)."""
    lines = text.splitlines()
    spans = _module_spans(text)
    if root not in spans:
        raise KeyError(f"@{root} not found among {len(spans)} modules")
    seen, order, missing = set(), [], []
    stack = [root]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in spans:
            missing.append(name)
            continue
        order.append(name)
        # `sorted` so the visit order (and therefore `included`) is reproducible run to run: the
        # reference set's iteration order is insertion-dependent, which made the returned list
        # vary between runs on the same input. The emitted TEXT was already stable (it is sorted
        # by source position below); only the reported order moved.
        for ref in sorted(_instances_in(lines, spans[name])):
            if ref not in seen:
                stack.append(ref)
    # top-level preamble: ops before the first hw.module (sv.macro.decl / emit.fragment defining
    # @SYNTHESIS, assert/printf/stop guards) — bodies reference these by symbol, so carry them.
    first_mod = min(s for s, _e, _x in spans.values())
    preamble = lines[1:first_mod]  # skip line 0 ("module attributes {...} {")
    body = list(preamble)
    for name in sorted(order, key=lambda n: spans[n][0]):
        s, e, _ = spans[name]
        body.extend(lines[s:e])
    out = "module {\n" + "\n".join(body) + "\n}\n"
    return out, order, sorted(set(missing))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    text = Path(a.src).read_text()
    out, included, missing = extract(text, a.root)
    Path(a.out).write_text(out)
    print(f"root=@{a.root}: {len(included)} modules, {len(missing)} unresolved refs "
          f"(externs/intrinsics: {missing[:8]}{'...' if len(missing) > 8 else ''})")
    print(f"wrote {a.out} ({out.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
