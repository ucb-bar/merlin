"""Extract a self-contained hw.module subtree (transitive closure) from a large CIRCT HW-dialect file.

For the arcilator middle-tier (#143): the elaborated SoC HW MLIR has 731 modules; to compile just the
Gemmini accelerator we need `@Gemmini` plus every module it transitively instantiates (and the externs it
references), emitted as a standalone `module { ... }`. This avoids the SoC-level blackboxes (e.g. the
EICG clock gate, which lives only in core/uncore) and yields a far smaller JIT.

Pure text/graph pass (no MLIR parse dependency): scan `hw.module[.extern|.generated] @NAME` definitions +
`hw.instance ... @NAME` references, BFS the instance graph from the root, emit the needed module blocks.

CLI: python -m merlin.targetgen.rtl.extract_module <soc.hw.mlir> --root Gemmini --out gemmini.hw.mlir
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_MOD_RE = re.compile(r"^\s*hw\.module(?:\.extern|\.generated)?\s+(?:private\s+)?@([A-Za-z0-9_]+)\b")
_INST_RE = re.compile(r"hw\.instance\s+\S+\s+@([A-Za-z0-9_]+)\b")


def _module_spans(text: str) -> dict[str, tuple[int, int, bool]]:
    """name -> (start_line_idx, end_line_idx_exclusive, is_extern). Top-level modules only (indent 2)."""
    lines = text.splitlines()
    starts = []
    for i, ln in enumerate(lines):
        m = _MOD_RE.match(ln)
        if m and (ln.startswith("  hw.module") or ln.startswith("hw.module")):
            starts.append((i, m.group(1), ".extern" in ln or ".generated" in ln))
    spans = {}
    for k, (i, name, ext) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        spans[name] = (i, end, ext)
    return spans


def _instances_in(lines: list[str], span: tuple[int, int, bool]) -> set[str]:
    s, e, _ = span
    refs = set()
    for ln in lines[s:e]:
        refs.update(_INST_RE.findall(ln))
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
        for ref in _instances_in(lines, spans[name]):
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
