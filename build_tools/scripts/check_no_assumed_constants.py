#!/usr/bin/env python3
"""Lint gate: no ASSUMED ISA constants (opcodes / funct / mesh dims / capacities) baked in core code.

The cardinal rule (CLAUDE.md prohibition #3): never hardcode an opcode (``0x7b``), a funct/func3
value, a mesh dimension, a scratchpad capacity, a memory base, or any ISA-identity constant. Every
such value is *derived* from the target's RTL facts (``rtl.facts.load_facts(target)`` — e.g.
``funct_decode_table.custom_opcode``) or its capability manifest, and compared as data. When a value
cannot be derived, **fail closed** (record ``UNKNOWN``), never substitute a baked default.

This gate is the automated enforcement that prohibition #3 previously lacked. To stay high-signal (a
gate that is mostly false positives gets everything allowlisted and enforces nothing), it does NOT
flag every numeric literal — bit masks (``0xFFFFFFFF``), struct offsets, alignment constants, and
loop bounds are legitimate. It flags the sharp, dangerous case only:

  a numeric literal (or a list/tuple of them) that is the DIRECT value of a name or dict-key whose
  identifier denotes an ISA-identity fact — ``*opcode*``, ``*funct*`` / ``*func3/5/7*``, ``*mesh*``,
  ``*scratchpad*``. So ``CUSTOM_OPCODE = 0x7B``, ``FUNCT3 = 0x3``, ``"mesh": [16, 16]``,
  ``"custom_opcode": 0x7B`` are caught; ``MASK32 = 0xFFFFFFFF`` and ``rows = (v >> 48) & 0xFFFF``
  (a bit-extraction expression, not a bare baked value) are not.

Detected structurally with ``ast`` (no regex on itself). In-scope trees:

  * ``merlin/python/merlin/**``
  * ``merlin/contract/**``
  * ``build_tools/scripts/**``

Allowed exceptions (checked in order):

  1. an inline ``# derived-ok: <rationale>`` on the offending line (preferred) — for a value that IS
     derived-then-cached, or a genuinely-universal standard constant (e.g. the RISC-V reset base) used
     only as a documented fallback. The rationale must cite the fact source / standard.
  2. a whole-file entry in ``build_tools/scripts/assumed_constants_allowlist.txt`` — for a file whose
     baked constant is a KNOWN overfit pending derivation (the entry names the tracking work). The
     allowlist only ever *shrinks*: as each constant is derived-or-failed-closed, delete its entry.

Run::

    python build_tools/scripts/check_no_assumed_constants.py            # full scan (exit 1)
    python build_tools/scripts/check_no_assumed_constants.py --staged   # only git-staged files
    python build_tools/scripts/check_no_assumed_constants.py --stop-hook # Claude Code Stop-hook JSON
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOW_FILE = ROOT / "build_tools" / "scripts" / "assumed_constants_allowlist.txt"
SCAN_ROOTS = ("merlin/python/merlin", "merlin/contract", "build_tools/scripts")
EXCLUDE_FRAGMENTS = ("/_data/",)
INLINE_MARKER = "# derived-ok:"

# Underscore-split identifier tokens that mark a name / dict-key as an ISA-identity fact. Matched by
# token EQUALITY (not substring) so ``functional`` / ``mesh_ran`` (a counter) do NOT match while
# ``CUSTOM_OPCODE`` (custom, opcode), ``FUNCT3`` (funct3), ``mesh``, ``scratchpad_bytes`` do. A
# numeric literal bound to one of these is an assumed constant unless derived (or a marked standard).
ISA_FACT_TOKENS = frozenset({
    "opcode", "funct3", "funct5", "funct7", "func3", "func5", "func7", "mesh", "scratchpad", "dim", "base",
})
# Trivial values that are never a baked opcode / dimension / capacity worth deriving (flags, inits).
_TRIVIAL_VALUES = frozenset({0, 1})


def _is_num(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool)


def _is_num_seq(n: ast.AST) -> bool:
    return isinstance(n, (ast.List, ast.Tuple)) and bool(n.elts) and all(_is_num(e) for e in n.elts)


def _magic(n: ast.AST) -> bool:
    """A bare baked numeric value (int literal, or a list/tuple of them) — not an expression."""
    return _is_num(n) or _is_num_seq(n)


def _value_repr(n: ast.AST) -> str:
    if _is_num(n):
        return repr(n.value)
    if _is_num_seq(n):
        return "[" + ", ".join(repr(e.value) for e in n.elts) + "]"
    return "?"


def _fact_label(name: str) -> str | None:
    toks = {t for t in name.lower().split("_") if t}
    return name if toks & ISA_FACT_TOKENS else None


class _ConstVisitor(ast.NodeVisitor):
    """Collect (lineno, label, value) for numeric literals bound to an ISA-fact-named target/key."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str, str]] = []

    def _check_target(self, label_name: str, value: ast.AST) -> None:
        lab = _fact_label(label_name)
        if lab is None or not _magic(value):
            return
        if _is_num(value) and value.value in _TRIVIAL_VALUES:
            return  # a flag/init 0 or 1, never a baked opcode/dim/capacity
        self.hits.append((getattr(value, "lineno", 0), lab, _value_repr(value)))

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                self._check_target(tgt.id, node.value)
            elif isinstance(tgt, ast.Attribute):
                self._check_target(tgt.attr, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            self._check_target(node.target.id, node.value)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                self._check_target(key.value, value)
        self.generic_visit(node)


def _load_allowlist() -> set[str]:
    allow: set[str] = set()
    if ALLOW_FILE.is_file():
        for line in ALLOW_FILE.read_text(encoding="utf-8").splitlines():
            path = line.split("#", 1)[0].strip()
            if path:
                allow.add(path)
    return allow


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    v = _ConstVisitor()
    v.visit(tree)
    if not v.hits:
        return []
    lines = src.splitlines()
    out = []
    for lineno, label, value in sorted(set(v.hits)):
        line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if INLINE_MARKER not in line:
            out.append((lineno, label, value))
    return out


def _iter_targets(staged: bool) -> list[Path]:
    if staged:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                             cwd=ROOT, capture_output=True, text=True).stdout
        rels = [ln for ln in out.splitlines() if ln.strip()]
    else:
        rels = []
        for root in SCAN_ROOTS:
            for p in sorted((ROOT / root).rglob("*.py")):
                rels.append(p.relative_to(ROOT).as_posix())
    targets = []
    for rel in rels:
        if not rel.endswith(".py"):
            continue
        if any(frag in f"/{rel}" for frag in EXCLUDE_FRAGMENTS):
            continue
        if any(rel.startswith(r + "/") or rel == r for r in SCAN_ROOTS):
            targets.append(Path(rel))
    return targets


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    allow = _load_allowlist()

    violations: list[str] = []
    for rel in _iter_targets(staged):
        relstr = rel.as_posix()
        if relstr in allow:
            continue
        for lineno, label, value in _scan_file(ROOT / rel):
            violations.append(f"{relstr}:{lineno}: assumed ISA constant {label!r} = {value} "
                              f"(derive it from rtl.facts / the manifest and fail closed on UNKNOWN, "
                              f"add `# derived-ok: <source>`, or allowlist the file)")

    if stop_hook:
        if violations:
            print(json.dumps({"decision": "block",
                              "reason": ("Assumed ISA constant outside the allowlist (see "
                                         "build_tools/scripts/assumed_constants_allowlist.txt):\n- "
                                         + "\n- ".join(violations))}))
        else:
            print(json.dumps({}))
        return 0

    if violations:
        print(f"[FAIL] no-assumed-constants: {len(violations)} assumed ISA constant(s) outside the allowlist:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"[  ok] no-assumed-constants: {len(allow)} allowlisted file(s); no baked ISA constant in scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
