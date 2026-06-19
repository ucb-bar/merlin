"""Typed C-intrinsic source extractor — the cross-check for the asm-lifted CCA.

Parses a kernel translation unit with ``clang -Xclang -ast-dump=json`` (NOT regex over C) and
reads the RVV decisions from the **resolved intrinsic types** — e.g. a ``__riscv_vfmacc_*`` call
returning ``vfloat32m4_t`` gives fused-FMA + SEW=32 + LMUL=4 from the canonical type spelling, not
a substring guess. ``asm`` is the authoritative substrate; this is the agreement cross-check
(``cca.cca_agree(lift_source, lift_asm)`` is the "good reconstruction" validity gate).

Header-dependent: needs the framework's ``riscv_vector.h`` include path (from its
``framework_contract``). When clang/headers are unavailable the extractor degrades gracefully
(returns an empty result) — the asm path does not depend on it.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ...llvmlower import toolchain


@dataclass
class IntrinsicCall:
    name: str                      # "__riscv_vfmacc_vf_f32m4"
    ret_type: str                  # resolved type, e.g. "vfloat32m4_t"
    sew: int | None = None
    lmul: float | None = None


@dataclass
class SourceFacts:
    intrinsics: list[IntrinsicCall] = field(default_factory=list)
    ok: bool = False               # False if clang/headers unavailable or parse failed

    def has(self, *needles: str) -> int:
        return sum(1 for c in self.intrinsics if any(n in c.name for n in needles))

    def dominant_vtype(self) -> tuple[int | None, float | None]:
        from collections import Counter
        seen = Counter((c.sew, c.lmul) for c in self.intrinsics if c.sew)
        return seen.most_common(1)[0][0] if seen else (None, None)


def _vtype_from_typename(t: str) -> tuple[int | None, float | None]:
    """Canonical RVV type spelling -> (SEW, LMUL). vfloat32m4_t->(32,4); vint8mf2_t->(8,0.5).
    Pure structured parse of the ISA's own type name (no fragile regex)."""
    s = t.strip().rstrip("_t")
    for pre in ("vfloat", "vint", "vuint", "vbool"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    else:
        return None, None
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    sew = int(s[:i]) if i else None
    rest = s[i:]
    lmul = None
    if rest.startswith("mf") and rest[2:].isdigit():
        lmul = 1.0 / int(rest[2:])
    elif rest.startswith("m") and rest[1:].isdigit():
        lmul = float(int(rest[1:]))
    return sew, lmul


def dump_ast(c_path: str | Path, include_dirs: list[str] | None = None,
             march: str = "rv64gcv") -> dict | None:
    """clang -ast-dump=json for a kernel TU, or None if clang/headers unavailable."""
    clang = toolchain.clang()
    if not Path(clang).is_file():
        return None
    cmd = [str(clang), "-Xclang", "-ast-dump=json", "-fsyntax-only",
           f"--target=riscv64-unknown-elf", f"-march={march}", "-mabi=lp64d"]
    for d in include_dirs or []:
        cmd += ["-I", d]
    cmd.append(str(c_path))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:  # noqa: BLE001
        return None
    if not p.stdout.strip():
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def _walk_calls(node: dict, out: list[IntrinsicCall]) -> None:
    """DFS the clang AST JSON for CallExprs to __riscv_* intrinsics, reading the resolved type."""
    if not isinstance(node, dict):
        return
    if node.get("kind") == "CallExpr":
        # the callee DeclRefExpr's referencedDecl.name is the intrinsic; node['type']['qualType']
        # is the call's resolved result type.
        name = _callee_name(node)
        if name and name.startswith("__riscv_"):
            rt = (node.get("type", {}) or {}).get("qualType", "")
            sew, lmul = _vtype_from_typename(rt)
            out.append(IntrinsicCall(name=name, ret_type=rt, sew=sew, lmul=lmul))
    for child in node.get("inner", []) or []:
        _walk_calls(child, out)


def _callee_name(call: dict) -> str | None:
    for child in call.get("inner", []) or []:
        if not isinstance(child, dict):
            continue
        ref = child.get("referencedDecl")
        if isinstance(ref, dict) and ref.get("name"):
            return ref["name"]
        n = _callee_name(child)
        if n:
            return n
    return None


def extract(c_path: str | Path, include_dirs: list[str] | None = None) -> SourceFacts:
    ast = dump_ast(c_path, include_dirs=include_dirs)
    if ast is None:
        return SourceFacts(ok=False)
    calls: list[IntrinsicCall] = []
    _walk_calls(ast, calls)
    return SourceFacts(intrinsics=calls, ok=True)


def facts_from_ast_json(ast: dict) -> SourceFacts:
    """Walk an already-parsed AST JSON (used by tests with a fixture; no clang needed)."""
    calls: list[IntrinsicCall] = []
    _walk_calls(ast, calls)
    return SourceFacts(intrinsics=calls, ok=True)
