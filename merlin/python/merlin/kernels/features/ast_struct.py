"""Structural facts via a tree-sitter C AST — the layer the intrinsic/regex extractors miss.

tree-sitter parses diverse kernel C *syntactically*, with no need to build or resolve framework
headers (unlike libclang) — exactly right for a corpus of microkernels with framework-specific
includes. Recovers loop nest depth + order, the streaming/prepack pointer-advance idiom, and
AST-accurate vector op counts (more reliable than regex). Returns {"struct": {...}}; a graceful
no-op (returns {}) when tree-sitter is unavailable (optional dep: the `kernels-ast` extra) or the
source doesn't parse, so the pipeline degrades to the regex/intrinsic layer rather than breaking.
"""
from __future__ import annotations

import re
from collections import Counter

from merlin.kernels.markers import target_family
from merlin.kernels.types import NormalizedKernel

try:  # optional dependency
    import tree_sitter_c
    from tree_sitter import Language, Parser
    _PARSER = Parser(Language(tree_sitter_c.language()))
except Exception:  # pragma: no cover - exercised when the extra isn't installed
    _PARSER = None

_LOOPS = {"for_statement", "while_statement", "do_statement"}


def available() -> bool:
    return _PARSER is not None


def _txt(src: bytes, n) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "ignore")


def _canon_call(name: str) -> str | None:
    if not name.startswith("__riscv_v"):
        return None
    t = name.replace("__riscv_", "").split("_")[0]
    return "vsetvl" if t.startswith("vset") else t


def _loop_cond_var(src: bytes, node) -> str:
    """A short identity for a loop: the first identifier in its condition (e.g. 'k', 'nc')."""
    cond = node.child_by_field_name("condition")
    if cond is None:
        for c in node.children:           # do-while: condition is a parenthesized_expression child
            if c.type == "parenthesized_expression":
                cond = c
                break
    if cond is None:
        return "?"
    m = re.search(r"[A-Za-z_]\w*", _txt(src, cond))
    return m.group(0) if m else "?"


def extract_ast_struct(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    if _PARSER is None or target_family(nk.target) != "rvv":
        return {}
    src = (nk.raw_text or "").encode("utf-8")
    if b"__riscv_v" not in src:
        return {}
    try:
        tree = _PARSER.parse(src)
    except Exception:
        return {}

    calls: Counter[str] = Counter()
    loop_order: list[str] = []
    state = {"maxdepth": 0, "ptr_advance": False}

    def walk(n, depth: int) -> None:
        is_loop = n.type in _LOOPS
        d = depth + (1 if is_loop else 0)
        if is_loop:
            state["maxdepth"] = max(state["maxdepth"], d)
            loop_order.append(_loop_cond_var(src, n))
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None:
                c = _canon_call(_txt(src, fn))
                if c:
                    calls[c] += 1
        if n.type == "assignment_expression":
            lhs = n.child_by_field_name("left")
            rhs = n.child_by_field_name("right")
            if lhs is not None and rhs is not None:
                lt = _txt(src, lhs).strip()
                rt = _txt(src, rhs).strip()
                # streaming/prepack idiom: `w = w + <stride>` (pointer advanced as data is consumed)
                if re.match(r"[A-Za-z_]\w*$", lt) and re.match(rf"\(?[^)]*\)?\s*{re.escape(lt)}\s*\+",
                                                               rt.replace("(const void*) ", "")):
                    state["ptr_advance"] = True
        for c in n.children:
            walk(c, d)

    walk(tree.root_node, 0)
    n_load = sum(v for k, v in calls.items() if k.startswith("vle"))
    # exclude vsetvl* — it also starts with "vse"
    n_store = sum(v for k, v in calls.items() if k.startswith("vse") and not k.startswith("vset"))
    n_fma = sum(v for k, v in calls.items() if k in ("vfmacc", "vmacc", "vwmacc", "vwmaccu"))
    return {"struct": {
        "loop_nest_depth": state["maxdepth"],
        "loop_order": loop_order,                 # outer->inner condition vars, e.g. ['nc','k']
        "pointer_advance_prepack": state["ptr_advance"],
        "n_vector_loads": n_load,
        "n_vector_stores": n_store,
        "n_fma_calls": n_fma,
        "ast_parsed": True,
    }}
