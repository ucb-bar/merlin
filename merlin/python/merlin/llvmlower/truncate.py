"""Truncate a linalg-on-tensors func to return an intermediate value.

The NaN-bisection tool (and the seed of the real outliner): given the model MLIR text,
produce a variant whose `@forward` returns the SSA value defined by the op at a chosen
*line index*, with everything after it dropped. Lowering + running the truncated module
and checking whether the output is finite localizes the first op that produces NaN —
no torch per-op golden needed (NaN is its own oracle).

Pure-text (operates before the xDSL/iree pipeline), mirroring passes_xdsl's textual
approach, so it composes with `lower_model(..., textual=True)`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A top-level op inside @forward that defines exactly one tensor result, e.g.
#   %123 = "op"(...) ... : (...) -> tensor<...>
# or the pretty form  %123 = op ... : tensor<...>
_RESULT_LINE = re.compile(
    r"^\s*(?P<res>%[A-Za-z0-9_]+) =.*-> (?P<ty>tensor<[^>]+>)\s*$")
_RESULT_LINE_PRETTY = re.compile(
    r"^\s*(?P<res>%[A-Za-z0-9_]+) = .* : (?P<ty>tensor<[^>]+>)\s*$")


@dataclass
class TensorDef:
    line_idx: int
    result: str
    type: str


# Any single-result op line: `%id = ...`. The result tensor type is the LAST
# `tensor<...>` on the line (outs type for transpose, dst for extract_slice, the
# `into` type for reshapes, etc.) — robust across op syntaxes.
_ANY_DEF = re.compile(r"^\s*(?P<res>%[A-Za-z0-9_]+) = ")
_LAST_TENSOR = re.compile(r"tensor<[^<>]*(?:<[^<>]*>[^<>]*)*>")


def tensor_defs(mlir_text: str) -> list[TensorDef]:
    """All single-tensor-result top-level defs inside the function body, in order.

    Covers every op syntax (linalg/tensor/arith), not just `-> tensor<...>` — so
    reshapes, transposes, concats and slices are bisectable too.
    """
    defs: list[TensorDef] = []
    for i, line in enumerate(mlir_text.splitlines()):
        if "^bb" in line or "linalg.yield" in line or "func.return" in line:
            continue
        m = _ANY_DEF.match(line)
        if not m or ", %" in line.split("=")[0]:   # skip multi-result defs
            continue
        tys = _LAST_TENSOR.findall(line)
        if not tys:
            continue
        defs.append(TensorDef(i, m["res"], tys[-1]))
    return defs


def truncate_to(mlir_text: str, target: TensorDef) -> str:
    """Return module text whose @forward returns ``target`` (drops later body lines).

    Keeps every line up to and including the target def, replaces the function's
    `func.return` with one returning the target, and rewrites the function result type.
    Relies on MLIR/clang dead-code elimination to prune now-unused weight args' uses;
    the arg list is unchanged (extra args are harmless).
    """
    lines = mlir_text.splitlines()
    out: list[str] = []
    indent = ""
    for i, line in enumerate(lines):
        if i <= target.line_idx:
            out.append(line)
            if line.strip().startswith("func.return"):
                indent = line[: len(line) - len(line.lstrip())]
        # drop body lines after the target, but keep closing braces of module/func
    # Rebuild: header..target, then return target, then close func + module.
    head = lines[: target.line_idx + 1]
    # function signature line -> patch result type
    new_head = []
    for line in head:
        if line.lstrip().startswith("func.func @forward") and "->" in line:
            line = re.sub(r"->\s*tensor<[^>]+>", f"-> {target.type}", line, count=1)
            line = re.sub(r"->\s*\(tensor<[^)]+\)", f"-> {target.type}", line, count=1)
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
        if line.lstrip().startswith("func.func @forward") and "->" in line:
            line = re.sub(r"->\s*tensor<[^>]+>", f"-> ({res_types})", line, count=1)
            line = re.sub(r"->\s*\(tensor<.*\)\s*(attributes|\{)",
                          lambda m: f"-> ({res_types}) " + m.group(1), line, count=1)
        new_head.append(line)
    return "\n".join(new_head) + "\n" + \
        f"    func.return {res_vals} : {res_types}\n  }}\n}}\n"
