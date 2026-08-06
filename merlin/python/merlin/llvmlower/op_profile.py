"""WHOLE-MODEL PER-OP PROFILER: instrument the top-level ops of ``func.func @forward``.

Motivation. The only whole-model instrumentation this repo had was a TWO-WAY split — a
matmul bucket (``-DMERLIN_DISPATCH_TIMING`` inside the routed GEMM shim) versus
"everything else". Measured on the K1, matmul is **1.3–6 %** of a model once the kernel is
fast, so 94–97 % of model time had never been attributed to anything. This module creates
the missing attribution.

METHOD (default-OFF; the un-instrumented build is byte-identical).
The board runs ONE monolithic ``_mlir_ciface_forward``, so there is no call boundary to hook.
We create one the same way the kernel backends do — by rewriting the IR — but instead of
replacing ops we *interleave* them with a zero-argument-cost marker:

    call @merlin_prof_mark(%id) : (i32) -> ()
    <op i>
    call @merlin_prof_mark(%id+1) : (i32) -> ()
    <op i+1>
    ...
    call @merlin_prof_mark(%sentinel)          (immediately before func.return)

The shim (``runtime/c/merlin_op_prof.c``) records ``rdtime`` at each mark and credits the
elapsed ticks to the PREVIOUS mark's id. So one call per op, not two, and
``ticks[i]`` is the cost of top-level op ``i``.

Why one marker per top-level op is safe here. The default K1 RVV pipeline
(:func:`merlin.llvmlower.pipeline.build_rvv_pipeline`) does **not** run
``linalg-fuse-elementwise-ops`` (it is env-gated behind ``MERLIN_FUSE_POST``), so
interleaving side-effecting calls cannot inhibit a fusion that would otherwise have
happened. The marks therefore cost a call + ``rdtime`` + two stores each, and change no
codegen decision for the ops themselves. This is an assumption about the pipeline, not a
proof — which is why the driver (``build_tools/scripts/k1_op_profile.py``) always measures
the instrumented wall against the un-instrumented wall and refuses to report a breakdown
whose total wall moved by more than the board noise floor.

KNOWN ATTRIBUTION LIMITS (stated, not hidden).
  * ``buffer-hoisting``/``buffer-loop-hoisting`` move ``memref.alloc``s toward the function
    entry, so allocation cost drifts to whichever mark interval the hoisted alloc lands in
    (typically the first). Allocation is measured in aggregate by the harness wall, not
    per-op.
  * ``tensor.empty`` / ``arith.constant`` ops are instrumented too and should read ~0; a
    non-zero reading there is the signature of hoisted work, and is reported as such.
  * ``rdtime`` is the 24 MHz platform counter (~41.7 ns/tick): a single op faster than ~42 ns
    reads 0 or 1 tick. Per-op numbers are only meaningful in aggregate (by op/family), which
    is how the driver reports them.

The op table (id -> op name + ``prov.*`` provenance + result type) is emitted alongside so
the board's ``PROF <id> <ticks>`` lines can be joined back to model semantics.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Name of the marker hook the instrumented IR calls (defined in runtime/c/merlin_op_prof.c).
MARK_SYM = "merlin_prof_mark"

#: Attributes lifted from each op into the table, when present. ``prov.fqn`` is the cross-compiler
#: join key (the deepest ``nn.Module`` path) that aligns a Merlin region with the SAME model layer
#: in another frontend (ExecuTorch/GGUF/ONNX) — see :mod:`merlin.baselines.contract` /
#: ``baselines/_et_export.py``. Captures that predate fqn-tagging still carry ``prov.region_id``,
#: which the driver uses as the fallback join key; ``join_key()`` encodes that preference.
_PROV_KEYS = ("prov.op", "prov.family", "prov.region_id", "prov.aten", "prov.module", "prov.fqn")


class OpProfileError(RuntimeError):
    pass


def _depth_delta(line: str) -> int:
    """Net ``{``-vs-``}`` nesting change of one MLIR line, ignoring quoted strings.

    String literals in this IR (``prov.*`` values, op names) never contain braces, but the
    scan skips them anyway so the depth tracking cannot be desynchronised by a future
    attribute that does.
    """
    depth = 0
    in_str = False
    prev = ""
    for ch in line:
        if in_str:
            if ch == '"' and prev != "\\":
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        prev = ch
    return depth


def _attr_value(line: str, key: str) -> str | None:
    """Value of ``key = "..."`` in an MLIR attribute dict, or None. Structured, not regex."""
    needle = key + ' = "'
    i = line.find(needle)
    if i < 0:
        return None
    j = i + len(needle)
    k = line.find('"', j)
    return None if k < 0 else line[j:k]


def _op_name(line: str) -> str:
    """Dialect-qualified op name of a top-level op line (``%3 = linalg.generic ...``)."""
    body = line.strip()
    eq = body.find(" = ")
    if eq >= 0 and body.startswith("%"):
        body = body[eq + 3:]
    tok = body.split(" ", 1)[0].split("(", 1)[0]
    return tok.rstrip(":")


def _result_type(line: str) -> str | None:
    """Best-effort result type: the last ``tensor<...>`` / ``memref<...>`` on the line."""
    for kind in ("tensor<", "memref<", "vector<"):
        i = line.rfind(kind)
        if i < 0:
            continue
        j = line.find(">", i)
        if j >= 0:
            return line[i:j + 1]
    return None


def _elem_count(ty: str | None) -> int | None:
    """Element count of a static shaped type string, or None if dynamic/unparsable."""
    if not ty:
        return None
    inner = ty[ty.find("<") + 1:ty.rfind(">")]
    dims = inner.split("x")[:-1]          # drop the element type
    n = 1
    for d in dims:
        d = d.strip()
        if not d.isdigit():
            return None
        n *= int(d)
    return n


def find_forward_ops(mlir_text: str) -> tuple[int, int, list[dict]]:
    """Locate the top-level ops of ``func.func @forward``.

    Returns ``(body_start_line, return_line, ops)`` where ``ops`` is a list of
    ``{"line": <0-based index>, "mlir_op": ..., "result_type": ..., prov...}`` in program
    order. Only ops at the function body's own nesting level are listed — the bodies of
    ``linalg.generic`` regions (and their ``^bb`` labels) are not.
    """
    lines = mlir_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("func.func @forward"):
            start = i
            break
    if start is None:
        raise OpProfileError("no `func.func @forward` in the module — cannot instrument")

    ops: list[dict] = []
    ret_line = None
    depth = 0                              # nesting relative to the function body
    for i in range(start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if depth == 0:
            # Both spellings. MLIR prints func's terminator as bare `return` in the pretty form and
            # `func.return` in the generic one, and which you get depends on who last round-tripped
            # the module -- the per-op tagging pass emits the pretty form. Matching only the
            # qualified name meant the scan ran off the end of the function, hit the module's closing
            # brace, and reported "unbalanced braces" for IR that was perfectly well formed. The
            # split on whitespace is so `returns_something` cannot masquerade as the terminator.
            if stripped.split(" ", 1)[0].rstrip(":") in ("return", "func.return"):
                ret_line = i
                break
            # A top-level op boundary is an SSA-assignment line (``%r = <op> ...``) at the
            # function body's own nesting level. This deliberately EXCLUDES depth-0 continuation
            # lines that some multi-line ops emit — e.g. ``linalg.reduce`` whose reduction region
            # ``(%a: f32, %b: f32) { ... }`` opens on the line AFTER its (brace-balanced) first
            # line: that continuation starts with ``(``, not ``%r =``, so it is not mistaken for a
            # new op and no marker is spliced into the middle of the reduce. Region bodies of ops
            # whose ``{`` opens on their first line are at depth>0 and already excluded.
            if stripped.startswith("%") and " = " in stripped.split("(", 1)[0]:
                ops.append({
                    "id": len(ops),
                    "line": i,
                    "mlir_op": _op_name(line),      # dialect op, e.g. linalg.generic
                    "result_type": _result_type(line),
                    # `prov.op`/`prov.family`/... land as op/family/region_id/aten/module below:
                    # the SEMANTIC identity (softmax, rms_norm, ...) the capture recorded.
                    **{k.split(".", 1)[1]: _attr_value(line, k) for k in _PROV_KEYS},
                })
        depth += _depth_delta(line)
        if depth < 0:                      # closed the function body without a return
            raise OpProfileError("unbalanced braces before the terminator of @forward")
    if ret_line is None:
        raise OpProfileError("no `return`/`func.return` found in @forward")
    for rec in ops:
        rec["elems"] = _elem_count(rec["result_type"])
    return start, ret_line, ops


def instrument(mlir_text: str) -> tuple[str, list[dict]]:
    """Interleave ``@merlin_prof_mark`` calls between the top-level ops of ``@forward``.

    Returns ``(instrumented_text, table)``. ``table`` has one record per mark id; the final
    id (``len(table)``) is the sentinel emitted before ``func.return`` and closes the last
    op's interval. Raises :class:`OpProfileError` if the module has no instrumentable
    ``@forward``.
    """
    lines = mlir_text.splitlines()
    fn_line, ret_line, ops = find_forward_ops(mlir_text)
    if not ops:
        raise OpProfileError("@forward has no top-level ops to instrument")

    # Marker insertions, keyed by the line they precede.
    def mark(mid: int, indent: str) -> list[str]:
        return [f"{indent}%prof_id_{mid} = arith.constant {mid} : i32",
                f"{indent}call @{MARK_SYM}(%prof_id_{mid}) : (i32) -> ()"]

    at: dict[int, list[str]] = {}
    for rec in ops:
        line = lines[rec["line"]]
        indent = line[:len(line) - len(line.lstrip())]
        at[rec["line"]] = mark(rec["id"], indent)
    sentinel = len(ops)
    rl = lines[ret_line]
    at[ret_line] = mark(sentinel, rl[:len(rl) - len(rl.lstrip())])

    out: list[str] = []
    for i, line in enumerate(lines):
        out.extend(at.get(i, ()))
        out.append(line)

    # Declare the hook just before @forward, at the function's own indentation.
    decl_indent = lines[fn_line][:len(lines[fn_line]) - len(lines[fn_line].lstrip())]
    decl = f"{decl_indent}func.func private @{MARK_SYM}(i32) -> ()"
    # `fn_line` shifted by the markers inserted above it (there are none — all insertions are
    # inside the body — so the index is stable, but recompute defensively).
    ins = out.index(lines[fn_line])
    out.insert(ins, decl)

    table = [{k: v for k, v in rec.items() if k != "line"} for rec in ops]
    return "\n".join(out) + "\n", table


def write_table(table: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1) + "\n")
    return path


def join_key(rec: dict) -> str:
    """The cross-compiler join key for one op record: prefer ``prov.fqn`` (the deepest
    ``nn.Module`` path that aligns with an ExecuTorch/GGUF/ONNX node — see
    :mod:`merlin.baselines.contract`), fall back to ``prov.region_id`` for captures that predate
    fqn-tagging, and finally to the MLIR op name so the key is never empty."""
    return rec.get("fqn") or rec.get("region_id") or rec.get("mlir_op") or "unknown"


def parse_prof_lines(console: str) -> dict[int, tuple[int, int]]:
    """Parse ``PROF <id> <ticks> <hits>`` lines from a board console into ``{id: (ticks, hits)}``."""
    out: dict[int, tuple[int, int]] = {}
    for line in console.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "PROF":
            try:
                out[int(parts[1])] = (int(parts[2]), int(parts[3]))
            except ValueError:
                continue
    return out
