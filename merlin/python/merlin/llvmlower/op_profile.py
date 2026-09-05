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
#: ``prov.role`` is stamped by a rewrite that SPLIT one captured op into several (the int8 datapath
#: emits a contraction plus a requant epilogue from one matmul, and both carry the source op's fqn);
#: without it the two pieces are indistinguishable under a single join key.
_PROV_KEYS = ("prov.op", "prov.family", "prov.region_id", "prov.aten", "prov.module", "prov.fqn",
              "prov.role")


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
    """Dialect-qualified op name of a top-level op line (``%3 = linalg.generic ...``).

    It is the name AS PRINTED, which is not always the dialect-qualified one: ops with a custom
    assembly format lose their dialect prefix, so a ``func.call`` reaches the table as ``call`` once the
    module has been round-tripped through ``mlir-opt`` (and as ``func.call`` before that). A consumer
    that selects records by ``mlir_op`` therefore has to accept both spellings; matching only the
    qualified one silently returns nothing, which reads as "those ops were never profiled" rather than
    as a failed match -- exactly how the matrix-unit calls first looked unmeasurable.
    """
    body = line.strip()
    eq = body.find(" = ")
    if eq >= 0 and body.startswith("%"):
        body = body[eq + 3:]
    tok = body.split(" ", 1)[0].split("(", 1)[0]
    return tok.rstrip(":")


def _callee(line: str) -> str | None:
    """Symbol a top-level ``call`` invokes (e.g. ``@merlin_opu_gemm_i8_1``), else ``None``.

    Recorded because ``mlir_op`` cannot tell two calls apart. Every routed matrix-unit entry point
    reaches the table as an indistinguishable ``call`` row, so the only way to attribute the routed
    region was to COUNT call rows and match that against the number of contractions the router
    reported. That is inference, not measurement: it is silently wrong the moment anything else in
    ``@forward`` lowers to a call, and it cannot separate one entry point from another when a model
    routes several distinct signatures. Both print forms are handled -- the pretty
    ``call @sym(%a, %b)`` and the generic ``"func.call"(%a) <{callee = @sym}>`` -- because which one
    a consumer sees depends on who last round-tripped the module (see :func:`_op_name`).
    """
    body = line.strip()
    eq = body.find(" = ")
    if eq >= 0 and body.startswith("%"):
        body = body[eq + 3:]
    head, _, rest = body.partition(" ")
    if head.split("(", 1)[0].rstrip(":") not in ("call", "func.call", '"func.call"'):
        return None
    for tok in rest.replace("(", " ").replace("{", " ").split():
        if tok.startswith("@"):
            return tok.split(")")[0].rstrip(",}>:")
    return None


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
                    "callee": _callee(line),        # the routed entry point, when this op is a call
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


# ==================================================================================================
# NORMALIZATION, COVERAGE AND ROLLUP — the arithmetic that turns raw `PROF` counters into shares.
#
# These live here (not in the driver script) because they are pure and board-free, so the rules
# below are unit-testable without a board.
#
# WHY THEY EXIST — a measured 144 % attribution.
# `k1_op_profile.py` on `small_llama_int8_consistent` reported 6.0 ms of attributed op time against
# a 4.2 ms wall (`profiler_cov=1.4407`) while the perturbation guard was CLEAN (profiled 4.21 ms vs
# un-instrumented 4.20 ms, delta 0.11 %). Every share it printed was therefore inflated by an
# unstated factor, and one of them ("transposes are 45.9 % of runtime") was quoted.
#
# The cause is a WINDOW MISMATCH, not a mis-measurement. The shim's accumulators run from process
# start, so they cover every execution of `@forward` in the image — the UNTIMED warmup passes
# included — while the harness reports a PER-ITERATION wall. That run used `--warmup 2 --iters 5`:
# all 996 ops reported `hits=7`, and the driver divided by `iters=5`. 7/5 = 1.400, and
# 1.4407 / 1.400 = 1.0291. The inflation was exactly the ratio of the two windows.
#
# Note what this rules OUT, because the alternatives are the ones people reach for first. Within ONE
# execution the attributed intervals TELESCOPE: op i is credited `rdtime(entry of mark i+1) -
# rdtime(exit of mark i)`, and those intervals are disjoint and lie inside the function, so their sum
# is bounded ABOVE by the function's own elapsed time. Neither the marks' own cost nor `rdtime`'s
# ~41.7 ns quantization can push a single execution past 100 % — a coverage above 1.0 can ONLY come
# from counting a different number of executions in the numerator than in the denominator.
#
# THE FIX: derive the divisor, never assume it. `hits` is the execution count the shim actually
# COUNTED for that op; `--iters` is what the CLI asked for. They differ under `--warmup`, under
# `MERLIN_SESSION_REPEATS`, and under any model whose session runs more than one step per iteration.
# So per-execution cost is `ticks / hits`, and the residual over 1.0 that remains is named rather
# than hidden (see `coverage_report`).
# ==================================================================================================

#: Sane band for `profiler_coverage` (attributed ticks per execution / measured wall ticks per timed
#: iteration). OUTSIDE this band the tool refuses to express any bucket as a percentage of runtime.
#:
#: The upper bound is not 1.0. Telescoping caps a single execution at 1.0, but the accumulation
#: window may still contain UNTIMED warmup executions, which are colder than the timed ones and so
#: pull the per-execution mean above the timed wall. Measured on the artifact above: 1.022 (op table)
#: / 1.029 (the `prof_total_ticks` metric, which also carries the sentinel's inter-execution gap)
#: with 2 warmup + 5 timed passes — consistent with the warmup passes being ~8 % slower. More than
#: 5 % over is a window/denominator defect rather than warmup, and is refused.
COVERAGE_BAND = (0.70, 1.05)

#: Families the RVV schedule vectorizes. Everything else lowers to scalar loops, so the sum of the
#: non-contraction ticks is the scalar work left on the table. This is the pipeline's INTENT, not
#: proof about the emitted asm.
VECTORIZED_FAMILIES = frozenset({"contraction"})

#: Named linalg ops that ARE a contraction, by op name alone. Used ONLY as the fallback when an op
#: carries no ``prov.family`` — which is the common case, not the rare one: the quantization and
#: blocking rewrites rebuild ops without re-stamping provenance, so on the int8 whole model 776 of
#: 996 profiled ops (including every ``linalg.matmul``) reach the table with ``family = None``. A
#: summary that buckets on the tag alone therefore put all of them in ``(none)`` and printed
#: ``contraction = 0.0 ms`` for a model whose matmuls measured 1.36 ms. A headline that says a
#: contraction costs zero is worse than no headline.
#:
#: Kept a SUPERSET of ``xdsl_dialects.lowering.contraction_coverage.MATMUL_OPS`` (asserted by a test
#: rather than by importing it, so this module stays pure-text and xDSL-free).
CONTRACTION_OPS = frozenset({
    "linalg.matmul", "linalg.batch_matmul", "linalg.quantized_matmul",
    "linalg.matmul_transpose_a", "linalg.matmul_transpose_b",
    "linalg.batch_matmul_transpose_a", "linalg.batch_matmul_transpose_b",
    "linalg.matvec", "linalg.vecmat", "linalg.batch_matvec", "linalg.dot",
    "linalg.quantized_batch_matmul", "linalg.contract",
})


def resolve_family(rec: dict) -> tuple[str, str]:
    """``(family, source)`` for one op record, falling back to the MLIR op name.

    ``prov.family`` wins when present. When it is absent the op NAME answers the question for every
    named op — a ``linalg.matmul`` is a contraction whether or not anything stamped it — and for an
    unnamed one (``linalg.generic``, ``tensor.concat``) the op name itself is the honest bucket
    label: it is what we actually know. The ``source`` is returned so a reader can tell a tagged
    family from a derived one instead of having to trust the label.
    """
    fam = rec.get("family")
    if fam:
        return str(fam), "prov.family"
    name = rec.get("mlir_op") or ""
    if name in CONTRACTION_OPS:
        return "contraction", "mlir_op"
    return name or "(unknown)", "mlir_op"


def is_unclassified_generic(rec: dict) -> bool:
    """True for an untagged ``linalg.generic`` — an op whose family we genuinely do not know.

    This is the one honest hole left by :func:`resolve_family`. The integer datapath rewrites a
    captured contraction into a ``linalg.generic`` with an ``i8 x i8 -> i32`` body and no
    ``prov.family``, so some of these ARE contractions; the profiler records only the printed op
    line and cannot see the body to tell. Their mass is reported as its own quantity so it reads as
    UNKNOWN rather than as "measured, and not a contraction".
    """
    return not rec.get("family") and (rec.get("mlir_op") or "") == "linalg.generic"


def per_execution_ticks(total_ticks: float, executions: int) -> float | None:
    """Ticks for ONE execution of an op, from the execution count the shim COUNTED.

    ``None`` when the op never executed (0 hits) — a fail-closed UNKNOWN, never a silent 0.0 that
    would enter a sum as if it had been measured and found free.
    """
    if not executions:
        return None
    return total_ticks / executions


def annotate_table(table: list[dict], *, timebase_hz: float) -> list[dict]:
    """Fill in the derived per-op fields, IN PLACE, and return the table.

    Sets ``executions`` (the measured hit count), ``ticks_avg``/``ms_avg`` (per execution),
    ``family_resolved``/``family_source``, ``vectorized``, ``join_key`` and
    ``category``/``category_source`` (:func:`resolve_category`). An op with 0 hits gets
    ``ticks_avg = None`` and is excluded from every sum by :func:`sum_attributed_ticks`.
    """
    ns_per_tick = 1e9 / timebase_hz
    for rec in table:
        hits = int(rec.get("hits") or 0)
        avg = per_execution_ticks(float(rec.get("ticks") or 0), hits)
        fam, src = resolve_family(rec)
        rec["executions"] = hits
        rec["ticks_avg"] = avg
        rec["ms_avg"] = None if avg is None else avg * ns_per_tick / 1e6
        rec["family_resolved"] = fam
        rec["family_source"] = src
        rec["vectorized"] = fam in VECTORIZED_FAMILIES
        rec["join_key"] = join_key(rec)
        cat, csrc = resolve_category(rec)
        rec["category"] = cat
        rec["category_source"] = csrc
    return table


def sum_attributed_ticks(table: list[dict]) -> float:
    """Total per-execution ticks over the ops that actually executed."""
    return sum(r["ticks_avg"] for r in table if r.get("ticks_avg") is not None)


def coverage_report(attributed_ticks: float, wall_ticks: float | None, *,
                    band: tuple[float, float] = COVERAGE_BAND,
                    executions: int | None = None,
                    timed_iterations: int | None = None,
                    executions_per_timed_iteration: int | None = None) -> dict:
    """Decide whether this profile may be quoted as a percentage of RUNTIME.

    ``attributed_ticks`` is per EXECUTION of ``@forward``; ``wall_ticks`` is per TIMED ITERATION.
    Those are the same window only when one iteration runs ``@forward`` once. A v1 session bundle
    runs it once per declared step (``steps: 256``), so the wall must first be divided by
    ``executions_per_timed_iteration`` (:func:`executions_per_iteration`) or the coverage comes back
    near 1/256 and a sound profile is refused. Passing ``None`` means "one execution per iteration",
    which is the plain-model case and what every pre-session caller meant; the value used is echoed
    back in the report so the denominator is never implicit.

    Fail-closed: an unknown or out-of-band coverage sets ``runtime_shares_reportable`` False, and
    every consumer must then express buckets as a share of ATTRIBUTED time with the coverage stated
    beside them. It is never left to the reader to notice that the denominator moved.
    """
    lo, hi = band
    per_iter = 1 if executions_per_timed_iteration is None else int(executions_per_timed_iteration)
    wall_per_execution = (wall_ticks / per_iter) if (wall_ticks and per_iter >= 1) else None
    cov = (attributed_ticks / wall_per_execution) if wall_per_execution else None
    over_window = (executions is not None and timed_iterations is not None
                   and executions > timed_iterations)
    if cov is None:
        refusal = ("cannot derive how many @forward executions one timed iteration contains, so "
                   "the per-execution wall has no divisor"
                   if wall_ticks else
                   "no measured wall ticks to divide by — a share of runtime has no denominator")
    elif cov < lo:
        refusal = (f"profiler coverage {cov:.4f} is below {lo:.2f}: more than "
                   f"{100 * (1 - cov):.1f}% of the wall was never attributed to any op, so a "
                   f"bucket's share of RUNTIME is unknowable from this profile")
    elif cov > hi:
        refusal = (f"profiler coverage {cov:.4f} is above {hi:.2f}: the ops were credited MORE "
                   f"time than the wall being measured, so every share of RUNTIME would be "
                   f"inflated by an unknown factor. Usual cause: the attribution window and the "
                   f"timed window count a different number of @forward executions")
    else:
        refusal = None
    return {
        "attributed_ticks": attributed_ticks,
        "wall_ticks": wall_ticks,
        #: The denominator actually divided by: the wall of ONE @forward execution. Equal to
        #: `wall_ticks` for a plain model and to `wall_ticks / steps` for a session bundle.
        "wall_ticks_per_execution": wall_per_execution,
        "executions_per_timed_iteration": per_iter,
        "profiler_coverage": None if cov is None else round(cov, 4),
        "band": [lo, hi],
        "in_band": refusal is None,
        "runtime_shares_reportable": refusal is None,
        "share_denominator": "wall" if refusal is None else "attributed",
        "executions_per_launch": executions,
        "timed_iterations_per_launch": timed_iterations,
        #: The shim COUNTED more executions of @forward than the harness TIMED. Untimed warmup
        #: passes do this, and so does a session that runs several steps per timed iteration. Either
        #: way the per-execution mean is over a wider window than the wall it is compared against,
        #: which is the direction that inflates coverage.
        "executions_exceed_timed_iterations": over_window,
        "refusal": refusal,
        "note": ("Shares are a fraction of ATTRIBUTED op time unless runtime_shares_reportable is "
                 "true. Coverage = attributed ticks per @forward EXECUTION / wall ticks per "
                 "@forward EXECUTION, where the latter is the harness's per-timed-iteration wall "
                 "divided by executions_per_timed_iteration. Both sides must count the same number "
                 "of executions: counting more in the numerator inflates the coverage above 1.0 "
                 "(untimed warmup passes accumulate into the shim), counting more in the "
                 "denominator deflates it (a session runs @forward once per declared step)."),
    }


def rollup(table: list[dict], keyfn, label: str, *, wall_ms: float | None = None,
           coverage: float | None = None) -> list[dict]:
    """Aggregate per-execution ms by ``keyfn``, sorted by cost.

    ``share_of_attributed`` is always present. ``share_of_runtime`` is present only when the caller
    passes ``wall_ms`` — which it must do ONLY when :func:`coverage_report` said the profile may be
    quoted that way; otherwise the field stays ``None`` so a consumer cannot read an inflated share
    as a runtime percentage.

    Every row also carries ``share_denominator`` and ``profiler_coverage``, so a row lifted out of
    the JSON and quoted on its own still says what its percentage is a percentage OF. The 144 %
    artifact was quoted one row at a time; a caveat that lives only in a sibling block is a caveat
    that does not travel.
    """
    agg: dict[str, dict] = {}
    for rec in table:
        ms = rec.get("ms_avg")
        if ms is None:
            continue
        a = agg.setdefault(keyfn(rec), {"ms": 0.0, "n_ops": 0, "hits": 0, "vectorized": None,
                                        "family_sources": set()})
        a["ms"] += ms
        a["n_ops"] += 1
        a["hits"] += int(rec.get("hits") or 0)
        v = bool(rec.get("vectorized"))
        a["vectorized"] = v if a["vectorized"] is None else (a["vectorized"] and v)
        a["family_sources"].add(rec.get("family_source") or "unknown")
    total = sum(a["ms"] for a in agg.values())
    rows = []
    for k, a in agg.items():
        rows.append({label: k, "ms": a["ms"], "n_ops": a["n_ops"], "hits": a["hits"],
                     "vectorized": a["vectorized"],
                     "family_sources": sorted(a["family_sources"]),
                     "share_of_attributed": (a["ms"] / total) if total else None,
                     "share_of_runtime": (a["ms"] / wall_ms) if wall_ms else None,
                     "share_denominator": "wall" if wall_ms else "attributed",
                     "profiler_coverage": coverage})
    rows.sort(key=lambda r: r["ms"], reverse=True)
    return rows


# ==================================================================================================
# ACTIONABLE CATEGORIES — the buckets an optimization decision is actually made in.
#
# `resolve_family` answers "what KIND of op is this" in the frontend's own vocabulary (contraction,
# layout, normalization, ...). That vocabulary is the right one for provenance and the wrong one for
# a decision: `normalization` merges layer_norm with softmax, `layout` merges a free `view` with a
# materializing transpose, and the whole int8 quantize chain is spread across `quantize`, `cast` and
# a set of `prov.role`s that no family name mentions.
#
# The categories below are the ones a lever can be aimed at. They are DERIVED, in this order:
#   1. `prov.role` — stamped by `passes_quant_int._carry_prov` when a rewrite SPLIT one captured op
#      into pieces. Its vocabulary is exactly {act_amax, act_scale, act_quantize, gather,
#      contraction, requant}, i.e. the quantize/requant chain itself, so it wins over every other
#      signal: the requant epilogue of a matmul carries the matmul's family and fqn, and bucketing it
#      as a contraction is the specific mistake this ordering exists to prevent.
#   2. `prov.op` — the semantic op, which splits a family whose members cost differently (`softmax`
#      is a reduction, `layer_norm` is not; `view` is free, `transpose` materializes).
#   3. `prov.family` — the frontend family.
#   4. the printed MLIR op name — for the ~40 % of ops the quantization and blocking rewrites rebuild
#      without re-stamping provenance.
# Anything none of the four can place lands in `unclassified:<mlir_op>` and is REPORTED as such. An
# untagged `linalg.generic` gets its own `unclassified_generic` bucket because it is the one case
# where the family is genuinely unknowable from the printed line (some of them ARE contractions).
# ==================================================================================================

#: What each `prov.role` is, in cost terms. Vocabulary from `passes_quant_int._carry_prov` call
#: sites; `epilogue_fusion` documents the contraction/requant pair.
ROLE_CATEGORY = {
    "contraction": "contraction",
    "requant": "quantize_requant",
    "act_amax": "quantize_requant",
    "act_scale": "quantize_requant",
    "act_quantize": "quantize_requant",
    "gather": "gather",
}

#: `prov.op` overrides that split a family whose members do not cost alike.
OP_CATEGORY = {
    "softmax": "reduction_softmax",
    "view": "layout_view",            # metadata-only reshape: should read ~free
    "reshape": "layout_view",
    "squeeze": "layout_view",
    "unsqueeze": "layout_view",
    "expand": "layout_view",
    "quantize": "quantize_requant",
    "dequantize": "quantize_requant",
    "requantize": "quantize_requant",
    "dtype_cast": "quantize_requant",
    "embedding": "gather",
    "index_gather": "gather",
    "convolution_im2col_matmul": "contraction",
}

#: `prov.family` -> category. The families are the ones the study models' own captures emit.
FAMILY_CATEGORY = {
    "contraction": "contraction",
    "quantize": "quantize_requant",
    "cast": "quantize_requant",
    "elementwise": "elementwise",
    "bitwise": "elementwise",
    "compare": "elementwise",
    "minmax": "elementwise",
    "normalization": "normalization",
    "reduce": "reduction_softmax",
    "arg_reduce": "reduction_softmax",
    "scan": "reduction_softmax",
    "pool": "reduction_softmax",
    "spectral": "spectral",
    "layout": "layout_copy",
    "concat": "layout_copy",
    "resize": "layout_copy",
    "gather_scatter": "gather",
    "fill": "fill_init",
    "iota": "fill_init",
}

#: Printed MLIR op name -> category, for ops that reach the table with no provenance at all.
MLIR_OP_CATEGORY = {
    "linalg.transpose": "layout_copy",
    "linalg.copy": "layout_copy",
    "linalg.broadcast": "elementwise",
    "linalg.reduce": "reduction_softmax",
    "linalg.fill": "fill_init",
    "tensor.insert_slice": "layout_copy",
    "tensor.extract_slice": "layout_copy",
    "tensor.concat": "layout_copy",
    "tensor.expand_shape": "layout_view",
    "tensor.collapse_shape": "layout_view",
    "tensor.empty": "alloc",
    "tensor.splat": "fill_init",
    "bufferization.alloc_tensor": "alloc",
    "memref.alloc": "alloc",
    "memref.alloca": "alloc",
    "memref.dealloc": "alloc",
    "memref.copy": "layout_copy",
    "arith.constant": "constant",
}

#: The categories the tool ranks, in the order a reader should see them. A category absent from a
#: given model simply does not appear; a category the profiler CANNOT see at all is listed in
#: :data:`CATEGORIES_NOT_ATTRIBUTABLE` so its absence is never read as "it costs nothing".
CATEGORY_ORDER = ("contraction", "quantize_requant", "elementwise", "normalization",
                  "reduction_softmax", "gather", "layout_copy", "spectral", "layout_view",
                  "fill_init", "alloc", "constant", "unclassified_generic")

#: What a per-op mark interval structurally CANNOT attribute, and why. Reported in the artifact so a
#: reader does not mistake a missing row for a measured zero.
CATEGORIES_NOT_ATTRIBUTABLE = {
    "allocator": ("`memref.alloc`/`free` are hoisted toward the function entry by "
                  "buffer-hoisting/buffer-loop-hoisting, so allocator cost drifts into whichever "
                  "mark interval the hoisted alloc lands in (typically the first) rather than into "
                  "the op that needed the buffer. Measure it differentially instead: build twice "
                  "with and without MERLIN_BUMP_MALLOC and diff the walls."),
    "fork_join": ("there is no fork/join inside a single-threaded @forward. Under "
                  "parallel_harts/OpenMP the join happens INSIDE one top-level op, below the "
                  "granularity of a mark interval, so it is charged to that op and cannot be "
                  "separated from it."),
    "intra_op": ("a mark interval is one whole top-level op. Anything inside it — a contraction's "
                 "inner loop versus its tail, an epilogue the compiler fused into the op — is below "
                 "this profiler's resolution."),
}


def resolve_category(rec: dict) -> tuple[str, str]:
    """``(category, source)`` for one op record — the bucket a lever can be aimed at.

    Resolution order is role -> op -> family -> mlir_op (see the module section above); ``source``
    names which one answered, so a reader can tell a stamped category from a derived one. An op none
    of them place returns ``unclassified:<mlir_op>`` rather than being folded into a neighbour.
    """
    role = rec.get("role")
    if role and role in ROLE_CATEGORY:
        return ROLE_CATEGORY[role], "prov.role"
    op = rec.get("op")
    if op and op in OP_CATEGORY:
        return OP_CATEGORY[op], "prov.op"
    fam = rec.get("family")
    if fam and fam in FAMILY_CATEGORY:
        return FAMILY_CATEGORY[fam], "prov.family"
    name = rec.get("mlir_op") or ""
    if name in MLIR_OP_CATEGORY:
        return MLIR_OP_CATEGORY[name], "mlir_op"
    if name in CONTRACTION_OPS:
        return "contraction", "mlir_op"
    if is_unclassified_generic(rec):
        return "unclassified_generic", "unknown"
    return f"unclassified:{name or '(unknown)'}", "unknown"


def executions_per_iteration(executions: int | None, timed_iterations: int | None,
                             warmup: int | None = 0) -> int | None:
    """How many ``@forward`` executions ONE timed iteration contains, or ``None`` if underivable.

    The shim's accumulators run from process start, so ``executions`` (the largest ``hits`` in the
    table) counts every execution in the image: ``(warmup + timed_iterations)`` launches, each of
    which runs ``@forward`` once for a plain model and once PER SESSION STEP for a session model.
    A v1 session bundle declares ``steps: 256``, so one timed iteration is 256 executions — and
    comparing a per-EXECUTION attributed total against a per-ITERATION wall then reports a coverage
    near 1/256 and refuses a profile that is in fact sound. This is the same window mismatch that
    produced the 144 % artifact, in the opposite direction.

    Fail-closed: a count that is not a positive whole multiple of the launches returns ``None``,
    which makes :func:`coverage_report` refuse rather than guess a divisor.
    """
    launches = int(timed_iterations or 0) + int(warmup or 0)
    if not executions or launches <= 0:
        return None
    q, r = divmod(int(executions), launches)
    return q if (r == 0 and q >= 1) else None


def table_blocker(table: list[dict] | None) -> str | None:
    """Why this op table cannot be a profile, or ``None`` if it can.

    Separated from the driver so the fail-closed rule is testable without a board. Two ways a run
    comes back carrying nothing, both of which used to sail through as a successful profile of a
    model with no attributable work:

    * an EMPTY table — the build never instrumented the IR (e.g. a build path that ignores the
      ``op_profile`` flag) or wrote no ``opprof_table.json``;
    * a table with ops but ZERO recorded hits — the marks were compiled in but never executed, or
      ``merlin_prof_dump()`` never ran, so no ``PROF`` line reached the console.

    Neither is a measurement. "Nothing was measured" and "there was nothing to measure" are
    different claims, and only the second is ever a finding.
    """
    if table is None:
        return "no op table was produced at all"
    if not table:
        return ("the instrumented run produced an EMPTY op table: the build emitted no "
                "opprof_table.json, or no `PROF <id> <ticks> <hits>` line reached the console. "
                "Nothing was measured — this is not a profile of a model with no ops.")
    if not any(int(r.get("hits") or 0) for r in table):
        return (f"the op table has {len(table)} ops and ZERO recorded hits: the marks were compiled "
                f"in but never executed, or merlin_prof_dump() never ran. Nothing was measured.")
    return None
