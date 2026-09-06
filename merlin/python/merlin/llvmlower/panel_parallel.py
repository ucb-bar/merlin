"""Turn bufferized, disjoint panel loops into OpenMP worksharing loops.

Both packed-weight and packed-im2col contractions use an outer tensor ``scf.for`` to carry the
accumulator while each iteration computes one disjoint output panel.  Parallelizing the inner
contraction changes the register block (and creates one fork per panel); parallelizing that carrier
before bufferization is illegal because it still has a tensor iter-arg.  After one-shot bufferization,
the accumulator is a destination memref, the iter-arg/result disappear, and the same loop is a
reduction-free collection of disjoint writes.  This runner rewrite changes only that marked loop to
``scf.parallel`` immediately before ``convert-scf-to-openmp``.

The marker is a zero-argument call inside the loop, emitted only for a multicore preparation.  A call
is intentional: xDSL's custom ``scf.for`` printer drops discardable attributes, while attributes on
tensor slices are dropped by one-shot bufferization.  The call survives every intervening pass and
is erased by this rewrite; if the rewrite ever fails to run, its private unresolved declaration makes
the build fail instead of silently producing a serial binary.  Still-result-carrying loops are left
alone, and a marked body containing an existing parallel region is refused so nested OpenMP forks
cannot be introduced silently.
"""
from __future__ import annotations

import json
from pathlib import Path


PANEL_MARKER_SYMBOL = "__merlin_parallel_panel_marker"


def marker_call():
    """The fail-closed marker inserted directly in one panel-loop body."""
    from xdsl.dialects.func import CallOp

    return CallOp(PANEL_MARKER_SYMBOL, [], [])


def ensure_marker_declaration(module) -> None:
    """Add the private marker declaration once to an xDSL module."""
    from xdsl.dialects.func import FuncOp
    from xdsl.ir import Region

    for op in module.walk():
        if isinstance(op, FuncOp) and op.sym_name.data == PANEL_MARKER_SYMBOL:
            return
    module.body.block.add_op(FuncOp(PANEL_MARKER_SYMBOL, ((), ()), Region(),
                                    visibility="private"))


# Executed in the model2MLIR venv, which owns the torch-mlir Python bindings.
RUNNER_PRELUDE = r'''
_MERLIN_PANEL_MARKER_SYMBOL = "__merlin_parallel_panel_marker"


def _pp_has_nested_parallel(op):
    """Whether ``op`` contains a parallel region below its own body."""
    def walk(inner):
        for region in inner.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name in ("scf.parallel", "scf.forall", "omp.parallel"):
                        return True
                    if walk(child):
                        return True
        return False
    return walk(op)


def _pp_marker_calls(op):
    """Marker calls below ``op``, preserved through bufferization by their side effect."""
    found = []
    def walk(inner):
        for region in inner.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name == "func.call":
                        try:
                            if str(child.attributes["callee"]) == "@" + _MERLIN_PANEL_MARKER_SYMBOL:
                                found.append(child)
                        except Exception:
                            pass
                    walk(child)
    walk(op)
    return found


def _parallelize_panel_loops(ctx, module):
    """Rewrite verified bufferized panel carriers and return the number changed."""
    from torch_mlir import ir as _ppir
    from torch_mlir.dialects import scf as _ppscf

    found = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name == "scf.for" and _pp_marker_calls(child):
                        found.append(child)
                    walk(child)
    walk(module.operation)

    rewritten = refused = nested = 0
    for op in found:
        body = op.regions[0].blocks[0]
        terms = [h.operation for h in body.operations]
        markers = _pp_marker_calls(op)
        # This is the post-bufferization proof boundary: a tensor carrier still has a result and a
        # block iter-arg.  Rewriting it as a reduction-free parallel loop would discard semantics.
        if (len(op.results) != 0 or len(op.operands) != 3 or len(body.arguments) != 1 or
                len(markers) != 1 or markers[0] not in terms or
                not terms or terms[-1].name != "scf.yield" or len(terms[-1].operands) != 0):
            refused += 1
            continue
        if _pp_has_nested_parallel(op):
            refused += 1
            nested += 1
            continue

        lb, ub, step = list(op.operands)
        old_iv = body.arguments[0]
        with ctx, _ppir.Location.unknown():
            par = _ppscf.ParallelOp([], [lb], [ub], [step], [], ip=_ppir.InsertionPoint(op))
            par.regions[0].blocks.append(lb.type)
            pbody = par.regions[0].blocks[0]
            with _ppir.InsertionPoint(pbody):
                reduce = _ppscf.ReduceOp([], 0)
        old_iv.replace_all_uses_with(pbody.arguments[0])
        for inner in terms[:-1]:
            if inner == markers[0]:
                inner.erase()
                continue
            inner.move_before(reduce.operation)
        terms[-1].erase()
        op.erase()
        rewritten += 1

    print("OK panel_parallel regions", len(found), "rewritten", rewritten,
          "refused", refused, "nested_parallel", nested)
    return rewritten
'''


# argv[10] is selected by ``pipeline.lower_to_llvm_ir`` only when a multicore build contains one of
# the panel-pack features.  It appends to parallel_grain's list so every runner variant executes both.
LATE_STAGE_SRC = r'''
_PANEL_PARALLEL = len(sys.argv) > 10 and sys.argv[10] == "1"
if _PANEL_PARALLEL:
    _LATE_STAGES.insert(0, ("panel_parallel", _parallelize_panel_loops))
'''


REPORT_PREFIX = "OK panel_parallel regions "
REPORT_FILE = "panel_parallel_lowering.json"


def require_complete_report(stdout: str, work: "str | Path") -> dict[str, int]:
    """Persist and validate the runner's proof that every surviving marker was consumed."""
    lines = [line for line in stdout.splitlines() if line.startswith(REPORT_PREFIX)]
    if len(lines) != 1:
        raise ValueError(f"expected one panel_parallel report, got {len(lines)}")
    tokens = lines[0].split()
    try:
        report = {tokens[i]: int(tokens[i + 1]) for i in range(2, len(tokens), 2)}
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed panel_parallel report: {lines[0]!r}") from exc
    if (report.get("regions", 0) < 1 or report.get("rewritten") != report.get("regions") or
            report.get("refused") != 0 or report.get("nested_parallel") != 0):
        raise ValueError(f"panel_parallel did not consume every marked carrier: {report}")
    Path(work, REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
