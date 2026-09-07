"""Coarsen adjacent OpenMP regions into one persistent team region.

``convert-scf-to-openmp`` emits one ``omp.parallel`` around every worksharing loop.  A straight-line
sequence therefore forks and joins the same team repeatedly even though the implicit barrier at the
end of each ``omp.wsloop`` already provides the ordering the sequence needs.  The default-off
``coarsen_openmp_regions`` feature merges only consecutive, result-free, operand-free
``omp.parallel`` operations in the same block.  It moves their worksharing bodies, in order, into
the first region; it does not move across an intervening operation or remove a worksharing barrier.

This is the compiler-level counterpart of a persistent worker pool: it reduces dispatch count while
leaving libomp's proven hot team, static partitioning, and loop bounds unchanged.  The rewrite fails
closed on non-canonical regions (operands, results, block arguments, or a missing terminator).

Measured on K1 LSTMNetVIT W8A8, on top of ``parallelize_residual_loops_0``: 780 static team regions
become 361, complete output stays bit-identical, and an interleaved eight-core A/B improves the
session medians from 87.74/88.93 ms to 86.86/87.12 ms.  This is a small but repeatable scheduling
gain; it does not close the much larger single-core code-generation/layout gap.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


FEATURE = "coarsen_openmp_regions"
REPORT_TOKEN = "OK parallel_coarsen original"


def require_report(stdout: str) -> None:
    """Fail closed when a named coarsening build did not execute or match the rewrite."""
    line = next((line for line in stdout.splitlines() if line.startswith(REPORT_TOKEN)), None)
    if line is None:
        raise ValueError("coarsen_openmp_regions was named but its runner stage did not execute")
    fields = line.split()
    try:
        original = int(fields[3])
        merged = int(fields[5])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed coarsening report: {line!r}") from exc
    if original == 0 or merged == 0:
        raise ValueError(
            f"coarsen_openmp_regions was named but matched no adjacent regions: {line}")


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="HEURISTIC",
            description=(
                "Merge consecutive canonical omp.parallel regions in one block into one team "
                "region containing the original ordered worksharing loops. Preserves every "
                "omp.wsloop barrier while removing redundant fork/join boundaries. Measured on "
                "K1 LSTMNetVIT W8A8: 780 to 361 static regions and about 1.5% lower eight-core "
                "latency in an interleaved A/B; default off and model-selected."),
        ))
    return FEATURE


RUNNER_PRELUDE = r'''
def _pc_canonical(op):
    """Whether ``op`` is a canonical result/operand-free OpenMP team region."""
    if op.name != "omp.parallel" or len(op.results) or len(op.operands):
        return False
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    block = op.regions[0].blocks[0]
    body = [handle.operation for handle in block.operations]
    return not len(block.arguments) and bool(body) and body[-1].name == "omp.terminator"


def _parallel_coarsen(ctx, module):
    """Merge maximal straight-line runs of canonical ``omp.parallel`` operations."""
    # Conversion materializes duplicate bound constants immediately before individual regions.
    # The normal pipeline removes them during later cleanup, but by then OpenMP has been lowered to
    # LLVM and its team boundaries are no longer editable.  Run that same cleanup here so only real
    # intervening computation separates runs; the pipeline's later canonicalize/CSE is idempotent.
    from torch_mlir.passmanager import PassManager as _pcPassManager
    _pcPassManager.parse("builtin.module(canonicalize,cse)", ctx).run(module.operation)
    merged = groups = original = 0

    def walk(op):
        nonlocal merged, groups, original
        for region in op.regions:
            for block in region.blocks:
                children = [handle.operation for handle in block.operations]
                # Process nested blocks before editing this block, so no recursion observes an
                # operation invalidated by erasing a later member of a run.
                for child in children:
                    walk(child)
                i = 0
                while i < len(children):
                    if not _pc_canonical(children[i]):
                        i += 1
                        continue
                    run = [children[i]]
                    j = i + 1
                    while j < len(children) and _pc_canonical(children[j]):
                        run.append(children[j])
                        j += 1
                    original += len(run)
                    if len(run) > 1:
                        groups += 1
                        target = run[0].regions[0].blocks[0]
                        terminator = [h.operation for h in target.operations][-1]
                        for later in run[1:]:
                            body = [h.operation for h in later.regions[0].blocks[0].operations]
                            for inner in body[:-1]:
                                inner.move_before(terminator)
                            later.erase()
                            merged += 1
                    i = j

    walk(module.operation)
    module.operation.verify()
    print("OK parallel_coarsen original", original, "merged", merged,
          "groups", groups, "remaining", original - merged)
    return merged
'''


STAGE_SRC = r'''
_PARALLEL_COARSEN = len(sys.argv) > 13 and sys.argv[13] == "1"
if _PARALLEL_COARSEN:
    _POST_OPENMP_STAGES = [*_POST_OPENMP_STAGES,
                           ("parallel_coarsen", _parallel_coarsen)]
'''


def apply_for_test(mlir_text: str) -> tuple[str, dict]:
    """Run the exact post-OpenMP rewrite used by the lowering pipeline."""
    from .toolchain import m2m_python

    work = Path(tempfile.mkdtemp(prefix="merlin_parallel_coarsen_"))
    src, script = work / "in.mlir", work / "run.py"
    src.write_text(mlir_text, encoding="utf-8")
    script.write_text(
        "import json, sys\n"
        "from torch_mlir import ir\n"
        + RUNNER_PRELUDE +
        "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f: module = ir.Module.parse(f.read(), ctx)\n"
        "merged = _parallel_coarsen(ctx, module)\n"
        "print('MERLIN_COARSEN ' + json.dumps({'merged': merged}))\n"
        "print('MERLIN_MODULE_BEGIN')\n"
        "print(module.operation)\n", encoding="utf-8")
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(src)], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"parallel-coarsen test rewrite failed:\n{proc.stdout}\n{proc.stderr}")
    record = next(line for line in proc.stdout.splitlines()
                  if line.startswith("MERLIN_COARSEN "))
    module = proc.stdout.split("MERLIN_MODULE_BEGIN\n", 1)[1]
    return module, json.loads(record[len("MERLIN_COARSEN "):])
