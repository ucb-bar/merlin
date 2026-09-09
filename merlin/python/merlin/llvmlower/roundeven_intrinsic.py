"""Lower ``math.roundeven`` to LLVM's exact round-even intrinsic after loop formation.

Dynamic activation quantization contains one round-to-nearest-even operation per element.  Leaving
it for ``convert-math-to-libm`` creates a scalar ``roundevenf`` call that blocks LLVM loop
vectorization.  Expanding it into ordinary arithmetic removes that barrier, but substantially
inflates large whole-model LLVM IR and compile time.  LLVM already has the operation we mean:
``llvm.intr.roundeven``.  This default-off lowering changes only the representation of the same
specified rounding operation and runs after linalg has become loops, where an LLVM-dialect scalar
intrinsic is legal and LLVM's loop vectorizer can form its vector counterpart.

The rewrite is deliberately structure-only: every scalar/vector ``math.roundeven`` is replaced,
without inspecting a model name, tensor shape, dtype recipe, or target.  Other math operations are
left for the existing libm path.  It is an alternative to ``fuse_quantize_round_convert``; enabling
both would erase the round before this stage and is rejected by the lowering entry point.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


FEATURE = "lower_roundeven_to_intrinsic"
MARKER = "__merlin_lower_roundeven_to_intrinsic__"
REPORT_TOKEN = "OK roundeven_intrinsic"


def _edit_pipeline(passes: list[str]) -> list[str]:
    """Insert the runner-owned marker after linalg-to-loop conversion."""
    matches = [i for i, p in enumerate(passes)
               if "convert-linalg-to-loops" in p or "convert-linalg-to-parallel-loops" in p]
    if not matches:
        raise ValueError(f"{FEATURE} requires a linalg-to-loops lowering stage")
    i = max(matches) + 1
    return [*passes[:i], MARKER, *passes[i:]]


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="PASS",
            description=(
                "Replace math.roundeven with the semantically identical llvm.intr.roundeven after "
                "linalg-to-loop conversion. This prevents convert-math-to-libm from emitting a "
                "scalar roundevenf call while avoiding the large ordinary-arithmetic expansion of "
                "fuse_quantize_round_convert. Structure-only, target/model/shape independent, and "
                "default-off; mutually exclusive with fuse_quantize_round_convert."),
            edit_pipeline=_edit_pipeline,
        ))
    return FEATURE


RUNNER_PRELUDE = r'''
def _lower_roundeven_intrinsics(ctx, module):
    """Replace every math.roundeven with llvm.intr.roundeven; return the exact count."""
    from torch_mlir import ir as _ri_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name == "math.roundeven":
                        todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    with ctx, _ri_ir.Location.unknown():
        for old in todo:
            with _ri_ir.InsertionPoint(old):
                new = _ri_ir.Operation.create(
                    "llvm.intr.roundeven", results=[old.results[0].type],
                    operands=[old.operands[0]])
            old.results[0].replace_all_uses_with(new.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)


_RI_MARKER = "__merlin_lower_roundeven_to_intrinsic__"
_RI_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _RI_MARKER not in passes:
        return _RI_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp,
                                   pre_generalize)
    i = passes.index(_RI_MARKER)
    # The marker is after linalg-to-loops: mid/pre-generalize belong to the head, while parallel
    # grain/team/coarsening stages still belong to the tail.  Keeping that routing explicit avoids
    # silently dropping another requested rewrite when this lever is composed with it.
    _RI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (),
                        pre_generalize)
    print('OK roundeven_intrinsic', _lower_roundeven_intrinsics(ctx, module))
    _RI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), 0, (), late,
                        post_openmp, ())
'''


def apply_for_test(mlir_text: str) -> tuple[str, int]:
    """Apply the shipped rewrite source to a module in the toolchain-owning Python."""
    from .toolchain import m2m_python

    work = Path(tempfile.mkdtemp(prefix="merlin_roundeven_intrinsic_"))
    src, script = work / "in.mlir", work / "run.py"
    src.write_text(mlir_text, encoding="utf-8")
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n"
        + RUNNER_PRELUDE.split("_RI_MARKER =", 1)[0]
        + "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f: module = ir.Module.parse(f.read(), ctx)\n"
        "n = _lower_roundeven_intrinsics(ctx, module)\n"
        "print('COUNT', n)\nprint('MODULE_BEGIN')\nprint(module.operation)\n",
        encoding="utf-8")
    proc = subprocess.run([str(m2m_python()), str(script), str(src)], capture_output=True,
                          text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"roundeven intrinsic rewrite failed:\n{proc.stdout}\n{proc.stderr}")
    count = int(next(line.split()[1] for line in proc.stdout.splitlines()
                     if line.startswith("COUNT ")))
    return proc.stdout.split("MODULE_BEGIN\n", 1)[1], count
