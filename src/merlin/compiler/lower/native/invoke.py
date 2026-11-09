"""Native diagnostic pass invocation and completed trace harvesting."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from merlin.common import compile_trace
from merlin.compiler.passes.contract import PipelineError
from merlin.llvmlower import fusion_guard
from merlin.llvmlower.dealloc_lifetime import DEALLOC_CHECK_PRELUDE
from merlin.llvmlower.optional_passes import child_environment
from merlin.llvmlower.toolchain import m2m_python


def apply_passes(mlir_text: str, pipeline: str, timeout: int = 600) -> str:
    """Run ``pipeline`` over ``mlir_text`` in the m2m venv and return the printed module.

    A diagnostic/test seam, not a build path: it drives the SAME pass registry the lowering drives
    (the torch-mlir wheel's), so a test that pins what a stage does to the IR cannot pass against a
    different LLVM than the one that ships.
    """
    work = Path(tempfile.mkdtemp(prefix="merlin_apply_passes_"))
    src, dst = work / "in.mlir", work / "out.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = work / "apply.py"
    script.write_text(
        fusion_guard.inject("import sys\nfrom torch_mlir import ir\nfrom torch_mlir.passmanager import PassManager\n")
        + "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "PassManager.parse('builtin.module(' + sys.argv[3] + ')', ctx).run(module.operation)\n"
        "with open(sys.argv[2], 'w') as f:\n"
        "    f.write(str(module.operation))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(src), str(dst), pipeline],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_environment(),
    )
    if proc.returncode != 0 or not dst.is_file():
        raise PipelineError(f"apply_passes failed:\n{proc.stdout}\n{proc.stderr}")
    return dst.read_text(encoding="utf-8")


def dealloc_placement_violations(mlir_text: str, timeout: int = 600, *, borrowed_calls=()) -> list[str]:
    """Violations of the placement rule in a post-deallocation memref module.

    Empty list == every allocation whose free was sunk into the alloc's own block is used only
    before that free. Runs the SAME source the build runs, out of process in the m2m venv (the MLIR
    Python bindings live there), so a test and a build cannot disagree about what is checked.
    Closed positive serial replacement loops additionally require a complete
    intrinsic generation/release proof; unknown calls/effects remain barriers.
    """
    import json

    from merlin.llvmlower.borrowed_buffer_calls import bind_native

    prelude = bind_native(DEALLOC_CHECK_PRELUDE, borrowed_calls)

    work = Path(tempfile.mkdtemp(prefix="merlin_dealloc_check_"))
    src = work / "module.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = work / "check.py"
    script.write_text(
        "import json, sys\n"
        "from torch_mlir import ir\n" + prelude + "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "module.operation.verify()\n"
        "bad, n_alloc, n_sunk = _dealloc_placement_violations(module)\n"
        "print('MERLIN_CHECK ' + json.dumps({'violations': bad, 'allocations': n_alloc, "
        "'sunk': n_sunk}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(src)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_environment(),
    )
    if proc.returncode != 0:
        raise PipelineError(f"dealloc placement check failed:\n{proc.stdout}\n{proc.stderr}")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("MERLIN_CHECK "))
    return json.loads(line[len("MERLIN_CHECK ") :])["violations"]


def _harvest_native(traced: "tuple[Path, bool] | None", produced: Path, failure: BaseException | None = None) -> None:
    """Index a native pass run's dumps into the open compile trace (``(directory, prune)``; None: no
    trace selects a native pass). A stop selected on a pass that ran before ``failure`` still stops.

    The pass manager ran its whole pipeline in the child before the trace could stop it, so on a stop
    its output (``produced``, the IR after the LAST pass) is removed: nothing past the stop survives."""
    if traced is None:
        return
    try:
        compile_trace.harvest_native(traced[0], pipeline="llvm", prune=traced[1])
    except compile_trace.StopAfterStage as stop:
        produced.unlink(missing_ok=True)
        stop.note = "the pass manager ran on in its child process; the output of the passes after it was discarded"
        if failure is None:
            raise
        stop.note += f"; a later pass failed: {str(failure).strip().splitlines()[0][:200]}"
        raise stop from failure
