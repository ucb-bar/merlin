"""Out-of-process MLIR diagnostics shared by pipeline public wrappers.

Diagnostic scripts and pass selection are kept separate from production runner
construction. The caller supplies its selected interpreter and public error type.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from . import fusion_guard


def apply_passes(mlir_text: str, pipeline: str, timeout: int, resolve_python, error_type) -> str:
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
        [str(resolve_python()), str(script), str(src), str(dst), pipeline],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0 or not dst.is_file():
        raise error_type(f"apply_passes failed:\n{proc.stdout}\n{proc.stderr}")
    return dst.read_text(encoding="utf-8")


def dealloc_placement_violations(
    mlir_text: str, timeout: int, resolve_python, check_source: str, error_type
) -> list[str]:
    """Violations of the placement rule in a post-deallocation memref module.

    Empty list == every allocation whose free was sunk into the alloc's own block is used only
    before that free. Runs the SAME source the build runs, out of process in the m2m venv (the MLIR
    Python bindings live there), so a test and a build cannot disagree about what is checked.
    """
    import json

    work = Path(tempfile.mkdtemp(prefix="merlin_dealloc_check_"))
    src = work / "module.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = work / "check.py"
    script.write_text(
        "import json, sys\n"
        "from torch_mlir import ir\n" + check_source + "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "bad, n_alloc, n_sunk = _dealloc_placement_violations(module)\n"
        "print('MERLIN_CHECK ' + json.dumps({'violations': bad, 'allocations': n_alloc, "
        "'sunk': n_sunk}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(resolve_python()), str(script), str(src)], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise error_type(f"dealloc placement check failed:\n{proc.stdout}\n{proc.stderr}")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("MERLIN_CHECK "))
    return json.loads(line[len("MERLIN_CHECK ") :])["violations"]
