"""Run the pipeline's leading ``canonicalize`` and ``cse`` where pass validation can see them.

Every lowering pipeline opens with ``canonicalize,cse`` on tensors. When ``cse`` is validated,
:func:`run` executes those two passes here, out of process with the same pass registry and fusion
guard as the build's runner, submits the exact before/after pair of ``cse``, and returns the
result for the pipeline. The pipeline then runs both passes again; :func:`run` proves that changes
nothing by running them once more here, and submits any change, so what the build lowers is
exactly what was checked. ``canonicalize`` itself is not validated (its folds are outside the
Lean checker's domain).

Only exact when the runner does nothing to the module before its pass manager, so a build whose
features or stages rewrite the input first is refused rather than silently left unvalidated.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from merlin.verify import pass_validation

from . import fusion_guard
from .toolchain import m2m_python

PASS = "cse"
#: The label of the check that the pipeline's own leading canonicalize,cse changes nothing more.
HEAD_RERUN = "cse:pipeline-head-rerun"

_SCRIPT = (
    "import sys\n"
    "from torch_mlir import ir\n"
    "from torch_mlir.passmanager import PassManager\n"
    "ctx = ir.Context()\n"
    "with open(sys.argv[1]) as f:\n"
    "    module = ir.Module.parse(f.read(), ctx)\n"
    "PassManager.parse('builtin.module(' + sys.argv[3] + ')', ctx).run(module.operation)\n"
    "with open(sys.argv[2], 'w') as f:\n"
    "    f.write(str(module.operation))\n"
)


def _apply(text: str, passes: str, directory: Path, name: str) -> str:
    source, result = directory / f"{name}.in.mlir", directory / f"{name}.mlir"
    source.write_text(text, encoding="utf-8")
    script = directory / "apply.py"
    if not script.is_file():
        script.write_text(fusion_guard.inject(_SCRIPT), encoding="utf-8")
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(source), str(result), passes],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0 or not result.is_file():
        raise RuntimeError(f"{passes} failed before the pipeline:\n{proc.stdout}\n{proc.stderr}")
    return result.read_text(encoding="utf-8")


def run(mlir_text: str, work: Path, *, features, stages: tuple) -> str:
    """``mlir_text`` after the pipeline's leading ``canonicalize,cse`` when ``cse`` is validated,
    else ``mlir_text`` unchanged. ``features`` are the build's requested features and ``stages`` its
    optional source stages; any of them may rewrite the module before the pass manager."""
    if not pass_validation.selected(PASS):
        return mlir_text
    from .impr_features import normalize
    from .optional_passes import selected_features

    effective = normalize(selected_features(features))  # exactly what lower_to_llvm_ir selects
    if effective or any(stage is not None for stage in stages):
        raise ValueError(
            "cse validation needs the pipeline to start from this module, but the selected features "
            f"{sorted(effective or ())} or source stages may rewrite it before the pass manager; "
            "validate the other passes only"
        )
    directory = Path(work) / "leading-cse"
    directory.mkdir(parents=True, exist_ok=True)
    canonical = _apply(mlir_text, "canonicalize", directory, "canonicalize")
    after = _apply(canonical, PASS, directory, PASS)
    pass_validation.submit(PASS, canonical, after)
    # The pipeline reruns its head on `after`; anything that changes is checked too, and reported as
    # the head not being a fixpoint rather than as cse itself.
    rerun = _apply(after, "canonicalize,cse", directory, "rerun")
    pass_validation.submit(PASS, after, rerun, label=HEAD_RERUN)
    return after
