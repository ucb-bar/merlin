"""Check every tensor-level ``cse`` the build's own runner runs, from the IR it actually ran on.

Every lowering pipeline opens with ``canonicalize,cse`` on tensors, and some pre-pipeline rewrites
run the same pair. When ``cse`` is validated, :func:`bind_runner` prepends a prelude to the
build's runner (whatever its variant or features) that wraps ``torch_mlir``'s ``PassManager``:
an invocation whose pass list opens with ``canonicalize,cse`` on a module with no ``memref`` runs
as ``canonicalize``, ``cse`` and the rest, in the same process and order, and the exact module
before and after that ``cse`` is written down. Nothing else changes, so the build is the one an
unvalidated build makes. :func:`submit` then checks each recorded pair; a build in which no
tensor-level ``cse`` was seen fails closed rather than passing unchecked.

Not validated: ``canonicalize`` (its folds are outside the Lean checker's domain) and ``cse`` on
bufferized or LLVM-dialect modules (outside the checker's typed tensor domain), which are listed
in the record as skipped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from merlin.verify import pass_validation

from . import fusion_guard
from .toolchain import m2m_python

PASS = "cse"
DIRECTORY = "leading-cse"

# Must not contain the text "PassManager.parse(": the IR-inspection binder rewrites that spelling.
_PRELUDE = r'''
import torch_mlir.passmanager as _lc_pm_module

_LC_NATIVE = _lc_pm_module.PassManager


def _lc_split(text):
    passes, depth, start = [], 0, 0
    for index, char in enumerate(text + ','):
        if char in '({':
            depth += 1
        elif char in ')}':
            depth -= 1
        elif char == ',' and depth == 0:
            if text[start:index].strip():
                passes.append(text[start:index].strip())
            start = index + 1
    return passes


def _lc_record(entry):
    import json
    from pathlib import Path

    index = Path(_LC_DIRECTORY) / 'index.json'
    entries = json.loads(index.read_text()) if index.is_file() else []
    entries.append(entry)
    index.write_text(json.dumps(entries, indent=2))
    return len(entries)


class _LCPassManager:
    """``PassManager`` that records the module around a leading tensor-level ``canonicalize,cse``."""

    def __init__(self, pipeline, context):
        self._pipeline, self._context, self._printing = pipeline, context, None

    @classmethod
    def parse(cls, pipeline, context=None):
        return cls(pipeline, context)

    def enable_ir_printing(self, **options):
        self._printing = options

    def _native(self, pipeline):
        manager = _LC_NATIVE.parse(pipeline, self._context)
        if self._printing is not None:
            manager.enable_ir_printing(**self._printing)
        return manager

    def run(self, operation):
        from pathlib import Path

        head, text = 'builtin.module(', self._pipeline
        passes = _lc_split(text[len(head):-1]) if text.startswith(head) and text.endswith(')') else []
        if passes[:2] != ['canonicalize', 'cse']:
            return self._native(text).run(operation)
        self._native(head + 'canonicalize)').run(operation)
        before = operation.get_asm(print_generic_op_form=True)
        if 'memref<' in before:
            _lc_record({'skipped': 'bufferized module (memref)', 'pipeline': text[:200]})
            self._native(head + ','.join(passes[1:]) + ')').run(operation)
            return None
        self._native(head + 'cse)').run(operation)
        after = operation.get_asm(print_generic_op_form=True)
        number = _lc_record({'pipeline': text[:200]})
        Path(_LC_DIRECTORY, '%04d.before.mlir' % number).write_text(before)
        Path(_LC_DIRECTORY, '%04d.after.mlir' % number).write_text(after)
        if passes[2:]:
            self._native(head + ','.join(passes[2:]) + ')').run(operation)
        return None


_lc_pm_module.PassManager = _LCPassManager
PassManager = _LCPassManager
'''


def bind_runner(source: str, work: Path) -> str:
    """``source`` with the recording prelude ahead of its own code when ``cse`` is validated."""
    if not pass_validation.selected(PASS):
        return source
    directory = Path(work) / DIRECTORY
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    return f"_LC_DIRECTORY = {str(directory)!r}\n" + _PRELUDE + source


def submit(work: Path) -> int:
    """Check every recorded ``cse`` pair; returns how many were submitted."""
    if not pass_validation.selected(PASS):
        return 0
    from merlin.verify.speculative_gate import GateFailure

    directory = Path(work) / DIRECTORY
    index = directory / "index.json"
    entries = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else []
    observed = [n for n, entry in enumerate(entries, 1) if "skipped" not in entry]
    if not observed:
        raise GateFailure(
            {
                "check": "tensor_refinement",
                "label": PASS,
                "verdict": "unsupported",
                "reason": "cse is validated but the build ran no tensor-level canonicalize,cse to check",
                "directory": str(directory),
            }
        )
    for number in observed:
        before = (directory / f"{number:04d}.before.mlir").read_text(encoding="utf-8")
        after = (directory / f"{number:04d}.after.mlir").read_text(encoding="utf-8")
        pass_validation.submit(PASS, before, after)
    return len(observed)


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


def apply(text: str, passes: str, directory: Path, name: str) -> str:
    """``text`` after ``passes``, run out of process with the build's pass registry and fusion guard
    (for tests and diagnostics)."""
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
        raise RuntimeError(f"{passes} failed:\n{proc.stdout}\n{proc.stderr}")
    return result.read_text(encoding="utf-8")
