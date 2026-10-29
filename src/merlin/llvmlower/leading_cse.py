"""Check every tensor-level ``cse`` the build's own runner runs, from the IR it actually ran on.

Every lowering pipeline opens with ``canonicalize,cse`` on tensors, and some pre-pipeline rewrites
run the same pair. When ``cse`` is validated, :func:`bind_runner` prepends a prelude to the
build's runner (whatever its variant or features) that wraps ``torch_mlir``'s ``PassManager``:
an invocation whose pass list opens with ``canonicalize,cse`` on a module with no ``memref`` runs
as ``canonicalize``, ``cse`` and the rest, in the same process and order, and the exact module
before and after that ``cse`` is written down. Nothing else changes, so the build is the one an
unvalidated build makes. :func:`submit` then checks each recorded pair; a build in which no
tensor-level ``cse`` was seen fails closed rather than passing unchecked.

The original explicitly selected scalar memory route may lend its live compiler
scope. That route records the real leading CSE pair and accepts exact raw
identity with zero jobs and zero proof credit; changed memory CSE refuses.
Outside that scope, bufferized/LLVM modules retain the legacy skip behavior.
Canonicalization and downstream native lowering receive no proof from this
observation wrapper.
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
    """``PassManager`` that records the actual module around leading ``canonicalize,cse``."""

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
        memory = globals().get('_LC_MEMORY')
        if memory is not None and text != 'builtin.module(' + memory['effective_pipeline'] + ')':
            raise RuntimeError('original memory CSE pipeline differs from its issued runner')
        if memory is not None:
            import hashlib
            import sys
            from torch_mlir import ir as _lc_ir

            actual_input = Path(memory['input_path'])
            if (actual_input.is_symlink() or actual_input.read_bytes() != _LC_UPSTREAM.encode('utf-8')
                    or len(sys.argv) < 4 or Path(sys.argv[1]).resolve() != actual_input
                    or Path(__file__).resolve() != Path(memory['runner_path'])
                    or Path(sys.argv[0]).resolve() != Path(memory['runner_path'])
                    or sys.argv[3] != memory['effective_pipeline']):
                raise RuntimeError('actual memory runner input/argv differs from its frozen upstream epoch')
            ctx = self._context if self._context is not None else operation.context
            try:
                expected_module = _lc_ir.Module.parse(_LC_UPSTREAM, ctx)
            except Exception as error:
                raise RuntimeError('original frozen upstream module could not be reparsed') from error
            expected_upstream = expected_module.operation.get_asm(print_generic_op_form=True)
            precanonical = operation.get_asm(print_generic_op_form=True)
            if precanonical != expected_upstream:
                raise RuntimeError('actual pre-canonicalize whole module differs from frozen upstream input')
        self._native(head + 'canonicalize)').run(operation)
        before = operation.get_asm(print_generic_op_form=True)
        self._native(head + 'cse)').run(operation)
        after = operation.get_asm(print_generic_op_form=True)
        if memory is None:
            entry = {'pipeline': text[:200]}
        else:
            import hashlib
            entry = dict(memory, pipeline=text,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                before_sha256=hashlib.sha256(before.encode('utf-8')).hexdigest(),
                after_sha256=hashlib.sha256(after.encode('utf-8')).hexdigest(),
                precanonical_sha256=hashlib.sha256(precanonical.encode('utf-8')).hexdigest(),
                upstream_generic_sha256=hashlib.sha256(expected_upstream.encode('utf-8')).hexdigest())
        number = _lc_record(entry)
        Path(_LC_DIRECTORY, '%04d.before.mlir' % number).write_text(before)
        Path(_LC_DIRECTORY, '%04d.after.mlir' % number).write_text(after)
        if memory is not None:
            Path(_LC_DIRECTORY, '%04d.precanonical.mlir' % number).write_text(precanonical)
            Path(_LC_DIRECTORY, '%04d.upstream.mlir' % number).write_text(expected_upstream)
        if passes[2:]:
            self._native(head + ','.join(passes[2:]) + ')').run(operation)
        return None


_lc_pm_module.PassManager = _LCPassManager
PassManager = _LCPassManager
'''


def bind_runner(source: str, work: Path, *, pipeline=None, upstream=None) -> str:
    """``source`` with the recording prelude ahead of its own code when ``cse`` is validated."""
    if not pass_validation.selected(PASS):
        return source
    from merlin.verify.compiler_memory_origin import current

    memory = current()
    directory = Path(work) / DIRECTORY
    if memory is not None:
        memory._work(work)
        memory.bind_input(upstream, work)
        token = memory.cse_record_token(pipeline=pipeline)
        upstream_text = memory.upstream_text(work)
    else:
        token = None
        upstream_text = None
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    issued = (
        f"_LC_DIRECTORY = {str(directory.resolve())!r}\n_LC_MEMORY = {token!r}\n"
        f"_LC_UPSTREAM = {upstream_text!r}\n" + _PRELUDE + source
    )
    return memory.bind_runner(issued, work, pipeline=pipeline) if memory is not None else issued


def _unsupported(directory, reason):
    from merlin.verify.speculative_gate import GateFailure

    raise GateFailure(
        {
            "check": "tensor_refinement",
            "label": PASS,
            "verdict": "unsupported",
            "reason": reason,
            "directory": str(directory),
        }
    )


def submit(work: Path) -> int:
    """Check real recorded pairs; unchanged original-memory CSE is zero jobs/credit."""
    if not pass_validation.selected(PASS):
        return 0
    from merlin.verify.compiler_memory_origin import current

    directory = Path(work) / DIRECTORY
    try:
        memory = current()
        index = directory / "index.json"
        if memory is not None:
            memory.validate_runner(work)
            if directory.is_symlink() or index.is_symlink():
                raise ValueError("original CSE capture may not use symlinks")
        entries = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else []
        if type(entries) is not list or any(type(entry) is not dict for entry in entries):
            raise ValueError("actual CSE capture list required")
        observed = [n for n, entry in enumerate(entries, 1) if "skipped" not in entry]
        if not observed:
            _unsupported(
                directory,
                "cse is validated but the build ran no tensor-level canonicalize,cse to check"
                if memory is None
                else "original memory compile ran no authentic canonicalize,cse capture",
            )
        if memory is not None and len(observed) != len(entries):
            raise ValueError("original memory CSE cannot use a skipped capture")
        for number in observed:
            before_path, after_path = (directory / f"{number:04d}.{side}.mlir" for side in ("before", "after"))
            if memory is not None and (before_path.is_symlink() or after_path.is_symlink()):
                raise ValueError("actual CSE buffers may not be symlinks")
            before, after = before_path.read_text(encoding="utf-8"), after_path.read_text(encoding="utf-8")
            entry = entries[number - 1]
            if memory is not None:
                upstream_paths = [directory / f"{number:04d}.{side}.mlir" for side in ("precanonical", "upstream")]
                if any(path.is_symlink() for path in upstream_paths):
                    raise ValueError("actual pre-canonicalize buffers may not be symlinks")
                precanonical, expected_upstream = [path.read_text(encoding="utf-8") for path in upstream_paths]
                memory.validate_cse_record(
                    entry,
                    work,
                    before=before,
                    after=after,
                    precanonical=precanonical,
                    expected_upstream=expected_upstream,
                )
                from merlin.verify.compiler_coverage import boundary
                from merlin.verify.compiler_profile import SourceOutsideProfile

                if before != after:
                    excluded = SourceOutsideProfile(
                        "typed-tensor-cse-index64-v1",
                        "explicit-memory",
                        "changed explicit-memory CSE is outside the tensor CSE model",
                    )
                    boundary(PASS, before, after, profile_id=excluded.profile_id, exclusion=excluded)
                else:
                    boundary(PASS, before, after, profile_id="raw-identity")
            else:
                if "schema" in entry:
                    raise ValueError("memory CSE capture has no original live memory scope")
                pass_validation.submit(PASS, before, after)
        if memory is not None:
            memory.finish_cse(work)
        # An observed raw identity submits nothing and earns no kernel/gate credit.
        return 0 if memory is not None else len(observed)
    except (ValueError, OSError, TypeError, KeyError) as error:
        _unsupported(directory, str(error))


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
