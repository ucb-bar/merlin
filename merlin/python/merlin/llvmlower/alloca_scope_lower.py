"""Lower structured ``memref.alloca_scope`` before SCF becomes a multi-block CFG.

The ordinary MemRef-to-LLVM conversion performs this lowering itself, but the RVV/OpenMP
pipeline must lower SCF first. A scope containing a fused contraction loop consequently becomes
multi-block before MemRef conversion and violates ``alloca_scope``'s verifier. This rewrite emits
the same stack-save/stack-restore lifetime boundary while the scope still has one structured block.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .toolchain import m2m_python


REPORT_FILE = "alloca_scope_pre_cfg_report.txt"


RUNNER_PRELUDE = r'''
def _lower_structured_alloca_scopes(ctx, module):
    """Replace zero-result one-block alloca scopes by explicit LLVM stack lifetime intrinsics."""
    scopes = []

    def visit(op):
        # Post-order makes nested scopes independent: lower the inner lifetime first.
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    visit(child)
        if op.operation.name == 'memref.alloca_scope':
            scopes.append(op.operation)

    for top in list(module.body.operations):
        visit(top)

    lowered = 0
    ptr = ir.Type.parse('!llvm.ptr', ctx)
    for scope in scopes:
        if len(list(scope.results)) != 0 or len(list(scope.regions)) != 1:
            raise RuntimeError('alloca-scope pre-CFG lowering only accepts zero-result scopes')
        blocks = list(scope.regions[0].blocks)
        if len(blocks) != 1:
            raise RuntimeError('alloca-scope reached pre-CFG lowering with ' + str(len(blocks)) +
                               ' blocks; expected exactly one structured block')
        body = blocks[0]
        operations = list(body.operations)
        if not operations or operations[-1].operation.name != 'memref.alloca_scope.return':
            raise RuntimeError('alloca-scope has no recognizable return terminator')
        terminator = operations[-1].operation
        if len(list(terminator.operands)) != 0:
            raise RuntimeError('zero-result alloca-scope return unexpectedly carries operands')

        with ir.InsertionPoint(scope):
            saved = ir.Operation.create(
                'llvm.intr.stacksave', results=[ptr], loc=scope.location).results[0]
        for child in operations[:-1]:
            child.operation.move_before(scope)
        with ir.InsertionPoint(scope):
            ir.Operation.create(
                'llvm.intr.stackrestore', operands=[saved], loc=scope.location)
        terminator.erase()
        scope.erase()
        lowered += 1
    return lowered


_LOWER_ALLOCA_SCOPES = len(sys.argv) > 16 and sys.argv[16] == '1'
if _LOWER_ALLOCA_SCOPES:
    _POST_OPENMP_STAGES = [*_POST_OPENMP_STAGES,
                           ('alloca_scope_pre_cfg', _lower_structured_alloca_scopes)]
'''


def _drive(source: str, *, passes: str | None = None) -> tuple[str, int]:
    """Test seam using the exact MLIR bindings and rewrite source used by the build runner."""
    with tempfile.TemporaryDirectory(prefix="merlin_alloca_scope_") as td:
        root = Path(td)
        src, dst, script = root / "in.mlir", root / "out.mlir", root / "run.py"
        src.write_text(source, encoding="utf-8")
        script.write_text(
            "import sys\n"
            "from torch_mlir import ir\n"
            "from torch_mlir.passmanager import PassManager\n" + RUNNER_PRELUDE.split(
                "_LOWER_ALLOCA_SCOPES =", 1)[0] +
            "ctx = ir.Context()\n"
            "with open(sys.argv[1]) as f: module = ir.Module.parse(f.read(), ctx)\n"
            + ("n = 0\nPassManager.parse('builtin.module(' + sys.argv[3] + ')', ctx).run(module.operation)\n"
               if passes is not None
               else "n = _lower_structured_alloca_scopes(ctx, module)\n") +
            "with open(sys.argv[2], 'w') as f: f.write(str(module.operation))\n"
            "print(n)\n", encoding="utf-8")
        command = [str(m2m_python()), str(script), str(src), str(dst)]
        if passes is not None:
            command.append(passes)
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not dst.is_file():
            raise RuntimeError(f"alloca-scope test driver failed:\n{proc.stdout}\n{proc.stderr}")
        return dst.read_text(encoding="utf-8"), int(proc.stdout.strip() or 0)


def lower_text_for_test(source: str) -> tuple[str, int]:
    return _drive(source)


def apply_passes_for_test(source: str, passes: str) -> str:
    return _drive(source, passes=passes)[0]


def require_report(stdout: str, work: Path) -> int:
    """Require and persist the real runner receipt; a requested but inert rewrite is an error."""
    prefix = "OK alloca_scope_pre_cfg "
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError(f"expected one {prefix.strip()!r} receipt, found {len(lines)}")
    try:
        count = int(lines[0][len(prefix):].strip())
    except ValueError as exc:
        raise ValueError(f"malformed alloca-scope receipt: {lines[0]!r}") from exc
    if count <= 0:
        raise ValueError("alloca-scope lowering was requested but rewrote no scopes")
    (Path(work) / REPORT_FILE).write_text(f"lowered={count}\n", encoding="utf-8")
    return count
