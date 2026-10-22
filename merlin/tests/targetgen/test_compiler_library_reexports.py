"""A neutral scheduling facade grants exact exports, never a package subtree."""
from pathlib import Path

import pytest

from merlin.targetgen.compiler_library import CompilerLibraryError, freeze_compiler_library
from merlin.targetgen.package_runtime import _py_imports_merlin


def _facade(tmp_path, *, tail="", public_extra=()):
    root = tmp_path / "library"
    source = Path(__file__).resolve().parents[3] / "src"
    modules = ("merlin", "merlin.sched", "merlin.sched.ir", "merlin.sched.ir.expr",
               "merlin.sched.ir.kernel", "merlin.sched.ir.parse")
    rows = []
    for module in modules:
        namespace = module in modules[:3]
        relative = module.replace(".", "/") + ("/__init__.py" if namespace else ".py")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / relative).read_bytes() if module.startswith("merlin.sched.ir") else b"")
        rows.append((relative, module))
    init = root / "merlin/sched/ir/__init__.py"
    init.write_text(init.read_text() + tail)
    return root, freeze_compiler_library(root, review_id="neutral-scheduling-facade",
                                        public_modules=modules[2:] + public_extra, sources=tuple(rows))


def test_actual_sched_facade_is_closed_and_does_not_grant_children(tmp_path):
    root, contract = _facade(tmp_path)
    contract.verify(root)
    assert _py_imports_merlin("from merlin.sched.ir import Kernel, TensorArg, parse_kernel", compiler_library=contract) is None
    for name in ("merlin.sched", "merlin.sched.ir.unreviewed", "merlin.runtime.reference"):
        assert not contract.permits(name)
        assert _py_imports_merlin(f"import {name}", compiler_library=contract) is not None


@pytest.mark.parametrize("tail", [
    "\nfrom merlin.runtime import reference\n", "\ndef helper(): return 1\n",
    "\nfrom .kernel import *\n", "\n__all__.append('unknown')\n",
])
def test_facade_cannot_add_executable_body_or_unreviewed_reexports(tmp_path, tail):
    with pytest.raises(CompilerLibraryError, match="static leaf re-exports"):
        _facade(tmp_path, tail=tail)


def test_other_public_namespace_still_refused(tmp_path):
    with pytest.raises(CompilerLibraryError, match="leaf modules"):
        _facade(tmp_path, public_extra=("merlin.sched",))
