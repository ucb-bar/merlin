"""Shared counted-index checks need no emitter, backend loader or numerical oracle."""

import subprocess
import sys

import pytest

from merlin.common.paths import repo_root
from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Ptr, Scaled, Select, Sum, Var
from merlin.sched.ir.integer_domain import IntegerDomainError, check_integer_domain
from merlin.targetgen.compiler_library import freeze_compiler_library


def program(value, extent=2):
    return Kernel("bounded", (), (Loop("i", extent, (Call("opaque", (("operand", value),)),)),))


@pytest.mark.parametrize("bits", [32, 64])
def test_signed_boundaries_and_complete_intermediates(bits):
    limit = (1 << (bits - 1)) - 1
    check_integer_domain(program(Const(-limit - 1)), bits)
    check_integer_domain(program(Scaled(Var("i"), limit)), bits)
    check_integer_domain(program(Select("i", 0, Const(limit), Const(-limit - 1))), bits)
    with pytest.raises(IntegerDomainError, match="sum intermediate"):
        check_integer_domain(program(Sum((Const(limit), Const(1), Const(-1)))), bits)
    with pytest.raises(IntegerDomainError, match="overflow"):
        check_integer_domain(program(Ptr("input", Scaled(Var("i"), limit + 1))), bits)
    with pytest.raises(IntegerDomainError, match="loop extent"):
        check_integer_domain(program(Const(0), limit + 1), bits)


@pytest.mark.parametrize("bits", [True, 0, 16, None])
def test_unselected_or_unsupported_index_width_refuses(bits):
    with pytest.raises(IntegerDomainError, match="explicit signed"):
        check_integer_domain(program(Const(0)), bits)


def test_unknown_expression_and_unbound_variable_are_diagnostics():
    with pytest.raises(IntegerDomainError, match="unknown integer expression"):
        check_integer_domain(program(Expr()), 64)
    with pytest.raises(IntegerDomainError, match="unbound integer"):
        check_integer_domain(program(Var("foreign")), 64)
    with pytest.raises(IntegerDomainError, match="unbound integer"):
        check_integer_domain(program(Select("foreign", 0, Const(0), Const(1))), 64)
    with pytest.raises(IntegerDomainError, match="unknown statement"):
        check_integer_domain(Kernel("unknown", (), (object(),)), 64)


def test_reviewed_leaf_imports_without_installed_packages_or_backend_access(tmp_path):
    modules = (
        "merlin", "merlin.sched", "merlin.sched.ir",
        "merlin.sched.ir.expr", "merlin.sched.ir.kernel", "merlin.sched.ir.parse",
        "merlin.sched.ir.integer_domain",
    )
    source_root = repo_root() / "src"
    sources = []
    for module in modules:
        relative = module.replace(".", "/")
        relative += "/__init__.py" if module in modules[:3] else ".py"
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
        sources.append((relative, module))
    contract = freeze_compiler_library(
        tmp_path, review_id="shared-index-leaf-v1", public_modules=(modules[-1],), sources=tuple(sources)
    )
    contract.verify(tmp_path)
    assert not contract.permits("merlin.sched.codegen")
    script = """import sys
sys.path.insert(0, sys.argv[1])
from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Var
from merlin.sched.ir.integer_domain import IntegerDomainError, check_integer_domain
kernel = Kernel('large', (), (Loop('i', 1 << 30, (Call('opaque', (('offset', Scaled(Var('i'), 8)),)),)),))
check_integer_domain(kernel, 64)
try:
    check_integer_domain(kernel, 32)
except IntegerDomainError:
    pass
else:
    raise AssertionError('unsafe narrower domain admitted')
forbidden = ('merlin.runtime', 'merlin.sched.codegen', 'merlin.sched.isa', 'numpy', 'xdsl')
assert not any(m.startswith(forbidden) for m in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
