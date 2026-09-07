"""Pure compiler imports must work while reference/evaluator modules are unavailable."""
import subprocess
import sys


def test_compiler_leaf_imports_do_not_load_reference_runtime():
    script = '''
import importlib.abc
import sys
class DenyAnswers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'merlin.runtime.simulator', 'merlin.runtime.reference'}:
            raise ModuleNotFoundError('answer module deliberately unavailable: ' + fullname)
sys.meta_path.insert(0, DenyAnswers())
from merlin.xdsl_dialects.lowering.canonical_matmul import is_integer_matmul
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer
from merlin.runtime import Tensor
from merlin.xdsl_dialects.lowering import CycleInterval
assert callable(is_integer_matmul) and callable(constant_integer)
assert CycleInterval.point(1).lo == 1
assert 'merlin.runtime.simulator' not in sys.modules
assert 'merlin.runtime.reference' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_lazy_public_apis_preserve_identity():
    from merlin import runtime, xdsl_dialects
    from merlin.runtime.simulator import simulate
    from merlin.xdsl_dialects.lowering import GlobalPlan
    from merlin.xdsl_dialects.lowering.global_plan import GlobalPlan as DirectPlan
    assert runtime.simulate is simulate
    assert GlobalPlan is DirectPlan
    assert len(xdsl_dialects.CORE_DIALECT_MODULES) == 5
    assert len(xdsl_dialects.get_all_dialects()) == 5
