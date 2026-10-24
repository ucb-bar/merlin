"""Generators use the installed namespace without legacy flat packages."""

import os
import subprocess
import sys

from merlin.common.paths import repo_root


def test_generate_import_without_legacy_pythonpath(tmp_path):
    source = repo_root() / "src"
    script = """
import importlib.abc
import sys

class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'common', 'targetgen', 'generate'} or fullname == 'merlin.targetgen.rtl.facts':
            raise AssertionError('legacy import: ' + fullname)

sys.meta_path.insert(0, RejectLegacy())
from merlin.targetgen import generate
from merlin.targetgen.target_registry import resolve
from merlin.common.artifacts import Artifact
assert resolve('synthetic').facts_path.name == 'facts.json'
for name in generate.__all__:
    assert getattr(generate, name)
assert all(isinstance(a, Artifact) for a in generate.llvm_plan.generate({}))
assert generate.target_repo.generate_skeleton('synthetic')
assert generate.zephyr_module.generate({})
assert not any('/merlin/python' in p for p in sys.path)
"""
    env = {**os.environ, "PYTHONPATH": str(source)}
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
