"""The pure core emitter serves corpus derivation without target-owned golden exporters."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract import matmul_interface


def test_core_emitter_requires_explicit_target():
    with pytest.raises(TypeError, match="target"):
        matmul_interface.emit_interface_mlir(
            lhs="A", weight="W", out="Y", M=2, K=3, N=4, epilogue=[], output_dtype="i32"
        )


def test_corpus_matmul_does_not_import_golden_exporter(tmp_path):
    root = repo_root()
    # -I/-S removes editable finders. Add the installed contract-validation
    # dependencies and explicitly refuse either optional evaluation dependency.
    names = ("yaml", "numpy", "jsonschema", "attrs", "attr", "referencing", "rpds", "jsonschema_specifications")
    dependencies = []
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            dependencies.append(str(Path(spec.origin).parent.parent))
    script = """
import importlib.abc
import sys
core, *dependencies = sys.argv[1:]
sys.path[:0] = [core, *dependencies]
class RejectResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('merlin.targetgen.model_slice_export', 'merlin.targetgen.capsule_golden'):
            raise AssertionError('core corpus imported optional golden evaluation: ' + fullname)
sys.meta_path.insert(0, RejectResearch())
from merlin.targetgen.corpus_spec import CorpusBinding, build_matmul
binding = CorpusBinding(target='probe_device', tile_dim=4, operand_dtype='int8',
    accum_dtype='i32', integer=True, tiers=['L0'], compare='exact_int', classes_for=lambda **_: [])
capsule, text = build_matmul(dict(name='probe', kind='layer', op='matmul', M=4, K=4, N=4,
    source_role='handauthored_compiler_test', source_reference='ownership regression'), binding)
assert capsule['name'] == 'probe'
assert 'merlin_iface.target = "probe_device"' in text
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root / "src"), *dict.fromkeys(dependencies)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
