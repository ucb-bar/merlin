"""The refactored SDK closes real imports and executes an explicitly edited pass."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from merlin.compiler.schedule.export import export_shared_scheduling, public_modules
from merlin.compiler.schedule.library_sources import library_modules
from merlin.targetgen import compiler_library_fork as fork
from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library


def test_refactored_sdk_imports_and_runs_selected_edited_pass(tmp_path):
    assert public_modules(25) == public_modules(24)
    assert library_modules() == library_modules(26)
    assert library_modules(25)[:len(library_modules(24))] == library_modules(24)
    assert all(name != "merlin.compile" and not name.startswith("merlin.compile.")
               for name in library_modules(26))
    root = tmp_path / "sdk"
    document = export_shared_scheduling(root, version=26)
    assert document["schema"] == "merlin.shared_scheduling_library.v20"
    contract = freeze_compiler_library(
        root, review_id=fork.REVIEW_PREFIX + "refactor-closure-test",
        public_modules=tuple(document["public_modules"]),
        sources=tuple((row["path"], row["module"]) for row in document["members"]),
    )
    selection = CompilerLibrarySelection(contract, root)
    package = tmp_path / "candidate"
    package.mkdir()
    fork.stage(selection, package)
    relative = "merlin/compiler/passes/xdsl.py"
    original = (root / relative).read_bytes()
    edited = original.replace(b"for rewrite in rewrites:", b"for rewrite in reversed(tuple(rewrites)):")
    assert edited != original
    candidate = package / fork.DIRECTORY / relative
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(edited)
    overrides, digest = fork.candidate_sources(selection, package)
    assert overrides == {relative: candidate} and digest

    def execute(directory):
        code = """
import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
for name in json.loads(sys.argv[2]):
    loaded = importlib.import_module(name)
    assert pathlib.Path(loaded.__file__).is_relative_to(root), name
from merlin.compiler.schedule.api_catalog import capability_effects, scheduling_capabilities
assert scheduling_capabilities()['inventory_version'] == 26
assert capability_effects()
for leaf in ('catalog_dense', 'catalog_spatial', 'catalog_source', 'catalog_effects'):
    loaded = importlib.import_module('merlin.compiler.schedule.' + leaf)
    assert pathlib.Path(loaded.__file__).is_relative_to(root), leaf
from merlin.compiler.frontend import preprocess
from merlin.compiler.quantization.linalg import lower_quant_ext
from merlin.compiler.transforms.dead import prune_dead_pure_tensor_ops
assert preprocess.lower_quant_ext is lower_quant_ext
assert preprocess.prune_dead_pure_tensor_ops is prune_dead_pure_tensor_ops
assert all(path.is_relative_to(root) for path in preprocess.implementation_paths())
assert 'merlin.llvmlower.passes_xdsl' not in sys.modules
assert preprocess.preprocess_text('builtin.module {}')[1]['quant_ext_lowered'] == 0
from merlin.compiler.passes.xdsl import Rewrite, run_rewrites
order = []
passes = [Rewrite(name, None, lambda _, name=name: order.append(name) or 1)
          for name in ('first', 'second')]
assert run_rewrites(None, passes) == {'first': 1, 'second': 1}
print(json.dumps(order))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(directory), json.dumps(document["public_modules"])],
            cwd=tmp_path, capture_output=True, text=True, timeout=120, env=dict(os.environ),
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])

    assert execute(root) == ["first", "second"]
    execution = tmp_path / "candidate-execution"
    shutil.copytree(root, execution)
    for relative, selected in overrides.items():
        (execution / relative).write_bytes(selected.read_bytes())
    assert execute(execution) == ["second", "first"]
    # Reversed order proves the edit executed, but would fail the original ordered
    # behavior obligation. Source selection does not certify a candidate's result.
    assert (root / relative).read_bytes() == original
    contract.verify(root)
    foreign = package / fork.DIRECTORY / "merlin/compiler/unreviewed.py"
    foreign.write_text("pass\n")
    with pytest.raises(ValueError, match="ungranted"):
        fork.candidate_sources(selection, package)
