"""Explicit public-library export, isolated imports and fail-closed membership."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def exporter(monkeypatch):
    directory = repo_root() / "build_tools/scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("export_compiler_library", directory / "export_compiler_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export


def selection():
    return dict(schema="merlin.compiler_library_selection.v1", review_id="test-review", public_modules=[
        "merlin.runtime.packed_float"], empty_namespaces=["merlin", "merlin.runtime"], sources=[
            dict(path=f"src/merlin/runtime/{name}.py", module=f"merlin.runtime.{name}")
            for name in ("fp8_formats", "packed_float")])


def test_export_is_importable_without_original_namespace_or_reference(exporter, tmp_path):
    contract = exporter(selection(), repo_root(), tmp_path / "library")
    contract.verify(tmp_path / "library")
    assert contract.permits("merlin.runtime.packed_float")
    assert not contract.permits("merlin.runtime")
    script = '''
import importlib.util, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from merlin.runtime.packed_float import PackedFloatLayout, pack_codes
assert pack_codes((1, 2), PackedFloatLayout("fp4_e2m1", 2, 1, "lsb-first", "little"))[0] == b"!"
assert importlib.util.find_spec("merlin.runtime.reference") is None
assert all(Path(module.__file__).is_relative_to(root) for name, module in sys.modules.items()
           if name == "merlin" or name.startswith("merlin."))
'''
    subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path / "library")], check=True, cwd=tmp_path)
    (tmp_path / "library/merlin/runtime/packed_float.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        contract.verify(tmp_path / "library")


@pytest.mark.parametrize("mutation", ["escape", "module", "missing", "parent_grant"])
def test_invalid_selection_never_leaves_a_usable_library(exporter, tmp_path, mutation):
    selected = selection()
    if mutation == "escape":
        selected["sources"][0]["path"] = "../outside.py"
    elif mutation == "module":
        selected["sources"][0]["module"] = "../outside"
    elif mutation == "missing":
        selected["sources"].pop(0)
    else:
        selected["public_modules"] = ["merlin.runtime"]
    with pytest.raises(ValueError):
        exporter(selected, repo_root(), tmp_path / "library")
    assert not (tmp_path / "library").exists()
