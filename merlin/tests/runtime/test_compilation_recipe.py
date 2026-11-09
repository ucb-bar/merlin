from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.compilation_recipe import FILENAME, CompilationRecipe
from merlin.runtime.backends import spike_model


def _read(work):
    return json.loads((work / FILENAME).read_text())


def test_failed_compiler_keeps_old_object_unqualified(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    source, obj = (tmp_path / "source.c", tmp_path / "source.o")
    source.write_text("int value(void) { return 7; }\n")
    recipe = CompilationRecipe(tmp_path, producer=Path(spike_model.__file__))
    cmd = [compiler, "-c", source, "-o", obj]

    def run(argv):
        return subprocess.run(argv, capture_output=True)

    recipe.run(cmd, runner=run, inputs=[source], output=obj)
    old_bytes = obj.read_bytes()
    source.write_text("not valid C\n")
    with pytest.raises(RuntimeError, match="compilation command returned"):
        recipe.run(cmd, runner=run, inputs=[source], output=obj)
    record = _read(tmp_path)
    assert record["status"] == "failed"
    assert record["commands"][-1]["status"] == "failed"
    assert "output" not in record["commands"][-1]
    assert obj.read_bytes() == old_bytes
    assert "executable" not in record
    with pytest.raises(ValueError, match="complete successful command sequence"):
        recipe.completed(obj)
