"""Compiler failures retain their first diagnostic without admitting stale output."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.proc import run_checked
from merlin.llvmlower.compilation_recipe import CompilationRecipe


def test_real_failure_keeps_full_diagnostics_and_rejects_stale_object(tmp_path):
    source = tmp_path / "compiler.py"
    source.write_text("import sys\nsys.stderr.write('FIRST ASSERTION\\n' + 'stack\\n' * 1000)\nsys.exit(7)\n")
    output = tmp_path / "stale.o"
    output.write_bytes(b"older output")
    recipe = CompilationRecipe(tmp_path, producer=source)
    with pytest.raises(RuntimeError, match="rc 7") as failed:
        recipe.run([sys.executable, source], runner=run_checked, inputs=[source], output=output)
    assert "FIRST ASSERTION" not in str(failed.value)
    command = json.loads(recipe.path.read_bytes())["commands"][0]
    assert command["status"] == "failed" and command["returncode"] == 7
    assert "output" not in command and command["elapsed_s"] >= 0
    identity = command["captured_diagnostics"]["stderr"]
    contents = Path(identity["path"]).read_bytes()
    assert contents.startswith(b"FIRST ASSERTION\n") and contents.endswith(b"stack\n")
    assert hashlib.sha256(contents).hexdigest() == identity["sha256"]
    with pytest.raises(ValueError, match="complete successful"):
        recipe.completed_product(output, kind="object")


@pytest.mark.parametrize("wrapped", [True, False])
def test_real_timeout_keeps_partial_diagnostics(tmp_path, wrapped):
    source = tmp_path / "compiler.py"
    source.write_text("import sys,time\nsys.stderr.write('BEFORE TIMEOUT\\n'); sys.stderr.flush()\ntime.sleep(5)\n")
    recipe = CompilationRecipe(tmp_path, producer=source)

    def runner(argv):
        return run_checked(argv, timeout=0.2, wrap_timeout=wrapped)

    with pytest.raises(RuntimeError if wrapped else subprocess.TimeoutExpired):
        recipe.run([sys.executable, source], runner=runner, inputs=[source], output=tmp_path / "absent.o")
    document = json.loads(recipe.path.read_bytes())
    command = document["commands"][0]
    assert document["status"] == command["status"] == "timeout"
    assert command["timeout_s"] == 0.2 and "output" not in command
    stderr = Path(command["captured_diagnostics"]["stderr"]["path"]).read_bytes()
    assert stderr == b"BEFORE TIMEOUT\n"
