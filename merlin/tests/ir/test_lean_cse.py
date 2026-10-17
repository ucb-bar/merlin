"""Translation validation of actual upstream MLIR CSE, without numerical assumptions."""

import json
import subprocess

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

RUN_CSE = """
import sys
from pathlib import Path
from torch_mlir import ir, passmanager
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    passmanager.PassManager.parse('builtin.module(cse)').run(module.operation)
    Path(sys.argv[2]).write_text(str(module))
"""


def test_actual_mlir_cse_and_defined_chain(tmp_path, lean_tools):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text("""func.func @forward() -> (i32, i32) {
      %a = arith.constant 7 : i32
      %b = arith.constant 7 : i32
      return %a, %b : i32, i32
    }""")
    result = subprocess.run(
        [str(lean_tools.m2m_python), "-c", RUN_CSE, str(before), str(after)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert before.read_text().count("arith.constant") == 2
    assert after.read_text().count("arith.constant") == 1
    (stage / "meta.json").write_text(json.dumps({"boundary": "upstream-cse"}))
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert (
        check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert chain["verdict"] == "verified", chain.get("reason")
    after.write_text(after.read_text().replace("constant 7", "constant 8"))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
