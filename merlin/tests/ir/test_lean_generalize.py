"""Check actual named-contraction generalization and reject independently corrupted output."""

import json
import subprocess

import pytest

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

RUN_GENERALIZE = """
import sys
from pathlib import Path
from torch_mlir import ir, passmanager
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    passmanager.PassManager.parse('builtin.module(linalg-generalize-named-ops)').run(module.operation)
    Path(sys.argv[2]).write_text(str(module))
"""


def _source(name="matmul", scalar="i8", *, extent=3, floating=False, unsigned=False):
    batch = "2x" if name == "batch_matmul" else ""
    output_scalar = "f32" if floating else "i32"
    a, b, c = (f"tensor<{batch}2x{extent}x{scalar}>", f"tensor<{batch}{extent}x2x{scalar}>",
               f"tensor<{batch}2x2x{output_scalar}>")
    cast = "{cast = #linalg.type_fn<cast_unsigned>} " if unsigned else ""
    return f"""func.func @forward(%a: {a}, %b: {b}, %c: {c}) -> {c} {{
      %r = linalg.{name} {cast}ins(%a, %b : {a}, {b}) outs(%c : {c}) -> {c}
      return %r : {c}
    }}"""


def _generalize(tmp_path, lean_tools, source, *, boundary="named-contraction-generalization"):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(source)
    result = subprocess.run([str(lean_tools.m2m_python), "-c", RUN_GENERALIZE, str(before), str(after)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "linalg.generic" in after.read_text()
    (stage / "meta.json").write_text(json.dumps({"boundary": boundary}))
    return stage, before, after


@pytest.mark.parametrize("name,scalar,unsigned", [("matmul", "i8", False), ("matmul", "i32", False),
                                                 ("batch_matmul", "i8", False), ("matmul", "i8", True)])
def test_actual_contraction_generalization_and_defined_chain(tmp_path, lean_tools, name, scalar, unsigned):
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(name, scalar, unsigned=unsigned))
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert check_support(before, "forward", tmp_path / "support",
                         m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    receipt = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    chain = compose_chain([stage], tmp_path / "support/receipt.json", "forward", tmp_path / "chain.json",
                          m2m_python=lean_tools.m2m_python, definedness=tmp_path / "defined/receipt.json")
    assert chain["verdict"] == "verified", chain.get("reason")


def test_float_generalization_preserves_order_without_float_laws(tmp_path, lean_tools):
    stage, _, _ = _generalize(tmp_path, lean_tools, _source(scalar="f32", floating=True))
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize("fault", ["body", "input_map", "output_map", "cast"])
def test_corrupted_generalization_is_rejected(tmp_path, lean_tools, fault):
    stage, _, after = _generalize(tmp_path, lean_tools, _source(extent=2))
    text = after.read_text()
    if fault == "body":
        corrupted = text.replace("arith.addi", "arith.subi")
    elif fault == "input_map":
        corrupted = text.replace("(d0, d2)", "(d2, d0)")
    elif fault == "output_map":
        corrupted = text.replace("(d0, d1)>", "(d1, d0)>")
    else:
        corrupted = text.replace("arith.extsi", "arith.extui")
    assert corrupted != text
    after.write_text(corrupted)
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_uninitialized_named_accumulator_is_not_proved(tmp_path, lean_tools):
    source = _source().replace("%c: tensor<2x2xi32>", "%unused: tensor<2x2xi32>")
    source = source.replace("%r =", "%c = tensor.empty() : tensor<2x2xi32>\n%r =")
    _, before, _ = _generalize(tmp_path, lean_tools, source)
    receipt = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")
