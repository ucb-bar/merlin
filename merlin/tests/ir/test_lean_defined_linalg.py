"""Source totality and initialization for real structured tensor programs."""

import json

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean import stages
from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

FIXTURE = repo_root() / "merlin/tests/fixtures/lean_initialized_matmul.mlir"


def _check(tmp_path, lean_tools, source):
    program = tmp_path / "program.mlir"
    program.write_text(source)
    return check_definedness(program, "forward", tmp_path / "proof", m2m_python=lean_tools.m2m_python)


@pytest.mark.parametrize("shape,scalar", [("2x3xi32", "i32"), ("0x3xi32", "i32"), ("i32", "i32"),
                                        ("2xf32", "f32")])
def test_fill_initializes_empty_storage(tmp_path, lean_tools, shape, scalar):
    receipt = _check(tmp_path, lean_tools, f"""func.func @forward(%v: {scalar}) -> tensor<{shape}> {{
      %e = tensor.empty() : tensor<{shape}>
      %r = linalg.fill ins(%v : {scalar}) outs(%e : tensor<{shape}>) -> tensor<{shape}>
      return %r : tensor<{shape}>
    }}""")
    assert receipt["verdict"] == "verified", receipt.get("reason")


def _reduction(extent=3, operation="addi", *, captures=False, multiple=False):
    result = "(tensor<i32>, tensor<i32>)" if multiple else "tensor<i32>"
    second_arg = ", %j: tensor<i32>" if multiple else ""
    second_out = ", %j : tensor<i32>, tensor<i32>" if multiple else " : tensor<i32>"
    body_args = "%x: i32, %acc: i32, %acc2: i32" if multiple else "%x: i32, %acc: i32"
    output_maps = ", affine_map<(d0)->()>" if multiple else ""
    lhs = "%r:2" if multiple else "%r"
    yielded = "%s, %s : i32, i32" if multiple else "%s : i32"
    returned = "%r#0, %r#1 : tensor<i32>, tensor<i32>" if multiple else "%r : tensor<i32>"
    rhs = "%captured" if captures else "%x"
    return f"""func.func @forward(%a: tensor<{extent}xi32>, %i: tensor<i32>{second_arg}) -> {result} {{
      %captured = arith.constant 7 : i32
      {lhs} = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->()>{output_maps}],
        iterator_types = ["reduction"]}} ins(%a : tensor<{extent}xi32>) outs(%i{second_out}) {{
      ^bb0({body_args}):
        %s = arith.{operation} %acc, {rhs} : i32
        linalg.yield {yielded}
      }} -> {result}
      return {returned}
    }}"""


@pytest.mark.parametrize("extent,captures,multiple", [(3, False, False), (0, False, False),
                                                    (3, True, False), (3, False, True)])
def test_initialized_reductions(tmp_path, lean_tools, extent, captures, multiple):
    receipt = _check(tmp_path, lean_tools, _reduction(extent, captures=captures, multiple=multiple))
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize("fault", ["uninitialized_accumulator", "out_of_bounds", "flagged", "division"])
def test_structured_definedness_refuses_missing_obligations(tmp_path, lean_tools, fault):
    source = FIXTURE.read_text()
    if fault == "uninitialized_accumulator":
        source = source.replace("outs(%z :", "outs(%e :")
    elif fault == "out_of_bounds":
        source = source.replace("affine_map<(d0,d1,d2)->(d2,d1)>", "affine_map<(d0,d1,d2)->(d1,d2)>")
    elif fault == "flagged":
        source = source.replace("arith.muli %xe, %ye :", "arith.muli %xe, %ye overflow<nsw> :")
    else:
        source = source.replace("arith.muli", "arith.divsi")
    receipt = _check(tmp_path, lean_tools, source)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_initialized_matmul_chain(tmp_path, monkeypatch, lean_tools):
    import jsonschema

    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    preprocess_text(FIXTURE.read_text())
    boundaries = stages.completed(tmp_path / "stages")
    for stage in boundaries:
        receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
        assert receipt["verdict"] == "verified", receipt.get("reason")
    source = boundaries[0] / "before.mlir"
    assert check_support(source, "forward", tmp_path / "support",
                         m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    receipt = check_definedness(source, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    chain = compose_chain(boundaries, tmp_path / "support/receipt.json", "forward", tmp_path / "chain.json",
                          m2m_python=lean_tools.m2m_python, definedness=tmp_path / "defined/receipt.json")
    schema = repo_root() / "merlin/contract/schemas/lean_chain.schema.json"
    jsonschema.validate(chain, json.loads(schema.read_text()))
    assert chain["verdict"] == chain["refinement_verdict"] == "verified", chain.get("reason")
    assert chain["source_definedness"]["verdict"] == "verified"
