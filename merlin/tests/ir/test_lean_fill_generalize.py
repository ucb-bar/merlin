"""Check MLIR's actual fill generalization on runtime ranks and initialized input domains."""

import pytest
from test_lean_generalize import _generalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _source(shape="2x3xi32", scalar="i32", *, dirty=True):
    ty = f"tensor<{shape}>"
    destination = "" if dirty else f", %init: {ty}"
    allocation = f"%init = tensor.empty() : {ty}" if dirty else ""
    return f"""func.func @forward(%value: {scalar}, %alternate: {scalar}{destination}) -> {ty} {{
      {allocation}
      %r = linalg.fill ins(%value : {scalar}) outs(%init : {ty}) -> {ty}
      return %r : {ty}
    }}"""


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize(
    "shape,scalar,dirty",
    [
        ("i32", "i32", True),
        ("0x3xi32", "i32", True),
        ("2x3xi32", "i32", True),
        ("2x3x4xi8", "i8", False),
        ("3x2xf32", "f32", True),
        ("2xf64", "f64", False),
        ("2xbf16", "bf16", True),
    ],
)
def test_actual_fill_generalization(tmp_path, lean_tools, shape, scalar, dirty, index_bits):
    stage, _, _ = _generalize(tmp_path, lean_tools, _source(shape, scalar, dirty=dirty), boundary="fill-generalization")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    assert result["index_bits"] == index_bits


@pytest.mark.parametrize("index_bits", [32, 64])
def test_fill_generalization_composes_with_source_definedness(tmp_path, lean_tools, index_bits):
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(), boundary="fill-generalization")
    for check, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = check(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert result["verdict"] == "verified", result.get("reason")
    assert result["theorem"] == "MerlinVerify.Check.chain_typed_extended_defined_refines"


@pytest.mark.parametrize("fault", ["input", "body", "returned_storage"])
def test_fill_generalization_rejects_corrupted_output(tmp_path, lean_tools, fault):
    stage, _, after = _generalize(tmp_path, lean_tools, _source(), boundary="fill-generalization")
    original = after.read_text()
    if fault == "input":
        corrupted = original.replace("ins(%arg0 : i32)", "ins(%arg1 : i32)")
    elif fault == "body":
        corrupted = original.replace("linalg.yield %in : i32", "linalg.yield %out : i32")
    else:
        corrupted = original.replace("return %1 :", "return %0 :")
    assert original != corrupted
    after.write_text(corrupted)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_fill_and_map_share_checked_intermediate_types(tmp_path, lean_tools):
    source = """func.func @forward(%x: i32) -> tensor<6xi32> {
      %e = tensor.empty() : tensor<2x3xi32>
      %filled = linalg.fill ins(%x : i32) outs(%e : tensor<2x3xi32>) -> tensor<2x3xi32>
      %flat = tensor.collapse_shape %filled [[0, 1]] : tensor<2x3xi32> into tensor<6xi32>
      %out = tensor.empty() : tensor<6xi32>
      %mapped = linalg.map ins(%flat : tensor<6xi32>) outs(%out : tensor<6xi32>)
        (%value: i32, %old: i32) { linalg.yield %value : i32 }
      return %mapped : tensor<6xi32>
    }"""
    stage, _, _ = _generalize(tmp_path, lean_tools, source, boundary="fill-and-map-generalization")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
