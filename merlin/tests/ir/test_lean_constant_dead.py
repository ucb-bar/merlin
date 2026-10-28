"""Actual canonicalization removes only successfully decoded unused scalar constants exactly."""

import pytest
from test_lean_constant_alias import CASES, stage_for
from test_lean_defined_conversions import certify
from test_lean_reshape_compose import _verify_mlir

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def source(element="i32", literal="7", *, escape=False):
    constant = literal if element == "i1" else f"{literal} : {element}"
    return f"""func.func @forward(%unknown: {element}, %keep: i32) -> ({element}, i32) {{
      %dead = arith.constant {constant}
      return %{"dead" if escape else "unknown"}, %keep : {element}, i32
    }}"""


def nested_source(depth):
    inner = """%dead1 = arith.constant 0x7FC01234 : f32
      %unused1 = arith.constant -1 : index
      %sum1 = arith.addi %x1, %old1 : i32
      linalg.yield %sum1 : i32"""
    body = "%sum0 = arith.addi %x0, %old0 : i32"
    result = "%sum0"
    if depth == 2:
        body = f"""%fresh = tensor.splat %x0 : tensor<i32>
        %inner = linalg.generic {{indexing_maps = [affine_map<()->()>, affine_map<()->()>],
            iterator_types = []}} ins(%fresh : tensor<i32>) outs(%fresh : tensor<i32>) {{
          ^bb0(%x1: i32, %old1: i32):
            {inner}
        }} -> tensor<i32>
        %scalar = tensor.extract %inner[] : tensor<i32>"""
        result = "%scalar"
    return f"""func.func @forward(%payload: tensor<2xi32>, %init: tensor<2xi32>) -> tensor<2xi32> {{
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%payload : tensor<2xi32>) outs(%init : tensor<2xi32>) {{
        ^bb0(%x0: i32, %old0: i32):
          %dead0 = arith.constant 17 : i32
          %unused0 = arith.constant 0x8000000000000000 : f64
          {body}
          linalg.yield {result} : i32
      }} -> tensor<2xi32>
      return %out : tensor<2xi32>
    }}"""


def captured_source():
    return """func.func @forward(%payload: tensor<2xi32>, %init: tensor<2xi32>) -> tensor<2xi32> {
      %live = arith.constant 7 : i32
      %dead = arith.constant 17 : i32
      %out = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]} ins(%payload : tensor<2xi32>) outs(%init : tensor<2xi32>) {
        ^bb0(%x: i32, %old: i32):
          %sum = arith.addi %x, %live : i32
          linalg.yield %sum : i32
      } -> tensor<2xi32>
      return %out : tensor<2xi32>
    }"""


@pytest.mark.parametrize("element,literal", CASES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_dead_scalar_constants(tmp_path, lean_tools, element, literal, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, source(element, literal))
    assert "arith.constant" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_dead_constants_inside_nested_generics(tmp_path, lean_tools, depth, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, nested_source(depth))
    assert after.read_text().count("linalg.generic") == depth
    assert "arith.constant" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_cleanup_preserves_a_captured_live_constant(tmp_path, lean_tools, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, captured_source())
    assert after.read_text().count("arith.constant") == 1 and "linalg.generic" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("fault", ["escape", "capture"])
def test_changed_live_constant_is_not_dead(tmp_path, lean_tools, fault):
    text = source(escape=True) if fault == "escape" else captured_source()
    stage, _, after = stage_for(tmp_path, lean_tools.m2m_python, text)
    after.write_text(after.read_text().replace("arith.constant 7 : i32", "arith.constant 8 : i32"))
    _verify_mlir(after, lean_tools.m2m_python)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_unknown_constant_attribute_has_no_strict_source_claim(tmp_path, lean_tools):
    text = source().replace(
        "arith.constant 7 : i32", '"arith.constant"() {value = 7 : i32, unknown.meaning} : () -> i32'
    )
    stage, before, _ = stage_for(tmp_path, lean_tools.m2m_python, text)
    # Universal DCE may still establish a conditional boundary. Strict source obligations
    # must refuse the actual unmodeled operation instead of silently declaring it defined.
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "abstracted", result.get("reason")


def mixed_source():
    return """func.func @forward(%value: i8, %keep: i32) -> (i32, index, i32) {
      %extended = arith.extsi %value : i8 to i32
      %a = arith.constant 17 : index
      %b = arith.constant 5 : index
      %sum = arith.addi %a, %b : index
      return %extended, %sum, %keep : i32, index, i32
    }"""


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_mixed_fold_cleanup_and_expression_motion(tmp_path, lean_tools, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, mixed_source())
    text = after.read_text()
    assert text.index("constant 22 : index") < text.index("arith.extsi")
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_changed_mixed_fold_constant_is_refused(tmp_path, lean_tools, index_bits):
    stage, _, after = stage_for(tmp_path, lean_tools.m2m_python, mixed_source())
    after.write_text(after.read_text().replace("constant 22 : index", "constant 23 : index"))
    _verify_mlir(after, lean_tools.m2m_python)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
