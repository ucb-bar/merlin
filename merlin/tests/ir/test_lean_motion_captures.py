"""Adjacent scalar motion preserves values captured by later recursive children.

The moved operations remain region-free. Region-bearing motion and operation
reordering inside regions have separate explicit conservative abstention cases.
"""

import pytest
from test_lean_defined_bounds import _chain
from test_lean_motion_independent import _stage
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def _captured(kind="parallel", *, region_barrier=False):
    cast = "%cast = arith.extsi %x : i8 to i32"
    constant = "%c = arith.constant 22 : i32"
    if kind == "reduction":
        args, shape, operands = (
            "%data: tensor<2xi32>, %dest: tensor<i32>",
            "tensor<i32>",
            ("ins(%data : tensor<2xi32>) outs(%dest : tensor<i32>)"),
        )
        maps = "affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>"
        block, iterator = "%element: i32, %old: i32", "reduction"
        body = """%delta = arith.addi %cast, %c : i32
          %term = arith.addi %element, %delta : i32
          %value = arith.addi %old, %term : i32
          linalg.yield %value : i32"""
    else:
        args, shape, operands = "%dest: tensor<2xi32>", "tensor<2xi32>", "outs(%dest : tensor<2xi32>)"
        maps, block, iterator = "affine_map<(d0) -> (d0)>", "%old: i32", "parallel"
        body = """%delta = arith.addi %cast, %c : i32
          %value = arith.addi %old, %delta : i32
          linalg.yield %value : i32"""
        if kind == "nested":
            # Allocate inside the outer iteration and overwrite completely. Do
            # not reuse the captured initialized destination implicated in S4.
            body = """%empty = tensor.empty() : tensor<i32>
              %inner = linalg.generic {
                indexing_maps = [affine_map<() -> ()>], iterator_types = []
              } outs(%empty : tensor<i32>) {
              ^bb0(%unused: i32):
                %delta = arith.addi %cast, %c : i32
                %value = arith.addi %old, %delta : i32
                linalg.yield %value : i32
              } -> tensor<i32>
              %extracted = tensor.extract %inner[] : tensor<i32>
              linalg.yield %extracted : i32"""
    if region_barrier:
        assert kind == "parallel"
        body = body.replace("%cast, %c", "%cast, %cast")
    generic = f"""%out = linalg.generic {{
          indexing_maps = [{maps}], iterator_types = ["{iterator}"]
        }} {operands} {{
        ^bb0({block}):
          {body}
        }} -> {shape}"""
    middle = f"{generic}\n{constant}" if region_barrier else f"{constant}\n{generic}"
    return f"""func.func @subject(%x: i8, {args}) -> (i32, i32, {shape}) {{
      {cast}
      {middle}
      return %cast, %c, %out : i32, i32, {shape}
    }}"""


def _adjacent_hoist(source):
    target = source.replace(
        "%cast = arith.extsi %x : i8 to i32\n      %c = arith.constant 22 : i32",
        "%c = arith.constant 22 : i32\n      %cast = arith.extsi %x : i8 to i32",
    )
    assert target != source
    return target


def _mutant(source, fault):
    if fault == "capture":
        return source.replace("arith.addi %cast, %c", "arith.addi %cast, %cast")
    if fault == "yield":
        return source.replace("linalg.yield %value", "linalg.yield %delta")
    if fault == "escaped-scalar":
        return source.replace("return %cast, %c, %out", "return %c, %c, %out")
    if fault == "escaped-tensor":
        return source.replace("return %cast, %c, %out", "return %cast, %c, %dest")
    raise AssertionError(fault)


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("kind", ["parallel", "nested", "reduction"])
def test_actual_adjacent_hoist_preserves_recursive_captures(tmp_path, lean_tools, bits, kind):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _captured(kind))
    assert before.read_text().index("arith.extsi") < before.read_text().index("arith.constant")
    assert after.read_text().index("arith.constant") < after.read_text().index("arith.extsi")
    _chain(tmp_path, stage, "subject", lean_tools, bits)


@pytest.mark.parametrize("kind", ["parallel", "nested"])
@pytest.mark.parametrize("fault", ["capture", "yield", "escaped-scalar", "escaped-tensor"])
def test_valid_changed_captured_or_escaped_values_refuse(tmp_path, lean_tools, kind, fault):
    source = _captured(kind)
    # Keep source identifiers while expressing the actual pass's adjacent hoist;
    # both full actual programs still receive native MLIR verification.
    target = _adjacent_hoist(source)
    mutant = _mutant(target, fault)
    assert mutant != target
    stage = _stage(tmp_path, source, mutant)
    for path in (stage / "before.mlir", stage / "after.mlir"):
        _verify_mlir(path, lean_tools.m2m_python)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_actual_hoist_across_region_bearing_op_remains_explicit_gap(tmp_path, lean_tools):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _captured(region_barrier=True))
    assert before.read_text().index("linalg.generic") < before.read_text().index("arith.constant")
    assert after.read_text().index("arith.constant") < after.read_text().index("linalg.generic")
    assert (
        check_support(before, "subject", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert (
        check_definedness(before, "subject", tmp_path / "defined", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def _body_motion(reverse=False):
    first = "%a = arith.addi %old, %capture : i32"
    second = "%b = arith.muli %old, %capture : i32"
    body = f"{second}\n{first}" if reverse else f"{first}\n{second}"
    return f"""func.func @subject(%capture: i32, %dest: tensor<2xi32>) -> tensor<2xi32> {{
      %out = linalg.generic {{
        indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]
      }} outs(%dest : tensor<2xi32>) {{
      ^bb0(%old: i32):
        {body}
        %sum = arith.addi %a, %b : i32
        linalg.yield %sum : i32
      }} -> tensor<2xi32>
      return %out : tensor<2xi32>
    }}"""


def test_operation_order_inside_generic_body_remains_explicit_gap(tmp_path, lean_tools):
    stage = _stage(tmp_path, _body_motion(), _body_motion(reverse=True))
    for path in (stage / "before.mlir", stage / "after.mlir"):
        _verify_mlir(path, lean_tools.m2m_python)
    assert (
        check_support(stage / "before.mlir", "subject", tmp_path / "support", m2m_python=lean_tools.m2m_python)[
            "verdict"
        ]
        == "verified"
    )
    assert (
        check_definedness(stage / "before.mlir", "subject", tmp_path / "defined", m2m_python=lean_tools.m2m_python)[
            "verdict"
        ]
        == "verified"
    )
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
