"""Generic bodies compose sound definedness books, including complete inner overwrites."""

import pytest
from test_lean_definedness import _check


def _nested_source(depth, *, dirty=False, read_destination=False, poison_input=False):
    def generic(level):
        destination = f"%init{level}" if dirty else "%a"
        allocation = f"%init{level} = tensor.empty() : tensor<i32>" if dirty else ""
        if level < depth:
            body = generic(level + 1)
            body += f"\n%scalar{level} = tensor.extract %result{level + 1}[] : tensor<i32>"
            value = f"%scalar{level}"
        else:
            operand = f"%old{level}" if read_destination else f"%element{level}"
            body = f"%sum{level} = arith.addi {operand}, %bias : i32"
            value = f"%sum{level}"
        source = "%poison" if poison_input else "%a"
        return f"""{allocation}
          %result{level} = linalg.generic {{indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
            iterator_types = []}} ins({source} : tensor<i32>) outs({destination} : tensor<i32>) {{
            ^bb0(%element{level}: i32, %old{level}: i32):
              {body}
              linalg.yield {value} : i32
          }} -> tensor<i32>"""

    poison = "%poison = tensor.empty() : tensor<i32>" if poison_input else ""
    return f"""func.func @forward(%a: tensor<i32>, %bias: i32) -> tensor<i32> {{
      {poison}
      {generic(1)}
      return %result1 : tensor<i32>
    }}"""


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("dirty", [False, True])
def test_nested_generics_preserve_initialized_results(tmp_path, lean_tools, index_bits, depth, dirty):
    receipt = _check(tmp_path, lean_tools, _nested_source(depth, dirty=dirty), index_bits=index_bits)
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize("fault", ["read_destination", "poison_input"])
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("depth", [3, 10])
def test_nested_body_does_not_hide_poison(tmp_path, lean_tools, fault, dirty, depth):
    receipt = _check(tmp_path, lean_tools, _nested_source(depth, dirty=dirty, **{fault: True}))
    expected = "verified" if fault == "read_destination" and not dirty else "abstracted"
    assert receipt["verdict"] == expected, receipt.get("reason")


def test_rank_zero_extraction_is_defined(tmp_path, lean_tools):
    receipt = _check(
        tmp_path,
        lean_tools,
        """func.func @forward(%a: tensor<i32>) -> i32 {
          %r = tensor.extract %a[] : tensor<i32>
          return %r : i32
        }""",
    )
    assert receipt["verdict"] == "verified", receipt.get("reason")


def test_positive_rank_extraction_still_needs_bounds(tmp_path, lean_tools):
    receipt = _check(
        tmp_path,
        lean_tools,
        """func.func @forward(%a: tensor<2xi32>, %i: index) -> i32 {
          %r = tensor.extract %a[%i] : tensor<2xi32>
          return %r : i32
        }""",
    )
    assert receipt["verdict"] == "abstracted", receipt.get("reason")
