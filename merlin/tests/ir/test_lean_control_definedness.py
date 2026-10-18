"""Defined comparisons and selection compose with real rewrites and retain poison checks."""

import pytest
from test_lean_definedness import _check
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

PREDICATES = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("ty,index_bits", [("i1", 64), ("i13", 64), ("i64", 64), ("index", 32), ("index", 64)])
def test_initialized_integer_comparisons(tmp_path, lean_tools, predicate, ty, index_bits):
    receipt = _check(
        tmp_path,
        lean_tools,
        f"""func.func @forward(%a: {ty}, %b: {ty}) -> i1 {{
          %r = arith.cmpi {predicate}, %a, %b : {ty}
          return %r : i1
        }}""",
        index_bits=index_bits,
    )
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize(
    "ty", ["i1", "i13", "index", "f16", "bf16", "f32", "f64", "tensor<i32>", "tensor<0xi32>", "tensor<2x3xf32>"]
)
def test_scalar_condition_selects_initialized_values(tmp_path, lean_tools, ty):
    receipt = _check(
        tmp_path,
        lean_tools,
        f"""func.func @forward(%c: i1, %a: {ty}, %b: {ty}) -> {ty} {{
          %r = arith.select %c, %a, %b : {ty}
          return %r : {ty}
        }}""",
    )
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize("shape", ["", "0x", "2x3x"])
@pytest.mark.parametrize("element", ["i1", "i13", "f32", "f64"])
def test_tensor_condition_selects_initialized_elements(tmp_path, lean_tools, shape, element):
    ty, condition = f"tensor<{shape}{element}>", f"tensor<{shape}i1>"
    receipt = _check(
        tmp_path,
        lean_tools,
        f"""func.func @forward(%c: {condition}, %a: {ty}, %b: {ty}) -> {ty} {{
          %r = arith.select %c, %a, %b : {condition}, {ty}
          return %r : {ty}
        }}""",
    )
    assert receipt["verdict"] == "verified", receipt.get("reason")


def comparison_select_source():
    return """func.func @forward(%x: i13, %y: i13, %a: i13, %b: i13) -> (i1, i13) {
      %c = arith.cmpi slt, %x, %y : i13
      %selected = arith.select %c, %a, %b : i13
      %zero = arith.constant 0 : i13
      %r = arith.addi %selected, %zero : i13
      return %c, %r : i1, i13
    }"""


@pytest.mark.parametrize("index_bits", [32, 64])
def test_control_source_domain_composes_with_actual_identity_rewrite(tmp_path, lean_tools, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, comparison_select_source())
    assert "arith.addi" not in after.read_text()
    assert "arith.cmpi" in after.read_text() and "arith.select" in after.read_text()
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        receipt = checker(
            before, "forward", tmp_path / directory, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
        assert receipt["verdict"] == "verified", receipt.get("reason")
    receipt = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize("dirty", ["condition", "true", "false"])
def test_selection_cannot_certify_poison_inputs(tmp_path, lean_tools, dirty):
    condition = "%dirty_condition" if dirty == "condition" else "%c"
    lhs = "%dirty_value" if dirty == "true" else "%a"
    rhs = "%dirty_value" if dirty == "false" else "%b"
    receipt = _check(
        tmp_path,
        lean_tools,
        f"""func.func @forward(%c: tensor<2xi1>, %a: tensor<2xi32>, %b: tensor<2xi32>) -> tensor<2xi32> {{
          %dirty_condition = tensor.empty() : tensor<2xi1>
          %dirty_value = tensor.empty() : tensor<2xi32>
          %r = arith.select {condition}, {lhs}, {rhs} : tensor<2xi1>, tensor<2xi32>
          return %r : tensor<2xi32>
        }}""",
    )
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


@pytest.mark.parametrize("operation", ["comparison", "selection"])
def test_scalar_control_cannot_read_uninitialized_extraction(tmp_path, lean_tools, operation):
    body = (
        "%r = arith.cmpi eq, %dirty, %a : i32"
        if operation == "comparison"
        else "%r = arith.select %c, %dirty, %a : i32"
    )
    result_type = "i1" if operation == "comparison" else "i32"
    receipt = _check(
        tmp_path,
        lean_tools,
        f"""func.func @forward(%c: i1, %a: i32) -> {result_type} {{
          %empty = tensor.empty() : tensor<i32>
          %dirty = tensor.extract %empty[] : tensor<i32>
          {body}
          return %r : {result_type}
        }}""",
    )
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_scalar_selection_cannot_certify_poison_condition(tmp_path, lean_tools):
    receipt = _check(
        tmp_path,
        lean_tools,
        """func.func @forward(%a: i32, %b: i32) -> i32 {
          %empty = tensor.empty() : tensor<i1>
          %c = tensor.extract %empty[] : tensor<i1>
          %r = arith.select %c, %a, %b : i32
          return %r : i32
        }""",
    )
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


@pytest.mark.parametrize("operation", ["comparison", "selection"])
def test_control_rules_refuse_unknown_semantic_attributes(tmp_path, lean_tools, operation):
    body = (
        "%r = arith.cmpi eq, %a, %b {unknown.numeric_meaning} : i13"
        if operation == "comparison"
        else "%r = arith.select %c, %a, %b {unknown.numeric_meaning} : i13"
    )
    ty = "i1" if operation == "comparison" else "i13"
    receipt = _check(
        tmp_path,
        lean_tools,
        f"""func.func @forward(%c: i1, %a: i13, %b: i13) -> {ty} {{
          {body}
          return %r : {ty}
        }}""",
    )
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_control_rules_work_in_generic_body(tmp_path, lean_tools):
    receipt = _check(
        tmp_path,
        lean_tools,
        """func.func @forward(%a: tensor<3xi32>, %b: tensor<3xi32>) -> tensor<3xi32> {
          %empty = tensor.empty() : tensor<3xi32>
          %r = linalg.generic {
            indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
            iterator_types = ["parallel"]
          } ins(%a, %b : tensor<3xi32>, tensor<3xi32>) outs(%empty : tensor<3xi32>) {
            ^bb0(%x: i32, %y: i32, %old: i32):
              %c = arith.cmpi slt, %x, %y : i32
              %v = arith.select %c, %x, %y : i32
              linalg.yield %v : i32
          } -> tensor<3xi32>
          return %r : tensor<3xi32>
        }""",
    )
    assert receipt["verdict"] == "verified", receipt.get("reason")
