"""Proved source definedness is distinct from progress and uses an explicit nonpoison domain."""

import json
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.verify.lean.check import check_definedness

BINARY_NAMES = ["addi", "subi", "muli", "andi", "ori", "xori", "maxsi", "minsi", "maxui", "minui"]
SCHEMA = repo_root() / "merlin/contract/schemas/lean_receipt.schema.json"


def _check(tmp_path, lean_tools, text, *, index_bits=64):
    program = tmp_path / "program.mlir"
    program.write_text(text)
    result = check_definedness(program, "forward", tmp_path / "proof", m2m_python=lean_tools.m2m_python,
                               index_bits=index_bits)
    import jsonschema

    jsonschema.validate(result, json.loads(SCHEMA.read_text()))
    assert result["claim_type"] == "defined"
    assert result["domain"].startswith("nonpoison inputs")
    assert result["index_bits"] == index_bits
    return result


@pytest.mark.parametrize("name", BINARY_NAMES)
def test_total_integer_operations(tmp_path, lean_tools, name):
    result = _check(tmp_path, lean_tools, f"""func.func @forward(%a: i32, %b: i32) -> i32 {{
      %r = arith.{name} %a, %b : i32
      return %r : i32
    }}""")
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("text", [
    "func.func @forward(%a: tensor<2xi32>) -> tensor<2xi32> { return %a : tensor<2xi32> }",
    "func.func @forward() -> i32 { %c = arith.constant 7 : i32 return %c : i32 }",
    """func.func @forward(%a: i32) -> i32 {
      %unused = tensor.empty() : tensor<2xi32>
      return %a : i32
    }""",
    """func.func @forward(%a: i32) -> tensor<2xi32> {
      %r = tensor.splat %a : tensor<2xi32>
      return %r : tensor<2xi32>
    }""",
    """func.func @forward(%a: i32, %b: i32) -> (i32, i32) {
      return %a, %b : i32, i32
    }""",
    "func.func @forward() { return }",
])
def test_defined_sources(tmp_path, lean_tools, text):
    result = _check(tmp_path, lean_tools, text)
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("body,result_type", [
    ("%r = tensor.empty() : tensor<2xi32> return %r : tensor<2xi32>", "tensor<2xi32>"),
    ("%r = arith.divsi %a, %b : i32 return %r : i32", "i32"),
    ("%r = arith.shli %a, %b : i32 return %r : i32", "i32"),
    ("%r = arith.addi %a, %b overflow<nsw> : i32 return %r : i32", "i32"),
    ("%r = arith.muli %a, %b overflow<nuw> : i32 return %r : i32", "i32"),
    ("%r = arith.constant dense<7> : tensor<2xi32> return %r : tensor<2xi32>", "tensor<2xi32>"),
    ("%r = arith.constant 7 : i32 %t = tensor.empty() : tensor<2xi32> "
     "return %r, %t : i32, tensor<2xi32>", "(i32, tensor<2xi32>)"),
])
def test_unproved_or_undefined_sources_abstain(tmp_path, lean_tools, body, result_type):
    result = _check(tmp_path, lean_tools,
                    f"func.func @forward(%a: i32, %b: i32) -> {result_type} {{ {body} }}")
    assert result["verdict"] == "abstracted", result.get("reason")


def test_index_width_is_bound(tmp_path, lean_tools):
    result = _check(tmp_path, lean_tools,
                    "func.func @forward(%a: index) -> index { return %a : index }", index_bits=32)
    assert result["verdict"] == "verified"
    assert "32-bit index" in result["domain"]


@pytest.mark.parametrize("args", [["0"], ["-1"], ["bad"], ["32", "64"]])
def test_definedness_cli_rejects_invalid_width(tmp_path, lean_tools, args):
    result = subprocess.run([str(lean_tools.lean_exe), "check-definedness", str(tmp_path / "missing"),
                             "forward", *args], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "positive integer" in result.stderr
