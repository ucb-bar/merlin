"""The installed map region contract, proved progress, and destination initialization."""

import json
import subprocess

import pytest

from merlin.verify.lean.check import check_definedness, check_support
from merlin.verify.lean.export import export_file


def map_source(shape="2x3xi32", scalar="i32", inputs=2, *, dirty=False, read_output=False,
               capture=False, shorthand=False):
    ty = f"tensor<{shape}>"
    arguments = [f"%a{i}: {ty}" for i in range(inputs)]
    if not dirty:
        arguments.append(f"%init: {ty}")
    ins = ("ins(" + ", ".join(f"%a{i}" for i in range(inputs)) + " : "
           + ", ".join([ty] * inputs) + ")") if inputs else ""
    body_args = ", ".join([f"%x{i}: {scalar}" for i in range(inputs)] + [f"%old: {scalar}"])
    value = "%captured" if capture else "%old" if read_output or not inputs else "%x0"
    prefix = f"%init = tensor.empty() : {ty}" if dirty else ""
    if capture:
        prefix += f"\n%captured = arith.constant 7 : {scalar}"
    payload = "{ arith.addi }" if shorthand else ""
    body = "" if shorthand else f"({body_args}) {{ linalg.yield {value} : {scalar} }}"
    return f"""func.func @forward({', '.join(arguments)}) -> {ty} {{
      {prefix}
      %r = linalg.map {payload} {ins} outs(%init : {ty}) {body}
      return %r : {ty}
    }}"""


@pytest.mark.parametrize("shape,scalar,inputs,dirty,read_output,capture,shorthand", [
    ("2x3xi32", "i32", 2, False, False, False, False),
    ("2x3xi32", "i32", 2, False, True, False, False),
    ("2x3xi32", "i32", 1, True, False, False, False),
    ("0x3xi32", "i32", 1, True, False, False, False),
    ("i32", "i32", 1, True, False, False, False),
    ("2xf32", "f32", 1, True, False, False, False),
    ("2xf32", "f32", 1, False, True, False, False),
    ("2xi32", "i32", 0, False, True, False, False),
    ("2xi32", "i32", 0, True, False, True, False),
    ("2xi32", "i32", 2, True, False, False, True),
])
def test_map_support_and_definedness(tmp_path, lean_tools, shape, scalar, inputs, dirty,
                                     read_output, capture, shorthand):
    source = tmp_path / "source.mlir"
    source.write_text(map_source(shape, scalar, inputs, dirty=dirty, read_output=read_output,
                                 capture=capture, shorthand=shorthand))
    for check, name in [(check_support, "support"), (check_definedness, "defined")]:
        receipt = check(source, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert receipt["verdict"] == "verified", receipt.get("reason")
        assert receipt["axioms"] == ["propext", "Classical.choice", "Quot.sound"]


def test_map_reading_dirty_destination_is_not_defined(tmp_path, lean_tools):
    source = tmp_path / "source.mlir"
    source.write_text(map_source(dirty=True, read_output=True))
    # Progress allows poison; definedness must independently refuse it.
    assert check_support(source, "forward", tmp_path / "support",
                         m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    receipt = check_definedness(source, "forward", tmp_path / "defined",
                               m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


@pytest.mark.parametrize("fault", ["legacy_body", "yield_type", "missing_output", "unknown_attribute"])
def test_map_malformed_export_is_refused(tmp_path, lean_tools, fault):
    source, exported = tmp_path / "source.mlir", tmp_path / "export.json"
    source.write_text(map_source())
    export_file(source, exported, backend="mlir", m2m_python=lean_tools.m2m_python)
    doc = json.loads(exported.read_text())
    function = doc["module"]["ops"][0]
    op = function["regions"][0]["blocks"][0]["ops"][0]
    assert op["name"] == "linalg.map"
    block = op["regions"][0]["blocks"][0]
    if fault == "legacy_body":
        block["args"].pop()
    elif fault == "yield_type":
        block["args"][0]["type"] = {"kind": "int", "width": 8}
    elif fault == "missing_output":
        op["operands"].pop()
    else:
        op["attrs"]["unknown_semantic_attribute"] = {"kind": "unit"}
    exported.write_text(json.dumps(doc))
    for command in ["check-support", "check-definedness"]:
        result = subprocess.run([str(lean_tools.lean_exe), command, str(exported), "forward"],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 1, result.stderr
        assert json.loads(result.stdout)["verdict"] == "unsupported"
