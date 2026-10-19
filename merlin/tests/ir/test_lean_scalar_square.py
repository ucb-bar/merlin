"""Real scalar-square boundaries abstain until a reviewed float law exists.

These are scope/refusal regressions, not numerical or paper-model qualification.
"""

import json
from io import StringIO

import pytest
from test_lean_defined_conversions import verify
from xdsl.dialects import arith, math
from xdsl.printer import Printer

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.scalar_square import canonicalize_scalar_squares
from merlin.verify.lean.check import check_typed_stage


def source(ty, exponent="2.0"):
    definition = "" if exponent == "dynamic" else f"%exp = arith.constant {exponent} : {ty}"
    rhs = "%y" if exponent == "dynamic" else "%exp"
    return f"""module {{
      func.func @forward(%x: {ty}, %y: {ty}) -> {ty} {{
        {definition}
        %r = math.powf %x, {rhs} : {ty}
        return %r : {ty}
      }}
    }}"""


def render(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def boundary(tmp_path, ty, exponent="2.0", mutation=None):
    module = parse_mlir_text(source(ty, exponent))
    before = render(module)
    count = canonicalize_scalar_squares(module)
    if mutation:
        multiply = next(op for op in module.walk() if isinstance(op, arith.MulfOp))
        block = multiply.parent
        x, y = block.args
        if mutation == "mixed-operand":
            multiply.operands = (x, y)
        elif mutation == "other-square":
            multiply.operands = (y, y)
        elif mutation == "add-instead":
            replacement = arith.AddfOp(x, x)
            block.insert_op_before(replacement, multiply)
            multiply.result.replace_all_uses_with(replacement.result)
            block.erase_op(multiply)
        elif mutation == "return-input":
            multiply.result.replace_all_uses_with(x)
        elif mutation == "return-other":
            multiply.result.replace_all_uses_with(y)
        else:
            raise AssertionError(mutation)
    module.verify()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "before.mlir").write_text(before)
    (stage / "after.mlir").write_text(render(module))
    (stage / "meta.json").write_text(json.dumps({"boundary": "prepare-scalar-squares"}))
    return module, count, stage


@pytest.mark.parametrize("ty", ["f32", "f64"])
@pytest.mark.parametrize("exponent", ["2.0", "3.0", "dynamic"])
def test_actual_pass_only_rewrites_constant_scalar_square(tmp_path, ty, exponent):
    module, count, stage = boundary(tmp_path, ty, exponent)
    assert count == (exponent == "2.0")
    powers = [op for op in module.walk() if isinstance(op, math.PowFOp)]
    multiplies = [op for op in module.walk() if isinstance(op, arith.MulfOp)]
    assert len(powers) == (exponent != "2.0")
    assert len(multiplies) == (exponent == "2.0")
    if multiplies:
        assert multiplies[0].lhs is multiplies[0].rhs
    else:
        assert (stage / "before.mlir").read_bytes() == (stage / "after.mlir").read_bytes()


@pytest.mark.parametrize("ty", ["f32", "f64"])
@pytest.mark.parametrize(
    "mutation", [None, "mixed-operand", "other-square", "add-instead", "return-input", "return-other"]
)
def test_square_and_valid_corruptions_are_unverified_without_float_law(tmp_path, lean_tools, ty, mutation):
    _, count, stage = boundary(tmp_path, ty, mutation=mutation)
    assert count == 1
    assert (stage / "before.mlir").read_bytes() != (stage / "after.mlir").read_bytes()
    verify(stage / "before.mlir", lean_tools)
    verify(stage / "after.mlir", lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result
