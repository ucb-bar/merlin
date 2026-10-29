"""Scalar float totality under arbitrary models, without fast-math promises or algebra laws."""

import pytest
from test_lean_definedness import _check
from test_lean_layout_generalize import Case, _source

from merlin.verify.lean.check import check_definedness


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("element", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("name", ["addf", "subf", "mulf", "divf", "remf", "maximumf", "minimumf", "maxnumf", "minnumf"])
def test_scalar_float_arithmetic_is_defined(tmp_path, lean_tools, index_bits, element, name):
    result = _check(
        tmp_path,
        lean_tools,
        f"""func.func @forward(%a: {element}, %b: {element}) -> {element} {{
          %r = arith.{name} %a, %b : {element}
          return %r : {element}
        }}""",
        index_bits=index_bits,
    )
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("name", ["addf", "mulf", "divf"])
@pytest.mark.parametrize("flags", ["nnan", "ninf", "fast"])
def test_fastmath_promises_remain_unqualified(tmp_path, lean_tools, name, flags):
    result = _check(
        tmp_path,
        lean_tools,
        f"""func.func @forward(%a: f32, %b: f32) -> f32 {{
          %r = arith.{name} %a, %b fastmath<{flags}> : f32
          return %r : f32
        }}""",
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("dirty_input", [False, True])
def test_float_reduction_does_not_initialize_poison(tmp_path, lean_tools, dirty_input):
    program = tmp_path / "source.mlir"
    # A dirty accumulator is read by addf; a dirty input likewise cannot produce clean output.
    program.write_text(_source(Case("reduce", (2, 3), (1,), "f32"), dirty=not dirty_input, dirty_input=dirty_input))
    receipt = check_definedness(program, "forward", tmp_path / "proof", m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_float_to_integer_conversion_still_needs_range_obligation(tmp_path, lean_tools):
    result = _check(
        tmp_path,
        lean_tools,
        """func.func @forward(%a: f32) -> i32 {
          %r = arith.fptosi %a : f32 to i32
          return %r : i32
        }""",
    )
    assert result["verdict"] == "abstracted", result.get("reason")
