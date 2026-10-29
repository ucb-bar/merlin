"""Source totality for scalar math operations, without floating algebra assumptions."""

import pytest
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain

UNARY = [
    "arith.negf",
    "math.absf",
    "math.sqrt",
    "math.rsqrt",
    "math.exp",
    "math.exp2",
    "math.expm1",
    "math.log",
    "math.log2",
    "math.log10",
    "math.log1p",
    "math.tanh",
    "math.erf",
    "math.sin",
    "math.cos",
    "math.ceil",
    "math.floor",
    "math.round",
    "math.roundeven",
    "math.trunc",
]


def program(name, element="f32", *, generic=False, attributes=""):
    operands = "%x, %x, %x" if name == "math.fma" else "%x"
    argument_types = ", ".join([element] * (3 if name == "math.fma" else 1))
    calculation = f'%r = "{name}"({operands}) {attributes} : ({argument_types}) -> {element}'
    if not generic:
        return f"""func.func @forward(%x: {element}, %keep: i32) -> ({element}, i32) {{
          %dead = arith.addi %keep, %keep : i32
          {calculation}
          return %r, %keep : {element}, i32
        }}"""
    tensor = f"tensor<2x{element}>"
    return f"""func.func @forward(%input: {tensor}, %initial: {tensor}, %keep: i32) -> ({tensor}, i32) {{
      %dead = arith.addi %keep, %keep : i32
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}} ins(%input : {tensor}) outs(%initial : {tensor}) {{
        ^bb0(%x: {element}, %old: {element}):
          {calculation}
          %sum = arith.addf %r, %old : {element}
          linalg.yield %sum : {element}
      }} -> {tensor}
      return %out, %keep : {tensor}, i32
    }}"""


@pytest.mark.parametrize("name", [*UNARY, "math.fma"])
@pytest.mark.parametrize("element", ["f16", "bf16", "f32", "f64"])
def test_initialized_floating_source_is_defined(tmp_path, lean_tools, name, element):
    source = tmp_path / "source.mlir"
    source.write_text(program(name, element))
    result = check_definedness(source, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("name", ["arith.negf", "math.exp", "math.sqrt", "math.roundeven", "math.fma"])
@pytest.mark.parametrize("element", ["f32", "f64"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_generic_cleanup_has_a_defined_chain(tmp_path, lean_tools, name, element, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, program(name, element, generic=True))
    assert "arith.addi" in before.read_text() and "arith.addi" not in after.read_text()
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(
            before, "forward", tmp_path / directory, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
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


@pytest.mark.parametrize("name", ["arith.negf", "math.sqrt", "math.fma"])
@pytest.mark.parametrize("attribute", ["fastmath = #arith.fastmath<nnan>", "unreviewed.meaning"])
def test_floating_promises_and_unknown_attributes_refuse(tmp_path, lean_tools, name, attribute):
    source = tmp_path / "source.mlir"
    source.write_text(program(name, attributes="{" + attribute + "}"))
    result = check_definedness(source, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_poison_float_capture_does_not_gain_initialized_facts(tmp_path, lean_tools):
    source = tmp_path / "source.mlir"
    source.write_text("""func.func @forward() -> f32 {
      %empty = tensor.empty() : tensor<f32>
      %poison = tensor.extract %empty[] : tensor<f32>
      %r = math.sqrt %poison : f32
      return %r : f32
    }""")
    result = check_definedness(source, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
