"""Qualify initialized raw-bit casts and conservatively retain unsupported float-to-float cases."""

import pytest
from test_lean_defined_conversions import certify, verify
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

PAIRS = [(f"i{w}", f"i{w}") for w in [1, 8, 13, 16, 32, 64]] + [
    pair
    for integer, floating in [("i16", "f16"), ("i16", "bf16"), ("i32", "f32"), ("i64", "f64")]
    for pair in [(integer, floating), (floating, integer)]
]
FLOAT_PAIRS = [(x, x) for x in ["f16", "bf16", "f32", "f64"]] + [("f16", "bf16"), ("bf16", "f16")]


def program(source, target, *, generic=False, attributes=""):
    calculation = f'%r = "arith.bitcast"(%x) {attributes} : ({source}) -> {target}'
    if not generic:
        return f"""func.func @forward(%x: {source}, %other: {source}, %keep: i32) -> ({target}, i32) {{
          %dead = arith.addi %keep, %keep : i32
          {calculation}
          return %r, %keep : {target}, i32
        }}"""
    return f"""func.func @forward(%input: tensor<3x{source}>, %initial: tensor<3x{target}>, %keep: i32)
        -> (tensor<3x{target}>, i32) {{
      %dead = arith.addi %keep, %keep : i32
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%input : tensor<3x{source}>) outs(%initial : tensor<3x{target}>) {{
        ^bb0(%x: {source}, %old: {target}):
          {calculation}
          linalg.yield %r : {target}
      }} -> tensor<3x{target}>
      return %out, %keep : tensor<3x{target}>, i32
    }}"""


def source_proofs(path, tools, work, *, verdict="verified", index_bits=64):
    verify(path, tools)
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(path, "forward", work / name, index_bits=index_bits, m2m_python=tools.m2m_python)
        assert result["verdict"] == verdict, result.get("reason")


@pytest.mark.parametrize("source,target", PAIRS)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_supported_bitcast_source_is_initialized(tmp_path, lean_tools, source, target, index_bits):
    path = tmp_path / "source.mlir"
    path.write_text(program(source, target))
    source_proofs(path, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("source,target", [pair for pair in PAIRS if pair[0] != pair[1]])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_generic_bitcast_cleanup_has_defined_chain(tmp_path, lean_tools, source, target, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, program(source, target, generic=True))
    assert "arith.addi" in before.read_text() and "arith.addi" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("source,target", [("i32", "f32"), ("f32", "i32")])
@pytest.mark.parametrize(
    "attribute",
    [
        "nonNeg",
        "fastmath = #arith.fastmath<nnan>",
        "overflowFlags = #arith.overflow<nsw>",
        "roundingmode = 0 : i32",
        "unreviewed.meaning",
    ],
)
def test_bitcast_promises_or_unknown_attributes_refuse(tmp_path, lean_tools, source, target, attribute):
    path = tmp_path / "source.mlir"
    path.write_text(program(source, target, attributes="{" + attribute + "}"))
    source_proofs(path, lean_tools, tmp_path, verdict="abstracted")


@pytest.mark.parametrize("source,target", [("i32", "f32"), ("f32", "i32")])
def test_poison_bitcast_input_cannot_supply_initialized_output(tmp_path, lean_tools, source, target):
    path = tmp_path / "source.mlir"
    path.write_text(f"""func.func @forward() -> {target} {{
      %empty = tensor.empty() : tensor<{source}>
      %x = tensor.extract %empty[] : tensor<{source}>
      %r = arith.bitcast %x : {source} to {target}
      return %r : {target}
    }}""")
    verify(path, lean_tools)
    result = check_support(path, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(path, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("dirty_input", [False, True])
def test_bitcast_overwrite_checks_the_actual_input_initialization(tmp_path, lean_tools, dirty_input):
    text = program("i32", "f32", generic=True)
    ty = "tensor<3xi32>" if dirty_input else "tensor<3xf32>"
    text = text.replace("%out = linalg.generic", f"%dirty = tensor.empty() : {ty}\n%out = linalg.generic")
    text = text.replace(
        "ins(%input :" if dirty_input else "outs(%initial :", "ins(%dirty :" if dirty_input else "outs(%dirty :"
    )
    path = tmp_path / "source.mlir"
    path.write_text(text)
    verify(path, lean_tools)
    result = check_definedness(path, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == ("abstracted" if dirty_input else "verified"), result.get("reason")


@pytest.mark.parametrize("source,target", FLOAT_PAIRS)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_legal_float_to_float_bitcast_remains_a_support_gap(tmp_path, lean_tools, source, target, index_bits):
    path = tmp_path / "source.mlir"
    path.write_text(program(source, target))
    source_proofs(path, lean_tools, tmp_path, verdict="abstracted", index_bits=index_bits)


def test_initialized_integer_identity_rewrite_is_not_overclaimed(tmp_path, lean_tools):
    stage, before, after = canonicalize(tmp_path, lean_tools, program("i13", "i13"))
    assert "arith.bitcast" not in after.read_text()
    source_proofs(before, lean_tools, tmp_path)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
