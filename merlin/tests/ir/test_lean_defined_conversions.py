"""Initialized scalar conversions are total without imposing numerical float laws."""

import subprocess

import pytest
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain

FORMATS = ["f16", "bf16", "f32", "f64"]
EXTENSIONS = [("f16", "f32"), ("f16", "f64"), ("bf16", "f32"), ("bf16", "f64"), ("f32", "f64")]
FLOAT_CASES = [("extf", a, b) for a, b in EXTENSIONS] + [("truncf", b, a) for a, b in EXTENSIONS]
INTEGER_CASES = [
    (name, f"i{width}", target) for name in ["sitofp", "uitofp"] for width in [1, 8, 13, 32, 64] for target in FORMATS
]


def program(name, source, target, *, generic=False, reduction=False, attributes="", accumulate=True):
    calculation = f'%r = "arith.{name}"(%x) {attributes} : ({source}) -> {target}'
    if not generic:
        return f"""func.func @forward(%x: {source}, %keep: i32) -> ({target}, i32) {{
          %dead = arith.addi %keep, %keep : i32
          {calculation}
          return %r, %keep : {target}, i32
        }}"""
    tensor_input = f"tensor<3x{source}>"
    tensor_output = f"tensor<{'' if reduction else '3x'}{target}>"
    outmap = "affine_map<(d0)->()>" if reduction else "affine_map<(d0)->(d0)>"
    addition = f"%sum = arith.addf %r, %old : {target}" if accumulate else ""
    return f"""func.func @forward(%input: {tensor_input}, %initial: {tensor_output}, %keep: i32)
        -> ({tensor_output}, i32) {{
      %dead = arith.addi %keep, %keep : i32
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, {outmap}],
        iterator_types = ["{"reduction" if reduction else "parallel"}"]}}
        ins(%input : {tensor_input}) outs(%initial : {tensor_output}) {{
        ^bb0(%x: {source}, %old: {target}):
          {calculation}
          {addition}
          linalg.yield %{"sum" if accumulate else "r"} : {target}
      }} -> {tensor_output}
      return %out, %keep : {tensor_output}, i32
    }}"""


def verify(path, tools):
    result = subprocess.run(
        [
            str(tools.m2m_python),
            "-c",
            "from pathlib import Path; from torch_mlir import ir; import sys\n"
            "with ir.Context():\n m=ir.Module.parse(Path(sys.argv[1]).read_text()); m.operation.verify()",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def certify(stage, before, tools, work, *, index_bits=64):
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", work / directory, index_bits=index_bits, m2m_python=tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        work / "support/receipt.json",
        "forward",
        work / "chain.json",
        definedness=work / "defined/receipt.json",
        m2m_python=tools.m2m_python,
    )
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("name,source,target", FLOAT_CASES + INTEGER_CASES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_initialized_conversion_source_is_defined(tmp_path, lean_tools, name, source, target, index_bits):
    path = tmp_path / "source.mlir"
    path.write_text(program(name, source, target))
    verify(path, lean_tools)
    result = check_definedness(
        path, "forward", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "verified", result.get("reason")


GENERIC_CASES = FLOAT_CASES + [(name, "i13", target) for name in ["sitofp", "uitofp"] for target in FORMATS]


@pytest.mark.parametrize("name,source,target", GENERIC_CASES)
@pytest.mark.parametrize("reduction", [False, True])
def test_actual_generic_conversion_cleanup_has_a_defined_chain(tmp_path, lean_tools, name, source, target, reduction):
    stage, before, after = canonicalize(
        tmp_path, lean_tools, program(name, source, target, generic=True, reduction=reduction)
    )
    assert "arith.addi" in before.read_text() and "arith.addi" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize(
    "name,source,target",
    [("extf", "f32", "f64"), ("truncf", "f64", "f32"), ("sitofp", "i8", "f32"), ("uitofp", "i8", "f32")],
)
@pytest.mark.parametrize(
    "attribute",
    [
        "fastmath = #arith.fastmath<nnan>",
        "roundingmode = 0 : i32",
        "nonNeg",
        "unreviewed.meaning",
    ],
)
def test_conversion_promises_or_unknown_attributes_refuse(tmp_path, lean_tools, name, source, target, attribute):
    path = tmp_path / "source.mlir"
    path.write_text(program(name, source, target, attributes="{" + attribute + "}"))
    verify(path, lean_tools)
    result = check_definedness(path, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize(
    "name,source,target",
    [("extf", "f32", "f64"), ("truncf", "f64", "f32"), ("sitofp", "i8", "f32"), ("uitofp", "i8", "f32")],
)
def test_poison_conversion_input_does_not_become_initialized(tmp_path, lean_tools, name, source, target):
    path = tmp_path / "source.mlir"
    path.write_text(f"""func.func @forward() -> {target} {{
      %empty = tensor.empty() : tensor<{source}>
      %x = tensor.extract %empty[] : tensor<{source}>
      %r = arith.{name} %x : {source} to {target}
      return %r : {target}
    }}""")
    verify(path, lean_tools)
    result = check_support(path, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(path, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("name", ["fptosi", "fptoui"])
def test_float_to_integer_still_requires_range_proof(tmp_path, lean_tools, name):
    path = tmp_path / "source.mlir"
    path.write_text(program(name, "f32", "i32"))
    verify(path, lean_tools)
    result = check_definedness(path, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize(
    "name,source,target",
    [("extf", "f16", "bf16"), ("truncf", "bf16", "f16"), ("sitofp", "index", "f32"), ("uitofp", "index", "f32")],
)
def test_pinned_upstream_rejects_equal_width_formats_and_direct_index(tmp_path, lean_tools, name, source, target):
    path = tmp_path / "source.mlir"
    path.write_text(program(name, source, target))
    with pytest.raises(AssertionError, match="Unable to parse module assembly"):
        verify(path, lean_tools)
