"""Actual widening/partial-truncation boundaries retain typed origins and escaped values."""

import pytest
from test_lean_integer_cast import cast_source, mutate
from test_lean_integer_cast_nested import certify, nested_source
from test_lean_integer_cast_nested import source as body_source
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify as certify_conditional

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

WIDTHS = [(1, 32, 8), (8, 32, 13), (13, 64, 32), (32, 128, 64), (8, 32, 1), (13, 32, 8), (64, 128, 13)]


def source(kind="extui", widths=(8, 32, 13), *, escape=False):
    return cast_source(kind, "trunci", widths, escape=escape)


def captured_source(kind="extui", widths=(8, 32, 13)):
    a, _, c = widths
    return (
        body_source(kind, "trunci", widths)
        .replace("%keep: i32)", f"%keep: i32, %capture: i{a})")
        .replace(f"arith.{kind} %x", f"arith.{kind} %capture")
        .replace(
            f"%sum = arith.addi %wide, %old : i{c}",
            f"%inputwide = arith.{kind if a < c else 'trunci'} %x : i{a} to i{c}\n"
            f"%sum0 = arith.addi %wide, %inputwide : i{c}\n"
            f"%sum = arith.addi %sum0, %old : i{c}",
        )
    )


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("widths", WIDTHS)
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_partial_casts_have_defined_chain(tmp_path, lean_tools, kind, widths, escape, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(kind, widths, escape=escape))
    assert sum(after.read_text().count(f"arith.{name}") for name in ["extui", "extsi", "trunci"]) == 1 + int(escape)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("widths", [(8, 32, 13), (13, 32, 8)])
@pytest.mark.parametrize("escape,reduction", [(False, False), (True, False), (False, True), (True, True)])
def test_partial_casts_inside_generic_keep_escaped_widening(tmp_path, lean_tools, kind, widths, escape, reduction):
    stage, before, after = canonicalize(
        tmp_path, lean_tools, body_source(kind, "trunci", widths, escape=escape, reduction=reduction)
    )
    assert "linalg.generic" in after.read_text()
    assert sum(after.read_text().count(f"arith.{name}") for name in ["extui", "extsi", "trunci"]) == 1 + int(escape)
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("widths", [(8, 32, 13), (13, 32, 8)])
def test_partial_casts_inside_nested_generic(tmp_path, lean_tools, kind, widths):
    stage, before, after = canonicalize(tmp_path, lean_tools, nested_source(kind, "trunci", widths))
    assert after.read_text().count("linalg.generic") == 2
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_local_origin_may_use_actual_captured_scalar(tmp_path, lean_tools, kind, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, captured_source(kind))
    assert after.read_text().count(f"arith.{kind}") == 2
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


def poison_source(kind):
    return f"""func.func @forward(%keep: i8) -> (i13, i8) {{
      %dirty = tensor.empty() : tensor<i8>
      %x = tensor.extract %dirty[] : tensor<i8>
      %mid = arith.{kind} %x : i8 to i32
      %r = arith.trunci %mid : i32 to i13
      return %r, %keep : i13, i8
    }}"""


@pytest.mark.parametrize("kind", ["extui", "extsi"])
def test_actual_poison_origin_does_not_gain_definedness(tmp_path, lean_tools, kind):
    stage, before, _ = canonicalize(tmp_path, lean_tools, poison_source(kind))
    certify_conditional(stage, before, lean_tools, tmp_path, defined=False)


@pytest.mark.parametrize(
    "side,fault",
    [
        ("target", "input"),
        ("target", "escaping_input"),
        ("target", "signedness"),
        ("target", "attribute"),
        ("source", "signedness"),
        ("source", "nonneg"),
    ],
)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_independently_verified_partial_cast_mutations_refuse(tmp_path, lean_tools, side, fault, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(escape=fault == "escaping_input"))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(before if side == "source" else after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    if fault == "nonneg":
        result = check_support(before, "forward", tmp_path / "mutant-support", m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
        result = check_definedness(before, "forward", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "abstracted", result.get("reason")
