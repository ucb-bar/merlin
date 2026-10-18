"""Check input-dependent cast-chain composition against actual upstream canonicalization."""

import subprocess

import pytest
from test_lean_integer_fold import _verify_mlir, canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain

FAMILIES = [("extui", "extui"), ("extsi", "extsi"), ("extui", "extsi"), ("trunci", "trunci")]
EXT_WIDTHS = [(1, 2, 8), (1, 8, 13), (8, 13, 32), (13, 32, 64), (32, 64, 128)]
TRUNC_WIDTHS = [(8, 2, 1), (13, 8, 1), (32, 13, 8), (64, 32, 13), (128, 64, 32)]


def cast_source(first, second, widths=(1, 8, 32), *, escape=False):
    a, b, c = widths
    types = [f"i{c}", f"i{a}"] + ([f"i{b}"] if escape else [])
    returned = ["%r", "%keep"] + (["%mid"] if escape else [])
    return f"""func.func @forward(%x: i{a}, %keep: i{a}) -> ({", ".join(types)}) {{
      %mid = arith.{first} %x : i{a} to i{b}
      %r = arith.{second} %mid : i{b} to i{c}
      return {", ".join(returned)} : {", ".join(types)}
    }}"""


CASES = [
    (first, second, widths)
    for first, second in FAMILIES
    for widths in (TRUNC_WIDTHS if first == "trunci" else EXT_WIDTHS)
]


@pytest.mark.parametrize("first,second,widths", CASES)
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_cast_chain(tmp_path, lean_tools, first, second, widths, escape, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, cast_source(first, second, widths, escape=escape))
    assert after.read_text().count("arith.") == 1 + int(escape)
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    for check, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = check(before, "forward", tmp_path / directory, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert result["verdict"] == "verified", result.get("reason")


def long_source(*, escape=False):
    body = []
    for width in range(2, 65):
        previous = "%x" if width == 2 else f"%v{width - 1}"
        # Alternating zero and signed extensions remain zero extensions from the i1 root.
        kind = "extui" if width % 2 == 0 else "extsi"
        body.append(f"%v{width} = arith.{kind} {previous} : i{width - 1} to i{width}")
    types = ["i64", "i1"] + (["i13", "i32"] if escape else [])
    returned = ["%v64", "%keep"] + (["%v13", "%v32"] if escape else [])
    return (
        f"func.func @forward(%x: i1, %keep: i1) -> ({', '.join(types)}) {{\n"
        + "\n".join(body)
        + f"\nreturn {', '.join(returned)} : {', '.join(types)}\n}}"
    )


@pytest.mark.parametrize("escape", [False, True])
def test_long_chain_retains_actual_escaping_intermediates(tmp_path, lean_tools, escape):
    stage, _, after = canonicalize(tmp_path, lean_tools, long_source(escape=escape))
    assert after.read_text().count("arith.") == 1 + 2 * int(escape)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    path, fault = Path(sys.argv[1]), sys.argv[2]
    module = ir.Module.parse(path.read_text())
    block = module.body.operations[0].regions[0].blocks[0]
    casts = [o for o in block.operations if o.name.startswith('arith.')]
    if fault == 'input':
        casts[-1].operation.operands[0] = block.arguments[1]
    elif fault == 'escaping_input':
        casts[0].operation.operands[0] = block.arguments[1]
    elif fault == 'attribute':
        casts[-1].operation.attributes['unknown.numeric_meaning'] = ir.UnitAttr.get()
    elif fault == 'nonneg':
        casts[0].operation.attributes['nonNeg'] = ir.UnitAttr.get()
    elif fault == 'overflow':
        pass  # Text form below preserves the registered enum property's actual syntax.
    elif fault == 'signedness':
        pass
    else:
        raise ValueError(fault)
    text = str(module)
    if fault == 'signedness':
        text = text.replace('arith.extui', 'arith.extsi', 1)
    elif fault == 'overflow':
        text = text.replace('arith.trunci %arg0', 'arith.trunci %arg0 overflow<nsw>', 1)
    module = ir.Module.parse(text)
    module.operation.verify()
    path.write_text(str(module))
"""


def mutate(path, tools, fault):
    original = path.read_bytes()
    result = subprocess.run(
        [str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() != original
    _verify_mlir(path, tools)


@pytest.mark.parametrize("fault", ["input", "escaping_input", "attribute", "signedness", "nonneg"])
def test_cast_rejects_independently_mutated_verified_target(tmp_path, lean_tools, fault):
    stage, _, after = canonicalize(tmp_path, lean_tools, cast_source("extui", "extui", escape=True))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("fault", ["input", "attribute", "signedness", "nonneg", "overflow"])
def test_cast_source_numerics_are_rechecked(tmp_path, lean_tools, fault):
    first = "trunci" if fault == "overflow" else "extui"
    widths = (64, 13, 1) if first == "trunci" else (1, 8, 32)
    stage, before, _ = canonicalize(tmp_path, lean_tools, cast_source(first, first, widths, escape=True))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    mutate(before, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    if fault in {"nonneg", "overflow"}:
        result = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
        result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("widths", [(1, 8, 32), (8, 13, 64)])
def test_sign_then_zero_is_not_flattened(tmp_path, lean_tools, widths):
    source = cast_source("extsi", "extui", widths)
    stage, _, after = canonicalize(tmp_path, lean_tools, source)
    assert after.read_text().count("arith.") == 2
    a, b, c = widths
    after.write_text(
        f"func.func @forward(%x: i{a}, %keep: i{a}) -> (i{c}, i{a}) {{\n"
        f"%r = arith.extsi %x : i{a} to i{c}\nreturn %r, %keep : i{c}, i{a}\n}}"
    )
    _verify_mlir(after, lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_tensor_casts_need_their_own_runtime_domain(tmp_path, lean_tools):
    source = """func.func @forward(%x: tensor<3xi1>) -> tensor<3xi32> {
      %a = arith.extui %x : tensor<3xi1> to tensor<3xi8>
      %r = arith.extui %a : tensor<3xi8> to tensor<3xi32>
      return %r : tensor<3xi32>
    }"""
    stage, before, after = canonicalize(tmp_path, lean_tools, source)
    assert after.read_text().count("arith.") == 1
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def computed_input_source():
    return """func.func @forward(%x: i8, %keep: i8) -> (i32, i13, i8) {
      %input = arith.xori %x, %keep : i8
      %mid = arith.extui %input : i8 to i13
      %r = arith.extsi %mid : i13 to i32
      return %r, %mid, %keep : i32, i13, i8
    }"""


def test_cast_input_type_comes_from_actual_prior_execution(tmp_path, lean_tools):
    from merlin.verify.lean.check import check_stage

    stage, _, after = canonicalize(tmp_path, lean_tools, computed_input_source())
    assert "arith.extsi" not in after.read_text()
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


def test_cast_rejects_inconsistent_declared_input_width(tmp_path, lean_tools):
    stage, before, _ = canonicalize(tmp_path, lean_tools, cast_source("extui", "extui", escape=True))
    before.write_text(before.read_text().replace("i1", "i2"))
    _verify_mlir(before, lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
