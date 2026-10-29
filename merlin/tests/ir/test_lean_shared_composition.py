"""Independent normalizers compose across actual shared-pass rewrite families."""

import pytest
from test_lean_scalar_carrier import _mixed_source, _stage

from merlin.verify.lean import check


@pytest.mark.parametrize("index_bits", [32, 64])
def test_carrier_exposes_nested_cast_composition(tmp_path, lean_tools, index_bits):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _mixed_source())
    target = after.read_text()
    assert "tensor.from_elements" not in target
    assert "tensor.extract" not in target
    assert "arith.extsi" not in target
    assert "linalg.generic" in target
    boundary = check.check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert boundary["verdict"] == "verified", boundary.get("reason")
    for checker, label in [(check.check_support, "support"), (check.check_definedness, "defined")]:
        receipt = checker(before, "forward", tmp_path / label, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
        assert receipt["verdict"] == "verified", receipt.get("reason")
    result = check.compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert result["verdict"] == "verified", result.get("reason")
    assert result["theorem"] == check.TYPED_EXTENDED_DEFINED_CHAIN_THEOREM


def _constructor_cast_index_source(*, escape=False, cancel=False, index_first=True):
    element, middle = ("i32", "i64") if cancel else ("i8", "i16")
    final_cast = "trunci" if cancel else "extsi"
    result_types = "i32, index" + (f", tensor<3x{element}>" if escape else "")
    results = "%narrow, %sum" + (", %carrier" if escape else "")
    index = """%left = arith.constant 17 : index
      %right = arith.constant 5 : index
      %sum = arith.addi %left, %right : index"""
    selected = f"""%carrier = tensor.from_elements %a, %b, %c : tensor<3x{element}>
      %one = arith.constant 1 : index
      %selected = tensor.extract %carrier[%one] : tensor<3x{element}>
      %wide = arith.extsi %selected : {element} to {middle}
      %narrow = arith.{final_cast} %wide : {middle} to i32"""
    statements = index + "\n" + selected if index_first else selected + "\n" + index
    return f"""func.func @forward(%a: {element}, %b: {element}, %c: {element}) -> ({result_types}) {{
      {statements}
      return {results} : {result_types}
    }}"""


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_constructor_selection_composes_with_casts_and_index_folding(tmp_path, lean_tools, index_bits, escape):
    from test_lean_defined_conversions import certify
    from test_lean_integer_fold import canonicalize

    stage, before, after = canonicalize(tmp_path, lean_tools, _constructor_cast_index_source(escape=escape))
    target = after.read_text()
    for operation in ("tensor.extract", "arith.trunci", "arith.addi"):
        assert operation not in target
    assert target.count("arith.extsi") == 1
    assert ("tensor.from_elements" in target) == escape
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("fault", ["selected-scalar", "index-result"])
def test_mixed_family_mutations_remain_defined_but_refuse_refinement(tmp_path, lean_tools, fault):
    from test_lean_defined_conversions import certify, verify
    from test_lean_integer_fold import canonicalize

    stage, before, after = canonicalize(tmp_path, lean_tools, _constructor_cast_index_source())
    certify(stage, before, lean_tools, tmp_path)
    import subprocess

    mutate = """
from pathlib import Path
import sys
from torch_mlir import ir
with ir.Context():
    path=Path(sys.argv[1]); module=ir.Module.parse(path.read_text())
    block=module.body.operations[0].regions[0].blocks[0]
    if sys.argv[2]=='selected-scalar':
        cast=next(o for o in block.operations if o.name=='arith.extsi')
        cast.operation.operands[0]=block.arguments[0]
    else:
        constant=next(o for o in block.operations if o.name=='arith.constant')
        attr=ir.IntegerAttr(constant.attributes['value'])
        constant.operation.attributes['value']=ir.IntegerAttr.get(attr.type,attr.value+1)
    module.operation.verify(); path.write_text(str(module))
"""
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", mutate, str(after), fault], capture_output=True, text=True
    )
    assert changed.returncode == 0, changed.stderr
    verify(after, lean_tools)
    defined = check.check_definedness(after, "forward", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python)
    assert defined["verdict"] == "verified", defined.get("reason")
    refused = check.check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert refused["verdict"] == "abstracted", refused.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_mixed_widen_truncate_roundtrip_has_strict_chain(tmp_path, lean_tools, index_bits):
    from test_lean_integer_fold import canonicalize

    stage, before, after = canonicalize(tmp_path, lean_tools, _constructor_cast_index_source(cancel=True))
    assert "arith.extsi" not in after.read_text() and "arith.trunci" not in after.read_text()
    for checker, name in [(check.check_support, "support"), (check.check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
        assert result["verdict"] == "verified", result.get("reason")
    accepted = check.check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert accepted["verdict"] == "verified", accepted.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_mixed_independent_constant_hoisting_has_strict_chain(tmp_path, lean_tools, index_bits):
    from test_lean_integer_fold import canonicalize

    stage, before, after = canonicalize(tmp_path, lean_tools, _constructor_cast_index_source(index_first=False))
    assert after.read_text().index("arith.constant") < after.read_text().index("arith.extsi")
    for checker, name in [(check.check_support, "support"), (check.check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
        assert result["verdict"] == "verified", result.get("reason")
    accepted = check.check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert accepted["verdict"] == "verified", accepted.get("reason")
