"""Actual identity-generic elimination is checked only at the function return boundary."""

import subprocess

import pytest
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def source(shape=(2, 2), element="i13", *, prefix=False):
    ty = f"tensor<{''.join(f'{n}x' for n in shape)}{element}>"
    dims = ",".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({dims})->({dims})>"
    iters = ", ".join('"parallel"' for _ in shape)
    return f"""func.func @forward(%data: {ty}, %initial: {ty}, %keep: i32)
        -> ({ty}, i32, {ty}, {ty}, {ty}) {{
      {"%one = arith.constant 1 : i32" if prefix else ""}
      {"%live = arith.addi %keep, %one : i32" if prefix else ""}
      %result = linalg.generic {{indexing_maps = [{identity}, {identity}], iterator_types = [{iters}]}}
          ins(%data : {ty}) outs(%initial : {ty}) {{
        ^bb0(%x: {element}, %old: {element}):
          linalg.yield %x : {element}
      }} -> {ty}
      return %result, {"%live" if prefix else "%keep"}, %data, %initial, %result
          : {ty}, i32, {ty}, {ty}, {ty}
    }}"""


@pytest.mark.parametrize("shape", [(), (0,), (1,), (2, 3), (1, 2, 1)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_identity_return_has_strict_defined_chain(tmp_path, lean_tools, shape, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(shape))
    assert before.read_text().count("linalg.generic") == 1 and "linalg.generic" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("element", ["i1", "i65", "f32", "f64", "index"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_identity_return_uses_exact_declared_element_domain(tmp_path, lean_tools, element, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(element=element))
    assert "linalg.generic" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_retained_live_prefix_remains_outside_whole_block_comparison(tmp_path, lean_tools, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(prefix=True))
    assert "arith.addi" in before.read_text() and "arith.addi" in after.read_text()
    assert "linalg.generic" not in after.read_text()
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(
            before, "forward", tmp_path / directory, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def mutate(path, tools, fault):
    script = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    m = ir.Module.parse(Path(sys.argv[1]).read_text())
    def walk(op):
        yield op
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)
    ops = list(walk(m.operation))
    if sys.argv[2] == 'return':
        fn = next(o for o in ops if o.name == 'func.func')
        ret = next(o for o in ops if o.name == 'func.return')
        ret.operands[0] = fn.regions[0].blocks[0].arguments[1]
    elif sys.argv[2] == 'yield':
        generic = next(o for o in ops if o.name == 'linalg.generic')
        term = next(o for o in ops if o.name == 'linalg.yield')
        term.operands[0] = generic.regions[0].blocks[0].arguments[1]
    elif sys.argv[2] == 'transpose':
        generic = next(o for o in ops if o.name == 'linalg.generic')
        generic.attributes['indexing_maps'] = ir.ArrayAttr.get([
            ir.AffineMapAttr.get(ir.AffineMap.get_permutation([1, 0])),
            ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1]))])
    else:
        name = 'func.return' if sys.argv[2] == 'return-attribute' else 'linalg.generic'
        op = next(o for o in ops if o.name == name)
        op.attributes['unmodeled_meaning'] = ir.UnitAttr.get()
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""
    result = subprocess.run(
        [str(tools.m2m_python), "-c", script, str(path), fault], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["return", "yield", "transpose", "attribute", "return-attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_mlir_verified_wrong_values_or_unknown_headers_refuse(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source())
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after if fault in {"return", "return-attribute"} else before, lean_tools, fault)
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
