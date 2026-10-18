"""Qualify exact integer operand ordering inside actual canonicalized generic regions."""

import json
import subprocess

import pytest

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain

NAMES = ["addi", "muli", "andi", "ori", "xori"]


def _type(shape, element):
    return "tensor<" + "".join(f"{d}x" for d in shape) + element + ">"


def _source(name="addi", *, width=32, shape=(2, 3), reduction=False, read_old=False, dirty=False, escape=False):
    element = f"i{width}"
    shaped = _type(shape, element)
    output = _type(() if reduction else shape, element)
    dims = ", ".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({dims})->({dims})>"
    outmap = f"affine_map<({dims})->({'' if reduction else dims})>"
    combine = "xori" if name == "addi" else "addi"
    body = f"%v = arith.{name} %constant, %x : {element}"
    if read_old or reduction:
        body += f"\n%combined = arith.{combine} %v, %old : {element}"
    value = "%combined" if read_old or reduction else "%v"
    result_types = f"({output}, {shaped})" if escape else output
    returns = f"%r, %a : {output}, {shaped}" if escape else f"%r : {output}"
    return f"""func.func @subject(%a: {shaped}, %init: {output},
        %other: {shaped}, %capture: {element}) -> {result_types} {{
      %constant = arith.constant 7 : {element}
      {f"%empty = tensor.empty() : {output}" if dirty else ""}
      %r = linalg.generic {{indexing_maps = [{identity}, {outmap}],
          iterator_types = [{", ".join(['"reduction"' if reduction else '"parallel"'] * len(shape))}]}}
          ins(%a : {shaped}) outs({"%empty" if dirty else "%init"} : {output}) {{
        ^bb0(%x: {element}, %old: {element}):
          {body}
          linalg.yield {value} : {element}
      }} -> {output}
      return {returns}
    }}"""


def _nested(name="addi", *, reduction=False):
    # A fresh inner destination at each outer iteration avoids the retained S4 discrepancy.
    combine = "xori" if name == "addi" else "addi"
    maps = "affine_map<(d0)->(d0)>, affine_map<(d0)->()>" if reduction else "affine_map<()->()>"
    setup = "%fresh = tensor.splat %seed : tensor<i32>"
    if reduction:
        setup += "\n%data = tensor.splat %seed : tensor<2xi32>"
    return f"""func.func @subject(%a: tensor<2xi32>, %init: tensor<2xi32>, %seed: i32) -> tensor<2xi32> {{
      %constant = arith.constant 7 : i32
      %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%a : tensor<2xi32>) outs(%init : tensor<2xi32>) {{
        ^bb0(%x: i32, %old: i32):
          {setup}
          %inner = linalg.generic {{indexing_maps = [{maps}],
              iterator_types = [{'"reduction"' if reduction else ""}]}}
              {"ins(%data : tensor<2xi32>)" if reduction else ""}
              outs(%fresh : tensor<i32>) {{
            ^bb0({"%unused: i32, " if reduction else ""}%previous: i32):
              %v = arith.{name} %constant, %x : i32
              %combined = arith.{combine} %v, %previous : i32
              linalg.yield %combined : i32
          }} -> tensor<i32>
          %scalar = tensor.extract %inner[] : tensor<i32>
          linalg.yield %scalar : i32
      }} -> tensor<2xi32>
      return %r : tensor<2xi32>
    }}"""


CANONICALIZE = """
import sys
from pathlib import Path
from torch_mlir import ir
from torch_mlir.passmanager import PassManager
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    module.operation.verify()
    def walk(operation):
        yield operation
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)
    def constant(value):
        owner = value.owner
        return hasattr(owner, 'operation') and owner.operation.name == 'arith.constant'
    def ordered(left):
        return sum(op.name in ['arith.addi', 'arith.muli', 'arith.andi', 'arith.ori', 'arith.xori']
                   and len(op.operands) == 2 and constant(op.operands[0 if left else 1])
                   for op in walk(module.operation))
    assert ordered(True) > 0, 'fixture has no constant-first integer operation'
    PassManager.parse('builtin.module(canonicalize)').run(module.operation)
    module.operation.verify()
    assert ordered(False) > 0, 'actual pass did not commute the fixture operation'
    Path(sys.argv[2]).write_text(str(module))
"""


def _stage(tmp_path, source, tools):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "before.mlir").write_text(source)
    subprocess.run(
        [
            str(tools.m2m_python if hasattr(tools, "m2m_python") else tools.compiler_python),
            "-c",
            CANONICALIZE,
            str(stage / "before.mlir"),
            str(stage / "after.mlir"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "arith." in (stage / "after.mlir").read_text()
    assert (stage / "before.mlir").read_text() != (stage / "after.mlir").read_text()
    (stage / "meta.json").write_text(json.dumps({"boundary": "canonicalize-integer-body"}))
    return stage


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"width": 13, "read_old": True},
        {"width": 8, "dirty": True},
        {"width": 4},
        {"width": 64, "shape": (2, 2, 2)},
        {"shape": (), "escape": True},
        {"shape": (0, 3)},
        {"reduction": True},
    ],
)
def test_actual_integer_body_canonicalization(tmp_path, lean_tools, name, options):
    stage = _stage(tmp_path, _source(name, **options), lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("nested,reduction", [(False, False), (False, True), (True, False), (True, True)])
def test_integer_body_defined_chain(tmp_path, lean_tools, index_bits, nested, reduction):
    source = _nested(reduction=reduction) if nested else _source(reduction=reduction, read_old=True, escape=True)
    stage = _stage(tmp_path, source, lean_tools)
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(
            stage / "before.mlir", "subject", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "subject",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert result["verdict"] == "verified", result.get("reason")
    assert result["theorem"] == "MerlinVerify.Check.chain_typed_defined_refines"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    function = module.body.operations[0]
    block = function.regions[0].blocks[0]
    generic = next(op for op in block.operations if op.operation.name == 'linalg.generic')
    body = generic.regions[0].blocks[0]
    arithmetic = next(op for op in body.operations if op.operation.name == 'arith.addi')
    fault = sys.argv[2]
    if fault == 'constant':
        constant = next(op for op in block.operations if op.operation.name == 'arith.constant')
        constant.attributes['value'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 8)
    elif fault == 'capture':
        arithmetic.operation.operands[1] = block.arguments[3]
    elif fault == 'input':
        generic.operation.operands[0] = block.arguments[2]
    elif fault == 'yield':
        list(body.operations)[-1].operation.operands[0] = body.arguments[1]
    elif fault == 'escape':
        list(block.operations)[-1].operation.operands[1] = block.arguments[2]
    elif fault == 'maps':
        generic.attributes['indexing_maps'] = ir.ArrayAttr.get([
            ir.AffineMapAttr.get(ir.AffineMap.get_permutation([1, 0])),
            ir.AffineMapAttr.get(ir.AffineMap.get_permutation([0, 1]))])
    elif fault == 'flag':
        arithmetic.attributes['overflowFlags'] = ir.Attribute.parse('#arith.overflow<nsw>')
    elif fault == 'unknown':
        arithmetic.attributes['unmodelled_semantics'] = ir.StringAttr.get('changed')
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""


@pytest.mark.parametrize("fault", ["constant", "capture", "input", "yield", "escape", "maps", "flag", "unknown"])
def test_verified_integer_body_mutants_refused(tmp_path, lean_tools, fault):
    stage = _stage(tmp_path, _source(shape=(2, 2), escape=True), lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(stage / "after.mlir"), fault],
        check=True,
        capture_output=True,
        text=True,
    )
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_dirty_reduction_keeps_definedness_obligation(tmp_path, lean_tools):
    stage = _stage(tmp_path, _source(reduction=True, dirty=True), lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(stage / "before.mlir", "subject", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


NESTED_MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    block = module.body.operations[0].regions[0].blocks[0]
    outer = next(op for op in block.operations if op.operation.name == 'linalg.generic')
    middle = outer.regions[0].blocks[0]
    inner = next(op for op in middle.operations if op.operation.name == 'linalg.generic')
    body = inner.regions[0].blocks[0]
    if sys.argv[2] == 'capture':
        arithmetic = next(op for op in body.operations if op.operation.name in ['arith.addi', 'arith.muli'])
        arithmetic.operation.operands[0] = middle.arguments[1]
    else:
        list(body.operations)[-1].operation.operands[0] = body.arguments[-1]
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""


@pytest.mark.parametrize("reduction", [False, True])
@pytest.mark.parametrize("fault", ["capture", "yield"])
def test_nested_integer_body_mutants_refused(tmp_path, lean_tools, reduction, fault):
    # Additive accumulation makes both reduction mutations observable: XOR repeated twice
    # would cancel the primary value and leave a semantically equivalent corrupted target.
    stage = _stage(tmp_path, _nested("muli" if reduction else "addi", reduction=reduction), lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    subprocess.run(
        [str(lean_tools.m2m_python), "-c", NESTED_MUTATE, str(stage / "after.mlir"), fault],
        check=True,
        capture_output=True,
        text=True,
    )
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_float_body_commutation_not_assumed(tmp_path, lean_tools):
    source = _source().replace("i32", "f32").replace("arith.addi", "arith.addf").replace("constant 7", "constant 7.0")
    # The ordinary canonicalization helper deliberately requires integer commutation. This
    # separate fixture exercises the float-law boundary without pretending it is in that domain.
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "before.mlir").write_text(source)
    code = CANONICALIZE.replace("    assert ordered(True) > 0, 'fixture has no constant-first integer operation'", "")
    code = code.replace("    assert ordered(False) > 0, 'actual pass did not commute the fixture operation'", "")
    subprocess.run(
        [str(lean_tools.m2m_python), "-c", code, str(stage / "before.mlir"), str(stage / "after.mlir")],
        check=True,
        capture_output=True,
        text=True,
    )
    (stage / "meta.json").write_text(json.dumps({"boundary": "float-law-refusal"}))
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_flagged_body_commutation_requires_separate_rule(tmp_path, lean_tools):
    source = _source().replace("%constant, %x : i32", "%constant, %x overflow<nsw> : i32")
    stage = _stage(tmp_path, source, lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def _composed_source():
    source = _source()
    source = source.replace(
        "%r = linalg.generic", "%uniform = tensor.splat %capture : tensor<2x3xi32>\n%r = linalg.generic"
    )
    source = source.replace("indexing_maps = [", "indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, ", 1)
    source = source.replace("ins(%a : tensor<2x3xi32>)", "ins(%a, %uniform : tensor<2x3xi32>, tensor<2x3xi32>)")
    source = source.replace("%x: i32, %old: i32", "%x: i32, %uniform_value: i32, %old: i32")
    source = source.replace(
        "linalg.yield %v : i32", "%combined = arith.xori %v, %uniform_value : i32\nlinalg.yield %combined : i32"
    )
    return source


def _composed_stage(tmp_path, tools):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.splat_inputs import scalarize_splat_inputs
    from merlin.xdsl_dialects._common import text

    source = _composed_source()
    module = parse_mlir_text(source)
    module.verify()
    assert scalarize_splat_inputs(module) == 1
    module.verify()
    stage = _stage(tmp_path, text(module), tools)
    (stage / "compacted.mlir").write_text((stage / "before.mlir").read_text())
    (stage / "before.mlir").write_text(source)
    return stage


def test_composed_splat_and_integer_body_normalization(tmp_path, lean_tools):
    stage = _composed_stage(tmp_path, lean_tools)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(stage / "before.mlir", "subject", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
