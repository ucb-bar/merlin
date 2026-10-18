"""Typed local reshape chains inside structured regions, with checked dead producers."""

import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize, _type

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _reduce(value, shape, name):
    dims = ", ".join(f"d{i}" for i in range(len(shape)))
    iterators = ", ".join('"reduction"' for _ in shape)
    tensor = _type(shape, "i32")
    return f"""%{name}_init = tensor.splat %seed : tensor<i32>
      %{name} = linalg.generic {{indexing_maps = [affine_map<({dims}) -> ({dims})>, affine_map<({dims}) -> ()>],
          iterator_types = [{iterators}]}}
        ins({value} : {tensor}) outs(%{name}_init : tensor<i32>) {{
        ^bb0(%{name}_x: i32, %{name}_acc: i32):
          %{name}_add = arith.addi %{name}_x, %{name}_acc : i32
          linalg.yield %{name}_add : i32
      }} -> tensor<i32>
      %{name}_scalar = tensor.extract %{name}[] : tensor<i32>
    """


def _source(depth=1, *, escape=False, cycles=2):
    operations = ["%s0 = tensor.collapse_shape %a [[0,1],[2,3]] : tensor<2x3x2x2xi32> into tensor<6x4xi32>"]
    current = "%s0"
    for n in range(cycles):
        operations += [
            f"%e{n} = tensor.expand_shape {current} [[0,1],[2]] output_shape [2,3,4] "
            ": tensor<6x4xi32> into tensor<2x3x4xi32>",
            f"%s{n + 1} = tensor.collapse_shape %e{n} [[0,1],[2]] : tensor<2x3x4xi32> into tensor<6x4xi32>",
        ]
        current = f"%s{n + 1}"
    operations += [f"%out = tensor.collapse_shape {current} [[0,1]] : tensor<6x4xi32> into tensor<24xi32>"]
    body = "\n".join(operations) + _reduce("%out", (24,), "sum")
    scalar = "%sum_scalar"
    if escape:
        body += _reduce("%s0", (6, 4), "capture")
        body += "%both = arith.addi %sum_scalar, %capture_scalar : i32\n"
        scalar = "%both"
    body += f"%adjusted = arith.addi {scalar}, %payload_value : i32\n"
    scalar = "%adjusted"
    for n in range(depth - 1):
        body = f"""%wrap_init{n} = tensor.splat %seed : tensor<i32>
          %wrap{n} = linalg.generic {{indexing_maps = [affine_map<() -> ()>], iterator_types = []}}
            outs(%wrap_init{n} : tensor<i32>) {{
            ^bb0(%wrap_old{n}: i32):
              {body}
              linalg.yield {scalar} : i32
          }} -> tensor<i32>
          %wrap_scalar{n} = tensor.extract %wrap{n}[] : tensor<i32>
        """
        scalar = f"%wrap_scalar{n}"
    return f"""func.func @forward(%a: tensor<2x3x2x2xi32>, %payload: tensor<2xi32>,
        %other: tensor<2x3x2x2xi32>, %seed: i32, %other_mid: tensor<6x4xi32>)
        -> (tensor<2xi32>, tensor<2x3x2x2xi32>) {{
      %init = tensor.splat %seed : tensor<2xi32>
      %result = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}} ins(%payload : tensor<2xi32>) outs(%init : tensor<2xi32>) {{
        ^bb0(%payload_value: i32, %old: i32):
          {body}
          linalg.yield {scalar} : i32
      }} -> tensor<2xi32>
      return %result, %a : tensor<2xi32>, tensor<2x3x2x2xi32>
    }}"""


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("escape", [False, True])
def test_actual_nested_reshape_chain(tmp_path, lean_tools, depth, escape):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(depth, escape=escape))
    assert "tensor.expand_shape" in before.read_text()
    assert "tensor.expand_shape" not in after.read_text()
    assert after.read_text().count("tensor.collapse_shape") == 1 + int(escape)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_nested_reshape_strict_chain(tmp_path, lean_tools, index_bits, escape):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(2, escape=escape))
    checked = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")
    checked = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert checked["verdict"] == "verified", checked.get("reason")
    assert checked["theorem"] == "MerlinVerify.Check.chain_typed_extended_defined_refines"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    m=ir.Module.parse(Path(sys.argv[1]).read_text())
    function=m.body.operations[0]
    block=function.regions[0].blocks[0]
    all_ops=[]
    def walk(b):
        for op in b.operations:
            all_ops.append(op)
            for region in op.regions:
                for nested in region.blocks:
                    walk(nested)
    walk(block)
    if sys.argv[2]=='root':
        reshape=next(op for op in all_ops if op.operation.name=='tensor.collapse_shape')
        reshape.operation.operands[0]=block.arguments[2]
    elif sys.argv[2]=='capture':
        generic=next(op for op in all_ops if op.operation.name=='linalg.generic' and
          len(op.operands)==2 and str(op.operands[0].type)=='tensor<6x4xi32>')
        generic.operation.operands[0]=block.arguments[4]
    elif sys.argv[2]=='yield':
        generic=next(op for op in block.operations if op.operation.name=='linalg.generic')
        body=generic.regions[0].blocks[0]
        list(body.operations)[-1].operation.operands[0]=body.arguments[0]
    else:
        reshape=next(op for op in all_ops if op.operation.name=='tensor.collapse_shape')
        reshape.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""


@pytest.mark.parametrize("fault", ["root", "capture", "yield", "attribute"])
def test_nested_reshape_valid_mlir_mutations(tmp_path, lean_tools, fault):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(2, escape=True))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def test_outer_region_provenance_is_not_assumed_inside_body(tmp_path, lean_tools):
    source = _source(1, cycles=1)
    producer = "%s0 = tensor.collapse_shape %a [[0,1],[2,3]] : tensor<2x3x2x2xi32> into tensor<6x4xi32>"
    assert source.count(producer) == 1
    source = source.replace(producer, "", 1)
    source = source.replace("%init = tensor.splat", producer + "\n      %init = tensor.splat", 1)
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    # Upstream folds across the captured producer boundary. The local shape fact domain starts
    # empty on entry to every block, so it does not assume this outer provenance relationship.
    assert after.read_text().count("tensor.collapse_shape") == 1
    assert "tensor.expand_shape" not in after.read_text()
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted", checked.get("reason")
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
