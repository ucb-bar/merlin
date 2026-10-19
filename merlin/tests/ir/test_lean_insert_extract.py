"""Actual scalar selection from tensor writes uses decoded in-bounds source coordinates."""

import subprocess

import pytest
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def tensor_type(shape, element="i13"):
    return f"tensor<{''.join(f'{size}x' for size in shape)}{element}>"


def source(shape=(3,), coords=(1,), *, escape=False, element="i13", computed=False, dirty=False):
    ty = tensor_type(shape, element)
    constants, indices = [], []
    for i, coordinate in enumerate(coords):
        if computed:
            constants.extend(
                [
                    f"%base{i} = arith.constant {coordinate - 1} : index",
                    f"%one{i} = arith.constant 1 : index",
                    f"%index{i} = arith.addi %base{i}, %one{i} : index",
                ]
            )
        else:
            constants.append(f"%index{i} = arith.constant {coordinate} : index")
        indices.append(f"%index{i}")
    results = ["%read", "%keep"] + (["%inserted"] if escape else [])
    result_types = [element, "i32"] + ([ty] if escape else [])
    return f"""func.func @forward(%x: {element}, %destination: {ty}, %keep: i32)
        -> ({", ".join(result_types)}) {{
      {chr(10).join(constants)}
      {f"%empty = tensor.empty() : {ty}" if dirty else ""}
      %inserted = tensor.insert %x into {"%empty" if dirty else "%destination"}[{", ".join(indices)}] : {ty}
      %read = tensor.extract %inserted[{", ".join(indices)}] : {ty}
      return {", ".join(results)} : {", ".join(result_types)}
    }}"""


def body_source(*, depth=1):
    inner = """%inserted = tensor.insert %x into %fresh[] : tensor<i13>
          %read = tensor.extract %inserted[] : tensor<i13>"""
    result = "%read"
    if depth == 2:
        inner += """
          %child = linalg.generic {indexing_maps = [affine_map<()->()>], iterator_types = []}
              outs(%fresh : tensor<i13>) {
            ^bb0(%previous: i13):
              %sum = arith.addi %read, %previous : i13
              linalg.yield %sum : i13
          } -> tensor<i13>
          %childScalar = tensor.extract %child[] : tensor<i13>"""
        result = "%childScalar"
    return f"""func.func @forward(%data: tensor<3xi13>, %initial: tensor<3xi13>, %keep: i32)
        -> (tensor<3xi13>, i32) {{
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%data : tensor<3xi13>) outs(%initial : tensor<3xi13>) {{
        ^bb0(%x: i13, %old: i13):
          %fresh = tensor.splat %old : tensor<i13>
          {inner}
          %mixed = arith.xori {result}, %old : i13
          linalg.yield %mixed : i13
      }} -> tensor<3xi13>
      return %out, %keep : tensor<3xi13>, i32
    }}"""


@pytest.mark.parametrize("shape,coords", [((), ()), ((1,), (0,)), ((3,), (1,)), ((2, 3), (1, 2))])
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_insert_extract_has_strict_defined_chain(tmp_path, lean_tools, shape, coords, escape, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(shape, coords, escape=escape))
    assert "tensor.extract" not in after.read_text()
    assert after.read_text().count("tensor.insert") == int(escape)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("shape,coords", [((3,), (1,)), ((2, 3), (1, 2))])
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_computed_source_coordinates_need_no_target_materialization(
    tmp_path, lean_tools, shape, coords, escape, index_bits
):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(shape, coords, computed=True, escape=escape))
    assert "arith.addi" in before.read_text() and "arith.addi" not in after.read_text()
    assert "tensor.extract" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("element", ["f32", "f64", "index"])
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_scalar_payload_bits_are_copied_without_arithmetic_laws(tmp_path, lean_tools, element, escape, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(element=element, escape=escape))
    assert "tensor.extract" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_insert_alias_updates_nested_captures_with_fresh_destinations(tmp_path, lean_tools, depth, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, body_source(depth=depth))
    assert "linalg.generic" in after.read_text()
    assert "tensor.insert" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_rankzero_write_completely_initializes_dirty_destination(tmp_path, lean_tools, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source((), (), dirty=True, escape=True))
    assert "tensor.extract" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_positive_rank_partial_poison_destination_remains_conditional(tmp_path, lean_tools, index_bits):
    stage, before, _ = canonicalize(tmp_path, lean_tools, source(dirty=True, escape=True))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits, defined=False)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_poison_payload_is_preserved_without_native_execution(tmp_path, lean_tools, index_bits):
    text = source((), (), escape=True).replace(
        "%inserted = tensor.insert %x",
        "%poison = tensor.empty() : tensor<i13>\n"
        "      %payload = tensor.extract %poison[] : tensor<i13>\n      %inserted = tensor.insert %payload",
    )
    stage, before, _ = canonicalize(tmp_path, lean_tools, text)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits, defined=False)


@pytest.mark.parametrize("coordinate", [-1, 3])
def test_out_of_bounds_writes_cannot_be_deleted_as_total(tmp_path, lean_tools, coordinate):
    stage, before, _ = canonicalize(tmp_path, lean_tools, source(coords=(coordinate,)))
    result = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path,fault=Path(sys.argv[1]),sys.argv[2]
    module=ir.Module.parse(path.read_text());body=module.body.operations[0].regions[0].blocks[0]
    ret=list(body.operations)[-1]
    def constant(ty,value,before):
        with ir.InsertionPoint(before):
            return ir.Operation.create('arith.constant',results=[ty],
                attributes={'value':ir.IntegerAttr.get(ty,value)},loc=ir.Location.unknown()).results[0]
    if fault=='selected':
        ret.operation.operands[0]=constant(ir.IntegerType.get_signless(13),0,ret)
    elif fault=='keep':
        ret.operation.operands[1]=constant(ir.IntegerType.get_signless(32),0,ret)
    elif fault=='tensor':
        ret.operation.operands[2]=body.arguments[1]
    elif fault=='coordinate':
        read=next(o for o in body.operations if o.name=='tensor.extract')
        read.operation.operands[1]=constant(ir.IndexType.get(),0,read)
    elif fault=='payload':
        insert=next(o for o in body.operations if o.name=='tensor.insert')
        insert.operation.operands[0]=constant(ir.IntegerType.get_signless(13),7,insert)
    elif fault=='attribute':
        insert=next(o for o in body.operations if o.name=='tensor.insert')
        insert.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    else: raise ValueError(fault)
    module.operation.verify();path.write_text(str(module))
"""


def mutate(path, tools, fault):
    original = path.read_bytes()
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() != original


@pytest.mark.parametrize("fault", ["selected", "keep", "tensor", "coordinate", "payload", "attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_mlir_verified_mutations_refuse_wrong_point_or_payload(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(escape=True))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(before if fault in {"coordinate", "payload", "attribute"} else after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
