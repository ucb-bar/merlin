"""Check actual nested generic compaction with explicit capture and initialized-source domains."""

import json
import subprocess

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.splat_inputs import scalarize_splat_inputs
from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain
from merlin.xdsl_dialects._common import text


def _type(shape, element):
    return "tensor<" + "".join(f"{d}x" for d in shape) + element + ">"


def _source(
    element="i32",
    *,
    outer_shape=(2,),
    inner_shape=(2, 3),
    rank_zero=False,
    captured=False,
    escape=False,
    multiple=False,
    tensor_capture=False,
    dirty=False,
    fresh_init=True,
):
    outer, full = _type(outer_shape, element), _type(inner_shape, element)
    inner = _type(() if rank_zero else inner_shape, element)
    dims = ", ".join(f"d{i}" for i in range(len(outer_shape)))
    outer_map = f"affine_map<({dims}) -> ({dims})>"
    inner_dims = "" if rank_zero else ", ".join(f"d{i}" for i in range(len(inner_shape)))
    coordinates = ", ".join("0" for _ in inner_shape) if rank_zero else inner_dims
    input_map = f"affine_map<({inner_dims}) -> ({coordinates})>"
    output_map = f"affine_map<({inner_dims}) -> ({'' if rank_zero else inner_dims})>"
    maps = [input_map, input_map] + ([input_map] if multiple else []) + [output_map]
    body_args = [f"%data_element: {element}", f"%uniform_element: {element}"]
    if multiple:
        body_args.append(f"%second_element: {element}")
    body_args.append(f"%previous: {element}")
    capture = ""
    uniform_value = "%uniform_element"
    if tensor_capture:
        indices = ", ".join("%index" for _ in inner_shape)
        capture = f"%captured_value = tensor.extract %uniform[{indices}] : {full}"
        uniform_value = "%captured_value"
    if element.startswith("i"):
        body = [capture, f"%sum = arith.addi %data_element, {uniform_value} : {element}"]
        value = "%sum"
        if multiple:
            body.append(f"%sum2 = arith.addi {value}, %second_element : {element}")
            value = "%sum2"
        body.append(f"%sum3 = arith.addi {value}, %old : {element}")
        value = "%sum3"
        if not dirty:
            body.append(f"%sum4 = arith.addi {value}, %previous : {element}")
            value = "%sum4"
        body.append(f"linalg.yield {value} : {element}")
    else:
        body = [f"linalg.yield {uniform_value} : {element}"]
    uniform = f"%uniform = tensor.splat {'%captured' if captured else '%value'} : {full}"
    indices = "" if rank_zero else ", ".join("%index" for _ in inner_shape)
    results = f"({outer}, {full})" if escape else outer
    returns = f"%r, %uniform : {outer}, {full}" if escape else f"%r : {outer}"
    assert not escape or captured
    return f"""func.func @subject(%payloads: {outer}, %outer_init: {outer}, %data: {full},
        %inner_init: {inner}, %captured: {element}, %alternate: {element}) -> {results} {{
      %index = arith.constant 0 : index
      {uniform if captured else ""}
      %r = linalg.generic {{ indexing_maps = [{outer_map}, {outer_map}],
          iterator_types = [{", ".join(['"parallel"'] * len(outer_shape))}] }}
          ins(%payloads : {outer}) outs(%outer_init : {outer}) {{
        ^bb0(%value: {element}, %old: {element}):
          {"" if captured else uniform}
          {f"%second = tensor.splat %alternate : {full}" if multiple else ""}
          {f"%dirty = tensor.empty() : {inner}" if dirty else ""}
          {f"%fresh = tensor.splat %captured : {inner}" if fresh_init and not dirty else ""}
          %inner = linalg.generic {{ indexing_maps = [{", ".join(maps)}],
              iterator_types = [{", ".join(['"parallel"'] * (0 if rank_zero else len(inner_shape)))}] }}
              ins(%data, %uniform{", %second" if multiple else ""} : {full}, {full}{", " + full if multiple else ""})
              outs({"%dirty" if dirty else "%fresh" if fresh_init else "%inner_init"} : {inner}) {{
            ^bb0({", ".join(body_args)}):
              {chr(10).join(body)}
          }} -> {inner}
          %scalar = tensor.extract %inner[{indices}] : {inner}
          linalg.yield %scalar : {element}
      }} -> {outer}
      return {returns}
    }}"""


def _stage(tmp_path, source):
    module = parse_mlir_text(source)
    module.verify()
    before = text(module)
    count = scalarize_splat_inputs(module)
    module.verify()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "before.mlir").write_text(before)
    (stage / "after.mlir").write_text(text(module))
    (stage / "meta.json").write_text(json.dumps({"boundary": "nested-splat-input-compaction"}))
    return stage, count


CASES = [
    {},
    {"rank_zero": True},
    {"rank_zero": True, "dirty": True},
    {"rank_zero": True, "captured": True, "escape": True},
    {"captured": True, "escape": True},
    {"multiple": True, "tensor_capture": True},
    {"rank_zero": True, "captured": True, "multiple": True},
    {"outer_shape": ()},
    {"outer_shape": (0,), "rank_zero": True},
    {"outer_shape": (2, 2), "inner_shape": (2,)},
    {"element": "i13", "multiple": True},
    {"element": "f32", "captured": True, "escape": True},
    {"element": "f64", "rank_zero": True},
    {"element": "bf16", "rank_zero": True},
]


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("options", CASES)
def test_actual_nested_splat_compaction(tmp_path, lean_tools, index_bits, options):
    stage, count = _stage(tmp_path, _source(**options))
    assert count == 1 + int(options.get("multiple", False))
    support = check_support(
        stage / "before.mlir", "subject", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert support["verdict"] == "verified", support.get("reason")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("dirty", [False, True])
def test_nested_splat_defined_chain(tmp_path, lean_tools, index_bits, dirty):
    stage, count = _stage(tmp_path, _source(rank_zero=True, captured=True, escape=True, multiple=True, dirty=dirty))
    assert count == 2
    before = stage / "before.mlir"
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "subject", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")

    def replay(name):
        return compose_chain(
            [stage],
            tmp_path / "support/receipt.json",
            "subject",
            tmp_path / name,
            definedness=tmp_path / "defined/receipt.json",
            m2m_python=lean_tools.m2m_python,
        )

    result = replay("chain.json")
    assert result["verdict"] == "verified", result.get("reason")
    assert result["theorem"] == "MerlinVerify.Check.chain_typed_extended_defined_refines"
    after = stage / "after.mlir"
    after.write_text(after.read_text() + "\n// changed actual nested output after proof publication\n")
    assert replay("stale.json")["verdict"] != "verified"


def test_constant_bounded_nonzero_rank_extraction_is_defined(tmp_path, lean_tools):
    stage, _ = _stage(tmp_path, _source())
    result = check_definedness(stage / "before.mlir", "subject", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    function = module.body.operations[0]
    block = function.regions[0].blocks[0]
    outer = next(op for op in block.operations if op.operation.name == 'linalg.generic')
    middle = outer.regions[0].blocks[0]
    inner = next(op for op in middle.operations if op.operation.name == 'linalg.generic')
    body = inner.regions[0].blocks[0]
    splat = next(op for op in middle.operations if op.operation.name == 'tensor.splat' and
                 ir.RankedTensorType(op.results[0].type).rank == 0)
    fault = sys.argv[2]
    if fault == 'payload':
        splat.operation.operands[0] = block.arguments[5]
    elif fault == 'captured_tensor':
        list(block.operations)[-1].operation.operands[1] = block.arguments[2]
    elif fault == 'captured_scalar':
        add = next(op for op in body.operations if op.operation.name == 'arith.addi' and
                   any(arg == middle.arguments[1] for arg in op.operands))
        add.operation.operands[1] = middle.arguments[0]
    elif fault == 'inner_destination':
        inner.operation.operands[-1] = block.arguments[2]
    elif fault == 'outer_destination':
        outer.operation.operands[-1] = block.arguments[0]
    elif fault == 'yield':
        list(middle.operations)[-1].operation.operands[0] = middle.arguments[1]
    elif fault == 'attribute':
        inner.operation.attributes['unreviewed.meaning'] = ir.UnitAttr.get()
    elif fault == 'body':
        list(body.operations)[-1].operation.operands[0] = body.arguments[-1]
    else:
        maps = ['affine_map<(d0,d1)->(d0,d1)>', 'affine_map<(d0,d1)->()>',
                'affine_map<(d0,d1)->(d0,d1)>']
        maps[0 if fault == 'input_map' else 2] = 'affine_map<(d0,d1)->(d1,d0)>'
        inner.operation.attributes['indexing_maps'] = ir.Attribute.parse('[' + ', '.join(maps) + ']')
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""


@pytest.mark.parametrize(
    "fault",
    [
        "payload",
        "captured_tensor",
        "captured_scalar",
        "inner_destination",
        "outer_destination",
        "yield",
        "attribute",
        "body",
        "input_map",
        "output_map",
    ],
)
def test_nested_splat_rejects_verified_mlir_mutations(tmp_path, lean_tools, fault):
    source = _source(inner_shape=(2, 2), captured=True, escape=True)
    # Observe a non-diagonal element so transposed maps change an actual function result.
    source = source.replace(
        "%index = arith.constant 0 : index", "%index = arith.constant 0 : index\n%one = arith.constant 1 : index"
    )
    source = source.replace("tensor.extract %inner[%index, %index]", "tensor.extract %inner[%one, %index]")
    stage, _ = _stage(tmp_path, source)
    accepted = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert accepted["verdict"] == "verified", accepted.get("reason")
    after = stage / "after.mlir"
    original = after.read_bytes()
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert after.read_bytes() != original
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def _three_level_source():
    source = _source(outer_shape=(), rank_zero=True)
    start, stop = source.index("%r = linalg.generic"), source.index("return %r")
    inner = source[start:stop].replace("outs(%outer_init :", "outs(%fresh_outer_init :")
    header = source[:start].replace(
        ") -> tensor<i32> {",
        ", %driver: tensor<2xi32>, %driver_init: tensor<2xi32>) -> tensor<2xi32> {",
        1,
    )
    return (
        header
        + f"""
      %result = linalg.generic {{ indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"] }} ins(%driver : tensor<2xi32>) outs(%driver_init : tensor<2xi32>) {{
        ^bb0(%driver_value: i32, %driver_old: i32):
          %fresh_outer_init = tensor.splat %driver_value : tensor<i32>
          {inner}
          %outer_scalar = tensor.extract %r[] : tensor<i32>
          linalg.yield %outer_scalar : i32
      }} -> tensor<2xi32>
      return %result : tensor<2xi32>
    }}"""
    )


@pytest.mark.parametrize("index_bits", [32, 64])
def test_three_nested_generics_preserve_scoped_payloads(tmp_path, lean_tools, index_bits):
    stage, count = _stage(tmp_path, _three_level_source())
    assert count == 1
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    defined = check_definedness(
        stage / "before.mlir", "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert defined["verdict"] == "verified", defined.get("reason")


def _four_level_source():
    source = _source(rank_zero=True)
    start, stop = source.index("%inner = linalg.generic"), source.index("%scalar = tensor.extract")
    innermost = source[start:stop]
    wrapped = f"""
      %middle_seed1 = tensor.splat %captured : tensor<i32>
      %middle1 = linalg.generic {{ indexing_maps = [affine_map<()->()>], iterator_types = [] }}
          outs(%middle_seed1 : tensor<i32>) {{
        ^bb0(%middle_old1: i32):
          %middle_seed2 = tensor.splat %captured : tensor<i32>
          %middle2 = linalg.generic {{ indexing_maps = [affine_map<()->()>], iterator_types = [] }}
              outs(%middle_seed2 : tensor<i32>) {{
            ^bb0(%middle_old2: i32):
              {innermost}
              %take_inner = tensor.extract %inner[] : tensor<i32>
              linalg.yield %take_inner : i32
          }} -> tensor<i32>
          %take_middle2 = tensor.extract %middle2[] : tensor<i32>
          linalg.yield %take_middle2 : i32
      }} -> tensor<i32>
    """
    return source[:start] + wrapped + source[stop:].replace("tensor.extract %inner[]", "tensor.extract %middle1[]")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_four_nested_generics_use_actual_depth_budget(tmp_path, lean_tools, index_bits):
    stage, count = _stage(tmp_path, _four_level_source())
    assert count == 1
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    defined = check_definedness(
        stage / "before.mlir", "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert defined["verdict"] == "verified", defined.get("reason")
