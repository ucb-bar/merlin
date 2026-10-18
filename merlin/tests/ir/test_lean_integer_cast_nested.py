"""Check scalar cast composition and exact dead producers through existing generic-body lifting."""

import subprocess

import pytest
from test_lean_integer_cast import FAMILIES
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def shaped(width, reduction=False):
    return f"tensor<{'' if reduction else '3x'}i{width}>"


def source(first="extui", second="extsi", widths=(1, 8, 32), *, escape=False, reduction=False, length=None):
    a, b, c = widths
    if length:
        a, b, c = 1, 13, length + 1
        casts = []
        for w in range(2, c + 1):
            previous = "%x" if w == 2 else f"%v{w - 1}"
            kind = "extui" if w % 2 == 0 else "extsi"
            casts.append(f"%v{w} = arith.{kind} {previous} : i{w - 1} to i{w}")
        mid, wide = "%v13", f"%v{c}"
    else:
        casts = [f"%mid = arith.{first} %x : i{a} to i{b}", f"%wide = arith.{second} %mid : i{b} to i{c}"]
        mid, wide = "%mid", "%wide"
    output, middle = shaped(c, reduction), shaped(b, reduction)
    outs = ["%init"] + (["%midinit"] if escape else [])
    outtypes = [output] + ([middle] if escape else [])
    result_types = outtypes + ["i32"]
    returned = (["%r#0", "%r#1"] if escape else ["%r"]) + ["%keep"]
    body = casts + [f"%sum = arith.addi {wide}, %old : i{c}"]
    if escape:
        body.append(f"%midsum = arith.addi {mid}, %midold : i{b}")
    body.append("linalg.yield %sum" + (", %midsum" if escape else "") + f" : i{c}" + (f", i{b}" if escape else ""))
    identity, outmap = "affine_map<(d0)->(d0)>", "affine_map<(d0)->()>" if reduction else "affine_map<(d0)->(d0)>"
    return f"""func.func @forward(%a: {shaped(a)}, %init: {output},
        %keep: i32{f", %midinit: {middle}" if escape else ""})
        -> ({", ".join(result_types)}) {{
      %r{":2" if escape else ""} = linalg.generic {{indexing_maps = [{", ".join([identity] + [outmap] * len(outs))}],
          iterator_types = ["{"reduction" if reduction else "parallel"}"]}}
          ins(%a : {shaped(a)}) outs({", ".join(outs)} : {", ".join(outtypes)}) {{
        ^bb0(%x: i{a}, %old: i{c}{f", %midold: i{b}" if escape else ""}):
          {chr(10).join(body)}
      }} -> ({", ".join(outtypes)})
      return {", ".join(returned)} : {", ".join(result_types)}
    }}"""


def nested_source(first, second, widths):
    a, b, c = widths
    return f"""func.func @forward(%a: {shaped(a)}, %init: {shaped(c)}, %keep: i32) -> ({shaped(c)}, i32) {{
      %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}}
          ins(%a : {shaped(a)}) outs(%init : {shaped(c)}) {{
        ^bb0(%x: i{a}, %old: i{c}):
          %data = tensor.splat %x : tensor<i{a}>
          %fresh = tensor.splat %old : tensor<i{c}>
          %inner = linalg.generic {{indexing_maps = [affine_map<()->()>, affine_map<()->()>], iterator_types = []}}
              ins(%data : tensor<i{a}>) outs(%fresh : tensor<i{c}>) {{
            ^bb0(%input: i{a}, %previous: i{c}):
              %mid = arith.{first} %input : i{a} to i{b}
              %wide = arith.{second} %mid : i{b} to i{c}
              %sum = arith.addi %wide, %previous : i{c}
              linalg.yield %sum : i{c}
          }} -> tensor<i{c}>
          %scalar = tensor.extract %inner[] : tensor<i{c}>
          linalg.yield %scalar : i{c}
      }} -> {shaped(c)}
      return %r, %keep : {shaped(c)}, i32
    }}"""


def certify(stage, before, tools, work, *, index_bits=64):
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    for check, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = check(before, "forward", work / directory, index_bits=index_bits, m2m_python=tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        work / "support/receipt.json",
        "forward",
        work / "chain.json",
        m2m_python=tools.m2m_python,
        definedness=work / "defined/receipt.json",
    )
    assert result["verdict"] == "verified", result.get("reason")


CASES = [
    (first, second, widths)
    for first, second in FAMILIES
    for widths in (
        [(8, 2, 1), (32, 13, 8), (64, 32, 13)] if first == "trunci" else [(1, 2, 8), (8, 13, 32), (13, 32, 64)]
    )
]


@pytest.mark.parametrize("first,second,widths", CASES)
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("reduction", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_cast_compositions_inside_generic(
    tmp_path, lean_tools, first, second, widths, escape, reduction, index_bits
):
    stage, before, after = canonicalize(
        tmp_path, lean_tools, source(first, second, widths, escape=escape, reduction=reduction)
    )
    assert sum(after.read_text().count(f"arith.{name}") for name in ["extui", "extsi", "trunci"]) == 1 + int(escape)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("first,second", FAMILIES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_casts_compose_in_actual_nested_generics(tmp_path, lean_tools, first, second, index_bits):
    widths = (64, 13, 8) if first == "trunci" else (8, 13, 32)
    stage, before, _ = canonicalize(tmp_path, lean_tools, nested_source(first, second, widths))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("length", [63, 128])
def test_long_cast_chains_inside_generic_have_no_fragment_limit(tmp_path, lean_tools, length):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(length=length))
    assert after.read_text().count("arith.extui") == 1
    certify(stage, before, lean_tools, tmp_path)


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    path, fault = Path(sys.argv[1]), sys.argv[2]
    module = ir.Module.parse(path.read_text())
    function = module.body.operations[0].regions[0].blocks[0]
    outer = next(o for o in function.operations if o.name == 'linalg.generic')
    block = outer.regions[0].blocks[0]
    casts = [o for o in block.operations if o.name in ['arith.extui', 'arith.extsi', 'arith.trunci']]
    if fault == 'input':
        cast = casts[-1]
        with ir.InsertionPoint(cast):
            value = ir.Operation.create('arith.constant', results=[cast.operands[0].type],
                attributes={'value': ir.IntegerAttr.get(cast.operands[0].type, 0)}, loc=ir.Location.unknown())
        cast.operation.operands[0] = value.results[0]
    elif fault == 'nonneg':
        casts[0].operation.attributes['nonNeg'] = ir.UnitAttr.get()
    elif fault == 'attribute':
        casts[0].operation.attributes['unknown.numeric_meaning'] = ir.UnitAttr.get()
    elif fault == 'keep':
        returned = list(function.operations)[-1]
        with ir.InsertionPoint(returned):
            value = ir.Operation.create('arith.constant', results=[ir.IntegerType.get_signless(32)],
                attributes={'value': ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)}, loc=ir.Location.unknown())
        returned.operation.operands[-1] = value.results[0]
    elif fault != 'signedness':
        raise ValueError(fault)
    text = str(module)
    if fault == 'signedness':
        text = text.replace('arith.extui', 'arith.extsi', 1)
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


@pytest.mark.parametrize("fault", ["input", "signedness", "keep", "nonneg", "attribute"])
def test_verified_wrong_cast_body_targets_refuse(tmp_path, lean_tools, fault):
    stage, before, after = canonicalize(tmp_path, lean_tools, source("extui", "extui", escape=True))
    certify(stage, before, lean_tools, tmp_path)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("fault", ["input", "signedness", "nonneg", "attribute"])
def test_actual_source_cast_semantics_are_rechecked(tmp_path, lean_tools, fault):
    stage, before, _ = canonicalize(tmp_path, lean_tools, source("extui", "extui", escape=True))
    mutate(before, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("dirty_input", [False, True])
@pytest.mark.parametrize("reduction", [False, True])
def test_nested_casts_do_not_initialize_dirty_operands(tmp_path, lean_tools, dirty_input, reduction):
    program = source("extui", "extui", reduction=reduction)
    ty = shaped(1) if dirty_input else shaped(32, reduction)
    program = program.replace("%r = linalg.generic", f"%dirty = tensor.empty() : {ty}\n      %r = linalg.generic")
    operand = "%a" if dirty_input else "%init"
    program = program.replace(
        f"{'ins' if dirty_input else 'outs'}({operand} :", f"{'ins' if dirty_input else 'outs'}(%dirty :"
    )
    stage, before, _ = canonicalize(tmp_path, lean_tools, program)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert result["verdict"] == "abstracted", result.get("reason")


def captured_source():
    return (
        source("extui", "extsi")
        .replace("%keep: i32)", "%keep: i32, %capture: i1)")
        .replace("arith.extui %x", "arith.extui %capture")
        .replace(
            "%sum = arith.addi %wide, %old : i32",
            "%inputwide = arith.extui %x : i1 to i32\n"
            "%sum0 = arith.addi %wide, %inputwide : i32\n"
            "%sum = arith.addi %sum0, %old : i32",
        )
    )


@pytest.mark.parametrize("index_bits", [32, 64])
def test_local_cast_origins_can_start_at_real_captured_scalar(tmp_path, lean_tools, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, captured_source())
    assert after.read_text().count("arith.extui") == 2
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
