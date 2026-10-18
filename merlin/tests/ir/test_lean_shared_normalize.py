"""One real pass boundary composes typed casts, mixed reshapes and generic-body normalization."""

import json
import subprocess

import pytest
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_stage, check_support, check_typed_stage, compose_chain

CASTS = [
    ("extui", "extui", 8, 16, 32),
    ("extsi", "extsi", 8, 16, 32),
    ("extui", "extsi", 8, 16, 32),
    ("trunci", "trunci", 64, 32, 13),
]


def combined_source(casts=CASTS[2], *, empty=False, compact=False):
    first, second, a, b, c = casts
    extent, count = (0, 0) if empty else (2, 3)
    shape, middle, result = f"tensor<{extent}x3x4xi32>", f"tensor<{extent * 3}x4xi32>", f"tensor<{extent}x3x2x2xi32>"
    tensor = f"tensor<{count}xi32>"
    uniform = f"%uniform = tensor.splat %keep : {tensor}" if compact else ""
    inputs = f"%body, %uniform : {tensor}, {tensor}" if compact else f"%body : {tensor}"
    input_maps = "affine_map<(i) -> (i)>, " * (2 if compact else 1)
    payload = "%payload: i32, " if compact else ""
    extra_sum = "%with_payload = arith.addi %sum, %payload : i32" if compact else ""
    sum_value = "%with_payload" if compact else "%sum"
    return f"""func.func @forward(%x: i{a}, %shape: {shape}, %body: {tensor}, %keep: i32,
        %other_shape: {shape}, %other_body: {tensor}, %other_mid: {middle})
        -> (i{c}, {result}, {tensor}, {shape}, {middle}, i32) {{
      %seven = arith.constant 7 : i32
      %empty = tensor.empty() : {tensor}
      {uniform}
      %cast1 = arith.{first} %x : i{a} to i{b}
      %cast2 = arith.{second} %cast1 : i{b} to i{c}
      %mid = tensor.collapse_shape %shape [[0, 1], [2]] : {shape} into {middle}
      %reshaped = tensor.expand_shape %mid [[0, 1], [2, 3]] output_shape [{extent}, 3, 2, 2]
        : {middle} into {result}
      %mapped = linalg.generic {{
        indexing_maps = [{input_maps}affine_map<(i) -> (i)>], iterator_types = ["parallel"]
      }} ins({inputs}) outs(%empty : {tensor}) {{
        ^bb0(%value: i32, {payload}%old: i32):
          %sum = arith.addi %seven, %value : i32
          {extra_sum}
          %cond = arith.cmpi slt, %value, %keep : i32
          %chosen = arith.select %cond, {sum_value}, %value : i32
          linalg.yield %chosen : i32
      }} -> {tensor}
      return %cast2, %reshaped, %mapped, %shape, %mid, %keep : i{c}, {result}, {tensor}, {shape}, {middle}, i32
    }}"""


def certify(tmp_path, tools, casts=CASTS[2], *, empty=False, compact=False, index_bits=64):
    source = combined_source(casts, empty=empty, compact=compact)
    if compact:
        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.llvmlower.splat_inputs import scalarize_splat_inputs
        from merlin.xdsl_dialects._common import text

        module = parse_mlir_text(source)
        module.verify()
        source = text(module)
        assert scalarize_splat_inputs(module) == 1
        module.verify()
        compacted = text(module)
        (tmp_path / "compacted.mlir").write_text(compacted)
        stage, before, after = canonicalize(tmp_path, tools, compacted)
        before.write_text(source)
    else:
        stage, before, after = canonicalize(tmp_path, tools, source)
    (stage / "meta.json").write_text(json.dumps({"boundary": "shared-normalization-batch"}))
    assert before.read_text().count("arith.ext") + before.read_text().count("arith.trunci") == 2
    assert after.read_text().count("arith.ext") + after.read_text().count("arith.trunci") == 1
    assert after.read_text().count("tensor.collapse_shape") == 1  # Escaping original intermediate.
    # The universal checker lacks the runtime type premises for both new typed families.
    assert check_stage(stage, m2m_python=tools.m2m_python)["verdict"] == "abstracted"
    receipt = check_typed_stage(stage, index_bits=index_bits, m2m_python=tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        receipt = checker(before, "forward", tmp_path / directory, index_bits=index_bits, m2m_python=tools.m2m_python)
        assert receipt["verdict"] == "verified", receipt.get("reason")
    receipt = replay(tmp_path, tools, stage)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert receipt["theorem"] == "MerlinVerify.Check.chain_typed_defined_refines"
    return stage, before, after


def replay(tmp_path, tools, stage):
    return compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=tools.m2m_python,
        definedness=tmp_path / "defined/receipt.json",
    )


@pytest.mark.parametrize("casts", CASTS)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_real_pass_composes_all_normalizer_families(tmp_path, lean_tools, casts, empty, index_bits):
    certify(tmp_path, lean_tools, casts, empty=empty, index_bits=index_bits)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_real_splat_compaction_composes_with_casts_reshapes_and_body_rewrites(tmp_path, lean_tools, empty, index_bits):
    certify(tmp_path, lean_tools, compact=True, empty=empty, index_bits=index_bits)


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    path = Path(sys.argv[1])
    module = ir.Module.parse(path.read_text())
    block = module.body.operations[0].regions[0].blocks[0]
    fault = sys.argv[2]
    operations = list(block.operations)
    returned = operations[-1]
    if fault == 'cast_signedness':
        text = str(module)
        assert text.count('arith.extui ') == 1
        module = ir.Module.parse(text.replace('arith.extui ', 'arith.extsi '))
    elif fault == 'reshape_operand':
        reshape = next(op for op in operations if op.name == 'tensor.expand_shape')
        reshape.operation.operands[0] = block.arguments[4]
    elif fault == 'body_constant':
        constant = next(op for op in operations if op.name == 'arith.constant')
        constant.operation.attributes['value'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 11)
    elif fault == 'escaping_input':
        returned.operation.operands[3] = block.arguments[4]
    elif fault == 'escaping_intermediate':
        returned.operation.operands[4] = block.arguments[6]
    elif fault == 'semantic_attribute':
        constant = next(op for op in operations if op.name == 'arith.constant')
        constant.operation.attributes['unknown.numeric_meaning'] = ir.UnitAttr.get()
    module.operation.verify()
    path.write_text(str(module))
"""


@pytest.mark.parametrize(
    "fault",
    [
        "cast_signedness",
        "reshape_operand",
        "body_constant",
        "escaping_input",
        "escaping_intermediate",
        "semantic_attribute",
    ],
)
def test_composed_checker_rejects_valid_mlir_mutations(tmp_path, lean_tools, fault):
    stage, _, after = certify(tmp_path, lean_tools)
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    assert replay(tmp_path, lean_tools, stage)["verdict"] == "abstracted"


def test_composed_chain_binds_actual_target_bytes(tmp_path, lean_tools):
    stage, _, after = certify(tmp_path, lean_tools)
    after.write_text(after.read_text() + "\n// changed after validation\n")
    assert replay(tmp_path, lean_tools, stage)["verdict"] == "abstracted"
