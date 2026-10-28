"""A total unused empty can disappear after exact aliases in surviving generic bodies."""

import subprocess

import pytest
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def source(*, depth=1, keep_empty=False):
    leaf = """%empty = tensor.empty() : tensor<i13>
        %inserted = tensor.insert %x into %empty[] : tensor<i13>
        %read = tensor.extract %inserted[] : tensor<i13>
        %mixed = arith.xori %read, %previous : i13"""
    if keep_empty:
        leaf += """
        %overwrite = linalg.generic {indexing_maps = [affine_map<()->()>], iterator_types = []}
            outs(%empty : tensor<i13>) {
          ^bb0(%ignored: i13):
            %twice = arith.addi %mixed, %x : i13
            linalg.yield %twice : i13
        } -> tensor<i13>
        %kept = tensor.extract %overwrite[] : tensor<i13>"""
    leaf_result = "%kept" if keep_empty else "%mixed"
    if depth == 2:
        body = f"""%local = tensor.splat %old : tensor<i13>
        %inner = linalg.generic {{indexing_maps = [affine_map<()->()>], iterator_types = []}}
            outs(%local : tensor<i13>) {{
          ^bb0(%previous: i13):
            {leaf}
            linalg.yield {leaf_result} : i13
        }} -> tensor<i13>
        %result = tensor.extract %inner[] : tensor<i13>"""
    else:
        body = leaf.replace("%previous", "%old") + f"\n%result = arith.addi {leaf_result}, %x : i13"
    return f"""func.func @forward(%data: tensor<3xi13>, %initial: tensor<3xi13>, %keep: i32)
        -> (tensor<3xi13>, i32) {{
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%data : tensor<3xi13>) outs(%initial : tensor<3xi13>) {{
        ^bb0(%x: i13, %old: i13):
          {body}
          linalg.yield %result : i13
      }} -> tensor<3xi13>
      return %out, %keep : tensor<3xi13>, i32
    }}"""


def dead_source(shape):
    ty = f"tensor<{''.join(f'{n}x' for n in shape)}i13>"
    return f"""func.func @forward(%x: i13) -> i13 {{
      %unused = tensor.empty() : {ty}
      return %x : i13
    }}"""


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_nested_alias_exposes_dead_empty(tmp_path, lean_tools, depth, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(depth=depth))
    assert "tensor.empty" in before.read_text() and "tensor.empty" not in after.read_text()
    assert "tensor.insert" not in after.read_text() and "linalg.generic" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_dead_empty_does_not_need_initialized_contents(tmp_path, lean_tools, shape, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, dead_source(shape))
    assert "tensor.empty" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_captured_empty_destination_remains_live(tmp_path, lean_tools, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(keep_empty=True))
    assert "tensor.empty" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


def test_escaping_poison_tensor_is_retained_and_never_strictly_claimed(tmp_path, lean_tools):
    text = """func.func @forward(%x: i13) -> (i13, tensor<2xi13>) {
      %unused = tensor.empty() : tensor<3xi13>
      %kept = tensor.empty() : tensor<2xi13>
      return %x, %kept : i13, tensor<2xi13>
    }"""
    stage, before, after = canonicalize(tmp_path, lean_tools, text)
    assert before.read_text().count("tensor.empty") == 2 and after.read_text().count("tensor.empty") == 1
    certify(stage, before, lean_tools, tmp_path, defined=False)


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
    if sys.argv[2] == 'payload':
        op = next(o for o in ops if o.name == 'arith.xori')
        op.operands[0] = op.operands[1]
    elif sys.argv[2] == 'return':
        op = next(o for o in ops if o.name == 'func.return')
        function = next(o for o in ops if o.name == 'func.func')
        op.operands[0] = function.regions[0].blocks[0].arguments[1]
    else:
        op = next(o for o in ops if o.name == 'tensor.empty')
        op.attributes['unmodeled'] = ir.UnitAttr.get()
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""
    result = subprocess.run(
        [str(tools.m2m_python), "-c", script, str(path), fault], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["payload", "return", "attribute"])
def test_mlir_verified_wrong_target_or_unknown_producer_refuses(tmp_path, lean_tools, fault):
    stage, before, after = canonicalize(tmp_path, lean_tools, source())
    certify(stage, before, lean_tools, tmp_path)
    mutate(before if fault == "attribute" else after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    if fault == "attribute":
        for checker, directory in [(check_support, "unknown-support"), (check_definedness, "unknown-defined")]:
            result = checker(before, "forward", tmp_path / directory, m2m_python=lean_tools.m2m_python)
            assert result["verdict"] == "abstracted", result.get("reason")


def test_observed_source_has_real_support_and_definedness(tmp_path, lean_tools):
    stage, before, _ = canonicalize(tmp_path, lean_tools, source())
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / directory, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    assert stage.is_dir()
