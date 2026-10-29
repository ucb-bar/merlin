"""Qualify the bounded repeated-input integer fusion family against the real guarded pass."""

import json
import subprocess

import pytest
from test_lean_defined_bounds import _chain
from test_lean_motion_independent import _stage as _manual_stage
from test_lean_reshape_compose import _verify_mlir

from merlin.llvmlower import fusion_guard
from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def source(shape=(2,), width=32, producer="addi", consumer="muli", *, captured=False):
    ty = "tensor<" + "".join(f"{dim}x" for dim in shape) + f"i{width}>"
    dims = ", ".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({dims})->({dims})>"
    iters = ", ".join(['"parallel"'] * len(shape))
    extra = f", %capture: i{width}" if captured else ""
    return f"""func.func @forward(%x: {ty}, %pinit: {ty}, %cinit: {ty}{extra}) -> {ty} {{
      %p = linalg.generic {{indexing_maps = [{identity}, {identity}], iterator_types = [{iters}]}}
          ins(%x : {ty}) outs(%pinit : {ty}) {{
        ^bb0(%a: i{width}, %old: i{width}):
          %v = arith.{producer} %a, {"%capture" if captured else "%a"} : i{width}
          linalg.yield %v : i{width}
      }} -> {ty}
      %r = linalg.generic {{indexing_maps = [{identity}, {identity}], iterator_types = [{iters}]}}
          ins(%p : {ty}) outs(%cinit : {ty}) {{
        ^bb0(%a: i{width}, %old: i{width}):
          %v = arith.{consumer} %a, %a : i{width}
          linalg.yield %v : i{width}
      }} -> {ty}
      return %r : {ty}
    }}"""


FUSE = """
import sys
from pathlib import Path
from torch_mlir import ir
from torch_mlir.passmanager import PassManager
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    module.operation.verify()
    PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
    module.operation.verify()
    Path(sys.argv[2]).write_text(module.operation.get_asm(print_generic_op_form=True))
    PassManager.parse('builtin.module(func.func(linalg-fuse-elementwise-ops))').run(module.operation)
    module.operation.verify()
    Path(sys.argv[3]).write_text(module.operation.get_asm(print_generic_op_form=True))
"""


def stage_for(tmp_path, compiler, program):
    stage = tmp_path / "stage"
    stage.mkdir(parents=True)
    original = tmp_path / "original.mlir"
    original.write_text(program)
    before, after = stage / "before.mlir", stage / "after.mlir"
    result = subprocess.run(
        [str(compiler), "-c", fusion_guard.runner_prelude() + FUSE, str(original), str(before), str(after)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "pass.stdout").write_text(result.stdout)
    (tmp_path / "pass.stderr").write_text(result.stderr)
    assert before.read_text().count('"linalg.generic"') == 2
    assert after.read_text().count('"linalg.generic"') == 1
    assert before.read_bytes() != after.read_bytes()
    (stage / "meta.json").write_text(json.dumps({"boundary": "actual-guarded-repeated-input-fusion"}))
    return stage, before, after


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize(
    "shape,width,producer,consumer",
    [
        ((2,), 13, "addi", "muli"),
        ((2, 3), 32, "muli", "addi"),
        ((), 8, "addi", "muli"),
        ((0, 3), 32, "addi", "muli"),
        ((1,), 64, "addi", "muli"),
    ],
)
def test_actual_repeated_input_fusion_has_strict_source_chain(
    tmp_path, lean_tools, bits, shape, width, producer, consumer
):
    stage, _, after = stage_for(tmp_path, lean_tools.m2m_python, source(shape, width, producer, consumer))
    assert '"tensor.empty"' in after.read_text()
    _chain(tmp_path, stage, "forward", lean_tools, bits)


def test_original_gate_fusion_gap_now_has_strict_source_chain(tmp_path, lean_tools):
    from test_lean_native_gate_structured_differential import FUSION_SOURCE

    _, _, after = stage_for(tmp_path / "actual", lean_tools.m2m_python, FUSION_SOURCE)
    stage = _manual_stage(tmp_path, FUSION_SOURCE, after.read_text())
    _chain(tmp_path, stage, "forward", lean_tools, 64)


def test_actual_captured_operand_fusion_stays_outside_bounded_family(tmp_path, lean_tools):
    stage, before, _ = stage_for(tmp_path, lean_tools.m2m_python, source(captured=True))
    for name, checker in (("support", check_support), ("definedness", check_definedness)):
        assert checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


@pytest.mark.parametrize("fault", ["primitive", "yield", "return", "old-output"])
def test_valid_actual_fused_target_mutants_are_refused(tmp_path, lean_tools, fault):
    _, _, after = stage_for(tmp_path / "actual", lean_tools.m2m_python, source())
    target = after.read_text()
    if fault == "primitive":
        assert '"arith.addi"' in target
        mutant = target.replace('"arith.addi"', '"arith.subi"', 1)
    else:
        # Use the compiler's real parser/value graph, so fixture SSA spellings
        # do not become a hidden mutation precondition.
        path = tmp_path / "mutant.mlir"
        path.write_text(target)
        subprocess.run(
            [str(lean_tools.m2m_python), "-c", MUTATE, str(path), fault],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        mutant = path.read_text()
    assert mutant != target
    stage = _manual_stage(tmp_path, (tmp_path / "actual/stage/before.mlir").read_text(), mutant)
    _verify_mlir(stage / "after.mlir", lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    function = module.body.operations[0]
    entry = function.regions[0].blocks[0]
    generic = next(op for op in entry.operations if op.operation.name == 'linalg.generic')
    body = generic.regions[0].blocks[0]
    operations = list(body.operations)
    if sys.argv[2] == 'yield':
        operations[-1].operands[0] = operations[0].results[0]
    elif sys.argv[2] == 'return':
        list(entry.operations)[-1].operands[0] = entry.arguments[2]
    elif sys.argv[2] == 'old-output':
        operations[0].operands[0] = body.arguments[1]
    else:
        raise AssertionError(sys.argv[2])
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""
