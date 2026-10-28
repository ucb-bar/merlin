"""Qualify exact scalar idempotence and SSA aliasing inside real canonicalized generic bodies."""

import json
import subprocess
import time

import pytest
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain

NAMES = ["andi", "ori", "maxsi", "minsi", "maxui", "minui"]


def source(name="andi", *, width=13, reduction=False, chain=False, dirty=False, chain_length=None):
    element = f"i{width}"
    tensor = f"tensor<2x3x{element}>"
    output = f"tensor<{element}>" if reduction else tensor
    outmap = "affine_map<(d0,d1)->()>" if reduction else "affine_map<(d0,d1)->(d0,d1)>"
    names = [NAMES[n % len(NAMES)] for n in range(chain_length)] if chain_length else (NAMES if chain else [name])
    body = []
    previous = "%x"
    for n, kind in enumerate(names):
        body.append(f"%same{n} = arith.{kind} {previous}, {previous} : {element}")
        previous = f"%same{n}"
    body.extend(
        [
            f"%masked = arith.xori {previous}, %capture : {element}",
            f"%sum = arith.addi %masked, %old : {element}",
            f"%result = arith.addi %sum, {previous} : {element}",
        ]
    )
    return f"""func.func @forward(%a: {tensor}, %init: {output}, %capture: {element}, %keep: i32)
        -> ({output}, i32) {{
      {f"%empty = tensor.empty() : {output}" if dirty else ""}
      %r = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, {outmap}],
          iterator_types = [{", ".join(['"reduction"' if reduction else '"parallel"'] * 2)}]}}
          ins(%a : {tensor}) outs({"%empty" if dirty else "%init"} : {output}) {{
        ^bb0(%x: {element}, %old: {element}):
          {chr(10).join(body)}
          linalg.yield %result : {element}
      }} -> {output}
      return %r, %keep : {output}, i32
    }}"""


def nested_source(name="andi"):
    return f"""func.func @forward(%a: tensor<3xi13>, %init: tensor<3xi13>, %keep: i32)
        -> (tensor<3xi13>, i32) {{
      %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%a : tensor<3xi13>) outs(%init : tensor<3xi13>) {{
        ^bb0(%x: i13, %old: i13):
          %same = arith.{name} %x, %x : i13
          %fresh = tensor.splat %old : tensor<i13>
          %inner = linalg.generic {{indexing_maps = [affine_map<()->()>], iterator_types = []}}
              outs(%fresh : tensor<i13>) {{
            ^bb0(%previous: i13):
              %sum = arith.addi %same, %previous : i13
              linalg.yield %sum : i13
          }} -> tensor<i13>
          %scalar = tensor.extract %inner[] : tensor<i13>
          linalg.yield %scalar : i13
      }} -> tensor<3xi13>
      return %r, %keep : tensor<3xi13>, i32
    }}"""


def certify(stage, before, tools, work, *, index_bits=64, defined=True):
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_support(before, "forward", work / "support", index_bits=index_bits, m2m_python=tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(before, "forward", work / "defined", index_bits=index_bits, m2m_python=tools.m2m_python)
    assert result["verdict"] == ("verified" if defined else "abstracted"), result.get("reason")
    result = compose_chain(
        [stage],
        work / "support/receipt.json",
        "forward",
        work / "chain.json",
        m2m_python=tools.m2m_python,
        **({"definedness": work / "defined/receipt.json"} if defined else {}),
    )
    assert result["verdict"] == ("verified" if defined else "abstracted"), result.get("reason")
    if not defined:
        assert "source definedness" in result.get("reason", "")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", [1, 8, 13, 64])
@pytest.mark.parametrize("reduction", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_generic_idempotence(tmp_path, lean_tools, name, width, reduction, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(name, width=width, reduction=reduction))
    assert f"arith.{name}" not in after.read_text()
    assert "linalg.generic" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("reduction", [False, True])
def test_all_six_rules_compose_without_losing_shared_uses(tmp_path, lean_tools, reduction):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(chain=True, reduction=reduction))
    assert all(f"arith.{name}" not in after.read_text() for name in NAMES)
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("name", NAMES)
def test_alias_substitution_reaches_nested_captures(tmp_path, lean_tools, name):
    stage, before, after = canonicalize(tmp_path, lean_tools, nested_source(name))
    assert f"arith.{name}" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("name", ["andi", "maxsi", "minui"])
def test_dirty_accumulator_does_not_gain_definedness(tmp_path, lean_tools, name):
    stage, before, _ = canonicalize(tmp_path, lean_tools, source(name, dirty=True))
    certify(stage, before, lean_tools, tmp_path, defined=False)


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    path, fault = Path(sys.argv[1]), sys.argv[2]
    module = ir.Module.parse(path.read_text())
    outer = next(o for o in module.body.operations[0].regions[0].blocks[0].operations if o.name == 'linalg.generic')
    block = outer.regions[0].blocks[0]
    ops = list(block.operations)
    if fault == 'input':
        first = next(o for o in ops if o.name.startswith('arith.'))
        first.operation.operands[0] = block.arguments[1]
    elif fault == 'attribute':
        first = next(o for o in ops if o.name.startswith('arith.'))
        first.operation.attributes['unknown.numeric_meaning'] = ir.UnitAttr.get()
    elif fault == 'source_operand':
        ops[0].operation.operands[1] = block.arguments[1]
    elif fault == 'escaping_keep':
        function_block = module.body.operations[0].regions[0].blocks[0]
        returned = list(function_block.operations)[-1]
        with ir.InsertionPoint(returned):
            value = ir.Operation.create('arith.constant', results=[ir.IntegerType.get_signless(32)],
                attributes={'value': ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)}, loc=ir.Location.unknown())
        returned.operation.operands[1] = value.results[0]
    else:
        raise ValueError(fault)
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


@pytest.mark.parametrize("fault", ["input", "attribute", "escaping_keep"])
def test_verified_target_mutations_are_rejected(tmp_path, lean_tools, fault):
    stage, before, after = canonicalize(tmp_path, lean_tools, source())
    certify(stage, before, lean_tools, tmp_path)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_source_idempotence_condition_is_rechecked(tmp_path, lean_tools):
    stage, before, _ = canonicalize(tmp_path, lean_tools, source())
    mutate(before, lean_tools, "source_operand")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_surrounding_overflow_promises_are_preserved(tmp_path, lean_tools):
    program = source().replace("addi %masked, %old :", "addi %masked, %old overflow<nsw> :")
    stage, before, after = canonicalize(tmp_path, lean_tools, program)
    assert "overflow<nsw>" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, defined=False)


@pytest.mark.parametrize("length", [128, 256])
def test_long_alias_chains_accept_with_retained_performance_evidence(tmp_path, lean_tools, length):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(chain_length=length))
    assert all(f"arith.{name}" not in after.read_text() for name in NAMES)
    certify(stage, before, lean_tools, tmp_path)
    receipt = json.loads((stage / "receipt.json").read_text())
    started = time.perf_counter()
    result = subprocess.run(
        [
            str(lean_tools.lean_exe),
            "check-module-typed",
            receipt["exports"]["before"]["path"],
            receipt["exports"]["after"]["path"],
            "64",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verdict"] == "verified"
    (tmp_path / "checker-performance.json").write_text(
        json.dumps(
            {
                "alias_count": length,
                "checker_seconds": elapsed,
                "before_bytes": before.stat().st_size,
                "after_bytes": after.stat().st_size,
                "checker_sha256": receipt["lean"]["checker_sha256"],
            },
            indent=2,
        )
        + "\n"
    )
