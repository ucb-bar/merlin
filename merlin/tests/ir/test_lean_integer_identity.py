"""Validate input-dependent scalar neutral elements against actual upstream canonicalization."""

import subprocess

import pytest
from test_lean_integer_fold import _verify_mlir, canonicalize

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

IDENTITIES = [
    ("addi", 0, False),
    ("addi", 0, True),
    ("subi", 0, False),
    ("muli", 1, False),
    ("muli", 1, True),
    ("andi", -1, False),
    ("andi", -1, True),
    ("ori", 0, False),
    ("ori", 0, True),
    ("xori", 0, False),
    ("xori", 0, True),
]


def identity_source(name, neutral, reverse, width=13, escape=False):
    ty = f"i{width}"
    operands = "%c, %x" if reverse else "%x, %c"
    types = [ty, ty] + ([ty] if escape else [])
    returns = ["%r", "%other"] + (["%c"] if escape else [])
    return f"""func.func @forward(%x: {ty}, %other: {ty}) -> ({", ".join(types)}) {{
      %c = arith.constant {neutral} : {ty}
      %r = arith.{name} {operands} : {ty}
      return {", ".join(returns)} : {", ".join(types)}
    }}"""


@pytest.mark.parametrize("name,neutral,reverse", IDENTITIES)
@pytest.mark.parametrize("width", [1, 8, 13, 32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_actual_scalar_integer_identities(tmp_path, lean_tools, name, neutral, reverse, width, escape):
    stage, before, after = canonicalize(tmp_path, lean_tools, identity_source(name, neutral, reverse, width, escape))
    assert f"arith.{name}" not in after.read_text()
    assert after.read_text().count("arith.constant") == int(escape)
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / directory, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_identity_defined_chain_and_stale_output(tmp_path, lean_tools, index_bits, escape):
    stage, before, after = canonicalize(tmp_path, lean_tools, identity_source("andi", -1, True, escape=escape))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(
            before, "forward", tmp_path / directory, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
        assert result["verdict"] == "verified", result.get("reason")

    def replay():
        return compose_chain(
            [stage],
            tmp_path / "support/receipt.json",
            "forward",
            tmp_path / "chain.json",
            m2m_python=lean_tools.m2m_python,
            definedness=tmp_path / "defined/receipt.json",
        )

    result = replay()
    assert result["verdict"] == "verified", result.get("reason")
    after.write_text(after.read_text() + "\n// output bytes changed after certification\n")
    assert replay()["verdict"] != "verified"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    path = Path(sys.argv[1])
    module = ir.Module.parse(path.read_text())
    block = module.body.operations[0].regions[0].blocks[0]
    fault = sys.argv[2]
    returned = list(block.operations)[-1]
    if fault == 'result_alias':
        returned.operation.operands[0] = block.arguments[1]
    elif fault == 'escaping_argument':
        returned.operation.operands[1] = block.arguments[0]
    elif fault == 'retained_constant':
        constant = next(o for o in block.operations if o.name == 'arith.constant')
        constant.operation.attributes['value'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(13), 1)
    elif fault == 'semantic_attribute':
        constant = next(o for o in block.operations if o.name == 'arith.constant')
        constant.operation.attributes['unknown.numeric_meaning'] = ir.UnitAttr.get()
    module.operation.verify()
    path.write_text(str(module))
"""


@pytest.mark.parametrize("fault", ["result_alias", "escaping_argument", "retained_constant", "semantic_attribute"])
def test_identity_rejects_verified_target_mutations(tmp_path, lean_tools, fault):
    stage, _, after = canonicalize(tmp_path, lean_tools, identity_source("addi", 0, False, escape=True))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    original = after.read_bytes()
    result = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert after.read_bytes() != original
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("fault", ["neutral", "overflow_flag", "reversed_subtraction", "attribute"])
def test_source_semantics_are_rechecked(tmp_path, lean_tools, fault):
    name = "subi" if fault == "reversed_subtraction" else "addi"
    stage, before, _ = canonicalize(tmp_path, lean_tools, identity_source(name, 0, False))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    source = before.read_text()
    if fault == "neutral":
        source = source.replace("constant 0 :", "constant 1 :")
    elif fault == "overflow_flag":
        source = source.replace("addi %x, %c :", "addi %x, %c overflow<nsw> :")
    elif fault == "reversed_subtraction":
        source = source.replace("subi %x, %c", "subi %c, %x")
    else:
        source = source.replace("addi %x, %c :", "addi %x, %c {unknown.numeric_meaning} :")
    before.write_text(source)
    _verify_mlir(before, lean_tools)
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_tensor_identity_requires_a_separate_runtime_domain(tmp_path, lean_tools):
    source = """func.func @forward(%x: tensor<3xi13>) -> tensor<3xi13> {
      %zero = arith.constant dense<0> : tensor<3xi13>
      %r = arith.addi %x, %zero : tensor<3xi13>
      return %r : tensor<3xi13>
    }"""
    stage, before, after = canonicalize(tmp_path, lean_tools, source)
    assert "arith.addi" not in after.read_text()
    result = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("name,neutral", [("addi", 0), ("muli", 1), ("andi", -1), ("ori", 0), ("xori", 0)])
def test_retained_identity_does_not_preempt_existing_commutation(tmp_path, lean_tools, name, neutral):
    source = identity_source(name, neutral, True, escape=True)
    stage, _, after = canonicalize(tmp_path, lean_tools, source)
    # Another valid boundary may commute this operation while retaining it. Preserve the older
    # commute rule instead of greedily consuming the constant and expecting the operation deleted.
    after.write_text(source.replace(f"{name} %c, %x", f"{name} %x, %c"))
    _verify_mlir(after, lean_tools)
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
