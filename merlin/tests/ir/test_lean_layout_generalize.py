"""Actual transpose/broadcast/reduce pass output, source definedness and byte-bound replay."""

import subprocess
from dataclasses import dataclass

import pytest
from test_lean_generalize import _generalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


@dataclass(frozen=True)
class Case:
    family: str
    shape: tuple[int, ...]
    dimensions: tuple[int, ...]
    element: str = "i32"
    broadcast_shape: tuple[int, ...] = ()
    outputs: int = 1

    @property
    def result_shape(self):
        if self.family == "transpose":
            return tuple(self.shape[d] for d in self.dimensions)
        if self.family == "broadcast":
            return self.broadcast_shape
        return tuple(size for d, size in enumerate(self.shape) if d not in self.dimensions)

    def tensor(self, shape):
        return "tensor<" + "".join(f"{n}x" for n in shape) + self.element + ">"


CASES = [
    Case("transpose", (2, 3, 4), (1, 2, 0)),
    Case("transpose", (2, 3, 4), (2, 0, 1), "i13"),
    Case("transpose", (0, 3), (1, 0)),
    Case("transpose", (2, 3), (1, 0), "f32"),
    Case("transpose", (2, 3), (1, 0), "f64"),
    Case("transpose", (2, 3), (1, 0), "bf16"),
    Case("broadcast", (), (0, 1), broadcast_shape=(2, 3)),
    Case("broadcast", (), (), broadcast_shape=()),
    Case("broadcast", (2, 3), (0, 2), broadcast_shape=(4, 2, 5, 3)),
    Case("broadcast", (3,), (0,), "i13", (0, 3)),
    Case("broadcast", (2,), (1,), "f32", (2, 3)),
    Case("broadcast", (2,), (0,), "f64", (3, 2)),
    Case("broadcast", (2,), (0,), "bf16", (3, 2)),
    Case("reduce", (2, 3, 4), (0, 2)),
    Case("reduce", (2, 3), (0, 1), "i8"),
    Case("reduce", (2, 3), (1,), "i13"),
    Case("reduce", (2, 0, 3), (1,)),
    Case("reduce", (0, 3), (1,)),
    Case("reduce", (2, 3), (), outputs=2),
    Case("reduce", (2, 3, 4), (0, 2), outputs=2),
]


def _source(case, *, dirty=False, dirty_input=False, capture=False):
    """Keep an alternate input and an escaping source to expose accidental SSA substitutions."""
    ni = case.outputs if case.family == "reduce" else 1
    tin, tout = case.tensor(case.shape), case.tensor(case.result_shape)
    arguments = [f"%a{i}: {tin}" for i in range(ni)] + [f"%alternate: {tin}"]
    arguments += [] if dirty else [f"%init{i}: {tout}" for i in range(case.outputs)]
    if capture:
        arguments += [f"%bias: {case.element}"]
    allocations = [f"%init{i} = tensor.empty() : {tout}" for i in range(case.outputs)] if dirty else []
    if dirty_input:
        allocations += [f"%poison = tensor.empty() : {tin}"]
    inputs = ["%poison" if dirty_input and i == 0 else f"%a{i}" for i in range(ni)]
    destinations = [f"%init{i}" for i in range(case.outputs)]
    ins = f"ins({', '.join(inputs)} : {', '.join([tin] * ni)})"
    outs = f"outs({', '.join(destinations)} : {', '.join([tout] * case.outputs)})"
    key = "permutation" if case.family == "transpose" else "dimensions"
    header = f"%r:{case.outputs} = linalg.{case.family} {ins} {outs} {key} = {list(case.dimensions)}"
    body = ""
    if case.family == "reduce":
        arguments_body = [f"%x{i}: {case.element}" for i in range(ni)]
        arguments_body += [f"%acc{i}: {case.element}" for i in range(case.outputs)]
        op = "arith.addf" if case.element.startswith("f") else "arith.addi"
        operations = []
        for i in range(ni):
            value = f"%x{i}"
            if capture:
                operations += [f"%biased{i} = {op} {value}, %bias : {case.element}"]
                value = f"%biased{i}"
            operations += [f"%sum{i} = {op} {value}, %acc{i} : {case.element}"]
        operations += [f"linalg.yield {', '.join(f'%sum{i}' for i in range(ni))} : {', '.join([case.element] * ni)}"]
        body = "(" + ", ".join(arguments_body) + ") {\n" + "\n".join(operations) + "\n}"
    types = [tout] * case.outputs + [tin]
    values = [f"%r#{i}" for i in range(case.outputs)] + ["%a0"]
    return (
        f"func.func @forward({', '.join(arguments)}) -> ({', '.join(types)}) {{\n"
        + "\n".join(allocations)
        + "\n"
        + header
        + "\n"
        + body
        + f"\nreturn {', '.join(values)} : {', '.join(types)}\n}}"
    )


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.family}-{c.shape}-{c.dimensions}-{c.element}-{c.outputs}")
def test_actual_layout_generalization(tmp_path, lean_tools, case, index_bits):
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(case, dirty=case.family != "reduce"))
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        receipt = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert receipt["verdict"] == "verified", receipt.get("reason")
    receipt = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")


CHAIN_CASES = [
    Case("transpose", (2, 3, 4), (1, 2, 0)),
    Case("broadcast", (2,), (0,), broadcast_shape=(3, 2)),
    Case("reduce", (2, 3, 4), (0, 2), outputs=2),
]


@pytest.mark.parametrize("case", CHAIN_CASES, ids=lambda c: c.family)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_layout_defined_chain_rejects_stale_source(tmp_path, lean_tools, case, index_bits):
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(case))
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    assert check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"

    def replay():
        return compose_chain(
            [stage],
            tmp_path / "support/receipt.json",
            "forward",
            tmp_path / "chain.json",
            definedness=tmp_path / "defined/receipt.json",
            m2m_python=lean_tools.m2m_python,
        )

    result = replay()
    assert result["verdict"] == "verified", result.get("reason")
    assert result["theorem"] == "MerlinVerify.Check.chain_typed_defined_refines"
    before.write_text(before.read_text() + "\n// changed live source\n")
    assert replay()["verdict"] != "verified"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    block = module.body.operations[0].regions[0].blocks[0]
    generic = next(op for op in block.operations if op.operation.name == 'linalg.generic')
    body = generic.regions[0].blocks[0]
    fault, family = sys.argv[2:]
    if fault == 'input':
        generic.operation.operands[0] = block.arguments[1]
    elif fault == 'body':
        list(body.operations)[-1].operation.operands[0] = body.arguments[-1]
    elif fault == 'escaping_input':
        list(block.operations)[-1].operation.operands[-1] = block.arguments[1]
    elif fault == 'attribute':
        generic.operation.attributes['unreviewed.meaning'] = ir.UnitAttr.get()
    elif fault == 'iterator':
        n = len(ir.ArrayAttr(generic.operation.attributes['iterator_types']))
        kinds = ','.join(['#linalg.iterator_type<parallel>'] * n)
        generic.operation.attributes['iterator_types'] = ir.Attribute.parse('[' + kinds + ']')
    else:
        maps = ir.ArrayAttr(generic.operation.attributes['indexing_maps'])
        index = 0 if fault == 'input_map' else 1
        changed = [str(m) for m in maps]
        if index == 0:
            results = 'd1,d0,d2' if family == 'reduce' else 'd2,d1'
        else:
            results = 'd1,d0,d2' if family == 'broadcast' else 'd1,d0'
        dims = 'd0,d1' if family == 'transpose' else 'd0,d1,d2'
        if family == 'transpose' and index == 0:
            results = 'd0,d1'
        changed[index] = 'affine_map<(' + dims + ')->(' + results + ')>'
        generic.operation.attributes['indexing_maps'] = ir.Attribute.parse('[' + ','.join(changed) + ']')
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""


MUTATION_CASES = [
    Case("transpose", (2, 2), (1, 0)),
    Case("broadcast", (2, 2), (0,), broadcast_shape=(2, 2, 2)),
    Case("reduce", (2, 2, 2), (2,)),
]


@pytest.mark.parametrize("case", MUTATION_CASES, ids=lambda c: c.family)
@pytest.mark.parametrize("fault", ["input", "body", "input_map", "output_map", "escaping_input", "attribute"])
def test_layout_rejects_legal_mlir_mutations(tmp_path, lean_tools, case, fault):
    stage, _, after = _generalize(tmp_path, lean_tools, _source(case))
    original = after.read_bytes()
    result = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault, case.family],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert after.read_bytes() != original
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


@pytest.mark.parametrize("case", MUTATION_CASES, ids=lambda c: c.family)
def test_poison_input_cannot_establish_definedness(tmp_path, lean_tools, case):
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(case, dirty_input=True))
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_reduction_requires_initialized_accumulator(tmp_path, lean_tools):
    case = Case("reduce", (2, 3), (1,))
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(case, dirty=True))
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_reduction_preserves_captured_values(tmp_path, lean_tools):
    case = Case("reduce", (2, 3), (1,))
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(case, capture=True))
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("case", [Case("transpose", (2,), (0,))])
def test_separate_identity_fold_is_explicitly_unverified(tmp_path, lean_tools, case):
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(case), expect_generic=False)
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert (
        check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("element", ["f32", "f64"])
def test_float_reduction_refinement_keeps_separate_definedness_obligation(tmp_path, lean_tools, element):
    case = Case("reduce", (2, 3, 4), (0, 2), element)
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(case))
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    # The current definedness leaf book has no floating arithmetic rule. Do not turn the
    # body-preserving rewrite proof into a claim that this missing obligation was discharged.
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


MIXED_SOURCE = """func.func @forward(%value: i32, %init: tensor<2xi32>) -> (tensor<2xi32>, tensor<3x2xi32>) {
  %e = tensor.empty() : tensor<2xi32>
  %seed = linalg.fill ins(%value : i32) outs(%e : tensor<2xi32>) -> tensor<2xi32>
  %b = tensor.empty() : tensor<3x2xi32>
  %broadcast = linalg.broadcast ins(%seed : tensor<2xi32>) outs(%b : tensor<3x2xi32>) dimensions = [0]
  %t = tensor.empty() : tensor<2x3xi32>
  %transpose = linalg.transpose ins(%broadcast : tensor<3x2xi32>) outs(%t : tensor<2x3xi32>) permutation = [1, 0]
  %sum = linalg.reduce ins(%transpose : tensor<2x3xi32>) outs(%init : tensor<2xi32>) dimensions = [1]
    (%x: i32, %acc: i32) {
      %next = arith.addi %x, %acc : i32
      linalg.yield %next : i32
    }
  return %sum, %broadcast : tensor<2xi32>, tensor<3x2xi32>
}"""


@pytest.mark.parametrize("index_bits", [32, 64])
def test_layout_families_compose_through_actual_ssa_values(tmp_path, lean_tools, index_bits):
    stage, before, _ = _generalize(tmp_path, lean_tools, MIXED_SOURCE)
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert result["verdict"] == "verified", result.get("reason")


def test_reduction_iterator_change_is_rejected(tmp_path, lean_tools):
    case = Case("reduce", (2, 2, 2), (2,))
    stage, _, after = _generalize(tmp_path, lean_tools, _source(case))
    result = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), "iterator", case.family],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
