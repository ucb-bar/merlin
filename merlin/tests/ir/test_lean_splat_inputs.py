"""Execute actual default splat compaction through both the Lean denotation and native MLIR.

The separate semantic kernel proves uniform-input compaction. These tests qualify the pass's
execution; they do not claim that a whole-program SSA rewrite checker has been implemented.
"""

import json

import pytest
from test_lean_differential import _tensor_samples

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.splat_inputs import scalarize_splat_inputs
from merlin.verify.lean import differential as diff
from merlin.xdsl_dialects._common import text


def _source(shape=(2, 3), element="i32", *, escape=False, multiple=False, affine=False):
    def tensor_type(dims):
        return "tensor<" + "".join(f"{d}x" for d in dims) + element + ">"

    ty = tensor_type(shape)
    uniform_ty = tensor_type((2,) if affine else shape)
    dims = ", ".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({dims}) -> ({dims})>"
    uniform_map = f"affine_map<({dims}) -> (d0 floordiv 2)>" if affine else identity
    maps = [identity, uniform_map] + ([identity] if multiple else []) + [identity]
    operands = f"%data, %uniform{', %second' if multiple else ''}"
    operand_types = f"{ty}, {uniform_ty}{', ' + ty if multiple else ''}"
    body_args = f"%data_element: {element}, %uniform_element: {element}"
    if multiple:
        body_args += f", %second_element: {element}"
    body_args += f", %old: {element}"
    if element == "i32":
        body = "%sum = arith.addi %data_element, %uniform_element : i32\n"
        if multiple:
            body += "%sum2 = arith.addi %sum, %second_element : i32\n"
        body += f"%computed = arith.addi {'%sum2' if multiple else '%sum'}, %old : i32\n"
        body += "linalg.yield %computed : i32"
    else:
        # Exact bit copies exercise arbitrary float payloads without introducing NaN arithmetic.
        body = f"linalg.yield %uniform_element : {element}"
    return f"""func.func @subject(%value: {element}, %data: {ty}, %init: {ty}, %other: {element})
        -> ({ty}{", " + uniform_ty if escape else ""}) {{
      %uniform = tensor.splat %value : {uniform_ty}
      {f"%second = tensor.splat %other : {ty}" if multiple else ""}
      %r = linalg.generic {{ indexing_maps = [{", ".join(maps)}],
        iterator_types = [{", ".join(['"parallel"'] * len(shape))}] }}
        ins({operands} : {operand_types}) outs(%init : {ty}) {{
        ^bb0({body_args}):
          {body}
      }} -> {ty}
      return %r{", %uniform" if escape else ""} : {ty}{", " + uniform_ty if escape else ""}
    }}"""


def _compact(source):
    module = parse_mlir_text(source)
    module.verify()
    before = text(module)
    count = scalarize_splat_inputs(module)
    module.verify()
    return before, text(module), count


@pytest.mark.parametrize(
    "shape,element,escape,multiple,affine",
    [
        ((2, 3), "i32", False, False, False),
        ((2, 3), "i32", True, False, False),
        ((2, 3), "i32", True, True, False),
        ((4, 3), "i32", False, False, True),
        ((0, 3), "i32", False, False, False),
        ((), "i32", False, False, False),
        ((2, 3), "f32", True, False, False),
        ((2,), "f64", False, False, False),
    ],
)
def test_actual_splat_compaction(tmp_path, fidelity_tools, shape, element, escape, multiple, affine):
    before, after, count = _compact(_source(shape, element, escape=escape, multiple=multiple, affine=affine))
    assert count == (1 + int(multiple) if shape else 0)
    samples = _tensor_samples(before, tmp_path, fidelity_tools)
    for name, source in [("before", before), ("after", after)]:
        diff.compare(source, samples, tmp_path / name, fidelity_tools)
    original = json.loads((tmp_path / "before/results.json").read_text())
    compacted = json.loads((tmp_path / "after/results.json").read_text())
    assert original["lean"] == compacted["lean"]
    assert original["native"]["results"] == compacted["native"]["results"]
    if element == "i32":
        # A separate modular-integer reference checks both evaluators and the pass outputs.
        for sample, result in zip(samples, original["native"]["results"], strict=True):
            value, data, init, other = sample
            expected = [
                (x + value[0] + y + (other[0] if multiple else 0)) % (1 << 32) for x, y in zip(data, init, strict=True)
            ]
            if escape:
                expected += [value[0]] * (2 if affine else len(data))
            assert [int(x) for x in result] == expected


def test_native_oracle_detects_wrong_splat_payload(tmp_path, fidelity_tools):
    from xdsl.dialects import tensor

    before, after, count = _compact(_source())
    assert count == 1
    module = parse_mlir_text(after)
    function = next(op for op in module.walk() if op.name == "func.func")
    compacted = next(op for op in module.walk() if isinstance(op, tensor.SplatOp))
    assert not compacted.result.type.get_shape()
    compacted.operands = [function.body.block.args[3]]
    module.verify()
    samples = [[[7], [0] * 6, [0] * 6, [13]]]
    for name, source in [("before", before), ("corrupted", text(module))]:
        diff.compare(source, samples, tmp_path / name, fidelity_tools)
    original = json.loads((tmp_path / "before/results.json").read_text())
    corrupted = json.loads((tmp_path / "corrupted/results.json").read_text())
    assert original["native"]["results"] == [["7"] * 6]
    assert corrupted["native"]["results"] == [["13"] * 6]


def test_nonuniform_inputs_do_not_acquire_a_splat_rewrite():
    source = _source().replace("%uniform = tensor.splat %value : tensor<2x3xi32>", "")
    source = source.replace("%data, %uniform :", "%data, %data :")
    before, after, count = _compact(source)
    assert count == 0
    assert before == after
