"""Execute actual default splat compaction through both the Lean denotation and native MLIR.

These tests independently compare the actual input and output with upstream execution.
The separate binding suite exercises the proof-backed whole-program checker.
"""

import json

import pytest
from test_lean_differential import _floats, _tensor_samples

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


def _captured_source():
    return _source(escape=True, multiple=True).replace(
        "%sum = arith.addi %data_element, %uniform_element : i32",
        "%captured = arith.addi %data_element, %other : i32\n%sum = arith.addi %captured, %uniform_element : i32",
    )


def _tensor_capture_source():
    return (
        _source()
        .replace("%uniform = tensor.splat %value", "%index = arith.constant 0 : index\n%uniform = tensor.splat %value")
        .replace(
            "%sum = arith.addi %data_element, %uniform_element : i32",
            "%captured = tensor.extract %uniform[%index, %index] : tensor<2x3xi32>\n"
            "%sum = arith.addi %data_element, %captured : i32",
        )
    )


@pytest.mark.parametrize("capture", ["scalar", "tensor"])
def test_splat_compaction_native_preserves_captures(tmp_path, fidelity_tools, capture):
    source = _captured_source() if capture == "scalar" else _tensor_capture_source()
    before, after, count = _compact(source)
    assert count == (2 if capture == "scalar" else 1)
    samples = _tensor_samples(before, tmp_path, fidelity_tools)
    expected = []
    for value, data, init, other in samples:
        result = [
            (x + value[0] + y + (2 * other[0] if capture == "scalar" else 0)) % (1 << 32)
            for x, y in zip(data, init, strict=True)
        ]
        if capture == "scalar":
            result += [value[0]] * len(data)
        expected.append([str(x) for x in result])
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("element", ["f32", "f64"])
def test_splat_compaction_preserves_exact_float_payloads(tmp_path, fidelity_tools, element):
    before, after, count = _compact(_source((2,), element, escape=True, multiple=True))
    assert count == 2
    samples = _tensor_samples(before, tmp_path, fidelity_tools)
    edges = _floats(int(element[1:])) + [0x7FC01234 if element == "f32" else 0x7FF8000000001234]
    samples.extend([[arg.copy() for arg in samples[0]] for _ in range(max(0, len(edges) - len(samples)))])
    for i, sample in enumerate(samples):
        sample[0] = [edges[i % len(edges)]]
        sample[3] = [edges[(i + 1) % len(edges)]]
    expected = [[str(sample[0][0])] * 4 for sample in samples]
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


def _repeated_source():
    source = _source(escape=True)
    generic = "%r =" + source.partition("%r =")[2].partition("return")[0]
    second = generic.replace("%r =", "%second =").replace("ins(%data,", "ins(%r,")
    return source.replace("return %r,", second + "return %second,")


def test_splat_compaction_native_repeated_producer_uses(tmp_path, fidelity_tools):
    before, after, count = _compact(_repeated_source())
    assert count == 2
    samples = _tensor_samples(before, tmp_path, fidelity_tools)
    expected = []
    for value, data, init, _other in samples:
        result = [(x + 2 * value[0] + 2 * y) % (1 << 32) for x, y in zip(data, init, strict=True)]
        expected.append([str(x) for x in result + [value[0]] * len(data)])
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected
