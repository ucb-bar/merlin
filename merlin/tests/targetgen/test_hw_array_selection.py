"""Public array order/index semantics retain all original finite scalar outputs."""

import dataclasses
import json
import os
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.rtl.hw_array_selection import (
    ArraySelectionLimits,
    preflight_array_selections,
    typed_array_selection,
)
from merlin.targetgen.rtl.hw_combinational import EvaluationLimits, prepare_combinational_observation
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_observations import _integer, _name

LIMITS = ArraySelectionLimits(4096, 65536, 4_000_000)
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def _module(body, inputs, outputs):
    block = "^bb0(" + ", ".join(f"%{name}: i{width}" for name, width in inputs) + "):\n" if inputs else ""
    ports = ", ".join(
        f"{direction} {name}: i{width}"
        for direction, roster in (("input", inputs), ("output", outputs))
        for name, width in roster
    )
    return (
        'module { "hw.module"() ({\n'
        + block
        + "\n".join(body)
        + '\n}) {sym_name="Select", module_type=!hw.modty<'
        + ports
        + ">, parameters=[]} : () -> ()\n}\n"
    )


def _source(count=4, width=7):
    index_width = (count - 1).bit_length()
    inputs = [(f"e{i}", width) for i in range(count)] + [("index", index_width)]
    body = [
        '%array = "hw.array_create"('
        + ",".join(f"%e{i}" for i in range(count))
        + ") : ("
        + ",".join([f"i{width}"] * count)
        + f") -> !hw.array<{count}xi{width}>",
        f'%value = "hw.array_get"(%array,%index) : (!hw.array<{count}xi{width}>,i{index_width}) -> i{width}',
        f'"hw.output"(%value) : (i{width}) -> ()',
    ]
    return _module(body, inputs, [("picked", width)])


def _selection(text, limits=LIMITS):
    parsed = parse_generic_hw(text, reject_dense_literals=True)
    module = next(op for op in parsed.walk() if _name(op) == "hw.module")
    operations = list(module.regions[0].block.ops)
    cost = preflight_array_selections({"Select": operations}, ["Select"], limits=limits, scalar_bits=64)
    op = next(op for op in operations if _name(op) == "hw.array_get")
    return typed_array_selection(op, scalar_bits=64, limits=limits), cost


@pytest.mark.parametrize("count", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("width", [1, 7, 64])
def test_original_creation_order_complete_indices_and_exact_bitvectors(count, width):
    selection, cost = _selection(_source(count, width))
    assert cost == {"operations": 2, "elements": count * 2, "aggregate_bits": count * width * 2}
    assert selection.elements == tuple(reversed(selection.original_creation_operands))
    assert selection.full_index_domain_defined == (count != 3)
    # Independent per-element patterns distinguish order, including one-bit elements.
    for seed in range(count + 3):
        values = tuple(((i + 1) * 37 + seed * 13) & (2**width - 1) for i in range(count))
        logical = tuple(reversed(values))
        assert tuple(selection.evaluate(values, index) for index in range(count)) == logical
    if count == 3:
        with pytest.raises(ValueError, match="defined original"):
            selection.evaluate(tuple(i & (2**width - 1) for i in range(count)), 3)


@pytest.mark.parametrize("field", ["operations", "elements", "aggregate_bits"])
def test_whole_occurrence_budgets_count_repeated_bodies_before_expansion(field):
    parsed = parse_generic_hw(_source(), reject_dense_literals=True)
    operations = list(next(op for op in parsed.walk() if _name(op) == "hw.module").regions[0].block.ops)
    exact = ArraySelectionLimits(4, 16, 112)
    assert preflight_array_selections({"Select": operations}, ["Select", "Select"], limits=exact, scalar_bits=64) == {
        "operations": 4,
        "elements": 16,
        "aggregate_bits": 112,
    }
    with pytest.raises(ValueError, match="pre-expansion"):
        preflight_array_selections(
            {"Select": operations},
            ["Select", "Select"],
            limits=dataclasses.replace(exact, **{field: getattr(exact, field) - 1}),
            scalar_bits=64,
        )


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "index_width",
        "result_type",
        "element_type",
        "count",
        "nested",
        "unknown_attribute",
        "duplicate_ownership",
        "opaque",
        "singleton",
        "signed",
        "zero_element",
        "creation_attribute",
    ],
)
def test_original_malformed_and_unsupported_aggregate_forms_refuse(case):
    source = _source()
    if case == "missing":
        source = source.replace('"hw.array_get"(%array,%index)', '"hw.array_get"(%array)').replace(
            "(!hw.array<4xi7>,i2)", "(!hw.array<4xi7>)"
        )
    elif case == "index_width":
        source = source.replace("index: i2", "index: i3").replace("4xi7>,i2", "4xi7>,i3")
    elif case == "result_type":
        source = (
            source.replace("-> i7", "-> i8", 1)
            .replace("output picked: i7", "output picked: i8")
            .replace('"hw.output"(%value) : (i7)', '"hw.output"(%value) : (i8)')
        )
    elif case == "element_type":
        source = (
            source.replace("%e0: i7", "%e0: i8")
            .replace("input e0: i7", "input e0: i8")
            .replace("(i7,i7,i7,i7)", "(i8,i7,i7,i7)")
        )
    elif case == "count":
        source = source.replace("4xi7", "5xi7").replace("index: i2", "index: i3").replace("5xi7>,i2", "5xi7>,i3")
    elif case == "nested":
        source = source.replace("4xi7", "4x!hw.array<2xi7>")
    elif case == "unknown_attribute":
        source = source.replace('"hw.array_get"(%array,%index) :', '"hw.array_get"(%array,%index) {opaque} :')
    elif case == "duplicate_ownership":
        source = source.replace(
            '"hw.array_get"(%array,%index) :', '"hw.array_get"(%array,%index) <{sv.namehint="x"}> {sv.namehint="x"} :'
        )
    elif case == "singleton":
        source = _source(1)
    elif case == "signed":
        source = source.replace("i7", "si7")
    elif case == "zero_element":
        source = source.replace("i7", "i0")
    elif case == "creation_attribute":
        source = source.replace('"hw.array_create"(%e0,%e1,%e2,%e3) :', '"hw.array_create"(%e0,%e1,%e2,%e3) {opaque} :')
    else:
        source = source.replace('"hw.array_create"', '"hw.unsupported_array_producer"')
    with pytest.raises(ValueError):
        _selection(source)


def test_undefined_indices_wrong_values_and_all_metadata_limits_refuse():
    selection, _ = _selection(_source(3))
    for values, index in [
        ((1, 2, 3), 3),
        ((1, 2, 3), -1),
        ((1, 2, 3), True),
        ((1, 2), 0),
        ((1, True, 3), 0),
        ((1, 2, 128), 0),
    ]:
        with pytest.raises(ValueError):
            selection.evaluate(values, index)
    for field in ArraySelectionLimits.__dataclass_fields__:
        for value in (0, -1, True, 1.0):
            with pytest.raises(ValueError):
                dataclasses.replace(LIMITS, **{field: value})


def _constants(count, width):
    body, outputs, expected = [], [], {}
    index_width = (count - 1).bit_length()
    for seed in range(count + 3):
        values = tuple(((i + 1) * 37 + seed * 13) & (2**width - 1) for i in range(count))
        for i, value in enumerate(values):
            body.append(f'%e{seed}_{i} = "hw.constant"() {{value={value}:i{width}}} : () -> i{width}')
        body.append(
            f'%a{seed} = "hw.array_create"('
            + ",".join(f"%e{seed}_{i}" for i in range(count))
            + ") : ("
            + ",".join([f"i{width}"] * count)
            + f") -> !hw.array<{count}xi{width}>"
        )
        for index, value in enumerate(reversed(values)):
            name = f"v{seed}_{index}"
            body.extend(
                [
                    f'%i{seed}_{index} = "hw.constant"() {{value={index}:i{index_width}}} : () -> i{index_width}',
                    f'%{name} = "hw.array_get"(%a{seed},%i{seed}_{index}) '
                    f": (!hw.array<{count}xi{width}>,i{index_width}) -> i{width}",
                ]
            )
            outputs.append((name, width))
            expected[name] = value
    body.append(
        '"hw.output"('
        + ",".join("%" + name for name, _ in outputs)
        + ") : ("
        + ",".join([f"i{width}"] * len(outputs))
        + ") -> ()"
    )
    return _module(body, [], outputs), expected


@pytest.fixture(params=["legacy", "modern"])
def native_tool(request):
    selected = os.environ.get("MERLIN_TEST_CIRCT_OPT" if request.param == "legacy" else "MERLIN_TEST_CIRCT_OPT_MODERN")
    if not selected:
        pytest.skip("array controls require explicitly selected coherent native CIRCT SDKs")
    return request.param, str(Path(selected).resolve(strict=True))


def _invoke(native_tool, tmp_path, source, stage, outputs=()):
    era, tool = native_tool
    return I.run(
        [
            tool,
            str(source),
            "--canonicalize",
            "--verify-each",
            "--mlir-print-op-generic",
            "-o",
            str(tmp_path / "folded.mlir"),
        ],
        directory=tmp_path,
        stage=stage + "_" + era,
        inputs=(source, Path(__file__)),
        outputs=outputs,
        dependencies=(
            module_source_path("merlin.targetgen.rtl.hw_array_selection"),
            module_source_path("xdsl.dialects.hw"),
            Path(I.__file__),
        ),
        env=ENVIRONMENT,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("count,width", [(2, 1), (3, 7), (4, 7), (8, 64), (16, 7)])
def test_native_both_eras_complete_original_element_index_outputs(native_tool, count, width, tmp_path):
    source = tmp_path / "original.mlir"
    text, expected = _constants(count, width)
    source.write_text(text)
    parsed = parse_generic_hw(text, reject_dense_literals=True)
    operations = list(next(op for op in parsed.walk() if _name(op) == "hw.module").regions[0].block.ops)
    preflight_array_selections({"Select": operations}, ["Select"], limits=LIMITS, scalar_bits=64)
    original_values = []
    for output_value in operations[-1].operands:
        selection = typed_array_selection(output_value.owner, scalar_bits=64, limits=LIMITS)
        # IntegerAttr stores signed representatives. Preserve the same original
        # source bits; this neither widens nor changes the declared signedness.
        values = tuple(
            _integer(value.owner, "value") & (2**selection.element_width - 1)
            for value in selection.original_creation_operands
        )
        index = _integer(selection.index.owner, "value") & (2**selection.index_width - 1)
        original_values.append(selection.evaluate(values, index))
    assert tuple(original_values) == tuple(expected.values())
    output = tmp_path / "folded.mlir"
    result = _invoke(native_tool, tmp_path, source, "native_array_complete_defined_outputs", (output,))
    assert result.returncode == 0, result.stderr
    observation = prepare_combinational_observation(
        output.read_text(), module="Select", limits=EvaluationLimits(1048576, 10000, 64, 1, 4000000)
    )
    assert all(row.kind == "hw.constant" for row in observation.expressions)
    assert observation.evaluate(({},)) == (expected,)
    assert [(port.name, port.width) for port in observation.outputs] == [(name, width) for name in expected]
    (tmp_path / "expected.json").write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
    for record in tmp_path.glob("invocations/*/invocation.json"):
        I.require_environment(record, environment=ENVIRONMENT)


@pytest.mark.parametrize("case", ["count", "index_width", "result_type", "element_type"])
def test_native_malformed_original_array_types_refuse(native_tool, case, tmp_path):
    source = _source()
    if case == "count":
        source = source.replace("4xi7", "3xi7")
    elif case == "index_width":
        source = source.replace("index: i2", "index: i3").replace("4xi7>,i2", "4xi7>,i3")
    elif case == "result_type":
        source = (
            source.replace("-> i7", "-> i8", 1)
            .replace("output picked: i7", "output picked: i8")
            .replace('"hw.output"(%value) : (i7)', '"hw.output"(%value) : (i8)')
        )
    else:
        source = (
            source.replace("%e0: i7", "%e0: i8")
            .replace("input e0: i7", "input e0: i8")
            .replace("(i7,i7,i7,i7)", "(i8,i7,i7,i7)")
        )
    path = tmp_path / "malformed.mlir"
    path.write_text(source)
    result = _invoke(native_tool, tmp_path, path, "native_array_schema_refusal")
    assert result.returncode != 0
    # Both selected SDKs return failure silently for creation/result count
    # mismatch. Preserve that refusal instead of inventing a diagnostic.
    if case != "count":
        assert result.stderr
    with pytest.raises(ValueError):
        _selection(source)
