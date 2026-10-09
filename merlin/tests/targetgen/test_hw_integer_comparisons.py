"""Selected public integer predicates preserve all finite source-bit outputs."""

import dataclasses
import json
import os
from pathlib import Path

import pytest
from xdsl.dialects.comb import ICMP_COMPARISON_OPERATIONS

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.rtl.hw_combinational import EvaluationLimits, prepare_combinational_observation

PREDICATES = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
LIMITS = EvaluationLimits(1_048_576, 20_000, 64, 4096, 4_000_000)
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def _module(body, inputs, outputs):
    arguments = ", ".join(f"%{name}: i{width}" for name, width in inputs)
    block = f"^bb0({arguments}):\n" if arguments else ""
    ports = ", ".join(
        f"{direction} {name}: i{width}"
        for direction, roster in (("input", inputs), ("output", outputs))
        for name, width in roster
    )
    return (
        'module { "hw.module"() ({\n'
        + block
        + "\n".join(body)
        + '\n}) {sym_name="Compare", module_type=!hw.modty<'
        + ports
        + ">, parameters=[]} : () -> ()\n}\n"
    )


def _dynamic_source(width, *, two_state=False):
    flag = ", twoState" if two_state else ""
    body = [
        f'%v{index} = "comb.icmp"(%left,%right) {{predicate={index}:i64{flag}}} : (i{width},i{width}) -> i1'
        for index in range(10)
    ]
    body.append('"hw.output"(' + ",".join(f"%v{i}" for i in range(10)) + ") : (" + ",".join(["i1"] * 10) + ") -> ()")
    return _module(body, [("left", width), ("right", width)], [(name, 1) for name in PREDICATES])


def _truth(width, left, right):
    # Mathematical signed representatives of the same original N source bits.
    domain = 2**width
    signed_left = left if left < domain // 2 else left - domain
    signed_right = right if right < domain // 2 else right - domain
    return dict(
        zip(
            PREDICATES,
            map(
                int,
                (
                    left == right,
                    left != right,
                    signed_left < signed_right,
                    signed_left <= signed_right,
                    signed_left > signed_right,
                    signed_left >= signed_right,
                    left < right,
                    left <= right,
                    left > right,
                    left >= right,
                ),
            ),
            strict=True,
        )
    )


def _cases(width):
    if width <= 3:
        values = range(2**width)
    else:
        half = 2 ** (width - 1)
        values = (0, 1, half - 1, half, half + 1, 2**width - 2, 2**width - 1)
    return tuple((left, right) for left in values for right in values)


def _prepare(source, *, limits=LIMITS):
    return prepare_combinational_observation(source, module="Compare", limits=limits)


@pytest.mark.parametrize("width", [1, 3, 9, 64])
@pytest.mark.parametrize("two_state", [False, True])
def test_complete_original_truth_roster_and_exact_predicate_widths(width, two_state):
    assert tuple(ICMP_COMPARISON_OPERATIONS) == PREDICATES
    prepared = _prepare(_dynamic_source(width, two_state=two_state))
    pairs = _cases(width)
    assert prepared.evaluate(tuple({"left": left, "right": right} for left, right in pairs)) == tuple(
        _truth(width, left, right) for left, right in pairs
    )
    assert tuple(row.parameter for row in prepared.expressions) == tuple(range(10))
    assert [(row.name, row.width) for row in prepared.inputs] == [("left", width), ("right", width)]
    assert [(row.name, row.width) for row in prepared.outputs] == [(name, 1) for name in PREDICATES]


def test_exhaustive_five_bit_truth_and_legacy_prepared_equality():
    source = _dynamic_source(5)
    pairs = tuple((left, right) for left in range(32) for right in range(32))
    prepared = _prepare(source)
    assert prepared.evaluate(tuple({"left": left, "right": right} for left, right in pairs)) == tuple(
        _truth(5, left, right) for left, right in pairs
    )
    legacy = dataclasses.replace(
        prepared, expressions=(dataclasses.replace(prepared.expressions[0], parameter=None), *prepared.expressions[1:])
    )
    assert legacy.evaluate(({"left": 31, "right": 0}, {"left": 31, "right": 31})) == prepared.evaluate(
        ({"left": 31, "right": 0}, {"left": 31, "right": 31})
    )


def _malformed(case):
    source = _dynamic_source(3, two_state=True)
    changes = {
        "missing_predicate": ("predicate=0:i64, twoState", "twoState"),
        "negative_predicate": ("predicate=0:i64", "predicate=-1:i64"),
        "unknown_predicate": ("predicate=0:i64", "predicate=99:i64"),
        "outside_selected_roster": ("predicate=0:i64", "predicate=10:i64"),
        "predicate_type": ("predicate=0:i64", "predicate=0:i32"),
        "predicate_float": ("predicate=0:i64", "predicate=0.0:f64"),
        "two_state_type": ("predicate=0:i64, twoState", "predicate=0:i64, twoState=false"),
        "unknown_attribute": ("predicate=0:i64, twoState", "predicate=0:i64, twoState, opaque"),
        "result_width": ("} : (i3,i3) -> i1", "} : (i3,i3) -> i2"),
        "duplicate_ownership": ("{predicate=0:i64, twoState}", "<{predicate=0:i64}> {predicate=0:i64, twoState}"),
    }
    if case == "result_width":
        source = source.replace("output eq: i1", "output eq: i2").replace(
            "(i1,i1,i1,i1,i1,i1,i1,i1,i1,i1) -> ()", "(i2,i1,i1,i1,i1,i1,i1,i1,i1,i1) -> ()"
        )
    if case == "operand_width":
        return (
            source.replace("%right: i3", "%right: i5")
            .replace("input right: i3", "input right: i5")
            .replace("(i3,i3)", "(i3,i5)")
        )
    if case == "signed_type":
        return source.replace("i3", "si3")
    if case == "zero_width":
        return source.replace("i3", "i0")
    before, after = changes[case]
    return source.replace(before, after, 1)


@pytest.mark.parametrize(
    "case",
    [
        "missing_predicate",
        "negative_predicate",
        "unknown_predicate",
        "outside_selected_roster",
        "predicate_type",
        "predicate_float",
        "two_state_type",
        "unknown_attribute",
        "result_width",
        "duplicate_ownership",
        "operand_width",
        "signed_type",
        "zero_width",
    ],
)
def test_malformed_original_fields_types_and_unsupported_rosters_refuse(case):
    with pytest.raises(ValueError):
        _prepare(_malformed(case))


def test_comparison_input_and_resource_limits_remain_explicit():
    prepared = _prepare(_dynamic_source(3))
    for case in ({"left": -1, "right": 0}, {"left": 8, "right": 0}, {"left": True, "right": 0}, {"left": 0}):
        with pytest.raises(ValueError):
            prepared.evaluate((case,))
    with pytest.raises(ValueError):
        _prepare(_dynamic_source(3), limits=dataclasses.replace(LIMITS, scalar_bits=2))
    with pytest.raises(ValueError):
        prepared.evaluate(tuple({"left": 0, "right": 0} for _ in range(LIMITS.cases + 1)))


def _constant_source(width, two_state):
    body, outputs, expected = [], [], {}
    flag = ", twoState" if two_state else ""
    for ordinal, (left, right) in enumerate(_cases(width)):
        body.extend(
            (
                f'%a{ordinal} = "hw.constant"() {{value={left}:i{width}}} : () -> i{width}',
                f'%b{ordinal} = "hw.constant"() {{value={right}:i{width}}} : () -> i{width}',
            )
        )
        for predicate, (name, value) in enumerate(_truth(width, left, right).items()):
            output = f"v{ordinal}_{name}"
            body.append(
                f'%{output} = "comb.icmp"(%a{ordinal},%b{ordinal}) '
                f"{{predicate={predicate}:i64{flag}}} : (i{width},i{width}) -> i1"
            )
            outputs.append((output, 1))
            expected[output] = value
    body.append(
        '"hw.output"('
        + ",".join("%" + name for name, _ in outputs)
        + ") : ("
        + ",".join(["i1"] * len(outputs))
        + ") -> ()"
    )
    return _module(body, [], outputs), expected


@pytest.fixture(params=["legacy", "modern"])
def native_tool(request):
    selected = os.environ.get("MERLIN_TEST_CIRCT_OPT" if request.param == "legacy" else "MERLIN_TEST_CIRCT_OPT_MODERN")
    if not selected:
        pytest.skip("comparison controls require explicitly selected native tools in both eras")
    return request.param, str(Path(selected).resolve(strict=True))


def _invoke(tool, tmp_path, source, stage, *, outputs=()):
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
        stage=stage,
        inputs=(source, Path(__file__)),
        outputs=outputs,
        dependencies=(
            module_source_path("merlin.targetgen.rtl.hw_combinational"),
            module_source_path("xdsl.dialects.comb"),
            Path(I.__file__),
        ),
        env=ENVIRONMENT,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("width", [1, 3, 9, 64])
@pytest.mark.parametrize("two_state", [False, True])
def test_native_both_eras_fold_complete_original_truth_roster(native_tool, width, two_state, tmp_path):
    era, tool = native_tool
    text, expected = _constant_source(width, two_state)
    source = tmp_path / "original.mlir"
    source.write_text(text)
    assert _prepare(text).evaluate(({},)) == (expected,)
    output = tmp_path / "folded.mlir"
    result = _invoke(tool, tmp_path, source, "native_integer_comparison_truth_" + era, outputs=(output,))
    assert result.returncode == 0, result.stderr
    folded = _prepare(output.read_text())
    assert all(row.kind == "hw.constant" for row in folded.expressions)
    assert folded.evaluate(({},)) == (expected,)
    assert tuple(port.name for port in folded.outputs) == tuple(expected)
    (tmp_path / "expected.json").write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
    for record in tmp_path.glob("invocations/*/invocation.json"):
        I.require_environment(record, environment=ENVIRONMENT)


@pytest.mark.parametrize(
    "case",
    ["negative_predicate", "unknown_predicate", "predicate_type", "operand_width", "result_width", "two_state_type"],
)
def test_native_original_malformed_schema_controls(native_tool, case, tmp_path):
    era, tool = native_tool
    source = tmp_path / "malformed.mlir"
    source.write_text(_malformed(case))
    result = _invoke(tool, tmp_path, source, "native_integer_comparison_schema_refusal_" + era)
    assert result.returncode != 0, (case, (tmp_path / "folded.mlir").read_text())
    assert result.stderr
    with pytest.raises(ValueError):
        _prepare(source.read_text())
