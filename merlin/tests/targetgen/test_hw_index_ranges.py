"""Typed unsigned domain facts never grant source definedness or memory effects."""

import dataclasses
import json
import os
from pathlib import Path

import pytest
from xdsl.dialects.builtin import IntegerType, Signedness

from merlin.common import invocation_record as I
from merlin.targetgen.rtl.hw_combinational import EvaluationLimits, prepare_combinational_observation
from merlin.targetgen.rtl.hw_index_ranges import IndexRangeLimits, known_unsigned_index_domain

LIMITS = IndexRangeLimits(128, 64, 64, 8192)
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


@pytest.mark.parametrize("width,depth", [(1, 1), (1, 2), (2, 3), (2, 4), (3, 5), (3, 6), (3, 8), (5, 32)])
def test_complete_small_known_bit_domain_is_conditional_only(width, depth):
    fact = known_unsigned_index_domain(IntegerType(width), depth, limits=LIMITS)
    assert fact["conditional_domain_contained"] is all(value < depth for value in range(2**width))
    assert fact["original_address_type"] == "i" + str(width)
    assert fact["unsigned_upper_bound_exclusive_power_of_two"] == width
    assert fact["address_definedness_proved"] is False
    assert "defined known bits" in fact["premise"]


@pytest.mark.parametrize(
    "case",
    [
        "zero_width",
        "width_bool",
        "signed",
        "width_budget",
        "depth_zero",
        "depth_bool",
        "depth_budget",
        "proof_budget",
        "limit_bool",
    ],
)
def test_malformed_and_oversized_original_domains_refuse(case):
    typ, depth, limits = IntegerType(3), 8, LIMITS
    if case == "zero_width":
        typ = IntegerType(0)
    elif case == "width_bool":
        typ = IntegerType(True)
    elif case == "signed":
        typ = IntegerType(3, Signedness.SIGNED)
    elif case == "width_budget":
        typ = IntegerType(65)
    elif case == "depth_zero":
        depth = 0
    elif case == "depth_bool":
        depth = True
    elif case == "depth_budget":
        depth = 1 << 64
    elif case == "proof_budget":
        limits = dataclasses.replace(limits, proof_bits=6)
    else:
        with pytest.raises(ValueError):
            dataclasses.replace(limits, addresses=True)
        return
    with pytest.raises(ValueError):
        known_unsigned_index_domain(typ, depth, limits=limits)


def test_symbolic_large_width_never_materializes_exponential_endpoints():
    fact = known_unsigned_index_domain(IntegerType(100000), 7, limits=IndexRangeLimits(1, 100000, 64, 100064))
    assert fact["unsigned_upper_bound_exclusive_power_of_two"] == 100000
    assert fact["conditional_domain_contained"] is False
    assert len(json.dumps(fact)) < 512


@pytest.fixture(params=["legacy", "modern"])
def native_tool(request):
    key = "MERLIN_TEST_CIRCT_OPT" + ("_MODERN" if request.param == "modern" else "")
    if not os.environ.get(key):
        pytest.skip("index-domain controls require explicitly selected native CIRCT SDKs")
    return request.param, Path(os.environ[key]).resolve(strict=True)


def _known_bit_comparisons(width, depth):
    wide = max(width, depth.bit_length()) + 1
    body = [
        f'%zero = "hw.constant"() {{value = 0 : i{wide - width}}} : () -> i{wide - width}',
        f'%depth = "hw.constant"() {{value = {depth} : i{wide}}} : () -> i{wide}',
    ]
    expected = {}
    for value in range(2**width):
        body.extend(
            [
                f'%a{value} = "hw.constant"() {{value = {value} : i{width}}} : () -> i{width}',
                f'%u{value} = "comb.concat"(%zero,%a{value}) : (i{wide - width},i{width}) -> i{wide}',
                f'%o{value} = "comb.icmp"(%u{value},%depth) {{predicate = 6 : i64}} : (i{wide},i{wide}) -> i1',
            ]
        )
        expected[f"v{value}"] = int(value < depth)
    body.append(
        '"hw.output"('
        + ",".join(f"%o{i}" for i in range(len(expected)))
        + ") : ("
        + ",".join(["i1"] * len(expected))
        + ") -> ()"
    )
    ports = ",".join("output " + name + ":i1" for name in expected)
    return 'module { "hw.module"() ({\n' + "\n".join(
        body
    ) + '\n}) {sym_name="Domain",module_type=!hw.modty<' + ports + ">,parameters=[]} : () -> () }\n", expected


@pytest.mark.parametrize("width,depth", [(1, 2), (2, 4), (3, 5), (3, 6), (3, 8), (5, 32)])
def test_both_native_eras_complete_unsigned_bit_domain_outputs(native_tool, width, depth, tmp_path):
    era, tool = native_tool
    source, expected = _known_bit_comparisons(width, depth)
    path = tmp_path / "original.mlir"
    path.write_text(source)
    output = tmp_path / "folded.mlir"
    result = I.run(
        [str(tool), str(path), "--canonicalize", "--verify-each", "--mlir-print-op-generic", "-o", str(output)],
        directory=tmp_path,
        cwd=tmp_path,
        stage="native_unsigned_index_domain_" + era,
        inputs=(path, Path(__file__)),
        outputs=(output,),
        env=ENVIRONMENT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    observation = prepare_combinational_observation(
        output.read_text(), module="Domain", limits=EvaluationLimits(1048576, 2048, 64, 1, 65536)
    )
    assert observation.evaluate(({},)) == (expected,)
    assert [(port.name, port.width) for port in observation.outputs] == [(name, 1) for name in expected]
    fact = known_unsigned_index_domain(IntegerType(width), depth, limits=LIMITS)
    assert fact["conditional_domain_contained"] is all(value == 1 for value in expected.values())
    (tmp_path / "expected.json").write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
    for record in tmp_path.glob("invocations/*/invocation.json"):
        I.require_environment(record, environment=ENVIRONMENT)
