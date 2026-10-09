"""Complete bounded bitvector observations never infer address/resource roles."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import merlin.targetgen.rtl.hw_combinational as observer_module
from merlin.common import invocation_record as I
from merlin.targetgen.rtl.hw_combinational import EvaluationLimits, prepare_combinational_observation
from merlin.targetgen.rtl.hw_graph import parse_generic_hw

LIMITS = EvaluationLimits(source_bytes=16384, nodes=64, scalar_bits=64, cases=4096, bit_work=2_000_000)


def _source():
    return """builtin.module {
      "hw.module"() ({
      ^bb0(%word: i6, %inc: i6, %mode: i1):
        %zero = "hw.constant"() {value = false} : () -> i1
        %all = "hw.constant"() {value = -1 : i4} : () -> i4
        %data = "comb.extract"(%word) {lowBit = 0 : i32} : (i6) -> i4
        %metadata = "comb.extract"(%word) {lowBit = 4 : i32} : (i6) -> i2
        %extended = "comb.concat"(%zero, %data) : (i1, i4) -> i5
        %delta = "comb.extract"(%inc) {lowBit = 0 : i32} : (i6) -> i5
        %sum = "comb.add"(%extended, %delta) : (i5, i5) -> i5
        %low = "comb.extract"(%sum) {lowBit = 0 : i32} : (i5) -> i4
        %small = "comb.extract"(%sum) {lowBit = 2 : i32} : (i5) -> i1
        %large = "comb.extract"(%sum) {lowBit = 4 : i32} : (i5) -> i1
        %carry = "comb.mux"(%mode, %small, %large) {twoState} : (i1, i1, i1) -> i1
        %joined = "comb.concat"(%metadata, %low) : (i2, i4) -> i6
        %equal = "comb.icmp"(%data, %all) {predicate = 0 : i64, twoState} : (i4, i4) -> i1
        %flag = "comb.and"(%mode, %equal) {twoState} : (i1, i1) -> i1
        "hw.output"(%metadata, %data, %joined, %carry, %flag) : (i2, i4, i6, i1, i1) -> ()
      }) {sym_name = "UnrelatedUnit", module_type = !hw.modty<input word : i6, input inc : i6,
        input mode : i1, output bank : i2, output row : i4, output updated : i6,
        output carry : i1, output flag : i1>, parameters = []} : () -> ()
    }"""


def _prepare(source=None, limits=LIMITS):
    return prepare_combinational_observation(
        _source() if source is None else source, module="UnrelatedUnit", limits=limits
    )


def test_complete_outputs_exhaustively_match_independent_modular_integer_expression():
    cases = tuple(
        {"word": word, "inc": delta, "mode": mode} for word in range(64) for delta in range(32) for mode in (0, 1)
    )
    expected = tuple(
        {
            "bank": case["word"] // 16,
            "row": case["word"] % 16,
            "updated": case["word"] // 16 * 16 + (case["word"] % 16 + case["inc"]) % 16,
            "carry": ((case["word"] % 16 + case["inc"]) // (4 if case["mode"] else 16)) % 2,
            "flag": int(case["mode"] and case["word"] % 16 == 15),
        }
        for case in cases
    )
    original = _prepare()
    assert original.evaluate(cases) == expected
    assert original.per_case_bit_work > sum(port.width for port in (*original.inputs, *original.outputs))
    renamed = _prepare(_source().replace("bank", "arbitrary_output"))
    assert tuple(renamed.evaluate(cases)[0]) == ("arbitrary_output", "row", "updated", "carry", "flag")


def test_changed_projection_changes_actual_complete_output_and_single_carry_is_not_range_check():
    case = {"word": 31, "inc": 1, "mode": 0}
    correct = _prepare().evaluate((case,))[0]
    broken = _prepare(
        _source().replace(
            '%metadata = "comb.extract"(%word) {lowBit = 4', '%metadata = "comb.extract"(%word) {lowBit = 3'
        )
    )
    wrong = broken.evaluate((case,))[0]
    assert correct["bank"] == 1 and wrong["bank"] == 3
    wrapped = _prepare().evaluate(({"word": 0, "inc": 32, "mode": 0},))[0]
    assert wrapped["updated"] == 0 and wrapped["carry"] == 0
    assert 32 >= 16  # This carry bit does not prove a full-width capacity check.


@pytest.mark.parametrize(
    "source",
    [
        _source().replace('"comb.add"', '"seq.firreg"'),
        _source().replace('"comb.add"', '"comb.xor"'),
        _source().replace("predicate = 0 : i64", "predicate = 99 : i64"),
        _source().replace("lowBit = 4 : i32} : (i6) -> i2", "lowBit = 5 : i32} : (i6) -> i2"),
        _source().replace("value = -1 : i4", "value = -1 : i3"),
        _source().replace("{twoState}", "{twoState = 1 : i1}"),
        _source().replace("parameters = []", 'parameters = [#hw.param.decl<"W" = 6 : i32> : i32]'),
        _source().replace("output flag : i1", "output flag : i2"),
        _source().replace("input mode : i1", "inout mode : i1"),
    ],
)
def test_opaque_or_ill_typed_semantics_refuse_before_stimuli(source):
    with pytest.raises(ValueError):
        _prepare(source)


def test_unreachable_unsupported_state_still_refuses_complete_module_observation():
    source = _source().replace(
        '"hw.output"(%metadata',
        '%unused = "seq.firreg"(%word) : (i6) -> i6\n "hw.output"(%metadata',
    )
    with pytest.raises(ValueError, match="unsupported"):
        _prepare(source)


@pytest.mark.parametrize("field,value", [("source_bytes", 16), ("nodes", 2), ("scalar_bits", 3), ("bit_work", 10)])
def test_source_derived_prepare_budget_refuses(field, value):
    with pytest.raises(ValueError, match="budget|bounded"):
        _prepare(limits=replace(LIMITS, **{field: value}))


def test_execution_budget_and_complete_unsigned_input_roster_refuse():
    prepared = _prepare(limits=replace(LIMITS, cases=1))
    case = {"word": 0, "inc": 1, "mode": 0}
    with pytest.raises(ValueError, match="budget"):
        prepared.evaluate((case, case))
    work_limited = _prepare(limits=replace(LIMITS, bit_work=prepared.per_case_bit_work))
    with pytest.raises(ValueError, match="budget"):
        work_limited.evaluate((case, case))
    for bad in (
        {"word": 0, "inc": 1},
        {**case, "other": 0},
        {**case, "word": -1},
        {**case, "mode": True},
        {**case, "inc": 64},
    ):
        with pytest.raises(ValueError, match="original"):
            prepared.evaluate((bad,))


def test_bool_and_unbounded_limit_or_stimulus_requests_refuse():
    with pytest.raises(ValueError, match="positive integers"):
        replace(LIMITS, cases=True)
    with pytest.raises(ValueError, match="bounded explicit sequence"):
        _prepare().evaluate(iter(({},)))


def test_small_dense_splat_source_refuses_before_large_parser_allocation():
    # Import from the actually selected package origin, including copied-wheel tests.
    package_root = Path(observer_module.__file__).resolve().parent.parent.parent.parent
    source = _source().replace("value = false", "value = dense<0> : tensor<1000000000xi8>")
    child = """import resource, sys, json
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
sys.path.insert(0, sys.argv[1])
from merlin.targetgen.rtl.hw_combinational import EvaluationLimits, prepare_combinational_observation
try:
    prepare_combinational_observation(
        json.loads(sys.argv[2]), module="UnrelatedUnit", limits=EvaluationLimits(16384,64,64,8,10000)
    )
except Exception as error:
    assert "unsupported before shaped parser allocation" in str(error), str(error)
else:
    raise AssertionError("dense source unexpectedly admitted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", child, str(package_root), json.dumps(source)],
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_historical_generic_parser_retains_small_dense_literals_without_scalar_mode():
    parsed = parse_generic_hw(_source().replace("value = false", "value = dense<0> : tensor<2xi8>"))
    module = next(op for op in parsed.walk() if op.attributes.get("sym_name") is not None)
    assert len(module.regions[0].block.first_op.attributes["value"]) == 2


@pytest.mark.parametrize("case", ["huge_integer", "nested_attributes", "dense_resource"])
def test_bounded_source_screen_refuses_before_integer_or_recursive_parser_allocation(case):
    package_root = Path(observer_module.__file__).resolve().parent.parent.parent.parent
    if case == "huge_integer":
        source = _source().replace("value = false", "value = 1 : i10000000000")
        message = "scalar width"
    elif case == "nested_attributes":
        source = "builtin.module attributes {x = " + "[" * 1500 + "0 : i1" + "]" * 1500 + "} {}"
        message = "nesting bound"
    else:
        source = _source().replace("value = false", "value = dense_resource<missing> : tensor<1000000000xi8>")
        message = "unsupported before shaped parser allocation"
    child = """import resource, sys, json
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
sys.path.insert(0, sys.argv[1])
from merlin.targetgen.contract.mlir_source_admission import MlirSourceUnavailable
from merlin.targetgen.rtl.hw_combinational import EvaluationLimits, prepare_combinational_observation
try:
    prepare_combinational_observation(
        json.loads(sys.argv[2]), module="UnrelatedUnit", limits=EvaluationLimits(16384,64,64,8,10000)
    )
except MlirSourceUnavailable as error:
    assert sys.argv[3] in str(error), str(error)
else:
    raise AssertionError("oversized source unexpectedly admitted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", child, str(package_root), json.dumps(source), message],
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_native_lowered_complete_outputs_match_independent_verilog_execution(tmp_path):
    selected = [os.environ.get(name) for name in ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_IVERILOG", "MERLIN_TEST_VVP")]
    if not all(selected):
        pytest.skip("explicit public firtool/Icarus selections are required")
    firtool, iverilog, vvp = (str(Path(path).resolve(strict=True)) for path in selected)
    source = tmp_path / "original.fir"
    source.write_text("""FIRRTL version 3.3.0
circuit NativeUnit :
  module NativeUnit :
    input word : UInt<6>
    input inc : UInt<6>
    input mode : UInt<1>
    output bank : UInt<2>
    output row : UInt<4>
    output updated : UInt<6>
    output carry : UInt<1>
    output flag : UInt<1>
    node data = bits(word, 3, 0)
    node metadata = bits(word, 5, 4)
    node sum = add(pad(data, 5), bits(inc, 4, 0))
    connect bank, metadata
    connect row, data
    connect updated, cat(metadata, bits(sum, 3, 0))
    connect carry, mux(mode, bits(sum, 2, 2), bits(sum, 4, 4))
    connect flag, and(mode, eq(data, UInt<4>(15)))
""")
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    hardware, verilog, bench, executable = (
        tmp_path / name for name in ("unit.mlir", "unit.sv", "bench.sv", "bench.vvp")
    )

    def invoke(argv, stage, inputs, outputs=()):
        return I.run(
            argv,
            directory=tmp_path / stage,
            stage=stage,
            inputs=(*inputs, Path(__file__)),
            outputs=outputs,
            dependencies=(Path(I.__file__),),
            env=environment,
            capture_output=True,
            check=True,
            timeout=30,
        )

    invoke(
        [firtool, str(source), "--ir-hw", "--mlir-print-op-generic", "-o", str(hardware)],
        "native_comb_hw",
        (source,),
        (hardware,),
    )
    invoke([firtool, str(source), "--verilog", "-o", str(verilog)], "native_comb_verilog", (source,), (verilog,))
    prepared = prepare_combinational_observation(hardware.read_text(), module="NativeUnit", limits=LIMITS)
    cases = tuple(
        {"word": word, "inc": delta, "mode": mode} for word in range(64) for delta in range(32) for mode in (0, 1)
    )
    observed = prepared.evaluate(cases)
    declarations = [f"logic [{port.width - 1}:0] {port.name};" for port in (*prepared.inputs, *prepared.outputs)]
    connections = ", ".join(f".{port.name}({port.name})" for port in (*prepared.inputs, *prepared.outputs))
    statements = []
    for ordinal, case in enumerate(cases):
        statements.extend(f"{port.name} = {port.width}'d{case[port.name]};" for port in prepared.inputs)
        values = ", ".join(port.name for port in prepared.outputs)
        statements.append(
            '#1; $display("ROW ' + str(ordinal) + " " + " ".join("%h" for _ in prepared.outputs) + '", ' + values + ");"
        )
    bench.write_text(
        "module Bench;\n"
        + "\n".join(declarations)
        + "\nNativeUnit dut("
        + connections
        + ");\ninitial begin\n"
        + "\n".join(statements)
        + "\n$finish;\nend\nendmodule\n"
    )
    invoke(
        [iverilog, "-g2012", "-s", "Bench", "-o", str(executable), str(verilog), str(bench)],
        "native_comb_compile",
        (verilog, bench),
        (executable,),
    )
    result = invoke([vvp, str(executable)], "native_comb_execute", (executable,))
    rows = [line.split() for line in result.stdout.decode().splitlines() if line.startswith("ROW ")]
    assert len(rows) == len(cases)
    for ordinal, (row, expected) in enumerate(zip(rows, observed, strict=True)):
        assert row[:2] == ["ROW", str(ordinal)] and len(row) == 2 + len(prepared.outputs)
        assert tuple(int(value, 16) for value in row[2:]) == tuple(expected[port.name] for port in prepared.outputs)
    for record in tmp_path.glob("*/invocations/*/invocation.json"):
        I.require_environment(record, environment=environment)
