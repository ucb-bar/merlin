"""Conditional original instance cones retain boundaries and complete port types."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.targetgen.rtl.hw_combinational import EvaluationLimits
from merlin.targetgen.rtl.hw_instance_inputs import prepare_instance_input_observation

LIMITS = EvaluationLimits(16384, 80, 64, 1024, 1_000_000)


def _source():
    return """builtin.module {
      "hw.module.extern"() {sym_name="Producer", parameters=[],
        module_type=!hw.modty<output word:i6, output ready:i1>} : () -> ()
      "hw.module.extern"() {sym_name="Consumer", parameters=[],
        module_type=!hw.modty<input enable:i1, input addr:i4>} : () -> ()
      "hw.module"() ({
      ^bb0(%exec:i1, %external:i4, %blocked:i1, %step:i4):
        %word, %ready = "hw.instance"() {instanceName="source", moduleName=@Producer,
          argNames=[], resultNames=["word","ready"],parameters=[]} : () -> (i6,i1)
        %bank = "comb.extract"(%word) {lowBit=4:i32} : (i6) -> i2
        %low = "comb.extract"(%word) {lowBit=0:i32} : (i6) -> i4
        %row = "comb.sub"(%low,%step) : (i4,i4) -> i4
        %zero = "hw.constant"() {value=0:i2} : () -> i2
        %one = "hw.constant"() {value=true} : () -> i1
        %selected = "comb.icmp"(%bank,%zero) {predicate=0:i64} : (i2,i2) -> i1
        %allowed = "comb.xor"(%blocked,%one) : (i1,i1) -> i1
        %dma = "comb.and"(%ready,%allowed,%selected) : (i1,i1,i1) -> i1
        %enable = "comb.or"(%exec,%dma) : (i1,i1) -> i1
        %addr = "comb.mux"(%exec,%external,%row) : (i1,i4,i4) -> i4
        "hw.instance"(%enable,%addr) {instanceName="sink", moduleName=@Consumer,
          argNames=["enable","addr"],resultNames=[],parameters=[]} : (i1,i4) -> ()
        "hw.output"() : () -> ()
      }) {sym_name="Parent",parameters=[],module_type=!hw.modty<input exec:i1,input external:i4,
        input blocked:i1,input step:i4>} : () -> ()
    }"""


def _prepare(source=None, ports=("enable", "addr"), limits=LIMITS):
    return prepare_instance_input_observation(
        _source() if source is None else source, module="Parent", instance="sink", ports=ports, limits=limits
    )


def _case(observation, **values):
    return {root.port.name: values[root.name] for root in observation.roots}


def test_actual_complete_selected_sinks_match_independent_bounded_conditional_expression():
    observed = _prepare()
    rows = tuple(
        {"word": word, "ready": ready, "exec": executing, "external": 13, "blocked": blocked, "step": 3}
        for word in range(64)
        for ready in (0, 1)
        for executing in (0, 1)
        for blocked in (0, 1)
    )
    expected = tuple(
        {
            "enable": int(row["exec"] or (row["ready"] and not row["blocked"] and row["word"] // 16 == 0)),
            "addr": row["external"] if row["exec"] else (row["word"] % 16 - row["step"]) % 16,
        }
        for row in rows
    )
    assert observed.evaluate(tuple(_case(observed, **row) for row in rows)) == expected
    assert tuple((p.direction, p.name, p.type) for p in observed.consumer_ports) == (
        ("input", "enable", "i1"),
        ("input", "addr", "i4"),
    )
    assert {(root.kind, root.instance, root.module, root.name) for root in observed.roots if root.instance} == {
        ("opaque_instance_result", "source", "Producer", "word"),
        ("opaque_instance_result", "source", "Producer", "ready"),
    }


def test_wrong_source_partition_changes_observed_guard_and_no_validity_is_invented():
    original = _prepare()
    broken = _prepare(_source().replace("lowBit=4:i32", "lowBit=3:i32"))
    values = {"word": 8, "ready": 1, "exec": 0, "external": 13, "blocked": 0, "step": 3}
    assert original.evaluate((_case(original, **values),))[0]["enable"] == 1
    assert broken.evaluate((_case(broken, **values),))[0]["enable"] == 0
    values["ready"] = 0
    # The local two-state expression still has an address, but no transfer is valid.
    assert original.evaluate((_case(original, **values),))[0] == {"enable": 0, "addr": 5}
    with pytest.raises(ValueError, match="unsigned bitvector"):
        original.evaluate((_case(original, **{**values, "word": 64}),))


@pytest.mark.parametrize(
    "source",
    [
        _source().replace('argNames=["enable","addr"]', 'argNames=["addr","enable"]'),
        _source().replace("input addr:i4", "input addr:i5"),
        _source().replace('resultNames=["word","ready"]', 'resultNames=["ready","word"]'),
        _source().replace("moduleName=@Producer", "moduleName=@Missing"),
        _source().replace('"comb.sub"', '"seq.firreg"'),
        _source().replace('"comb.sub"', '"unknown.expression"'),
        _source().replace("input blocked:i1", "inout blocked:i1"),
    ],
)
def test_incomplete_bindings_state_and_unsupported_cones_refuse(source):
    with pytest.raises(ValueError):
        _prepare(source)


def test_selected_sink_roster_and_source_derived_budget_are_enforced():
    for ports in ((), ("enable", "enable"), ("not_an_input",), ["enable"]):
        with pytest.raises(ValueError):
            _prepare(ports=ports)
    for field, value in (("source_bytes", 16), ("nodes", 8), ("scalar_bits", 3), ("bit_work", 10)):
        with pytest.raises(ValueError):
            _prepare(limits=replace(LIMITS, **{field: value}))


def test_explicit_partial_sink_selection_preserves_unobserved_complete_consumer_roster():
    source = _source().replace('"comb.sub"', '"seq.firreg"')
    observed = _prepare(source, ports=("enable",))
    assert tuple(p.name for p in observed.consumer_ports) == ("enable", "addr")
    assert tuple(p.name for p in observed.expression.outputs) == ("enable",)
    with pytest.raises(ValueError, match="stateful"):
        _prepare(source)


def test_native_original_instance_inputs_match_complete_independent_verilog_outputs(tmp_path):
    tools = [os.environ.get(name) for name in ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_IVERILOG", "MERLIN_TEST_VVP")]
    if not all(tools):
        pytest.skip("explicit public firtool/Icarus selections are required")
    firtool, iverilog, vvp = (str(Path(path).resolve(strict=True)) for path in tools)
    source, hw, verilog, bench, image = (
        tmp_path / name for name in ("original.fir", "original.mlir", "original.sv", "bench.sv", "bench.vvp")
    )
    source.write_text("""FIRRTL version 3.3.0
circuit Parent :
  module Producer :
    input inword : UInt<6>
    input inready : UInt<1>
    output word : UInt<6>
    output ready : UInt<1>
    connect word, inword
    connect ready, inready
  module Consumer :
    input enable : UInt<1>
    input addr : UInt<4>
    output outenable : UInt<1>
    output outaddr : UInt<4>
    connect outenable, enable
    connect outaddr, addr
  module Parent :
    input word : UInt<6>
    input ready : UInt<1>
    input exec : UInt<1>
    input external : UInt<4>
    input blocked : UInt<1>
    input step : UInt<4>
    output enable : UInt<1>
    output addr : UInt<4>
    inst source of Producer
    inst sink of Consumer
    connect source.inword, word
    connect source.inready, ready
    node low = bits(source.word,3,0)
    node selected = eq(bits(source.word,5,4), UInt<2>(0))
    node dma = and(and(source.ready, not(blocked)), selected)
    connect sink.enable, or(exec,dma)
    connect sink.addr, mux(exec,external,bits(sub(low,step),3,0))
    connect enable, sink.outenable
    connect addr, sink.outaddr
""")
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}

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
        [firtool, str(source), "--ir-hw", "--mlir-print-op-generic", "-o", str(hw)], "native_cone_hw", (source,), (hw,)
    )
    invoke([firtool, str(source), "--verilog", "-o", str(verilog)], "native_cone_sv", (source,), (verilog,))
    observed = prepare_instance_input_observation(
        hw.read_text(), module="Parent", instance="sink", ports=("enable", "addr"), limits=LIMITS
    )
    rows = tuple(
        {"word": word, "ready": ready, "exec": executing, "external": 13, "blocked": blocked, "step": 3}
        for word in range(64)
        for ready in (0, 1)
        for executing in (0, 1)
        for blocked in (0, 1)
    )
    outputs = observed.evaluate(tuple(_case(observed, **row) for row in rows))
    bench.write_text(
        "module Bench; logic [5:0] word; logic ready, exec, blocked, enable; logic [3:0] external, step, addr;\n"
        "Parent dut(.*); initial begin\n"
        + "\n".join(
            ";".join(f"{name}={value}" for name, value in row.items()) + ';#1;$display("ROW %h %h",enable,addr);'
            for row in rows
        )
        + "\n$finish;end endmodule\n"
    )
    invoke(
        [iverilog, "-g2012", "-s", "Bench", "-o", str(image), str(verilog), str(bench)],
        "native_cone_compile",
        (verilog, bench),
        (image,),
    )
    executed = invoke([vvp, str(image)], "native_cone_execute", (image,))
    actual = [line.split() for line in executed.stdout.decode().splitlines() if line.startswith("ROW ")]
    assert len(actual) == len(outputs)
    assert tuple(tuple(int(value, 16) for value in row[1:]) for row in actual) == tuple(
        (row["enable"], row["addr"]) for row in outputs
    )
    for record in tmp_path.glob("*/invocations/*/invocation.json"):
        I.require_environment(record, environment=environment)
