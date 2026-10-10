"""Explicit native SDKs check original bit order and structural memory joins."""

import json
import os
from pathlib import Path

import pytest
from test_hw_partition_memory_bindings import HIERARCHY, LIMITS, LOCAL

from merlin.common import invocation_record as I
from merlin.targetgen.rtl.hw_combinational import EvaluationLimits, prepare_combinational_observation
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_partition_memory_bindings import partition_memory_bindings
from merlin.targetgen.rtl.source_selection import produce_selection

ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


@pytest.fixture(params=["legacy", "modern"])
def native_tools(request):
    suffix = "_MODERN" if request.param == "modern" else ""
    keys = ("MERLIN_TEST_FIRTOOL" + suffix, "MERLIN_TEST_CIRCT_OPT" + suffix)
    if any(not os.environ.get(key) for key in keys):
        pytest.skip("partition controls require two explicitly selected coherent native CIRCT pairs")
    return request.param, *(Path(os.environ[key]).resolve(strict=True) for key in keys)


def _firrtl(kind):
    source = "FIRRTL version 2.0.0\ncircuit Root :\n"
    if kind == "opaque":
        source += "  extmodule Unknown :\n    output word : UInt<8>\n    defname = Unknown\n"
    source += (
        "  module Cell :\n    input clock : Clock\n    input index : UInt<2>\n"
        "    input consent : UInt<1>\n    input word : UInt<8>\n    output seen : UInt<8>\n"
    )
    for index in range(2):
        source += (
            f"    mem m{index} :\n      data-type => UInt<4>\n      depth => 3\n"
            "      read-latency => 0\n      write-latency => 1\n      reader => r\n      writer => w\n"
            "      read-under-write => undefined\n"
            f"    m{index}.r.addr <= index\n    m{index}.r.en <= consent\n    m{index}.r.clk <= clock\n"
            f"    m{index}.w.addr <= index\n    m{index}.w.en <= consent\n    m{index}.w.clk <= clock\n"
            f"    m{index}.w.data <= bits(word, {index * 4 + 3}, {index * 4})\n"
            f"    m{index}.w.mask <= consent\n"
        )
    source += "    seen <= cat(m1.r.data, m0.r.data)\n"
    source += (
        "  module Root :\n    input clock : Clock\n    input index : UInt<2>\n    input consent : UInt<1>\n"
        "    input choice : UInt<1>\n    input x : UInt<8>\n    input y : UInt<8>\n    output seen : UInt<8>\n"
    )
    if kind == "state":
        source += "    reg retained : UInt<8>, clock\n    retained <= x\n    node word = retained\n"
    elif kind == "opaque":
        source += "    inst other of Unknown\n    node word = other.word\n"
    elif kind == "unsupported":
        source += "    node word = div(x, y)\n"
    elif kind == "mux":
        source += "    node word = mux(choice, x, y)\n"
    else:
        source += "    node word = x\n"
    for name, word in (("a", "word"), ("b", "a.seen" if kind == "read" else "y")):
        source += (
            f"    inst {name} of Cell\n    {name}.clock <= clock\n    {name}.index <= index\n"
            f"    {name}.consent <= consent\n    {name}.word <= {word}\n"
        )
    return source + "    seen <= xor(a.seen, b.seen)\n"


@pytest.mark.parametrize("kind", ["direct", "mux", "state", "read", "opaque", "unsupported"])
def test_both_native_eras_preserve_all_memory_ports_and_intermediate_partition_stops(native_tools, kind, tmp_path):
    era, firtool, circt_opt = native_tools
    original = tmp_path / "original.fir"
    original.write_text(_firrtl(kind))
    bundle = produce_selection(
        target="test_unit",
        firrtl=original,
        generator="test_unit",
        config="IndependentBitConnections",
        core_root="Root",
        firtool=firtool,
        output=tmp_path / "production",
    )
    selected = json.loads(bundle.read_bytes())
    core = Path(selected["sources"]["core_hw"]["path"])
    generic = tmp_path / "original.generic.mlir"
    I.run(
        [str(circt_opt), str(core), "--verify-each", "--mlir-print-op-generic", "-o", str(generic)],
        directory=tmp_path / "serialization",
        stage="native_partition_memory_source_" + era,
        inputs=(original, bundle, core, Path(__file__)),
        outputs=(generic,),
        cwd=tmp_path,
        env=ENVIRONMENT,
        capture_output=True,
        check=True,
        timeout=30,
    )
    record = partition_memory_bindings(
        parse_generic_hw(generic.read_text(), reject_dense_literals=True),
        root="Root",
        local_limits=LOCAL,
        hierarchy_limits=HIERARCHY,
        limits=LIMITS,
    )
    parts = record["local_partitions"]["partitions"]
    cell = next(index for index, row in enumerate(parts) if row["module"] == "Cell")
    intervals = [
        (piece["frame"], piece["source_low_bit"], piece["destination_low_bit"], piece["width"])
        for row in record["ports"]
        for piece in row["intervals"]
        if piece["partition"] == cell
    ]
    occurrences = [row["id"] for row in record["hierarchy"]["frames"] if row["module"] == "Cell"]
    assert len(occurrences) == 2
    assert sorted(intervals) == [(frame, low, 0, 4) for frame in occurrences for low in (0, 4)]
    assert len(record["ports"]) == 8
    assert sum(row["data_status"] == "no_write_data_operand" for row in record["ports"]) == 4
    nodes = {row["id"]: row for row in record["hierarchy"]["expressions"]}
    cuts = [nodes[identity] for row in record["ports"] for identity in row["cuts"]]
    expected = {
        "state": "state_result",
        "read": "memory_read_result",
        "opaque": "opaque_instance_result",
        "unsupported": "unsupported_result",
    }
    if kind in expected:
        assert expected[kind] in {row["kind"] for row in cuts}
    if kind == "mux":
        assert "comb.mux" in {row.get("expression") for row in cuts}
    assert all(memory["declaration"]["read_under_write"] == "undefined" for memory in record["hierarchy"]["memories"])
    assert record["source_values_evaluated"] is False
    assert record["packing_mapping_admission"] is False
    (tmp_path / "conditional-bit-bindings.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


def test_both_native_eras_complete_extract_concat_truth_roster(native_tools, tmp_path):
    era, _, tool = native_tools
    body, outputs, expected = [], [], {}
    for value in range(256):
        body.extend(
            [
                f'%v{value} = "hw.constant"() {{value={value}:i8}} : () -> i8',
                f'%l{value} = "comb.extract"(%v{value}) {{lowBit=0:i32}} : (i8) -> i4',
                f'%h{value} = "comb.extract"(%v{value}) {{lowBit=4:i32}} : (i8) -> i4',
                f'%s{value} = "comb.concat"(%l{value},%h{value}) : (i4,i4) -> i8',
            ]
        )
        for prefix, width, answer in (
            ("l", 4, value & 15),
            ("h", 4, value >> 4),
            ("s", 8, (value & 15) * 16 + (value >> 4)),
        ):
            outputs.append((prefix + str(value), width))
            expected[prefix + str(value)] = answer
    body.append(
        '"hw.output"('
        + ",".join("%" + name for name, _ in outputs)
        + ") : ("
        + ",".join("i" + str(width) for _, width in outputs)
        + ") -> ()"
    )
    source = (
        'module { "hw.module"() ({\n'
        + "\n".join(body)
        + '\n}) {sym_name="Truth",module_type=!hw.modty<'
        + ",".join("output " + name + ":i" + str(width) for name, width in outputs)
        + ">,parameters=[]} : () -> () }\n"
    )
    original, folded = tmp_path / "original.mlir", tmp_path / "folded.mlir"
    original.write_text(source)
    I.run(
        [str(tool), str(original), "--canonicalize", "--verify-each", "--mlir-print-op-generic", "-o", str(folded)],
        directory=tmp_path / "folding",
        stage="native_partition_complete_bit_truth_" + era,
        inputs=(original, Path(__file__)),
        outputs=(folded,),
        cwd=tmp_path,
        env=ENVIRONMENT,
        capture_output=True,
        check=True,
        timeout=30,
    )
    observation = prepare_combinational_observation(
        folded.read_text(), module="Truth", limits=EvaluationLimits(1048576, 4096, 16, 1, 1048576)
    )
    assert [(port.name, port.width) for port in observation.outputs] == outputs
    assert observation.evaluate(({},)) == (expected,)
    (tmp_path / "complete-truth-roster.json").write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
