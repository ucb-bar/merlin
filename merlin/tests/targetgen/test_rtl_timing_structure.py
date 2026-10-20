"""Structural timing facts read from a small synthetic elaborated HW design."""

import json

import pytest

from merlin.targetgen.rtl import timing_structure as ts
from merlin.targetgen.rtl.hw_design import Stages, load_design


def _module(name, ports, body, results):
    args = ", ".join(f"%{port}: {typ}" for direction, port, typ in ports if direction == "input")
    modty = ", ".join(f"{direction} {port} : {typ}" for direction, port, typ in ports)
    outs = ", ".join(value for value, _ in results)
    types = ", ".join(typ for _, typ in results)
    return (
        '  "hw.module"() ({\n'
        f"  ^bb0({args}):\n"
        + "".join(f"    {line}\n" for line in body)
        + f'    "hw.output"({outs}) : ({types}) -> ()\n'
        f'  }}) {{module_type = !hw.modty<{modty}>, parameters = [], sym_name = "{name}"}} : () -> ()\n'
    )


CLK = ("input", "clock", "!seq.clock")
MEM = (
    '"seq.firmem"() <{name = "ram", prefix = "", readLatency = 0 : i32, ruw = 0 : i32, '
    "writeLatency = 1 : i32, wuw = 1 : i32}> : () -> !seq.firmem<4 x 8>"
)


def _design_text() -> str:
    fifo = _module(
        "Fifo",
        [
            CLK,
            ("input", "io_enq", "i8"),
            ("input", "io_wen", "i1"),
            ("input", "io_ren", "i1"),
            ("output", "io_deq", "i8"),
        ],
        [
            '%one = "hw.constant"() {value = 1 : i2} : () -> i2',
            f"%mem = {MEM}",
            '%wp = "seq.firreg"(%wn, %clock) <{name = "wptr"}> : (i2, !seq.clock) -> i2',
            '%rp = "seq.firreg"(%rn, %clock) <{name = "rptr"}> : (i2, !seq.clock) -> i2',
            '%wa = "comb.add"(%wp, %one) : (i2, i2) -> i2',
            '%wn = "comb.mux"(%io_wen, %wa, %wp) : (i1, i2, i2) -> i2',
            '%ra = "comb.add"(%rp, %one) : (i2, i2) -> i2',
            '%rn = "comb.mux"(%io_ren, %ra, %rp) : (i1, i2, i2) -> i2',
            '"seq.firmem.write_port"(%mem, %wp, %clock, %io_wen, %io_enq) '
            "<{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!seq.firmem<4 x 8>, i2, !seq.clock, i1, i8) -> ()",
            '%deq = "seq.firmem.read_port"(%mem, %rp, %clock) : (!seq.firmem<4 x 8>, i2, !seq.clock) -> i8',
        ],
        [("%deq", "i8")],
    )
    enables = [f"io_en{i}" for i in range(5)]
    table_body = []
    for i in range(5):
        table_body += [
            f'%s{i} = "seq.firreg"(%n{i}, %clock) <{{name = "slot{i}"}}> : (i8, !seq.clock) -> i8',
            f'%n{i} = "comb.mux"(%{enables[i]}, %io_alloc, %s{i}) : (i1, i8, i8) -> i8',
        ]
    table_body += [
        '%ia = "comb.mux"(%io_sel0, %s1, %s0) : (i1, i8, i8) -> i8',
        '%ib0 = "comb.mux"(%io_sel0, %s3, %s2) : (i1, i8, i8) -> i8',
        '%ib = "comb.mux"(%io_sel1, %s4, %ib0) : (i1, i8, i8) -> i8',
    ]
    table = _module(
        "Table",
        [CLK, ("input", "io_alloc", "i8"), *[("input", e, "i1") for e in enables]]
        + [("input", "io_sel0", "i1"), ("input", "io_sel1", "i1")]
        + [("output", "io_issue_a_data", "i8"), ("output", "io_issue_b_data", "i8")],
        table_body,
        [("%ia", "i8"), ("%ib", "i8")],
    )
    consumer = _module(
        "Consumer",
        [CLK, ("input", "io_cmd", "i8"), ("input", "io_go", "i1"), ("output", "io_out", "i8")],
        [
            '%q = "hw.instance"(%clock, %io_cmd, %io_go, %io_go) {argNames = ["clock", "io_enq", "io_wen", "io_ren"], '
            'instanceName = "cmd_q", moduleName = @Fifo, parameters = [], resultNames = ["io_deq"]} '
            ": (!seq.clock, i8, i1, i1) -> i8",
            '%work = "seq.firreg"(%q, %clock) <{name = "work"}> : (i8, !seq.clock) -> i8',
        ],
        [("%work", "i8")],
    )
    slots = [f'%e{i} = "seq.firreg"(%m{i}, %clock) <{{name = "entry{i}"}}> : (i8, !seq.clock) -> i8' for i in range(4)]
    slots += [f'%m{i} = "comb.mux"(%io_alloc, %x_a_bits_data_lo, %e{i}) : (i1, i8, i8) -> i8' for i in range(4)]
    client = _module(
        "Client",
        [
            CLK,
            ("input", "io_alloc", "i1"),
            ("input", "io_big", "i1"),
            ("input", "x_a_bits_data_lo", "i8"),
            ("input", "x_a_bits_data_hi", "i8"),
            ("input", "x_d_bits_source", "i2"),
            ("output", "x_a_bits_data", "i16"),
            ("output", "x_a_bits_size", "i3"),
            ("output", "x_a_bits_source", "i2"),
            ("output", "io_peek", "i8"),
        ],
        [
            '%two = "hw.constant"() {value = 2 : i3} : () -> i3',
            '%three = "hw.constant"() {value = 3 : i3} : () -> i3',
            '%size = "comb.mux"(%io_big, %three, %two) : (i1, i3, i3) -> i3',
            '%data = "comb.concat"(%x_a_bits_data_hi, %x_a_bits_data_lo) : (i8, i8) -> i16',
            '%id = "seq.firreg"(%id, %clock) <{name = "next_id"}> : (i2, !seq.clock) -> i2',
            *slots,
            '%arr = "hw.array_create"(%e3, %e2, %e1, %e0) : (i8, i8, i8, i8) -> !hw.array<4xi8>',
            '%peek = "hw.array_get"(%arr, %x_d_bits_source) : (!hw.array<4xi8>, i2) -> i8',
        ],
        [("%data", "i16"), ("%size", "i3"), ("%id", "i2"), ("%peek", "i8")],
    )
    pipe = _module(
        "Pipe",
        [CLK, ("input", "io_in", "i8"), ("output", "io_out", "i8")],
        [
            '%r1 = "seq.firreg"(%io_in, %clock) <{name = "r1"}> : (i8, !seq.clock) -> i8',
            '%r2 = "seq.firreg"(%r1, %clock) <{name = "r2"}> : (i8, !seq.clock) -> i8',
        ],
        [("%r2", "i8")],
    )
    hold = _module(
        "Hold",
        [CLK, ("input", "io_in", "i8"), ("input", "io_en", "i1"), ("output", "io_out", "i8")],
        [
            '%h = "seq.firreg"(%hn, %clock) <{name = "h"}> : (i8, !seq.clock) -> i8',
            '%hn = "comb.mux"(%io_en, %io_in, %h) : (i1, i8, i8) -> i8',
        ],
        [("%h", "i8")],
    )
    acc = _module(
        "Acc",
        [CLK, ("input", "io_in", "i8"), ("output", "io_out", "i8")],
        [
            '%a = "seq.firreg"(%an, %clock) <{name = "acc"}> : (i8, !seq.clock) -> i8',
            '%an = "comb.add"(%a, %io_in) : (i8, i8) -> i8',
        ],
        [("%a", "i8")],
    )
    cell = _module(
        "Cell",
        [
            CLK,
            ("input", "io_a", "i8"),
            ("input", "io_b", "i8"),
            ("input", "io_sel", "i1"),
            ("output", "io_out_b", "i8"),
            ("output", "io_out_c", "i8"),
        ],
        [
            '%p = "comb.mul"(%io_a, %io_b) : (i8, i8) -> i8',
            '%ob = "comb.mux"(%io_sel, %p, %io_b) : (i1, i8, i8) -> i8',
            '%c = "seq.firreg"(%cn, %clock) <{name = "c"}> : (i8, !seq.clock) -> i8',
            '%sum = "comb.add"(%c, %p) : (i8, i8) -> i8',
            '%cn = "comb.mux"(%io_sel, %c, %sum) : (i1, i8, i8) -> i8',
        ],
        [("%ob", "i8"), ("%c", "i8")],
    )
    unit_inputs = [
        CLK,
        ("input", "io_alloc", "i8"),
        *[("input", e, "i1") for e in enables],
        ("input", "io_sel0", "i1"),
        ("input", "io_sel1", "i1"),
        ("input", "io_go", "i1"),
        ("input", "io_cfg", "i1"),
        ("input", "io_cfg_en", "i1"),
        ("input", "io_x", "i8"),
        ("input", "io_d_source", "i2"),
    ]
    enable_args = ", ".join(f"%{e}" for e in enables)
    unit = _module(
        "Unit",
        [*unit_inputs, ("output", "io_a", "i8"), ("output", "io_b", "i8"), ("output", "io_m", "i8")],
        [
            f'%t:2 = "hw.instance"(%clock, %io_alloc, {enable_args}, %io_sel0, %io_sel1) '
            '{argNames = ["clock", "io_alloc", "io_en0", "io_en1", "io_en2", "io_en3", "io_en4", "io_sel0", "io_sel1"], '
            'instanceName = "table", moduleName = @Table, parameters = [], '
            'resultNames = ["io_issue_a_data", "io_issue_b_data"]} '
            ": (!seq.clock, i8, i1, i1, i1, i1, i1, i1, i1) -> (i8, i8)",
            '%ua = "hw.instance"(%clock, %t#0, %io_go) {argNames = ["clock", "io_cmd", "io_go"], instanceName = "unit_a", '
            'moduleName = @Consumer, parameters = [], resultNames = ["io_out"]} : (!seq.clock, i8, i1) -> i8',
            '%ub = "hw.instance"(%clock, %t#1, %io_go) {argNames = ["clock", "io_cmd", "io_go"], instanceName = "unit_b", '
            'moduleName = @Consumer, parameters = [], resultNames = ["io_out"]} : (!seq.clock, i8, i1) -> i8',
            '%cl:4 = "hw.instance"(%clock, %io_go, %io_cfg, %io_x, %io_x, %io_d_source) '
            '{argNames = ["clock", "io_alloc", "io_big", "x_a_bits_data_lo", "x_a_bits_data_hi", "x_d_bits_source"], '
            'instanceName = "client", moduleName = @Client, parameters = [], '
            'resultNames = ["x_a_bits_data", "x_a_bits_size", "x_a_bits_source", "io_peek"]} '
            ": (!seq.clock, i1, i1, i8, i8, i2) -> (i16, i3, i2, i8)",
            '%cfg = "seq.firreg"(%cfgn, %clock) <{name = "cfg"}> : (i1, !seq.clock) -> i1',
            '%cfgn = "comb.mux"(%io_cfg_en, %io_cfg, %cfg) : (i1, i1, i1) -> i1',
            '%ce:2 = "hw.instance"(%clock, %io_x, %io_x, %cfg) {argNames = ["clock", "io_a", "io_b", "io_sel"], '
            'instanceName = "cell", moduleName = @Cell, parameters = [], resultNames = ["io_out_b", "io_out_c"]} '
            ": (!seq.clock, i8, i8, i1) -> (i8, i8)",
            '%pp = "hw.instance"(%clock, %io_x) {argNames = ["clock", "io_in"], instanceName = "pipe", moduleName = @Pipe, '
            'parameters = [], resultNames = ["io_out"]} : (!seq.clock, i8) -> i8',
            '%hh = "hw.instance"(%clock, %io_x, %io_go) {argNames = ["clock", "io_in", "io_en"], instanceName = "hold", '
            'moduleName = @Hold, parameters = [], resultNames = ["io_out"]} : (!seq.clock, i8, i1) -> i8',
            '%aa = "hw.instance"(%clock, %io_x) {argNames = ["clock", "io_in"], instanceName = "acc", moduleName = @Acc, '
            'parameters = [], resultNames = ["io_out"]} : (!seq.clock, i8) -> i8',
            '%m1 = "comb.xor"(%ce#0, %ce#1) : (i8, i8) -> i8',
            '%m2 = "comb.xor"(%m1, %pp) : (i8, i8) -> i8',
            '%m3 = "comb.xor"(%m2, %hh) : (i8, i8) -> i8',
            '%m4 = "comb.xor"(%m3, %aa) : (i8, i8) -> i8',
            '%m5 = "comb.xor"(%m4, %cl#3) : (i8, i8) -> i8',
        ],
        [("%ua", "i8"), ("%ub", "i8"), ("%m5", "i8")],
    )
    return (
        '"builtin.module"() ({\n'
        + fifo
        + table
        + consumer
        + client
        + pipe
        + hold
        + acc
        + cell
        + unit
        + "}) : () -> ()\n"
    )


DECLARATIONS = {
    "dispatch": {"module": "Table", "classes": {"a": "io_issue_a_", "b": "io_issue_b_"}},
    "memory_protocol": {
        "request": "a_bits_",
        "response": "d_bits_",
        "id": "source",
        "data": "data",
        "size": "size",
        "size_encoding": "log2_bytes",
    },
    "memory_clients": {"mem": {"module": "Client", "port_prefix": "x_"}},
    "dataflow": {"cell_module": "Cell", "select_port": "io_sel"},
    "latencies": {
        "pipe": {"module": "Pipe", "from": "io_in", "to": "io_out"},
        "hold": {"module": "Hold", "from": "io_in", "to": "io_out"},
        "acc": {"module": "Acc", "from": "io_in", "to": "io_out"},
        "cell_ws": {"module": "Cell", "from": "io_a", "to": "io_out_b", "mode": "weight_stationary"},
        "missing": {"module": "Pipe", "from": "io_nope", "to": "io_out"},
    },
}


@pytest.fixture(scope="module")
def record():
    return ts.derive(load_design(_design_text()), root="Unit", decl=DECLARATIONS)


def test_dispatch_entries_and_command_queues_are_read_from_storage(record):
    classes = record["command_classes"]["classes"]
    assert classes["a"]["entries"]["value"] == 2
    assert classes["b"]["entries"]["value"] == 3
    assert record["command_classes"]["total_entries"] == 5
    for name, unit in (("a", "unit_a"), ("b", "unit_b")):
        queue = classes[name]["command_queue"]
        assert queue["chain"] == [{"instance": f"Unit/{unit}/cmd_q", "module": "Fifo", "entries": 4}]
        assert queue["entries_total"] == 4


def test_memory_client_widths_sizes_and_in_flight_slots(record):
    client = record["memory_clients"]["clients"]["mem"]
    assert (client["request_data_bits"], client["beat_bytes"], client["id_bits"], client["id_space"]) == (16, 2, 2, 4)
    assert (client["request_sizes_log2"], client["max_request_bytes"], client["max_request_beats"]) == ([2, 3], 8, 4)
    assert client["request_channels"] == 1
    assert client["in_flight"]["value"] == 4
    assert client["in_flight"]["evidence"] == [{"kind": "indexed_array", "instance": "Unit/client", "slots": 4}]


def test_stage_counts_separate_fixed_enabled_and_feedback_paths(record):
    latencies = record["fixed_latencies"]
    assert (latencies["pipe"]["status"], latencies["pipe"]["min_stages"]) == ("fixed", 2)
    assert (latencies["hold"]["status"], latencies["hold"]["min_stages"]) == ("fixed_when_enabled", 1)
    assert (latencies["acc"]["status"], latencies["acc"]["max_stages"]) == ("variable", None)
    assert (latencies["cell_ws"]["status"], latencies["cell_ws"]["min_stages"]) == ("fixed", 0)
    assert latencies["missing"]["status"] == "unknown"
    assert any(item.startswith("fixed_latencies.missing") for item in record["unknown"])


def test_dataflow_modes_are_classified_from_the_product_destinations(record):
    flow = record["dataflow"]
    assert flow["modes"] == {"0": "output_stationary", "1": "weight_stationary"}
    assert (flow["support"], flow["select_runtime_driven"], flow["cell_occurrences"]) == ("both", True, 1)


def test_buffer_census_and_measurement_fits_are_recorded(record):
    census = {row["module"]: row["locations"] for row in record["buffers"]}
    assert census["Fifo"] == [4] and census["Table"] == [2, 3] and census["Client"] == [4]
    assert "memory_round_trip_latency_cycles" in record["fit_from_measurement"]
    assert record["schema"] == ts.SCHEMA


def test_profile_mapping_is_explicit_and_unmapped_fields_stay_unknown(record):
    mapping = {
        "unit.a": {
            "capacity": {"buffer_instance": "Unit/unit_a/cmd_q"},
            "queue_capacity": {
                "from": "command_classes.classes.a.entries.value",
                "minus": "memory_clients.clients.mem.request_channels",
            },
            "basis": "test",
        },
        "unit.dma": {"capacity": "memory_clients.clients.mem.request_channels", "queue_capacity": None},
        "unit.bad": {"capacity": "command_classes.classes.zzz.entries.value"},
    }
    resources = ts.profile_resources(record, mapping)
    assert (resources["unit.a"]["capacity"], resources["unit.a"]["queue_capacity"]) == (4, 1)
    assert (resources["unit.dma"]["capacity"], resources["unit.dma"]["queue_capacity"]) == (1, None)
    assert "queue_capacity" in resources["unit.dma"]["unknown"] and resources["unit.bad"]["capacity"] is None
    profile = {
        "resources": {"unit.a": {"capacity": None, "queue_capacity": None}},
        "target": {"prices": {"k": {"x": None}}},
    }
    applied = ts.apply_to_profile(profile, {"facts": {"timing_structure": record}}, mapping)
    assert applied["resources"]["unit.a"] == {"capacity": 4, "queue_capacity": 1}
    assert "target.prices.k.x" in applied["structural_mapping"]["fit_from_measurement"]
    assert profile["resources"]["unit.a"]["capacity"] is None  # the input profile is not mutated
    json.dumps(applied)


def test_stage_algebra():
    assert Stages(1, 1).chain(Stages(2, None)).maximum is None
    assert Stages(1, 2).join(Stages(3, 3)) == Stages(1, 3)
    assert Stages(2, 2, held=True).status == "fixed_when_enabled"
    assert Stages(2, 3).status == "variable" and Stages(0, 0, opaque=True).status == "unknown"
