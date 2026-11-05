"""Stream effects over a decoded dynamic command stream: declared roles and kinds only."""

from __future__ import annotations

from merlin.kernels import stream_effects as S

ACC_BIT = 1 << 15  # accumulator_select at msb_offset 0 of a 16-bit local address
FLAGS = {"sel": {"msb_offset": 0}, "acc": {"msb_offset": 1}}

# A bare RTL-facts body: 8x8 array, i8 operand rows (8 B), i32 accumulator rows (32 B).
# Operand store: 1024 B / 8 B = 128 rows, 64 per bank -> 2 banks. Accumulator: 2048 / 32 = 64 rows, 2 banks.
RTL = {
    "memories": [
        {"name": "scratchpad", "bytes": 1024, "depth": 64, "source": "test"},
        {"name": "accumulator", "bytes": 2048, "depth": 32, "source": "test"},
    ],
    "arrays": [{"name": "grid", "rows": 8, "cols": 8, "source": "test"}],
    "datapaths": [
        {"name": "input", "dtype": "i8", "evidence": "scratchpad UInt<8>"},
        {"name": "accumulator", "dtype": "i32", "evidence": "accumulator SInt<32>"},
    ],
}

ROLES = {
    "LDA": ["operand_load"],
    "LDB": ["weight_load"],
    "MAC": ["accumulate"],
    "OUT": ["readout"],
    "CFGL": ["config"],
    "CFGS": ["config"],
    "CFGX": ["config"],
    "FENCE": ["sync"],
}


def _kinds():
    kinds = {}
    for k in ("LDA", "LDB", "OUT"):
        kinds.update(
            {
                f"{k}.addr": "local_address",
                f"{k}.dram": "dram_address",
                f"{k}.rows": {"kind": "extent", "axis": "rows"},
                f"{k}.cols": {"kind": "extent", "axis": "cols"},
            }
        )
    kinds.update(
        {
            "LDA.is_acc": "accumulator_select",
            "MAC.a": "local_address",
            "CFGL.stride": "stride",
            "CFGL.state": "load_state",
            "CFGL.scale": "scale",
            "CFGS.act": "activation",
            "CFGS.scale": "scale",
            "OUT.act": "activation",
        }
    )
    return kinds


STRUCTURE = {
    "instruction_roles": ROLES,
    "operand_field_kinds": _kinds(),
    "operand_protocol": {
        "roles": {
            "local_address_flags": FLAGS,
            "classes": {"LDA": {"fields": [{"name": "is_acc", "as": "flag", "flag": "sel"}]}},
        }
    },
    "encoding": {"addr_len": 16},
    "load_state_ports": {"LDA": 0, "LDB": 1},
    "field_identity": {"CFGS.act": 0, "OUT.act": 0},
}


def _cmd(klass, host=None, **fields):
    return {"class": klass, "fields": fields, "host_instructions_since_prev": host}


def _stream():
    """Two fence-delimited compute blocks with double-buffered weights in two banks."""
    out = [
        _cmd("CFGX", host=10),
        _cmd("CFGL", host=3, stride=8, state=0, scale=1.0),
        _cmd("CFGL", host=3, stride=64, state=1, scale=1.0),
        _cmd("CFGS", host=3, act=0, scale=1.0),
    ]
    for block, weight_base in enumerate((0x40, 0x00, 0x40)):  # banks 1, 0, 1 -> alternating
        out.append(_cmd("LDA", host=1, addr=0, is_acc=False, dram=4096 * block, rows=8, cols=8))
        out.append(_cmd("LDB", host=1, addr=weight_base, dram=["w", 64 * block], rows=8, cols=8))
        out += [_cmd("MAC", host=0, a=0) for _ in range(4)]
        out.append(_cmd("OUT", host=6, addr=ACC_BIT | (1 << 14), dram=8 * block, rows=8, cols=8, act=0))
        out.append(_cmd("FENCE", host=2))
    for i, entry in enumerate(out):
        entry["seq"], entry["pc"] = i, 0x1000 + 4 * i
    return out


def test_every_fact_has_the_dynamic_scope_and_a_status():
    facts = S.derive(_stream(), STRUCTURE, rtl_facts=RTL)
    assert set(facts) == {
        "host_issue_rate",
        "reconfiguration",
        "local_capacity_use",
        "dram_contiguity",
        "epilogue_placement",
        "resource_occupancy",
        "transfer_legality",
        "post_store_reuse",
    }
    for fact in facts.values():
        assert fact["scope"] == "proved_dynamic_command_stream"
        assert fact["status"] in ("observed", "unknown")
        assert isinstance(fact["evidence"], dict)


def test_host_issue_rate_quantiles_roles_and_shares():
    fact = S.derive(_stream(), STRUCTURE)["host_issue_rate"]
    assert fact["status"] == "observed"
    value = fact["value"]
    # 4 configs (10,3,3,3) + 3 blocks x (1,1,0,0,0,0,6,2) = 28 commands
    assert value["overall"]["n"] == 28
    assert value["overall"]["max"] == 10
    assert value["overall"]["median"] == 1
    assert value["by_role"]["accumulate"] == {"n": 12, "median": 0, "p90": 0, "max": 0}
    assert value["by_role"]["readout"]["median"] == 6
    assert value["share_below"]["2"] == round(18 / 28, 6)
    assert value["share_below"]["8"] == round(27 / 28, 6)


def test_host_issue_rate_unknown_without_counts():
    stream = [{**e, "host_instructions_since_prev": None} for e in _stream()]
    fact = S.derive(stream, STRUCTURE)["host_issue_rate"]
    assert fact["status"] == "unknown" and fact["value"] is None and fact["reason"]


def test_reconfiguration_hoisted_versus_inside_the_compute_run():
    stream = _stream()
    fact = S.derive(stream, STRUCTURE)["reconfiguration"]
    assert fact["status"] == "observed"
    assert fact["value"]["config_commands"] == 4
    assert fact["value"]["hoisted"] == 4 and fact["value"]["inside_compute_run"] == 0
    assert fact["value"]["config_per_block"] == {"0": 4}
    # A reconfiguration between two blocks' computes sits inside the compute run.
    stream.insert(16, _cmd("CFGS", act=0, scale=1.0))
    fact = S.derive(stream, STRUCTURE)["reconfiguration"]
    assert fact["value"]["inside_compute_run"] == 1
    assert fact["value"]["config_per_block"] == {"0": 4, "1": 1}


def test_reconfiguration_uses_roles_from_the_structure_not_the_class():
    stream = _stream()
    renamed = {**ROLES, "CFGX": []}
    fact = S.derive(stream, {**STRUCTURE, "instruction_roles": renamed})["reconfiguration"]
    assert fact["value"]["config_commands"] == 3


def test_local_capacity_rows_against_rtl_capacity_and_bank_spread():
    fact = S.derive(_stream(), STRUCTURE, rtl_facts=RTL)["local_capacity_use"]
    assert fact["status"] == "observed"
    value = fact["value"]
    # block 0: operand rows 0..7 and weight rows 64..71 -> 16 scratchpad rows; output 8 acc rows
    assert value["per_block"]["0"]["scratchpad"]["distinct_rows"] == 16
    assert value["per_block"]["0"]["accumulator"]["distinct_rows"] == 8
    cap = value["capacity"]
    assert cap["scratchpad"]["capacity_rows"] == 128 and cap["scratchpad"]["fraction"] == 16 / 128
    assert cap["accumulator"]["capacity_rows"] == 64
    spread = value["bank_spread"]
    assert spread["operand_load"]["banks"] == {"0": 3} and spread["operand_load"]["alternating"] is None
    alternating = spread["weight_load"]["alternating"]
    assert alternating["banks"] == [[0], [1]] and alternating["same_bank"] is False
    # MAC addresses carry no extent: counted as addresses, not footprints.
    assert value["addresses_without_footprint"] == 12


def test_alternating_buffers_in_one_bank_are_reported():
    stream = _stream()
    for entry in stream:
        if entry["class"] == "LDB":
            entry["fields"]["addr"] = 0x40 + (16 if entry["fields"]["addr"] else 0)
    fact = S.derive(stream, STRUCTURE, rtl_facts=RTL)["local_capacity_use"]
    alternating = fact["value"]["bank_spread"]["weight_load"]["alternating"]
    assert alternating["same_bank"] is True and alternating["banks"] == [[1], [1]]


def test_local_capacity_without_rtl_facts_is_unknown_capacity():
    fact = S.derive(_stream(), STRUCTURE)["local_capacity_use"]
    value = fact["value"]
    assert value["capacity"]["scratchpad"]["capacity_rows"] is None
    assert value["capacity"]["scratchpad"]["reason"] == "no rtl_facts given"
    assert value["bank_spread"] == {}  # no row footprint without the row width
    assert value["per_block"]["0"]["scratchpad"]["distinct_addresses"] == 2


def test_local_capacity_unknown_without_address_kinds():
    kinds = {k: v for k, v in _kinds().items() if v != "local_address"}
    fact = S.derive(_stream(), {**STRUCTURE, "operand_field_kinds": kinds}, rtl_facts=RTL)["local_capacity_use"]
    assert fact["status"] == "unknown" and "local_address" in fact["reason"]


def test_address_space_unknown_without_a_declared_selector():
    structure = {**STRUCTURE, "operand_protocol": {}, "encoding": {}}
    kinds = {k: v for k, v in _kinds().items() if v != "accumulator_select"}
    fact = S.derive(_stream(), {**structure, "operand_field_kinds": kinds}, rtl_facts=RTL)["local_capacity_use"]
    assert set(fact["value"]["capacity"]) == {"undetermined"}
    assert fact["value"]["capacity"]["undetermined"]["capacity_rows"] is None


def test_dram_contiguity_per_load_role_and_alignment():
    fact = S.derive(_stream(), STRUCTURE, rtl_facts=RTL)["dram_contiguity"]
    assert fact["status"] == "observed"
    per_role = fact["value"]["per_role"]
    # operand rows: 8 cols x 1 B == state-0 stride 8 -> contiguous; weights: stride 64 != 8
    assert per_role["operand_load"]["contiguous"] == 3 and per_role["operand_load"]["strided"] == 0
    assert per_role["weight_load"]["strided"] == 3
    assert per_role["weight_load"]["stride_vs_row_bytes"] == {"stride=64,row_bytes=8": 3}
    assert fact["value"]["dram_alignment_histogram"] == {"zero": 1, "4096": 1, "8192": 1}
    assert fact["value"]["dram_offset_alignment_relative_to_symbol"] == {"zero": 1, "64": 1, "128": 1}


def test_dram_contiguity_unknown_without_element_widths_or_stride_kind():
    fact = S.derive(_stream(), STRUCTURE)["dram_contiguity"]
    assert fact["status"] == "unknown"
    assert fact["value"]["per_role"]["operand_load"]["unknown"] == 3
    kinds = {k: v for k, v in _kinds().items() if v != "stride"}
    fact = S.derive(_stream(), {**STRUCTURE, "operand_field_kinds": kinds}, rtl_facts=RTL)["dram_contiguity"]
    assert fact["status"] == "unknown" and "stride" in fact["reason"]


def test_dram_contiguity_needs_a_port_for_stateful_configs():
    structure = {**STRUCTURE, "load_state_ports": {"LDA": 0}}
    fact = S.derive(_stream(), structure, rtl_facts=RTL)["dram_contiguity"]
    assert fact["value"]["per_role"]["weight_load"]["unknown"] == 3
    assert fact["value"]["per_role"]["operand_load"]["contiguous"] == 3


def test_epilogue_identity_everywhere_is_not_on_accelerator():
    fact = S.derive(_stream(), STRUCTURE)["epilogue_placement"]
    assert fact["status"] == "observed" and fact["value"]["placement"] == "not_on_accelerator"
    # The load-state config's scale is input scaling, not an output epilogue.
    assert "CFGL.scale" not in fact["value"]["fields"]


def test_epilogue_non_identity_on_readout_is_on_accelerator():
    stream = _stream()
    stream[3]["fields"]["scale"] = 0.25
    fact = S.derive(stream, STRUCTURE)["epilogue_placement"]
    assert fact["value"]["placement"] == "on_accelerator"
    assert fact["value"]["fields"]["CFGS.scale"]["non_identity"] == 1
    stream = _stream()
    for entry in stream:
        if entry["class"] == "OUT":
            entry["fields"]["act"] = 1
    assert S.derive(stream, STRUCTURE)["epilogue_placement"]["value"]["placement"] == "on_accelerator"


def test_epilogue_unknown_without_kinds_or_identity():
    kinds = {k: v for k, v in _kinds().items() if v not in ("activation", "scale")}
    fact = S.derive(_stream(), {**STRUCTURE, "operand_field_kinds": kinds})["epilogue_placement"]
    assert fact["status"] == "unknown"
    fact = S.derive(_stream(), {**STRUCTURE, "field_identity": {}})["epilogue_placement"]
    assert fact["status"] == "unknown" and fact["value"]["fields"]["OUT.act"]["identity_unknown"] == 3


def test_epilogue_absent_fields_is_not_on_accelerator():
    stream = [e for e in _stream() if e["class"] != "CFGS"]
    for entry in stream:
        entry["fields"].pop("act", None)
    fact = S.derive(stream, STRUCTURE)["epilogue_placement"]
    assert fact["value"]["placement"] == "not_on_accelerator" and fact["value"]["fields"] == {}


def test_empty_stream_and_structure_never_raise():
    facts = S.derive([], {})
    assert all(f["status"] == "unknown" for f in facts.values())


# ----------------------------------------------------------------------------- dump-format adapter
# A lifted dynamic dump lists ``[class, pc, fields]`` triples without roles or host counts.
DUMP = [
    ["FENCE", "0x22", {}],
    ["CFGX", "0x4e", {}],
    ["CFGL", "0x5a", {"stride": 32, "scale": 1.0, "state": 0}],
    ["CFGL", "0x5e", {"stride": 32, "scale": 1.0, "state": 1}],
    ["CFGS", "0x64", {"act": 0, "scale": 1.0}],
    ["LDA", "0x78", {"is_acc": False, "dram": ["arg1", 0], "rows": 8, "cols": 32, "addr": 0}],
    ["LDA", "0x78", {"is_acc": False, "dram": ["arg1", 32], "rows": 8, "cols": 32, "addr": 32}],
    ["FENCE", "0x8c", {}],
    ["LDB", "0xf4", {"dram": ["arg0", 0], "rows": 8, "cols": 32, "addr": 64}],
    ["MAC", "0x140", {"a": 0}],
    ["OUT", "0x378", {"dram": ["arg2", 0], "rows": 8, "cols": 8, "addr": ACC_BIT | (1 << 13), "act": 0}],
    ["FENCE", "0x380", {}],
]


def _adapt(dump):
    return [
        {"seq": i, "pc": int(pc, 16), "class": c, "fields": f, "host_instructions_since_prev": None}
        for i, (c, pc, f) in enumerate(dump)
    ]


def test_dump_triples_through_the_adapter():
    facts = S.derive(_adapt(DUMP), STRUCTURE, rtl_facts=RTL)
    assert facts["host_issue_rate"]["status"] == "unknown"
    assert facts["reconfiguration"]["value"]["hoisted"] == 4
    use = facts["local_capacity_use"]["value"]
    # 8 rows x ceil(32 / 8) = 32 rows per operand transfer, two transfers -> 64 rows in block 1
    assert use["per_block"]["1"]["scratchpad"]["distinct_rows"] == 64
    # 32 cols x 1 B = 32 B rows, equal to the declared stride -> contiguous
    assert facts["dram_contiguity"]["value"]["per_role"]["operand_load"]["contiguous"] == 2
    assert facts["epilogue_placement"]["value"]["placement"] == "not_on_accelerator"
