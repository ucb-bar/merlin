"""Derived static effects from declared roles/protocol fields over synthetic decoded commands."""

from __future__ import annotations

import inspect

from merlin.kernels import decoded_effects
from merlin.kernels.decoded_effects import derive_effects

# A neutral synthetic target: class names, field names and widths are test data only.
ROLES = {
    "WLOAD": ["weight_load"],
    "MAC": ["accumulate"],
    "CFG_IN": ["config"],
    "CFG_OUT": ["config"],
    "IN0": ["operand_load"],
    "IN1": ["operand_load"],
    "OUT": ["readout", "commit"],
}
PROTOCOL = {
    "address_field": {"bundle": "OutRs", "field": "laddr"},
    "roles": {
        "local_address_flags": {"to_acc": {"msb_offset": 0}, "accum": {"msb_offset": 1}},
        "classes": {
            "WLOAD": {
                "fields": [
                    {"bundle": "WRs", "field": "laddr", "name": "src", "operand": "rs1"},
                    {"bundle": "WRs", "field": "laddr", "name": "dst", "operand": "rs2"},
                    {"as": "flag", "bundle": "WRs", "field": "laddr", "flag": "accum", "name": "acc", "operand": "rs2"},
                ]
            },
            "MAC": {
                "fields": [
                    {"bundle": "MRs", "field": "laddr", "name": "a", "operand": "rs1"},
                    {"as": "all_ones", "bundle": "MRs", "field": "laddr", "name": "none", "operand": "rs2"},
                ]
            },
            "CFG_IN": {
                "fields": [
                    {"as": "raw", "name": "step", "operand": "rs2"},
                    {"bundle": "CfgInRs1", "field": "sel", "name": "sel", "operand": "rs1"},
                ]
            },
            "CFG_OUT": {
                "fields": [
                    {"bundle": "CfgOutRs1", "field": "act", "name": "act", "operand": "rs1"},
                    {"bundle": "CfgOutRs1", "names": ["win", "hop"], "operand": "rs1"},
                ]
            },
            "IN0": {
                "fields": [
                    {"bundle": "InRs2", "field": "nr", "name": "r", "operand": "rs2"},
                    {"bundle": "InRs2", "field": "nc", "name": "c", "operand": "rs2"},
                    {"bundle": "InRs2", "field": "laddr", "name": "at", "operand": "rs2"},
                ]
            },
            "IN1": {
                "fields": [
                    {"bundle": "InRs2", "field": "nr", "name": "r", "operand": "rs2"},
                    {"bundle": "InRs2", "field": "nc", "name": "c", "operand": "rs2"},
                ]
            },
            "OUT": {
                "fields": [
                    {"bundle": "OutRs", "field": "nr", "name": "r", "operand": "rs2"},
                    {"bundle": "OutRs", "field": "nc", "name": "c", "operand": "rs2"},
                    {"bundle": "OutRs", "field": "laddr", "name": "at", "operand": "rs2"},
                ]
            },
        },
    },
}
LAYOUTS = [
    {"bundle": "WRs", "name": "laddr", "offset": 0, "width": 8},
    {"bundle": "WRs", "name": "nr", "offset": 16, "width": 5},
    {"bundle": "MRs", "name": "laddr", "offset": 0, "width": 8},
]
STRUCTURE = {
    "instruction_roles": ROLES,
    "operand_protocol": PROTOCOL,
    "field_layouts": LAYOUTS,
    "load_state_selection": {
        "config_class": "CFG_IN",
        "config_field": {"bundle": "CfgInRs1", "name": "sel"},
        "stride_input": "q.cmd.rs2",
        "transfer_bound": {"rows": {"bundle": "InRs2", "name": "nr"}, "cols": {"bundle": "InRs2", "name": "nc"}},
    },
    "load_state_ports": {"IN0": 0, "IN1": 1},
    "encoding": {"addr_len": 8},
    "output_epilogue": {
        "activation": {
            "carriers": [{"bundle": "CfgOutRs1", "field": "act"}],
            "functions": {"clamp_low": {"codes": [1]}},
        },
        "pool": {"carriers": [{"bundle": "CfgOutRs1", "field": "win"}, {"bundle": "CfgOutRs1", "field": "hop"}]},
    },
    "compute_units": [{"kind": "array"}],
}


def _cmds(*rows):
    out = []
    for index, (klass, fields, *rest) in enumerate(rows):
        record = dict(index=index, pc=0x1000 + 4 * index, instruction=klass, fields=fields, status="decoded")
        if rest:
            record["operands"] = rest[0]
        out.append(record)
    return out


def _analysis(commands):
    classes = {}
    for record in commands:
        classes[record["instruction"]] = classes.get(record["instruction"], 0) + 1
    return {"region_id": "region-test", "mechanism_projection": {"instruction_classes": classes}}


def test_retention_follows_roles_and_declared_sentinel():
    commands = _cmds(
        ("WLOAD", {"src": 3, "dst": 0x40, "acc": True}),
        ("MAC", {"a": 1, "none": False}),
        ("WLOAD", {"src": 255, "dst": 0x41, "acc": True}),
        ("MAC", {"a": 2, "none": False}),
        ("MAC", {"a": 3, "none": False}),
        ("WLOAD", {"src": None, "dst": 0x42, "acc": True}),
        ("MAC", {"a": 4, "none": False}),
    )
    fact = derive_effects(_analysis(commands), STRUCTURE, commands=commands)["facts"]["operand_retention"]
    assert fact["value"]["compute_sites"] == 4
    assert (fact["value"]["retained"], fact["value"]["reloaded"], fact["value"]["unknown"]) == (2, 1, 1)
    assert fact["status"] == "partial"
    # The flag-carrying slot is the destination; only the flag-free address is the loaded operand.
    assert fact["scope"]["weight_operand_fields"] == {"WLOAD": ["src"]}


def test_first_compute_without_intervening_load_is_inherited_unknown():
    commands = _cmds(("MAC", {"a": 1, "none": False}), ("MAC", {"a": 2, "none": False}))
    value = derive_effects(_analysis(commands), STRUCTURE, commands=commands)["facts"]["operand_retention"]["value"]
    assert value["unknown_breakdown"] == {"unknown_inherited_state": 1}
    assert value["retained"] == 1


def test_stride_zero_configs_bind_movement_through_declared_ports():
    commands = _cmds(
        ("IN1", {"r": 4, "c": 4}),
        ("CFG_IN", {"step": 0, "sel": 1}),
        ("CFG_IN", {"step": 64, "sel": 0}),
        ("IN1", {"r": 4, "c": 4}),
        ("IN0", {"r": 4, "c": 8, "at": 5}),
    )
    fact = derive_effects(_analysis(commands), STRUCTURE, commands=commands)["facts"]["broadcast_loads"]
    assert fact["value"]["stride_zero_by_state"] == {"1": 1}
    assert fact["value"]["broadcast_movement_sites"] == {"IN1": 1}
    assert fact["value"]["movement_sites_inherited_state"] == {"IN1": 1}
    assert fact["status"] == "observed"


def test_tiles_and_flagged_windows_use_declared_extents():
    commands = _cmds(
        ("IN0", {"r": 4, "c": 8, "at": 0x10}),
        ("IN0", {"r": 4, "c": 8, "at": 0x14}),
        ("OUT", {"r": 2, "c": 8, "at": 0x80 | 0x20}),
        ("OUT", {"r": None, "c": 8, "at": None}),
        ("WLOAD", {"src": 255, "dst": 0x80 | 0x30, "acc": False}, {"rs2": {"kind": "const", "raw": (6 << 16) | 0xB0}}),
    )
    facts = derive_effects(_analysis(commands), STRUCTURE, commands=commands)["facts"]
    tiles = facts["transfer_tiles"]
    assert tiles["value"]["by_class"]["IN0"] == [{"rows": 4, "cols": 8, "sites": 2}]
    assert tiles["value"]["unresolved_sites"] == {"OUT": 1}
    assert tiles["status"] == "partial"
    windows = facts["local_address_windows"]["value"]
    acc = windows["by_declared_flag"]["to_acc"]
    # OUT base 0x20 (2 rows) and WLOAD dst base 0x30 whose 6-row extent comes from the RTL layout.
    assert (acc["min"], acc["max"], acc["span_with_extent"]) == (0x20, 0x30, 0x36 - 0x20)
    assert windows["sentinel_sites"] == {"WLOAD.src": 1}
    assert windows["by_declared_flag"]["none"]["span_with_extent"] == 0x18 - 0x10


def test_epilogue_reports_declared_carriers_functions_and_pooled_readouts():
    commands = _cmds(
        ("OUT", {"r": 1, "c": 8, "at": 0x80}),
        ("CFG_OUT", {"act": 1, "win": 3, "hop": 2}),
        ("OUT", {"r": 1, "c": 8, "at": 0x81}),
        ("CFG_OUT", {"act": 0, "win": 0, "hop": 0}),
        ("OUT", {"r": 1, "c": 8, "at": 0x82}),
    )
    fact = derive_effects(_analysis(commands), STRUCTURE, commands=commands)["facts"]["epilogue_configuration"]
    assert fact["value"]["fields"]["CFG_OUT.act"]["declared_functions_observed"] == ["clamp_low"]
    assert fact["value"]["readout_sites_after_pool_configuration"] == {"unknown": 1, "pooled": 1, "not_pooled": 1}


def test_without_commands_ordered_facts_and_semantics_stay_unknown():
    analysis = {
        "mechanism_projection": {
            "instruction_classes": {"CFG_IN": 2, "MAC": 1},
            "field_profiles": {"CFG_IN.step": [{"value": 0, "sites": 1}, {"value": 8, "sites": 1}]},
        }
    }
    block = derive_effects(analysis, STRUCTURE)
    facts = block["facts"]
    assert block["basis"] == "analysis_summary"
    for name in ("operand_retention", "transfer_tiles", "local_address_windows", "dataflow", "operation"):
        assert facts[name]["status"] == "unknown" and facts[name]["reason"], name
    assert facts["broadcast_loads"]["value"]["stride_zero_config_sites"] == 1
    assert facts["layer_identity"]["status"] == "unknown"
    assert facts["hardware_loop"]["value"] == {"loop_descriptor_sites": 0, "uses_hardware_loop": False}
    assert {row["fact"] for row in block["unknowns"]} >= {"operand_retention", "dataflow", "operation"}


def test_undeclared_roles_and_host_regions_fail_closed():
    commands = _cmds(("MAC", {"a": 1, "none": False}))
    block = derive_effects(_analysis(commands), {**STRUCTURE, "instruction_roles": {}}, commands=commands)
    assert block["facts"]["operand_retention"]["status"] == "unknown"
    assert block["facts"]["hardware_loop"]["status"] == "unknown"
    host = derive_effects({"kind": "static_host_region", "cca": None}, STRUCTURE)
    assert all(fact["status"] == "unknown" for fact in host["facts"].values())


def test_replay_difference_is_surfaced():
    commands = _cmds(("MAC", {"a": 1, "none": False}))
    analysis = {"mechanism_projection": {"instruction_classes": {"MAC": 2}}}
    block = derive_effects(analysis, STRUCTURE, commands=commands)
    assert block["replay"]["status"] == "differs"
    assert any(row["fact"] == "*" for row in block["unknowns"])


def test_dataflow_only_from_declared_compute_unit():
    commands = _cmds(("MAC", {"a": 1, "none": False}))
    declared = {**STRUCTURE, "compute_units": [{"kind": "array", "dataflow": "declared_mode"}]}
    fact = derive_effects(_analysis(commands), declared, commands=commands)["facts"]["dataflow"]
    assert fact == {
        "value": ["declared_mode"],
        "status": "declared",
        "scope": {},
        "evidence": ["structure#compute_units"],
    }


def test_module_reads_no_historical_map_or_instruction_spelling():
    source = inspect.getsource(decoded_effects)
    assert "REFERENCE_MAP" not in source and "reference-map" not in source
    for spelling in ("MVIN", "PRELOAD", "COMPUTE", "weight_spad"):
        assert spelling not in source
