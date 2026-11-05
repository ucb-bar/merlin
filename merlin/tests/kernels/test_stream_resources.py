"""Queue routing, queue occupancy, transfer bounds and post-store reuse over a decoded command stream."""

from __future__ import annotations

import copy

from merlin.kernels import stream_effects
from merlin.kernels import stream_resources as R
from merlin.kernels.command_routing import command_queue_routing

ACC, ACCUMULATE = 1 << 15, 1 << 14  # accumulator_select / accumulate at msb offsets 0 / 1 of 16 bits


def _queue(name, depth, chain):
    return {"entries": {"status": "derived", "value": depth},
            "command_queue": {"status": "derived", "entries_total": 2,
                              "chain": [{"instance": chain, "module": "Q"}]}}


# Operand store 1024 B of 8 B rows = 128 rows; accumulator 2048 B of 32 B rows = 64 rows.
FACTS = {
    "memories": [
        {"name": "scratchpad", "bytes": 1024, "depth": 64, "source": "test"},
        {"name": "accumulator", "bytes": 2048, "depth": 32, "source": "test"},
    ],
    "arrays": [{"name": "grid", "rows": 8, "cols": 8, "source": "test"}],
    "datapaths": [
        {"name": "input", "dtype": "i8", "evidence": "scratchpad UInt<8>"},
        {"name": "accumulator", "dtype": "i32", "evidence": "accumulator SInt<32>"},
    ],
    "timing_structure": {
        "unit_root": "Unit",
        "command_classes": {"status": "derived", "classes": {
            "ld": _queue("ld", 4, "Unit/ld_ctl/q"),
            "ex": _queue("ex", 8, "Unit/ex_ctl/q"),
            "st": _queue("st", 2, "Unit/st_ctl/q"),
        }},
        "memory_clients": {"status": "derived", "clients": {
            "rd": {"request_data_bits": 128, "response_data_bits": 128, "max_request_bytes": 16, "beat_bytes": 16},
            "wr": {"request_data_bits": 128, "response_data_bits": None, "max_request_bytes": 32, "beat_bytes": 16},
        }},
    },
}

HIERARCHY = {"instance_name": "top", "module_name": "Top", "instances": [
    {"instance_name": "u", "module_name": "Unit", "instances": [
        {"instance_name": n, "module_name": m, "instances": [{"instance_name": "q", "module_name": "Q"}]}
        for n, m in (("ld_ctl", "LoadUnit"), ("ex_ctl", "ExecUnit"), ("st_ctl", "StoreUnit"))]}]}


def _f(bundle, field, name, operand="rs2", **extra):
    return dict(bundle=bundle, field=field, name=name, operand=operand, **extra)


PROTOCOL = {
    "address_field": {"bundle": "LdRs", "field": "laddr"},
    "roles": {
        "local_address_flags": {"accumulator_select": {"msb_offset": 0}, "accumulate": {"msb_offset": 1}},
        "classes": {
            "LD": {"fields": [_f("LdRs", "laddr", "addr"), _f("LdRs", "n_rows", "rows"), _f("LdRs", "n_cols", "cols")]},
            "ST": {"fields": [_f("StRs", "laddr", "addr"), _f("StRs", "n_rows", "rows"), _f("StRs", "n_cols", "cols")]},
            "EX": {"fields": [_f("ExRs", "laddr", "a", "rs1"), _f("ExRs", "laddr", "c"),
                              _f("ExRs", "n_rows", "rs1_rows", "rs1"), _f("ExRs", "n_rows", "rs2_rows")]},
        },
    },
}
LAYOUTS = [{"bundle": b, "name": "laddr", "module": m, "source_lines": [i]}
           for i, (b, m) in enumerate((("LdRs", "LoadUnit"), ("ExRs", "ExecUnit"), ("StRs", "StoreUnit")))]
BASE = {
    "instruction_roles": {"LD": ["operand_load"], "EX": ["accumulate"], "ST": ["readout"], "FENCE": ["sync"]},
    "operand_protocol": PROTOCOL,
    "field_layouts": LAYOUTS,
    "load_state_selection": {"transfer_bound": {"rows": {"name": "n_rows"}, "cols": {"name": "n_cols"}}},
    "encoding": {"addr_len": 16},
}


def _structure(facts=FACTS, hierarchy=HIERARCHY):
    out = dict(BASE)
    out["command_queue_routing"] = command_queue_routing(facts, out, hierarchy=hierarchy)
    out["stream_bounds"] = R.stream_bounds(facts)
    return out


def _cmd(klass, **fields):
    return {"class": klass, "fields": fields}


def _seq(entries):
    for i, e in enumerate(entries):
        e["seq"] = i
    return entries


# ----------------------------------------------------------------------------- routing
def test_routing_joins_bundle_consumers_to_queue_chains():
    routing = command_queue_routing(FACTS, BASE, hierarchy=HIERARCHY)
    assert {k: v["queue"] for k, v in routing["classes"].items()} == {"LD": "ld", "EX": "ex", "ST": "st"}
    assert routing["classes"]["LD"]["controller_instances"] == ["Unit/ld_ctl"]
    assert routing["queues"]["ld"]["dispatch_entries"] == 4
    assert routing["status"] == "derived"


def test_routing_unknown_when_no_chain_sits_under_the_consumer():
    tree = copy.deepcopy(HIERARCHY)
    tree["instances"][0]["instances"][0]["module_name"] = "Elsewhere"
    row = command_queue_routing(FACTS, BASE, hierarchy=tree)["classes"]["LD"]
    assert row["status"] == "unknown" and "no dispatch queue" in row["reason"]


def test_routing_names_missing_command_classes():
    facts = {k: v for k, v in FACTS.items() if k != "timing_structure"}
    routing = command_queue_routing(facts, BASE, hierarchy=HIERARCHY)
    assert routing["status"] == "unknown"
    assert any("command_classes" in m for m in routing["missing"])


# ----------------------------------------------------------------------------- resource occupancy
def test_occupancy_counts_routed_commands_since_sync_against_entries():
    stream = _seq([_cmd("LD", addr=8 * i, rows=1, cols=8) for i in range(6)] + [_cmd("FENCE")]
                  + [_cmd("LD", addr=0, rows=1, cols=8)] * 3)
    fact = R.derive(stream, _structure())["resource_occupancy"]
    assert fact["status"] == "observed"
    ld = fact["value"]["per_queue"]["ld"]
    assert ld["max_outstanding_since_sync"] == 6 and ld["dispatch_entries"] == 4
    assert ld["points_exceeding_dispatch_entries"] == 2  # the 5th and 6th before the fence
    assert ld["points_exceeding_dispatch_and_queue_entries"] == 0
    assert ld["first_exceeding_seq"] == 4
    assert fact["value"]["unrouted_commands"] == {}


def test_occupancy_window_closes_when_another_queue_consumes_its_rows():
    loads = [_cmd("LD", addr=8 * i, rows=8, cols=8) for i in range(3)]
    use = _cmd("EX", a=0, c=ACC | 0, rs1_rows=8, rs2_rows=8)  # reads operand rows 0..7 written by a load
    fact = R.derive(_seq(loads + [use] + loads + loads), _structure())["resource_occupancy"]
    ld = fact["value"]["per_queue"]["ld"]
    assert ld["dependency_closures"] == 1 and ld["max_outstanding_since_sync"] == 9
    assert ld["max_outstanding_before_dependency"] == 6 and ld["points_exceeding_dispatch_entries"] == 2


def test_occupancy_unknown_without_queue_depths_names_the_fact():
    facts = {k: v for k, v in FACTS.items() if k != "timing_structure"}
    fact = R.derive(_seq([_cmd("LD", addr=0, rows=1, cols=8)]), _structure(facts))["resource_occupancy"]
    assert fact["status"] == "unknown" and "command_classes" in fact["reason"]


# ----------------------------------------------------------------------------- transfer legality
def test_transfer_row_bytes_past_the_largest_request_is_flagged_with_its_source():
    stream = _seq([_cmd("LD", addr=0, rows=8, cols=8), _cmd("LD", addr=16, rows=2, cols=24),
                   _cmd("ST", addr=ACC | 0, rows=8, cols=8)])
    fact = R.derive(stream, _structure())["transfer_legality"]
    assert fact["status"] == "observed"
    load = fact["value"]["per_role"]["operand_load"]
    assert (load["transfers"], load["within"], load["exceeds_bound"]) == (2, 1, 1)
    assert load["exceeding"][0]["cols"] == 24
    bound = load["bounds"]["row_bytes_vs_largest_dma_request"]
    assert bound["bound"] == 16 and bound["source"].endswith("clients.rd.max_request_bytes")
    # Readout from the accumulator: 8 cols x (8..32 bits) = 8..32 B, within the 32 B outbound request.
    assert fact["value"]["per_role"]["readout"]["within"] == 1


def test_transfer_footprint_past_the_store_rows_is_flagged():
    fact = R.derive(_seq([_cmd("LD", addr=124, rows=8, cols=8)]), _structure())["transfer_legality"]
    load = fact["value"]["per_role"]["operand_load"]
    assert load["exceeds_bound"] == 1 and load["bounds"]["local_rows_vs_store_rows"]["bound"] == 128


def test_transfer_bound_unknown_without_memory_clients():
    facts = copy.deepcopy(FACTS)
    facts["timing_structure"].pop("memory_clients")
    stream = _seq([_cmd("LD", addr=0, rows=8, cols=8)])
    fact = R.derive(stream, _structure(facts))["transfer_legality"]
    assert fact["status"] == "unknown" and "max_request_bytes" in fact["reason"]
    assert R.derive(stream, BASE)["transfer_legality"]["status"] == "unknown"


# ----------------------------------------------------------------------------- post-store reuse
def test_post_store_reuse_classifies_the_next_touch_and_the_syncs_between():
    stream = _seq([
        _cmd("ST", addr=ACC | 0, rows=8, cols=8),
        _cmd("FENCE"),
        _cmd("EX", a=0, c=ACC | ACCUMULATE | 4, rs1_rows=8, rs2_rows=8),  # accumulates into rows 4..11
        _cmd("ST", addr=ACC | 16, rows=8, cols=8),
        _cmd("LD", addr=ACC | 16, rows=8, cols=8),  # overwrites 16..23
        _cmd("ST", addr=ACC | 40, rows=8, cols=8),
        _cmd("ST", addr=ACC | 44, rows=4, cols=8),  # re-reads part of 40..47
    ])
    fact = R.derive(stream, _structure())["post_store_reuse"]
    assert fact["status"] == "observed"
    row = fact["value"]["per_role"]["readout"]
    assert row["stores"] == 4
    assert (row["accumulated_into"], row["rewritten"], row["read_again"], row["not_touched_again"]) == (1, 1, 1, 1)
    assert row["syncs_between"] == {"1": 1, "0": 3} and row["sync_classes"] == {"FENCE": 1}
    assert row["examples"][0] == dict(store_seq=0, outcome="accumulated_into", touch_seq=2, space="accumulator",
                                      rows=[0, 8], syncs_between=1)


def test_post_store_reuse_unknown_without_an_accumulator_flag():
    structure = copy.deepcopy(_structure())
    structure["operand_protocol"]["roles"]["local_address_flags"].pop("accumulator_select")
    fact = R.derive(_seq([_cmd("ST", addr=0, rows=8, cols=8)]), structure)["post_store_reuse"]
    assert fact["status"] == "unknown" and "accumulator_select" in fact["reason"]


def test_stream_effects_derive_carries_the_three_facts():
    facts = stream_effects.derive(_seq([_cmd("LD", addr=0, rows=8, cols=8)]), _structure())
    assert {"resource_occupancy", "transfer_legality", "post_store_reuse"} <= set(facts)
