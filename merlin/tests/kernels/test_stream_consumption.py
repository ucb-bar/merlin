"""Buffering, load-to-first-reader distance, consumer ports and output groups over synthetic proved streams.

A neutral synthetic target: every class, field, flag and module name and every width is test data.
Loads write ``ceil(cols / row elements)`` row blocks at the per-state block step a configuration issue
sets; a consumer is a later issue whose declared local-address field reads a written row.
"""

from __future__ import annotations

from merlin.kernels.decoded_effects import BUFFERING_DEFINITION, stream_facts

ROLES = {"LD_A": ["operand_load"], "LD_B": ["operand_load"], "STAGE": ["weight_load"], "MAC": ["accumulate"],
         "OUT": ["readout"], "CFG": ["config"], "CFG_OUT": ["config"], "FENCE": ["sync"]}


def _load_fields():
    return [{"as": "operand", "name": "src", "operand": "rs1"},
            {"bundle": "InRs2", "field": "nr", "name": "r", "operand": "rs2"},
            {"bundle": "InRs2", "field": "nc", "name": "c", "operand": "rs2"},
            {"bundle": "InRs2", "field": "laddr", "name": "at", "operand": "rs2"}]


PROTOCOL = {
    "address_field": {"bundle": "OutRs", "field": "laddr"},
    "roles": {
        "local_address_flags": {"to_acc": {"msb_offset": 0}},
        "classes": {
            "LD_A": {"fields": _load_fields()},
            "LD_B": {"fields": _load_fields()},
            "STAGE": {"fields": [{"bundle": "SRs", "field": "laddr", "name": "w", "operand": "rs1"},
                                 {"bundle": "SRs", "field": "nr", "name": "w_rows", "operand": "rs1"}]},
            "MAC": {"fields": [{"bundle": "MRs", "field": "laddr", "name": "a", "operand": "rs1"},
                               {"bundle": "MRs", "field": "nr", "name": "a_rows", "operand": "rs1"},
                               {"as": "all_ones", "bundle": "MRs", "field": "laddr", "name": "none",
                                "operand": "rs2"}]},
            "OUT": {"fields": [{"as": "operand", "name": "dst", "operand": "rs1"},
                               {"bundle": "OutRs", "field": "nr", "name": "r", "operand": "rs2"},
                               {"bundle": "OutRs", "field": "nc", "name": "c", "operand": "rs2"},
                               {"bundle": "OutRs", "field": "laddr", "name": "at", "operand": "rs2"}]},
            "CFG": {"fields": [{"as": "raw", "name": "mem_step", "operand": "rs2"},
                               {"bundle": "CfgRs1", "field": "sel", "name": "sel", "operand": "rs1"},
                               {"bundle": "CfgRs1", "field": "blk", "name": "blk", "operand": "rs1"}]},
            "CFG_OUT": {"fields": [{"bundle": "CfgOutRs2", "field": "pitch", "name": "pitch", "operand": "rs2"}]},
        },
    },
}
LAYOUTS = [{"bundle": "MRs", "name": "laddr", "offset": 0, "width": 16},
           {"bundle": "SRs", "name": "laddr", "offset": 0, "width": 16},
           {"bundle": "OutRs", "name": "nc", "module": "Storer", "offset": 32, "width": 8}]
KINDS = [{"label": "CFG.blk", "kind": "stride", "module": "Loader", "sink": "steps[sel]"},
         {"label": "CFG_OUT.pitch", "kind": "stride", "module": "Storer", "sink": "pitch"}]
STRUCTURE = {
    "instruction_roles": ROLES,
    "operand_protocol": PROTOCOL,
    "field_layouts": LAYOUTS,
    "load_state_selection": {"config_class": "CFG", "config_field": {"bundle": "CfgRs1", "name": "sel"},
                             "state": "sel", "stride_input": "q.cmd.rs2",
                             "transfer_bound": {"rows": {"bundle": "InRs2", "name": "nr"},
                                                "cols": {"bundle": "InRs2", "name": "nc"}}},
    "load_state_ports": {"LD_A": 0, "LD_B": 1},
    "operand_field_kind_sources": KINDS,
    "local_geometry": {"uniform_row_elements": 4},
    "encoding": {"addr_len": 16},
}


def _arg(index, offset=0):
    return {"kind": "pointer", "pointer_base": f"arg{index}", "byte_offset": offset}


def config(state, step):
    return ("CFG", dict(sel=state, blk=step, mem_step=64))


def load_b(at, cols=8):  # tensor argument a2, consumed by the staging port
    return ("LD_B", dict(src=_arg(2), r=4, c=cols, at=at))


def load_a(at):  # tensor argument a1, consumed by the compute port
    return ("LD_A", dict(src=_arg(1), r=4, c=4, at=at))


def stage(at):
    return ("STAGE", dict(w=at, w_rows=4))


def mac(at):
    return ("MAC", dict(a=at, a_rows=4, none=0xFFFF))


FENCE = ("FENCE", {})


def stream(*rows):
    return [dict(seq=i, pc=None, **{"class": k}, roles=ROLES.get(k, []), fields=f) for i, (k, f) in enumerate(rows)]


def ring(early=False):
    """Two 16-row regions of wide (two-block) weight loads; each region is refilled while the other is
    consumed, with fences between issue pairs (fences must not shape the regions). ``early`` issues the
    first load into the other region before the current region's first read."""
    rows = [config(1, 8), config(0, 4), load_a(64), load_b(0), load_b(4)]  # prologue fills region 0
    region = 0
    for _ in range(4):
        other = 16 - region
        head = [load_b(other), FENCE, stage(region), mac(64)] if early else [stage(region), mac(64), FENCE,
                                                                            load_b(other)]
        rows += head + [stage(region + 8), mac(64), FENCE, load_b(other + 4), stage(region + 4), mac(64),
                        stage(region + 12), mac(64)]
        region = other
    return stream(*rows)


def facts_of(rows, structure=STRUCTURE):
    return stream_facts(rows, structure, evidence_id="synthetic")


def test_wide_loads_rotate_two_regions_independent_of_fences():
    facts = facts_of(ring())
    entry = facts["operand_buffering"]["value"]["operand_load"]["a2"]
    assert entry["definition"] == BUFFERING_DEFINITION
    assert (entry["disjoint_windows"], entry["alternating"]) == (2, True)
    # Each wide load writes two blocks at the configured step: regions span 16 rows each.
    assert [(w["lo"], w["hi"], w["rows"]) for w in entry["windows"]] == [(0, 15, 16), (16, 31, 16)]
    assert entry["window_sequence_prefix"][:5] == [0, 1, 0, 1, 0] and entry["block_steps"] == [8]
    assert entry["consumer_port"] == "weight_load.w"
    a = facts["operand_buffering"]["value"]["operand_load"]["a1"]
    assert (a["disjoint_windows"], a["alternating"], a["consumer_port"]) == (1, False, "accumulate.a")


def test_rounds_that_straddle_regions_leave_buffering_unknown():
    entry = facts_of(ring(early=True))["operand_buffering"]["value"]["operand_load"]["a2"]
    assert entry["status"] == "unknown" and entry["disjoint_windows"] is None and entry["straddling_rounds"] > 0
    assert "regions are not established" in entry["reason"]


def test_just_in_time_single_region_and_its_distance():
    rows = [config(1, 8)]
    for _ in range(4):
        rows += [load_b(0), stage(0), stage(8), mac(64)]
    facts = facts_of(stream(load_a(64), *rows))
    entry = facts["operand_buffering"]["value"]["operand_load"]["a2"]
    assert (entry["disjoint_windows"], entry["alternating"], entry["fill_rounds"]) == (1, False, 4)
    distance = facts["prefetch_distance"]["value"]["by_operand"]["a2"]
    assert distance["median_loads_between"] == 0 and distance["min_loads_between"] == 0
    assert distance["fraction_with_compute_ahead"] == 0.0 and distance["consumer_port"] == "weight_load.w"


def test_prefetch_distance_counts_intervening_loads_and_computes():
    facts = facts_of(ring())
    distance = facts["prefetch_distance"]["value"]["by_operand"]["a2"]
    # Each load waits for one or two further loads; the last region's refill is never read.
    assert (distance["loads"], distance["consumed_loads"], distance["unconsumed_loads"]) == (10, 8, 2)
    assert distance["min_loads_between"] == 1 and distance["median_loads_between"] == 1.5
    # Only the first prologue load is read before any compute issues after it.
    assert distance["fraction_with_compute_ahead"] == 0.875
    assert facts["prefetch_distance"]["status"] == "observed"
    assert "not a timing or speed claim" in facts["prefetch_distance"]["limits"][0]


def test_wide_load_without_a_configured_block_step_stays_unknown():
    facts = facts_of(stream(load_b(0), stage(0), mac(0)))
    entry = facts["operand_buffering"]["value"]["operand_load"]["a2"]
    assert entry["status"] == "unknown" and entry["disjoint_windows"] is None
    assert "block step of this load's state not set" in entry["reason"]
    assert facts["prefetch_distance"]["status"] == "unknown"


def test_unresolved_load_names_the_missing_local_address_not_the_extent():
    # An issue-time register the proved execution does not know (e.g. an unproved call return).
    unknown = {"kind": "unknown", "raw": None}
    rows = [config(1, 8), ("LD_B", dict(src=_arg(2), r=4, c=4, at=unknown)), stage(0), mac(64)]
    entry = facts_of(stream(*rows))["operand_buffering"]["value"]["operand_load"]["a2"]
    assert entry["status"] == "unknown"
    assert entry["reason"] == "1 loads: load local address unresolved"


def test_a_zero_row_load_leaves_its_operand_buffering_unknown():
    for r, c in ((0, 4), (4, 0)):
        rows = [config(1, 8), ("LD_B", dict(src=_arg(2), r=r, c=c, at=0)), stage(0), mac(64)]
        entry = facts_of(stream(*rows))["operand_buffering"]["value"]["operand_load"]["a2"]
        assert entry["status"] == "unknown" and "zero transfer rows or columns" in entry["reason"]


def test_block_step_needs_the_rtl_field_kind_citation():
    structure = dict(STRUCTURE, operand_field_kind_sources=[])
    entry = facts_of(ring(), structure)["operand_buffering"]["value"]["operand_load"]["a2"]
    assert entry["status"] == "unknown" and "RTL field-kind sources" in entry["reason"]
    # Single-block loads need no block step.
    facts = facts_of(stream(load_a(64), mac(64)), structure)
    assert facts["operand_buffering"]["value"]["operand_load"]["a1"]["disjoint_windows"] == 1


def test_row_geometry_comes_from_the_structure_or_stays_unknown():
    structure = dict(STRUCTURE, local_geometry={"uniform_row_elements": None, "reason": "stores differ"})
    entry = facts_of(ring(), structure)["operand_buffering"]["value"]["operand_load"]["a1"]
    assert entry["status"] == "unknown" and "stores differ" in entry["reason"]
    # A row of eight elements makes the same transfers single-block: four-row footprints, no step needed.
    structure = dict(STRUCTURE, local_geometry={"uniform_row_elements": 8}, operand_field_kind_sources=[])
    entry = facts_of(ring(), structure)["operand_buffering"]["value"]["operand_load"]["a2"]
    assert entry["status"] == "observed" and entry["block_steps"] == []


def test_output_columns_per_block_use_the_configured_readout_pitch():
    def out(offset):
        return ("OUT", dict(dst=_arg(3, offset), r=4, c=4, at=0x8000))

    rows = stream(("CFG_OUT", dict(pitch=64)), mac(0), out(0), out(16), out(256), out(272), FENCE, mac(0), out(32))
    value = facts_of(rows)["output_group_extent"]["value"]
    first, second = value["blocks_prefix"]
    assert (first["column_groups"], first["row_bands"], first["output_columns"]) == (2, 2, 8)
    assert second["output_columns"] == 4 and value["median_output_columns"] == 6
    assert value["readout_pitch_fields"] == {"CFG_OUT": "pitch"}
    unknown = facts_of(stream(mac(0), out(0)))["output_group_extent"]["value"]
    assert unknown["median_output_columns"] is None


def test_every_operand_entry_names_its_consumer_port():
    facts = facts_of(ring())
    width = facts["role_transfer_width"]["value"]["data_movement"]["operand_load"]["by_operand"]
    assert {k: v["consumer_port"] for k, v in width.items()} == {"a1": "accumulate.a", "a2": "weight_load.w"}
    overlap = facts["issue_overlap"]["value"]["by_operand"]
    assert {k: v["consumer_port"] for k, v in overlap.items()} == {"a1": "accumulate.a", "a2": "weight_load.w"}


def _relative(symbol, offset, width=16):
    """An issue register whose low ``width`` bits are ``symbol + offset`` (an unproved call return)."""
    return {"kind": "unknown", "raw": None, "window": {"symbol": symbol, "offset": offset, "at": 0, "width": width}}


RELATIVE = dict(STRUCTURE, field_layouts=[*LAYOUTS, {"bundle": "InRs2", "name": "laddr", "offset": 0, "width": 16}])


def relative_stream(rows):
    """Rows of ``(class, fields, {slot: symbolic register})``: the address fields stay undecoded."""
    out = stream(*[(k, f) for k, f, _ in rows])
    for entry, (_, _, registers) in zip(out, rows):
        if registers:
            entry["symbolic_operands"] = registers
    return out


def relative_ring(reader="S"):
    """:func:`ring` with every weight-tile address relative to one allocation symbol ``S``."""

    def load(offset):
        return ("LD_B", dict(src=_arg(2), r=4, c=8, at=None), {"rs2": _relative("S", offset)})

    def read(offset):
        return [("STAGE", dict(w=None, w_rows=4), {"rs1": _relative(reader, offset)}), (*mac(64), None)]

    rows = [(*config(1, 8), None), (*config(0, 4), None), (*load_a(64), None), load(0), load(4)]
    region = 0
    for _ in range(4):
        other = 16 - region
        rows += [*read(region), load(other), *read(region + 8), load(other + 4), *read(region + 4),
                 *read(region + 12)]
        region = other
    return relative_stream(rows)


def test_allocation_relative_loads_rotate_windows_within_one_base():
    facts = stream_facts(relative_ring(), RELATIVE, evidence_id="synthetic")
    entry = facts["operand_buffering"]["value"]["operand_load"]["a2"]
    assert entry["status"] == "observed" and entry["windows_relative_to"] == "allocation"
    assert entry["allocation"] == "S" and (entry["disjoint_windows"], entry["alternating"]) == (2, True)
    assert entry["consumer_port"] == "weight_load.w"
    assert any("disjointness or overlap across allocations is not established" in limit
               for limit in facts["operand_buffering"]["limits"])
    distance = facts["prefetch_distance"]["value"]["by_operand"]["a2"]
    # The last refill is unread by S readers; a later exact-address read may alias it: not established.
    assert distance["consumed_loads"] == 8 and distance["unestablished_consumer_loads"] == 2


def test_a_reader_of_another_base_is_never_matched():
    facts = stream_facts(relative_ring(reader="T"), RELATIVE, evidence_id="synthetic")
    distance = facts["prefetch_distance"]["value"]["by_operand"]["a2"]
    # Rows of T are not comparable with rows of S: no pair is matched and none is called unconsumed.
    assert distance["consumed_loads"] == 0 and distance["unestablished_consumer_loads"] == 10
    assert facts["operand_buffering"]["value"]["operand_load"]["a2"]["consumer_port"] is None


def test_loads_of_one_operand_across_two_bases_get_one_entry_per_base():
    rows = [(*config(1, 8), None)]
    for symbol in ("S", "T"):
        rows.append(("LD_B", dict(src=_arg(2), r=4, c=8, at=None), {"rs2": _relative(symbol, 0)}))
        rows.append(("STAGE", dict(w=None, w_rows=4), {"rs1": _relative(symbol, 0)}))
    value = stream_facts(relative_stream(rows), RELATIVE, evidence_id="synthetic")["operand_buffering"]["value"]
    entries = value["operand_load"]
    assert set(entries) == {"a2@S", "a2@T"}  # same offsets, but never compared across bases
    for symbol in ("S", "T"):
        entry = entries["a2@" + symbol]
        assert entry["allocation"] == symbol and entry["windows_relative_to"] == "allocation"
        assert (entry["status"], entry["disjoint_windows"], entry["fill_rounds"]) == ("observed", 1, 1)
