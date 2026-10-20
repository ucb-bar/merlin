"""Source-selected banks retain exact state identity and storage width."""

import copy
import dataclasses
import hashlib

import pytest

from merlin.targetgen.rtl.firrtl_load_states import derive_load_state_mapping
from merlin.targetgen.rtl.firrtl_operand_slices import derive_operand_layouts

SOURCE = """FIRRTL version 4.2.0
circuit Selector:
  module Selector:
    input clock : Clock
    input cmd : { rs1 : UInt<64>, rs2 : UInt<64>, inst : { funct : UInt<7> } }
    output address : UInt<65>
    wire cfg : { state : UInt<1>, reserved : UInt<1> }
    node config_bits = bits(cmd.rs1, 3, 3)
    connect cfg.state, config_bits
    invalidate cfg.reserved
    reg strides : UInt<5>[2], clock
    reg row : UInt<3>, clock
    node is_config = eq(cmd.inst.funct, UInt<1>(0))
    node is_second = eq(cmd.inst.funct, UInt<4>(9))
    node selected = mux(is_second, UInt<1>(1), UInt<1>(0))
    node state = mux(is_config, cfg.state, selected)
    node offset = mul(row, strides[state])
    node actual_address = add(cmd.rs1, offset)
    connect address, actual_address
    when is_config :
      connect strides[state], cmd.rs2
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "selected.fir"
    path.write_text(source)
    layouts = derive_operand_layouts(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selections=[
            {"bundle": "LoadCfg", "name": "state", "module": "Selector", "field": "cfg.state", "input": "cmd.rs1"},
        ],
    )
    selection = {
        "module": "Selector",
        "command": "cmd.inst.funct",
        "state": "state",
        "config_state": "cfg.state",
        "config_field": {"bundle": "LoadCfg", "name": "state"},
        "config_class": "CONFIG_LD",
        "stride_register": "strides",
        "stride_input": "cmd.rs2",
        "dram_address": "cmd.rs1",
        "row_index": "row",
        "stride_sink": "address",
        "load_classes": ["MVIN", "MVIN2"],
    }
    isa = {
        "FUNCT_CLASS": {0: "CFG", 7: "MVIN", 9: "MVIN2"},
        "CONFIG_CLASS": "CFG",
        "CONFIG_SUBTYPE": {1: "CONFIG_LD"},
        "OPERAND_FIELDS": {
            "CONFIG_LD": {
                "fields": [
                    {"name": "state_id", "as": "int", "operand": "rs1", "offset": 3, "width": 1},
                    {"name": "stride", "as": "raw", "operand": "rs2"},
                ]
            },
            **{klass: {"fields": [{"name": "dram", "as": "operand", "operand": "rs1"}]} for klass in ("MVIN", "MVIN2")},
        },
    }
    return path, layouts, selection, isa


def test_actual_bank_mapping_and_width_are_immutable(tmp_path):
    _path, layouts, selection, isa = fixture(tmp_path)
    mapping = derive_load_state_mapping(layouts, selection, isa)
    assert mapping.ports == (("MVIN", 0), ("MVIN2", 1))
    assert mapping.states == (0, 1)
    assert mapping.stride_width == 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        mapping.stride_width = 64


@pytest.mark.parametrize("mutation", ["source", "foreign_source", "field", "decoder", "unknown_class", "missing"])
def test_selection_identity_decoder_and_missing_fields_refuse(tmp_path, mutation):
    path, layouts, selection, isa = fixture(tmp_path)
    if mutation == "source":
        path.write_text(SOURCE.replace("UInt<5>[2]", "UInt<6>[2]"))
    elif mutation == "foreign_source":
        layouts["source_sha256"] = "0" * 64
    elif mutation == "field":
        selection["config_state"] = "cfg.reserved"
    elif mutation == "decoder":
        isa["OPERAND_FIELDS"]["CONFIG_LD"]["fields"][0]["offset"] = 4
    elif mutation == "unknown_class":
        selection["load_classes"].append("UNKNOWN")
    else:
        selection.pop("stride_register")
    with pytest.raises(ValueError):
        derive_load_state_mapping(layouts, selection, isa)


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("UInt<1>(1), UInt<1>(0)", "UInt<2>(3), UInt<1>(0)"),
        SOURCE.replace("connect strides[state], cmd.rs2", "connect strides[state], cmd.rs1"),
        SOURCE.replace("when is_config :", "when is_second :"),
        SOURCE.replace("mul(row, strides[state])", "mul(row, strides[0])"),
        SOURCE.replace("connect cfg.state, config_bits", "connect cfg.state, config_bits\n    invalidate cfg.state"),
        SOURCE.replace("node state =", "node state = xor(selected, cfg.state)\n    node other ="),
        SOURCE.replace("reg strides : UInt<5>[2]", "reg strides : SInt<5>[2]"),
        SOURCE.replace("eq(cmd.inst.funct, UInt<4>(9))", "eq(cfg.state, UInt<1>(0))"),
    ],
)
def test_unsafe_bank_writer_selector_address_and_invalid_data_refuse(tmp_path, source):
    _path, layouts, selection, isa = fixture(tmp_path, source)
    with pytest.raises(ValueError):
        derive_load_state_mapping(layouts, selection, isa)


def test_original_metadata_is_not_mutated(tmp_path):
    _path, layouts, selection, isa = fixture(tmp_path)
    before = copy.deepcopy((layouts, selection, isa))
    derive_load_state_mapping(layouts, selection, isa)
    assert (layouts, selection, isa) == before


def transfer_fixture(tmp_path, expression=None):
    expression = expression or (
        "shr(mux(is_second, mul(mul(extent.cols, selected_rows), UInt<6>(32)), "
        "mul(mul(extent.cols, selected_rows), UInt<4>(8))), 3)"
    )
    source = SOURCE + """    wire extent : { rows : UInt<2>, cols : UInt<2> }
    connect extent.rows, bits(cmd.rs2, 5, 4)
    connect extent.cols, bits(cmd.rs2, 3, 2)
    node selected_rows = mux(is_second, UInt<1>(1), extent.rows)
    node byte_count = """ + expression + "\n"
    path, _layouts, selection, isa = fixture(tmp_path, source)
    fields = [{"bundle": "LoadCfg", "name": "state", "module": "Selector",
               "field": "cfg.state", "input": "cmd.rs1"}]
    fields += [{"bundle": "Extent", "name": axis, "module": "Selector",
                "field": f"extent.{axis}", "input": "cmd.rs2"} for axis in ("rows", "cols")]
    layouts = derive_operand_layouts(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     selections=fields)
    for klass in ("MVIN", "MVIN2"):
        isa["OPERAND_FIELDS"][klass]["fields"] += [
            {"name": axis, "operand": "rs2", "width": 2, "offset": offset}
            for axis, offset in [("rows", 4), ("cols", 2)]
        ]
    selection["transfer_bound"] = {"byte_count": "byte_count",
                                   **{axis: {"bundle": "Extent", "name": axis} for axis in ("rows", "cols")}}
    return layouts, selection, isa


def test_transfer_bound_covers_both_element_widths_and_repeated_rows(tmp_path):
    layouts, selection, isa = transfer_fixture(tmp_path)
    assert derive_load_state_mapping(layouts, selection, isa).max_element_bytes == 4


def test_transfer_bound_comes_from_source_constants(tmp_path):
    layouts, selection, isa = transfer_fixture(tmp_path, "shr(mul(mul(extent.rows, extent.cols), UInt<7>(64)), 3)")
    assert derive_load_state_mapping(layouts, selection, isa).max_element_bytes == 8


@pytest.mark.parametrize("expression", [
    "mul(extent.rows, extent.rows)", "mul(extent.cols, cmd.rs1)",
    "UInt<1>(0)", "missing", "div(extent.rows, extent.cols)",
])
def test_unbound_or_non_linear_transfer_counts_refuse(tmp_path, expression):
    layouts, selection, isa = transfer_fixture(tmp_path, expression)
    with pytest.raises(ValueError):
        derive_load_state_mapping(layouts, selection, isa)


def test_transfer_extents_must_match_the_actual_decoder(tmp_path):
    layouts, selection, isa = transfer_fixture(tmp_path)
    isa["OPERAND_FIELDS"]["MVIN"]["fields"][1]["offset"] = 3
    with pytest.raises(ValueError, match="original operand decoder"):
        derive_load_state_mapping(layouts, selection, isa)
