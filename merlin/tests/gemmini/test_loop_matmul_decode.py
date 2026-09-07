"""Target-edge pin and emitted-field mutation checks, without execution."""
import importlib.util
import hashlib
import json

import pytest

from merlin.common.paths import repo_root


def adapter():
    path = repo_root() / "merlin/targets/gemmini/backend/gemmini_loop_matmul_decode.py"
    spec = importlib.util.spec_from_file_location("loop_matmul_decode_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(offset=7):
    hardware = f'''hw.module @LoopMatmul(in %clk : !seq.clock, in %ready : i1) {{
    %selector = hw.constant 39 : i7
    %match = comb.icmp bin eq %cmd_q.io_deq_bits_cmd_inst_funct, %selector : i7
    %gate = comb.and bin %ready, %match : i1
    %a = comb.extract %cmd_q.io_deq_bits_cmd_rs1 from {offset} : (i32) -> i5
    %b = comb.extract %cmd_q.io_deq_bits_cmd_rs2 from 0 : (i32) -> i5
    %next = comb.mux bin %gate, %a, %loops_0_pad_i : i5
    %loops_0_pad_i = seq.firreg %next clock %clk : i5
    hw.output
  }}'''
    facts = {"inputs": {"core_hw_sha256": hashlib.sha256(hardware.encode()).hexdigest()},
        "facts": {"interfaces": [{"name": "funct_decode_table", "names": {"39": "LOOP_WS_CONFIG_BOUNDS"},
            "legal_funct": [39]}]}}
    return hardware, facts


def writeback_fixture():
    hardware = '''hw.module @LoopMatmul(in %clk : !seq.clock, in %ready : i1) {
    %dc_selector = hw.constant 11 : i7
    %dc_match = comb.icmp bin eq %cmd_q.io_deq_bits_cmd_inst_funct, %dc_selector : i7
    %dc_gate = comb.and bin %ready, %dc_match : i1
    %c_addr = comb.extract %cmd_q.io_deq_bits_cmd_rs2 from 0 : (i64) -> i40
    %c_next = comb.mux bin %dc_gate, %c_addr, %loops_0_c_dram_addr : i40
    %loops_0_c_dram_addr = seq.firreg %c_next clock %clk : i40
    %run_selector = hw.constant 8 : i7
    %run_match = comb.icmp bin eq %cmd_q.io_deq_bits_cmd_inst_funct, %run_selector : i7
    %run_gate = comb.and bin %ready, %run_match : i1
    %full = comb.extract %cmd_q.io_deq_bits_cmd_rs1 from 1 : (i64) -> i1
    %full_next = comb.mux bin %run_gate, %full, %loops_0_full_c : i1
    %loops_0_full_c = seq.firreg %full_next clock %clk : i1
    hw.output
  }'''
    facts = {
        "inputs": {"core_hw_sha256": hashlib.sha256(hardware.encode()).hexdigest()},
        "facts": {
            "target": "gemmini",
            "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
            "datapaths": [
                {"name": "input", "dtype": "i8", "evidence": "synthetic scratchpad path"},
                {"name": "accumulator", "dtype": "i32", "evidence": "synthetic accumulator path"},
            ],
            "memories": [
                {"name": "scratchpad", "bytes": 4096, "depth": 16, "source": "synthetic RTL"},
                {"name": "accumulator", "bytes": 4096, "depth": 16, "source": "synthetic RTL"},
            ],
            "interfaces": [{
                "name": "funct_decode_table",
                "names": {"8": "LOOP_WS", "11": "LOOP_WS_CONFIG_ADDRS_DC"},
                "legal_funct": [8, 11],
            }],
        },
    }
    return hardware, facts


def test_target_adapter_uses_rtl_bits_and_records_missing_fields():
    hardware, facts = fixture()
    result = adapter().derive_layouts(facts_text=json.dumps(facts), hardware_text=hardware)
    row = result["layouts"]["39"]["layout"]["registers"][0]
    assert (row["offset"], row["width"], row["field"]) == (7, 5, "pad_i")
    assert "full_c" in result["absent_selected_fields"]
    instructions = [{"index": 0, "funct": 39, "rs1": {"kind": "const", "raw": 9 << 7}}]
    observed = adapter().observe_instructions(instructions, layouts=result)
    assert observed["instructions"][0]["fields"][0]["value"] == 9
    instructions[0]["rs1"]["raw"] += 1 << 7
    assert adapter().observe_instructions(instructions, layouts=result)["instructions"][0]["fields"][0]["value"] == 10
    assert observed["descriptor_arithmetic_equivalence"] == "UNKNOWN"


def test_stale_hardware_refuses_even_if_structurally_valid():
    hardware, facts = fixture()
    with pytest.raises(ValueError, match="does not match"):
        adapter().derive_layouts(facts_text=json.dumps(facts), hardware_text=hardware + "\n")


def test_header_only_selector_cannot_become_authority():
    hardware, facts = fixture()
    facts["facts"]["interfaces"][0]["legal_funct"] = []
    with pytest.raises(ValueError, match="header-only"):
        adapter().derive_layouts(facts_text=json.dumps(facts), hardware_text=hardware)


@pytest.mark.parametrize(("full_c", "expected_width"), [(0, 8), (1, 32)])
def test_actual_loop_descriptor_exposes_abi_destination_and_physical_width(full_c, expected_width):
    hardware, facts = writeback_fixture()
    layouts = adapter().derive_layouts(
        facts_text=json.dumps(facts), hardware_text=hardware)
    instructions = [
        {"index": 0, "funct": 11,
         "rs1": {"kind": "const", "raw": 0},
         "rs2": {"kind": "argbase", "arg_index": 4, "offset": 96}},
        {"index": 1, "funct": 8,
         "rs1": {"kind": "const", "raw": full_c << 1},
         "rs2": {"kind": "const", "raw": 0}},
    ]

    evidence = adapter().derive_writebacks(instructions, layouts=layouts)

    assert evidence["coverage_status"] == "complete"
    assert evidence["covered_instruction_indices"] == [0, 1]
    assert evidence["writebacks"] == [{
        "instruction_index": 1,
        "instruction_name": "LOOP_WS",
        "destination": {"arg_index": 4, "byte_offset": 96},
        "physical_readout": {"encoding": "signed_integer", "width_bits": expected_width},
        "selector": {"field": "full_c", "value": full_c},
    }]


def test_unresolved_loop_destination_fails_closed():
    hardware, facts = writeback_fixture()
    layouts = adapter().derive_layouts(
        facts_text=json.dumps(facts), hardware_text=hardware)
    instructions = [
        {"index": 0, "funct": 11,
         "rs1": {"kind": "const", "raw": 0}, "rs2": {"kind": "unknown"}},
        {"index": 1, "funct": 8,
         "rs1": {"kind": "const", "raw": 0},
         "rs2": {"kind": "const", "raw": 0}},
    ]

    evidence = adapter().derive_writebacks(instructions, layouts=layouts)

    assert evidence["coverage_status"] == "UNKNOWN"
    assert evidence["writebacks"] == []
    assert evidence["unresolved_writebacks"] == [{
        "instruction_index": 1,
        "reason": "c_dram_addr update is not derived for every possible descriptor slot",
    }]
