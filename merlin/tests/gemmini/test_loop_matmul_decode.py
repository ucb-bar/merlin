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
