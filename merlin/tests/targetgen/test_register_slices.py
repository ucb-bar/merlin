"""Conditional field recovery never promotes unknown data into a layout."""
from merlin.targetgen.rtl.register_slices import derive_register_slices, decode_register_slices


def hardware(offset=4, branch="%field", selector=37):
    return f'''hw.module @Engine(in %clk : !seq.clock, in %command : i7, in %payload : i32, in %ready : i1) {{
    %selector = hw.constant {selector} : i7
    %match = comb.icmp bin eq %command, %selector : i7
    %gate = comb.and bin %ready, %match : i1
    %field = comb.extract %payload from {offset} : (i32) -> i5
    %next = comb.mux bin %gate, {branch}, %state : i5
    %state = seq.firreg %next clock %clk : i5
    hw.output
  }}'''


def derive(text, selector=37):
    return derive_register_slices(text, module="Engine", registers=["%state"],
        selector="%command", selector_value=selector, inputs={"%payload": ("word", 32)})


def test_actual_slice_and_unknown_acceptance_are_separate():
    result = derive(hardware())
    field = result["registers"][0]
    assert (field["status"], field["offset"], field["width"]) == ("derived", 4, 5)
    assert result["arithmetic_and_sequencer_progress"] == "UNPROVEN"
    decoded = decode_register_slices(result, {"word": {"kind": "const", "raw": 0b111010000}})
    assert decoded[0]["value"] == 29


def test_selector_mutation_removes_update():
    assert derive(hardware(), selector=38)["registers"][0]["status"] == "no_non_hold_update"


def test_rtl_offset_mutation_changes_layout_not_assumed_bits():
    assert derive(hardware(9))["registers"][0]["offset"] == 9


def test_unknown_data_branch_refuses_field():
    assert derive(hardware(branch="%unmodeled"))["registers"][0]["status"] == "UNKNOWN"


def test_unknown_alternative_data_branch_refuses_even_with_known_slice():
    text = hardware().replace("%field = comb.extract", "%choice = comb.mux bin %ready, %field, %unmodeled : i5\n    %field = comb.extract")
    text = text.replace("%next = comb.mux bin %gate, %field", "%next = comb.mux bin %gate, %choice")
    assert derive(text)["registers"][0]["status"] == "UNKNOWN"


def test_pointer_slice_keeps_range_obligation():
    result = decode_register_slices(derive(hardware(0)),
        {"word": {"kind": "argbase", "arg_index": 2, "offset": 64}})[0]
    assert result["symbolic_value"] == {"argument": 2, "byte_offset": 64,
        "extract_offset": 0, "extract_width": 5, "address_range_validated": False}


def test_unresolved_operand_does_not_claim_field_value():
    assert decode_register_slices(derive(hardware()), {"word": {"kind": "unknown"}})[0]["status"] == "UNKNOWN"


def test_second_engine_width_and_selector_are_data():
    text = hardware(offset=7, selector=61).replace("i5", "i11")
    row = derive(text, selector=61)["registers"][0]
    assert (row["offset"], row["width"]) == (7, 11)


def test_reset_cannot_be_ignored_when_claiming_register_update():
    text = hardware().replace("clock %clk : i5", "clock %clk reset sync %ready, %other : i5")
    row = derive(text)["registers"][0]
    assert row["status"] == "UNKNOWN"
    assert row["reason"] == "reset update not modeled"
