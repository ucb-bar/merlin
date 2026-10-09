"""Complete local partitions stop at state and preserve exact root ownership."""

from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_packing import equal_partitions


def _hardware(body):
    return parse_generic_hw(
        'builtin.module { "hw.module"() ({ ^bb0(%a: i12, %b: i12):\n'
        + body
        + '\n"hw.output"() : () -> ()\n'
        + '}) {sym_name = "ArbitraryName", module_type = !hw.modty<input a : i12, input b : i12>} : () -> () }'
    )


def test_nested_extracts_and_contiguous_reassembly_keep_complete_root_partition():
    facts = equal_partitions(
        _hardware("""
      %low = "comb.extract"(%a) {lowBit = 0 : i32} : (i12) -> i6
      %high = "comb.extract"(%a) {lowBit = 6 : i32} : (i12) -> i6
      %same = "comb.concat"(%high, %low) : (i6, i6) -> i12
      %s0 = "comb.extract"(%same) {lowBit = 0 : i32} : (i12) -> i3
      %s1 = "comb.extract"(%same) {lowBit = 3 : i32} : (i12) -> i3
      %s2 = "comb.extract"(%same) {lowBit = 6 : i32} : (i12) -> i3
      %s3 = "comb.extract"(%same) {lowBit = 9 : i32} : (i12) -> i3
    """)
    )
    assert facts["complete_resource_domain"] is False
    assert [(row["slice_width"], row["slice_count"]) for row in facts["partitions"]] == [(3, 4), (6, 2)]
    assert all(row["root"] == {"kind": "module_input", "ordinal": 0, "name": "a"} for row in facts["partitions"])


def test_gap_duplicate_and_multiple_input_slices_cannot_form_complete_partition():
    facts = equal_partitions(
        _hardware("""
      %s0 = "comb.extract"(%a) {lowBit = 0 : i32} : (i12) -> i3
      %s1 = "comb.extract"(%a) {lowBit = 3 : i32} : (i12) -> i3
      %duplicate = "comb.extract"(%a) {lowBit = 3 : i32} : (i12) -> i3
      %other = "comb.extract"(%b) {lowBit = 6 : i32} : (i12) -> i3
      %s3 = "comb.extract"(%a) {lowBit = 9 : i32} : (i12) -> i3
    """)
    )
    assert facts["partitions"] == [] and facts["incomplete_candidates"] == 2


def test_nontransparent_state_and_cross_root_concat_remain_unknown():
    facts = equal_partitions(
        _hardware("""
      %state = "seq.firreg"(%a) : (i12) -> i12
      %sl = "comb.extract"(%state) {lowBit = 0 : i32} : (i12) -> i6
      %sh = "comb.extract"(%state) {lowBit = 6 : i32} : (i12) -> i6
      %mixed = "comb.concat"(%a, %b) : (i12, i12) -> i24
      %ml = "comb.extract"(%mixed) {lowBit = 0 : i32} : (i24) -> i12
      %mh = "comb.extract"(%mixed) {lowBit = 12 : i32} : (i24) -> i12
    """)
    )
    assert facts["partitions"] == []
