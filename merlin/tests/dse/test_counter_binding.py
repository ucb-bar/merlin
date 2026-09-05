"""CIRCT counter binding extraction must retain UNKNOWN physical semantics."""
from __future__ import annotations

import textwrap

from merlin.perf.counter_binding import extract_external_additive_counters


_HEADER = textwrap.dedent("""\
    #define EXTERNAL_BASE 10
    #define OPAQUE_BYTES (EXTERNAL_BASE + 4)
    """)


def _hw() -> str:
    text = textwrap.dedent("""\
        module {
          hw.module private @Leaf(in %clock : !seq.clock, in %extent : i4,
                                  in %fire : i1, in %clear : i1, out count : i32) {
            %zero16 = hw.constant 0 : i16
            %zero32 = hw.constant 0 : i32
            %one16 = hw.constant 1 : i16
            %acc = seq.firreg %next clock %clock reset sync %clear, %zero32 : i32
            %wide_extent = comb.concat %zero16, %extent : i16, i4
            %shifted = comb.shl bin %one16, %wide_extent : i16
            %wide_increment = comb.concat %zero16, %shifted : i16, i16
            %sum = comb.add bin %acc, %wide_increment : i32
            %update = comb.mux bin %fire, %sum, %acc : i32
            %next = comb.mux bin %clear, %zero32, %update : i32
            hw.output %acc : i32
          }
          hw.module private @Pipe(in %clock : !seq.clock, in %extent : i4,
                                  in %fire : i1, in %clear : i1, out value : i32) {
            LEAF_INSTANCE
            hw.output %leaf.count : i32
          }
          hw.module private @File(in %external_4 : i32, out data : i32) {
            %zero = hw.constant 0 : i32
            %array = hw.array_create %zero, %external_4, %zero, %zero, %zero, %zero : i32
            %index = hw.constant 4 : i3
            %selected = hw.array_get %array[%index] : !hw.array<6xi32>, i3
            hw.output %selected : i32
          }
          hw.module private @Controller(in %external_4 : i32, out data : i32) {
            %file.data = hw.instance "file" @File(external_4: %external_4: i32) -> (data: i32)
            hw.output %file.data : i32
          }
          hw.module @Top(in %clock : !seq.clock, in %extent : i4,
                         in %fire : i1, in %clear : i1, out data : i32) {
            PIPE_INSTANCE
            %controller.data = hw.instance "controller" @Controller(external_4: %pipe.value: i32) -> (data: i32)
            hw.output %controller.data : i32
          }
        }
        """)
    leaf = ('%leaf.count = hw.instance "leaf" @Leaf(clock: %clock: !seq.clock, '
            'extent: %extent: i4, fire: %fire: i1, clear: %clear: i1) -> (count: i32)')
    pipe = ('%pipe.value = hw.instance "pipe" @Pipe(clock: %clock: !seq.clock, '
            'extent: %extent: i4, fire: %fire: i1, clear: %clear: i1) -> (value: i32)')
    return text.replace("LEAF_INSTANCE", leaf).replace("PIPE_INSTANCE", pipe)


def _extract(hw: str | None = None) -> dict:
    return extract_external_additive_counters(
        hw or _hw(), _HEADER, top_module="Top", counter_module="Controller",
        counter_file_module="File", external_port_prefix="external_",
        external_base_define="EXTERNAL_BASE", declared_unit="BYTES",
        source="synthetic.hw.mlir", header_source="synthetic.h")


def test_structure_is_proved_but_physical_binding_is_withheld() -> None:
    result = _extract()
    assert result["status"] == "unknown"
    assert result["counter_facts"] == []
    assert result["structurally_proved_candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["status"] == "structurally_proved"
    assert candidate["accumulator_proof"]["increment_formula"] == "1 << encoded_extent"
    assert candidate["direction"] is None and candidate["unit_bytes"] is None
    assert len(result["inputs"]["circt_core_hw"]["sha256"]) == 64


def test_header_name_is_not_treated_as_direction_or_byte_unit_proof() -> None:
    renamed = _HEADER.replace("OPAQUE_BYTES", "READ_BYTES")
    result = extract_external_additive_counters(
        _hw(), renamed, top_module="Top", counter_module="Controller",
        counter_file_module="File", external_port_prefix="external_",
        external_base_define="EXTERNAL_BASE", declared_unit="BYTES")
    candidate = result["candidates"][0]
    assert candidate["direction"] is None and candidate["unit_bytes"] is None
    assert result["counter_facts"] == []


def test_non_additive_update_fails_closed() -> None:
    result = _extract(_hw().replace("comb.add bin %acc, %wide_increment",
                                    "comb.xor bin %acc, %wide_increment"))
    assert result["structurally_proved_candidates"] == 0
    assert result["candidates"][0]["status"] == "unknown"


def test_missing_external_reset_fails_closed() -> None:
    result = _extract(_hw().replace("comb.mux bin %clear, %zero32, %update",
                                    "comb.mux bin %fire, %update, %acc"))
    assert result["structurally_proved_candidates"] == 0


def test_swapped_external_array_ordinal_fails_closed() -> None:
    # Move the external value from logical index four to logical index three.
    result = _extract(_hw().replace(
        "%zero, %external_4, %zero, %zero, %zero, %zero",
        "%zero, %zero, %external_4, %zero, %zero, %zero"))
    assert result["structurally_proved_candidates"] == 0
    assert "index disagree" in result["candidates"][0]["counter_file_proof"]["why"]


def test_zero_is_a_valid_external_array_ordinal() -> None:
    header = _HEADER.replace("EXTERNAL_BASE + 4", "EXTERNAL_BASE + 0")
    hw = _hw().replace("external_4", "external_0").replace(
        "%zero, %external_0, %zero, %zero, %zero, %zero",
        "%zero, %zero, %zero, %zero, %zero, %external_0")
    result = extract_external_additive_counters(
        hw, header, top_module="Top", counter_module="Controller",
        counter_file_module="File", external_port_prefix="external_",
        external_base_define="EXTERNAL_BASE", declared_unit="BYTES")

    assert result["structurally_proved_candidates"] == 1
    assert result["candidates"][0]["external_index"] == 0


def test_combinational_transformation_in_value_route_is_not_identity_routing() -> None:
    result = _extract(_hw().replace("hw.output %leaf.count : i32",
                                    "%changed = comb.add %leaf.count, %leaf.count : i32\n"
                                    "            hw.output %changed : i32"))
    assert result["structurally_proved_candidates"] == 0
    assert "not a seq.firreg" in result["candidates"][0]["accumulator_proof"]["why"]
