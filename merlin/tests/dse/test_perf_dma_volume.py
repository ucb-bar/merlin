"""Movement-volume prediction must degrade to a floor, and must never read a size by position.

Two failures these pin, both of which return a plausible number instead of raising:

* summing only the descriptors that resolved. That understates the footprint in the FLATTERING
  direction -- a smaller predicted volume makes the compiler look better and the model look more
  accurate at once -- so any unresolved descriptor must demote the whole kernel to a lower bound.
* reading the size from a fixed operand position. It works on the form you tested and silently
  returns the wrong field on the next one.
"""
from __future__ import annotations

import json

from merlin.perf.dma_volume import (
    Descriptor,
    compare_to_counters,
    compare_to_measured,
    descriptors_from_program,
    kernel_volume,
    physical_volume_from_counters,
    propagate_constants,
    size_field_for,
    traffic_report_from_trace,
    volume_from_program,
)


class _Isa:
    """A stand-in ISA model exposing only the layout accessor this module is allowed to use."""

    def __init__(self, layouts):
        self._l = layouts

    def fields_of(self, mnemonic):
        return self._l[mnemonic]


_DIGEST = "a" * 64


def _semantic(**values):
    return {"fact_kind": "instruction_effect", "artifact_sha256": _DIGEST,
            "derived_from_rtl": True, "provenance": "generated RTL effect probe", **values}


def _field(**values):
    return {"fact_kind": "descriptor_layout", "artifact_sha256": _DIGEST,
            "derived_from_tool": True, "provenance": "generated ISA layout",
            "size_semantics": "static_product", **values}


def _counter(**values):
    return {"fact_kind": "counter_byte_binding", "artifact_sha256": _DIGEST,
            "derived_from_rtl": True, "provenance": "generated RTL counter binding", **values}


def _desc(i, direction="read", size=64, reason=None, field="len"):
    return Descriptor(index=i, form="move", channel=0, direction=direction,
                      size_bytes=size, size_field=field if size is not None else None,
                      unresolved_reason=reason)


def test_a_resolved_kernel_reports_a_total() -> None:
    v = kernel_volume("k", [_desc(0, "read", 64), _desc(1, "write", 32)])
    assert (v.read_bytes, v.write_bytes, v.total_bytes) == (64, 32, 96)
    assert v.is_lower_bound is False
    assert "AT LEAST" not in v.claim()


def test_one_unresolved_descriptor_demotes_the_whole_kernel_to_a_floor() -> None:
    """The flattering-direction guard: a partial sum must not present as a total."""
    v = kernel_volume("k", [_desc(0, "read", 64),
                            _desc(1, "read", None, reason="size came from a runtime value")])
    assert v.is_lower_bound is True
    assert v.total_bytes == 64                     # the resolved part is still reported
    assert "AT LEAST" in v.claim()                 # but never as a total
    assert v.unresolved and "runtime value" in v.unresolved[0]


def test_a_floor_below_its_measurement_is_consistent_not_a_match() -> None:
    """A lower bound cannot agree with a measurement; every unresolved descriptor could close the gap."""
    v = kernel_volume("k", [_desc(0, "read", 64), _desc(1, "read", None, reason="unresolved")])
    out = compare_to_measured(v, 4096)
    assert out["verdict"] == "consistent_lower_bound"
    assert "cannot match" in out["note"]


def test_a_floor_above_its_measurement_falsifies_an_input() -> None:
    """The same rule the headline experiment applies: a bound over its measurement is a refutation."""
    v = kernel_volume("k", [_desc(0, "read", 8192), _desc(1, "read", None, reason="unresolved")])
    out = compare_to_measured(v, 4096)
    assert out["verdict"] == "bound_violated"
    assert "falsifies" in out["note"]


def test_the_size_field_comes_from_the_declared_layout() -> None:
    isa = _Isa({"move.a": {"rd": [7], "rs1": [15], "len": [20, 21]}})
    assert size_field_for(isa, "move.a") == "len"


def test_a_form_that_declares_no_size_is_unknown_not_a_guess() -> None:
    """The pick-by-position bug: with no size field declared, the answer is None, not 'operand 2'."""
    isa = _Isa({"move.b": {"rd": [7], "rs1": [15], "vd": [20]}})
    assert size_field_for(isa, "move.b") is None


def test_the_layout_decides_which_field_even_when_forms_differ() -> None:
    """Two forms of one family may carry the size in different fields; position would confuse them."""
    isa = _Isa({"move.a": {"rd": [7], "len": [20]}, "move.b": {"rd": [7], "rs2": [20]}})
    assert size_field_for(isa, "move.a") == "len"
    assert size_field_for(isa, "move.b") == "rs2"


def test_propagation_kills_a_register_it_cannot_evaluate() -> None:
    """A register rewritten by an opaque instruction must go UNKNOWN, never keep a stale constant."""
    prog = [{"form": "li", "operands": {"rd": 2, "imm": 1024}},
            {"form": "opaque", "operands": {"rd": 2}},
            {"form": "move", "operands": {"rd": 0, "rs2": 2}}]
    states = propagate_constants(prog, immediate_forms={"li": "imm"})
    assert states[0][2] == 1024
    assert states[2][2] is None, "a stale constant survived an intervening write"


def test_a_backward_branch_invalidates_every_constant() -> None:
    """Inside a loop the 'constant' differs per iteration, so it is not a constant."""
    prog = [{"form": "li", "operands": {"rd": 2, "imm": 1024}},
            {"form": "br", "operands": {}, "branches_backward": True},
            {"form": "move", "operands": {"rd": 0, "rs2": 2}}]
    states = propagate_constants(prog, immediate_forms={"li": "imm"})
    assert states[2].get(2) is None, "a loop-carried value was treated as a constant"


def test_register_zero_is_never_marked_unknown() -> None:
    """Writing x0 discards; it must not poison the state map."""
    prog = [{"form": "opaque", "operands": {"rd": 0}}]
    assert propagate_constants(prog, immediate_forms={})[0].get(0) is None


def test_program_extraction_gets_direction_and_fields_only_from_supplied_facts() -> None:
    """Neither instruction identity nor an operand's position says whether or how many bytes move."""
    isa = _Isa({
        "constant.form": {"dst": [1], "literal": [2]},
        "opaque.a": {"route": [4], "extent": [8]},
        "opaque.b": {"payload": [3], "route": [9]},
        "opaque.c": {"route": [5]},
    })
    program = [
        {"form": "constant.form", "operands": {"dst": 6, "literal": 48}},
        {"form": "opaque.a", "operands": {"route": 7, "extent": 6}},
        {"form": "opaque.b", "operands": {"payload": 16, "route": 9}},
        {"form": "opaque.c", "operands": {"route": 7}},
    ]
    semantics = {
        "constant.form": _semantic(traffic=False),
        "opaque.a": _semantic(direction="read", provenance="rtl-effect-probe:a"),
        "opaque.b": _semantic(direction="write", provenance="rtl-effect-probe:b"),
        "opaque.c": _semantic(direction="sync", provenance="rtl-effect-probe:c"),
    }
    fields = {
        "opaque.a": _field(size_field="extent", size_source="register", unit_bytes=1,
                           channel_field="route", channel_source="value",
                           provenance="isa-layout:a"),
        "opaque.b": _field(size_field="payload", size_source="value", unit_bytes=1,
                           channel_field="route", channel_source="value",
                           provenance="isa-layout:b"),
        "opaque.c": _field(channel_field="route", channel_source="value",
                           provenance="isa-layout:c"),
    }

    descriptors = descriptors_from_program(
        program, isa, semantic_facts=semantics, field_facts=fields,
        immediate_forms={"constant.form": {
            "destination_field": "dst", "value_field": "literal",
        }})
    volume = kernel_volume("k", descriptors)

    assert [d.direction for d in descriptors] == ["read", "write", "sync"]
    assert [d.size_bytes for d in descriptors] == [48, 16, None]
    assert [d.channel for d in descriptors] == [7, 9, 7]
    assert descriptors[0].semantic_provenance == "rtl-effect-probe:a"
    assert descriptors[0].field_provenance == "isa-layout:a"
    assert (volume.read_bytes, volume.write_bytes, volume.total_bytes) == (48, 16, 64)
    assert volume.basis == "scheduled_descriptors"
    assert volume.scheduled_total_bytes == 64
    assert volume.read_provenance == ("rtl-effect-probe:a; isa-layout:a",)
    assert volume.write_provenance == ("rtl-effect-probe:b; isa-layout:b",)
    assert volume.exact_total_bytes == 64
    assert volume.is_lower_bound is False


def test_decoded_container_and_identity_key_are_explicit_not_assumed() -> None:
    """A runner may call its identity ``class`` and expose decoded fields separately."""
    isa = _Isa({"shape.form": {"height": [1], "width": [2], "lane": [3]}})
    program = [{"class": "shape.form", "decoded": {"lane": 4, "height": 3, "width": 5}}]
    semantics = {"shape.form": _semantic(direction="read", provenance="rtl")}
    fields = {"shape.form": _field(
        size_fields=["height", "width"], size_source="value", unit_bytes=2,
        channel_field="lane", channel_source="value", provenance="isa")}

    volume = volume_from_program(
        "shaped", program, isa, semantic_facts=semantics, field_facts=fields,
        identity_field="class", operands_field="decoded")

    assert volume.descriptors[0].size_fields == ("height", "width")
    assert volume.descriptors[0].size_bytes == 30
    assert volume.descriptors[0].channel == 4
    assert volume.total_bytes == 30


def test_unknown_direction_demotes_even_when_size_is_known() -> None:
    isa = _Isa({"opaque": {"extent": [8]}})
    descriptors = descriptors_from_program(
        [{"form": "opaque", "operands": {"extent": 64}}], isa,
        semantic_facts={"opaque": _semantic(direction=None, provenance="probe inconclusive")},
        field_facts={"opaque": _field(size_field="extent", size_source="value",
                                        unit_bytes=1, provenance="isa")})

    assert descriptors[0].direction == "unknown"
    assert descriptors[0].size_bytes == 64       # known evidence is retained, not rewritten to zero
    volume = kernel_volume("k", descriptors)
    assert volume.is_lower_bound is True
    assert volume.exact_total_bytes is None
    assert "direction" in volume.unresolved[0]


def test_missing_size_fact_is_unknown_never_a_zero_byte_transfer() -> None:
    isa = _Isa({"opaque": {"anything": [1]}})
    descriptors = descriptors_from_program(
        [{"form": "opaque", "operands": {"anything": 4096}}], isa,
        semantic_facts={"opaque": _semantic(direction="write", provenance="rtl")},
        field_facts={})

    assert descriptors[0].size_bytes is None
    assert descriptors[0].unresolved_reason
    volume = kernel_volume("k", descriptors)
    assert volume.total_bytes == 0                # this is explicitly a floor, never an exact zero
    assert volume.exact_total_bytes is None
    assert volume.is_lower_bound is True


def test_a_field_value_is_not_used_unless_the_isa_declares_that_field() -> None:
    """A fact/decoder mismatch fails closed instead of trusting a plausible decoded integer."""
    isa = _Isa({"opaque": {"different": [1]}})
    descriptors = descriptors_from_program(
        [{"form": "opaque", "operands": {"extent": 999}}], isa,
        semantic_facts={"opaque": _semantic(direction="read")},
        field_facts={"opaque": _field(size_field="extent", size_source="value",
                                        unit_bytes=1)})
    assert descriptors[0].size_bytes is None
    assert "ISA layout" in descriptors[0].unresolved_reason


def test_transfer_before_a_backedge_is_also_loop_carried_and_unknown() -> None:
    """The backedge appears later in program order, so a one-pass 'seen branch' test is unsound."""
    isa = _Isa({"constant": {"literal": [1]}, "opaque": {"extent": [2]}, "branch": {}})
    program = [
        {"form": "constant", "operands": {"rd": 8, "literal": 128}},
        {"form": "opaque", "operands": {"extent": 8}},
        {"form": "branch", "operands": {}, "branches_backward": True, "branch_target": 1},
    ]
    descriptors = descriptors_from_program(
        program, isa,
        semantic_facts={"constant": _semantic(traffic=False),
                        "opaque": _semantic(direction="read"),
                        "branch": _semantic(traffic=False)},
        field_facts={"opaque": _field(size_field="extent", size_source="register",
                                        unit_bytes=1)},
        immediate_forms={"constant": "literal"})

    assert descriptors[0].size_bytes is None
    assert "loop" in descriptors[0].unresolved_reason
    assert volume_from_program(
        "loop", program, isa,
        semantic_facts={"constant": _semantic(traffic=False),
                        "opaque": _semantic(direction="read"),
                        "branch": _semantic(traffic=False)},
        field_facts={"opaque": _field(size_field="extent", size_source="register",
                                        unit_bytes=1)},
        immediate_forms={"constant": "literal"}).exact_total_bytes is None


def test_instructions_with_no_semantic_fact_make_the_trace_a_lower_bound() -> None:
    isa = _Isa({"ordinary": {"extent": [1]}})
    descriptors = descriptors_from_program(
        [{"form": "ordinary", "operands": {"extent": 512}}], isa,
        semantic_facts={}, field_facts={})
    assert len(descriptors) == 1 and not descriptors[0].traffic_resolved
    assert kernel_volume("k", descriptors).exact_total_bytes is None


def test_physical_bytes_require_explicit_counter_bindings_and_units() -> None:
    """Counter field spellings and beat sizes are target facts, never names embedded in the bridge."""
    physical = physical_volume_from_counters(
        {"alpha": 4, "omega": 2},
        counter_facts=[
            _counter(counter_field="alpha", direction="read", unit_bytes=32,
                     provenance="rtl-counter:a"),
            _counter(counter_field="omega", direction="write", unit_bytes=16,
                     provenance="rtl-counter:b"),
        ])

    assert physical.basis == "physical_counters"
    assert (physical.read_bytes, physical.write_bytes, physical.total_bytes) == (128, 32, 160)
    assert physical.read_provenance == ("rtl-counter:a",)
    assert physical.write_provenance == ("rtl-counter:b",)
    assert physical.is_lower_bound is False


def test_unknown_counter_direction_keeps_physical_totals_unknown() -> None:
    physical = physical_volume_from_counters(
        {"alpha": 4, "mystery": 3},
        counter_facts=[
            _counter(counter_field="alpha", direction="read", unit_bytes=32,
                     provenance="rtl-counter:a"),
            _counter(counter_field="mystery", direction=None, unit_bytes=16,
                     provenance="probe-inconclusive"),
        ])

    assert physical.read_bytes is None
    assert physical.write_bytes is None
    assert physical.total_bytes is None
    assert physical.known_lower_bound_bytes == 176
    assert physical.unattributed_bytes == 48
    assert physical.is_lower_bound is True
    assert "direction" in physical.unresolved[0]


def test_counter_comparison_reports_scheduled_and_physical_bases() -> None:
    scheduled = kernel_volume("k", [_desc(0, "read", 128), _desc(1, "write", 32)])
    result = compare_to_counters(
        scheduled, {"one": 4, "two": 2},
        counter_facts=[
            _counter(counter_field="one", direction="read", unit_bytes=32,
                     provenance="counter:one"),
            _counter(counter_field="two", direction="write", unit_bytes=16,
                     provenance="counter:two"),
        ])
    assert result["verdict"] == "match"
    assert result["scheduled"]["basis"] == "scheduled_descriptors"
    assert result["physical"]["basis"] == "physical_counters"
    assert result["physical"]["total_bytes"] == 160
    assert result["directional"]["read"]["verdict"] == "match"


def test_counter_comparison_refuses_an_unknown_physical_binding() -> None:
    scheduled = kernel_volume("k", [_desc(0, "read", 128)])
    result = compare_to_counters(
        scheduled, {"mystery": 4},
        counter_facts=[_counter(counter_field="mystery", direction=None, unit_bytes=32,
                                provenance="inconclusive")])
    assert result["verdict"] == "unknown_measurement"
    assert result["physical"]["total_bytes"] is None


def test_equal_aggregate_bytes_do_not_hide_a_directional_mismatch() -> None:
    scheduled = kernel_volume("k", [_desc(0, "read", 128), _desc(1, "write", 32)])
    result = compare_to_counters(
        scheduled, {"one": 100, "two": 60},
        counter_facts=[
            _counter(counter_field="one", direction="read", unit_bytes=1),
            _counter(counter_field="two", direction="write", unit_bytes=1),
        ])
    assert result["physical"]["total_bytes"] == scheduled.total_bytes
    assert result["verdict"] == "directional_mismatch"
    assert result["directional"]["read"]["verdict"] == "mismatch"


def test_scheduled_and_physical_serialization_is_json_ready() -> None:
    scheduled = kernel_volume("k", [
        Descriptor(0, "a", 3, "read", 64, "extent", semantic_provenance="rtl:a",
                   field_provenance="isa:a"),
        Descriptor(1, "b", 3, "write", None, "amount", "runtime value",
                   semantic_provenance="rtl:b", field_provenance="isa:b"),
    ])
    physical = physical_volume_from_counters(
        {"one": 2, "two": 1},
        counter_facts=[
            _counter(counter_field="one", direction="read", unit_bytes=32,
                     provenance="counter:one"),
            _counter(counter_field="two", direction="write", unit_bytes=32,
                     provenance="counter:two"),
        ])

    scheduled_record = scheduled.to_dict()
    physical_record = physical.to_dict()
    json.dumps({"scheduled": scheduled_record, "physical": physical_record})
    assert scheduled_record["basis"] == "scheduled_descriptors"
    assert scheduled_record["exact_total_bytes"] is None
    assert scheduled_record["descriptors"][0]["provenance"]["semantic"] == "rtl:a"
    assert isinstance(scheduled_record["unresolved"], list)
    assert physical_record["basis"] == "physical_counters"
    assert physical_record["exact_total_bytes"] == 96
    assert physical_record["provenance"]["write"] == ["counter:two"]


def test_trace_report_uses_only_the_fact_bundles_explicit_adapter() -> None:
    isa = _Isa({"ordinary": {}, "traffic.form": {"extent": [2], "route": [3]}})
    trace = {
        "ops-list": [
            {"tag": "ordinary", "args-map": {}},
            {"tag": "traffic.form", "args-map": {"extent": 4, "route": 9}},
        ],
        "bus-sample": {"alpha": 4, "omega": 0},
    }
    bundle = {
        "provenance": {"semantic": "rtl-probe", "fields": "isa-model", "adapter": "runner-schema"},
        "adapter": {
            "instructions_field": "ops-list", "identity_field": "tag",
            "operands_field": "args-map", "counter_readings_field": "bus-sample",
        },
        "semantic_facts": {
            "ordinary": _semantic(traffic=False, provenance="rtl:ordinary"),
            "traffic.form": _semantic(traffic=True, direction="read", provenance="rtl:traffic"),
        },
        "field_facts": {
            "traffic.form": _field(size_field="extent", size_source="value", unit_bytes=1,
                                   channel_field="route", channel_source="value",
                                   provenance="isa:traffic"),
        },
        "counter_facts": [
            _counter(counter_field="alpha", direction="read", unit_bytes=1,
                     provenance="counter:alpha"),
            _counter(counter_field="omega", direction="write", unit_bytes=1,
                     provenance="counter:omega"),
        ],
    }

    report = traffic_report_from_trace("kernel", trace, isa, fact_bundle=bundle)
    json.dumps(report)
    assert report["status"] == "exact"
    assert report["provenance"] == bundle["provenance"]
    assert report["scheduled"]["exact_total_bytes"] == 4
    assert len(report["scheduled"]["descriptors"]) == 1
    assert report["physical"]["exact_total_bytes"] == 4
    assert report["comparison"]["verdict"] == "match"


def test_trace_report_explicitly_refuses_missing_adapter_facts() -> None:
    report = traffic_report_from_trace(
        "kernel", {"ops": []}, _Isa({}),
        fact_bundle={
            "provenance": "synthetic-facts",
            "adapter": {"instructions_field": "ops", "identity_field": "tag"},
            "semantic_facts": {}, "field_facts": {},
        })
    json.dumps(report)
    assert report["status"] == "unknown"
    assert report["scheduled"] is None
    assert "operands_field" in report["unresolved"][0]


def test_trace_report_missing_semantic_binding_is_a_floor_not_an_exact_zero() -> None:
    trace = {"ops": [{"tag": "unbound", "args": {"extent": 4096}}]}
    report = traffic_report_from_trace(
        "kernel", trace, _Isa({"unbound": {"extent": [1]}}),
        fact_bundle={
            "provenance": "synthetic-facts",
            "adapter": {"instructions_field": "ops", "identity_field": "tag",
                        "operands_field": "args"},
            "semantic_facts": {}, "field_facts": {},
        })

    assert report["status"] == "lower_bound"
    assert report["scheduled"]["exact_total_bytes"] is None
    assert report["scheduled"]["known_lower_bound_bytes"] == 0
    assert "semantic binding" in report["unresolved"][0]


def test_trace_report_requires_fact_bundle_provenance() -> None:
    report = traffic_report_from_trace(
        "kernel", {"ops": []}, _Isa({}),
        fact_bundle={
            "adapter": {"instructions_field": "ops", "identity_field": "tag",
                        "operands_field": "args"},
            "semantic_facts": {}, "field_facts": {},
        })
    assert report["status"] == "unknown"
    assert "provenance" in report["unresolved"][0]


def test_extra_and_duplicate_counter_bindings_can_never_form_an_exact_total() -> None:
    physical = physical_volume_from_counters(
        {"read": 3, "write": 2, "omitted": 999},
        counter_facts=[
            _counter(counter_field="read", direction="read", unit_bytes=1),
            _counter(counter_field="read", direction="read", unit_bytes=1),
            _counter(counter_field="write", direction="write", unit_bytes=1),
        ])

    assert physical.total_bytes is None and physical.is_lower_bound
    assert physical.known_lower_bound_bytes == 5
    assert any("duplicate" in reason for reason in physical.unresolved)
    assert any("no exhaustive byte binding" in reason for reason in physical.unresolved)


def test_unproven_counter_bindings_and_zero_units_are_unknown() -> None:
    physical = physical_volume_from_counters(
        {"read": 3, "write": 2},
        counter_facts=[
            {"counter_field": "read", "direction": "read", "unit_bytes": 1},
            _counter(counter_field="write", direction="write", unit_bytes=0),
        ])

    assert physical.total_bytes is None and physical.known_lower_bound_bytes == 0
    assert any("unproven" in reason for reason in physical.unresolved)
    assert any("byte unit" in reason for reason in physical.unresolved)


def test_stateful_descriptor_semantics_are_unknown_until_trace_resolved() -> None:
    isa = _Isa({"configured.move": {"rows": [1], "cols": [2]}})
    descriptor = descriptors_from_program(
        [{"form": "configured.move", "operands": {"rows": 0, "cols": 16}}], isa,
        semantic_facts={"configured.move": _semantic(direction="write")},
        field_facts={"configured.move": {
            **_field(size_fields=["rows", "cols"], size_source="value", unit_bytes=1),
            "size_semantics": "state_machine",
        }})[0]

    assert descriptor.size_bytes is None
    assert "stateful/dynamic" in descriptor.unresolved_reason
