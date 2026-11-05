"""Full source-local state/getter timelines retain all timer unknowns."""

import hashlib
from dataclasses import asdict, replace

import pytest

from merlin.targetgen.rtl.hw_combinational import EvaluationLimits
from merlin.targetgen.rtl.hw_counter_intervals import CounterInterval
from merlin.targetgen.rtl.hw_counter_state_timelines import (
    StateGetterSelection,
    StatePhaseSample,
    StateTimelineLimits,
    observe_state_getter_timeline,
)
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_observations import _name

LIMITS = StateTimelineLimits(EvaluationLimits(65536, 256, 128, 1024, 4_000_000), 16, 64)


def _source(low=2, high=3, *, direct_clock=False, secondary=False):
    clock_type = "!seq.clock" if direct_clock else "i1"
    conversion = "" if direct_clock else '%tick = "seq.to_clock"(%c) : (i1) -> !seq.clock'
    tick = "%c" if direct_clock else "%tick"
    secondary_input = ", %other: !seq.clock" if secondary else ""
    secondary_port = ", input other : !seq.clock" if secondary else ""
    control_clock = "%other" if secondary else tick
    return f"""builtin.module {{
      "hw.module"() ({{
      ^bb0(%c: {clock_type}, %reset: i1, %stall: i1, %inhibit_write: i1, %inhibit_value: i1,
           %write: i1, %write_value: i{low + high}{secondary_input}):
        {conversion}
        %zero_low = "hw.constant"() {{value = 0 : i{low}}} : () -> i{low}
        %zero_high = "hw.constant"() {{value = 0 : i{high}}} : () -> i{high}
        %zero_bit = "hw.constant"() {{value = 0 : i1}} : () -> i1
        %one_bit = "hw.constant"() {{value = 1 : i1}} : () -> i1
        %one_high = "hw.constant"() {{value = 1 : i{high}}} : () -> i{high}
        %not_stall = "comb.xor"(%stall, %one_bit) : (i1, i1) -> i1
        %extend_low = "comb.concat"(%zero_bit, %lo) : (i1, i{low}) -> i{low + 1}
        %extend_inc = "comb.concat"(%zero_low, %not_stall) : (i{low}, i1) -> i{low + 1}
        %sum = "comb.add"(%extend_low, %extend_inc) {{twoState}} : (i{low + 1}, i{low + 1}) -> i{low + 1}
        %tail = "comb.extract"(%sum) {{lowBit = 0 : i32}} : (i{low + 1}) -> i{low}
        %carry = "comb.extract"(%sum) {{lowBit = {low} : i32}} : (i{low + 1}) -> i1
        %not_inhibit = "comb.xor"(%blocked, %one_bit) : (i1, i1) -> i1
        %high_enable = "comb.and"(%carry, %not_inhibit) : (i1, i1) -> i1
        %high_sum = "comb.add"(%hi, %one_high) : (i{high}, i{high}) -> i{high}
        %high_auto = "comb.mux"(%high_enable, %high_sum, %hi) : (i1, i{high}, i{high}) -> i{high}
        %low_auto = "comb.mux"(%blocked, %lo, %tail) : (i1, i{low}, i{low}) -> i{low}
        %write_low = "comb.extract"(%write_value) {{lowBit = 0 : i32}} : (i{low + high}) -> i{low}
        %write_high = "comb.extract"(%write_value) {{lowBit = {low} : i32}} : (i{low + high}) -> i{high}
        %low_write = "comb.mux"(%write, %write_low, %low_auto) : (i1, i{low}, i{low}) -> i{low}
        %high_write = "comb.mux"(%write, %write_high, %high_auto) : (i1, i{high}, i{high}) -> i{high}
        %low_next = "comb.mux"(%reset, %zero_low, %low_write) : (i1, i{low}, i{low}) -> i{low}
        %control_next = "comb.mux"(%inhibit_write, %inhibit_value, %blocked) : (i1, i1, i1) -> i1
        %lo = "seq.firreg"(%low_next, {tick}) {{name = "opaque_a", firrtl.random_init_start = 20 : ui64}}
          : (i{low}, !seq.clock) -> i{low}
        %hi = "seq.firreg"(%high_write, {tick}, %reset, %zero_high) {{name = "opaque_b"}}
          : (i{high}, !seq.clock, i1, i{high}) -> i{high}
        %blocked = "seq.firreg"(%control_next, {control_clock}, %reset, %zero_bit) {{name = "opaque_c"}}
          : (i1, !seq.clock, i1, i1) -> i1
        %value = "comb.concat"(%hi, %lo) : (i{high}, i{low}) -> i{low + high}
        "hw.output"(%value, %blocked, %carry) : (i{low + high}, i1, i1) -> ()
      }}) {{sym_name = "StateCell", parameters = [], module_type = !hw.modty<input pulse : {clock_type},
        input clear : i1, input stall : i1, input set_control : i1, input control_value : i1,
        input replace : i1, input replacement : i{low + high}{secondary_port},
        output visible : i{low + high}, output control : i1, output carry : i1>}} : () -> ()
    }}"""


def _selection(source):
    parsed = parse_generic_hw(source)
    module = next(op for op in parsed.walk() if _name(op) == "hw.module")
    indices = {
        op.results[0].name_hint: ordinal
        for ordinal, op in enumerate(module.regions[0].block.ops)
        if _name(op) == "seq.firreg"
    }
    return StateGetterSelection(
        hashlib.sha256(source.encode()).hexdigest(), "StateCell", (indices["hi"], indices["lo"]), "visible", "pulse"
    )


def _samples(cycles, *, low=2, high=3, initial=30, blocked=0, secondary=False):
    # Independent whole unsigned integer expectation. Source checker evaluates
    # separate simultaneous register transfers and the original concat getter.
    modulus, mask = 2 ** (low + high), 2**low - 1
    value, samples = initial, []
    other_prior = 0
    for reset, stall, set_control, control, write, replacement, *other_levels in cycles:
        for phase in (0, 1):
            other = other_levels[phase] if secondary else phase
            prior_blocked = blocked
            if phase:
                if reset:
                    value = 0
                elif write:
                    value = replacement
                elif not (prior_blocked or stall):
                    value = (value + 1) % modulus
            if not other_prior and other:
                blocked = 0 if reset else control if set_control else blocked
            other_prior = other
            carry = int((value & mask) + int(not stall) > mask)
            inputs = (phase, reset, stall, set_control, control, write, replacement)
            if secondary:
                inputs += (other,)
            samples.append(
                StatePhaseSample(len(samples), inputs, (value & mask, value >> low, blocked), (value, blocked, carry))
            )
    return tuple(samples)


def _observe(source, samples, *, selection=None, limits=LIMITS, intervals=None, expected=None):
    return observe_state_getter_timeline(
        source,
        selection=_selection(source) if selection is None else selection,
        limits=limits,
        expected_phases=len(samples) if expected is None else expected,
        samples=samples,
        intervals=(CounterInterval(0, len(samples) - 1),) if intervals is None else intervals,
    )


@pytest.mark.parametrize("low,high,direct", [(1, 1, False), (2, 3, True), (6, 58, False), (16, 16, True)])
def test_split_counter_complete_carry_wrap_stall_and_original_states(low, high, direct):
    source = _source(low, high, direct_clock=direct)
    samples = _samples(
        ((0, 0, 0, 0, 0, 0),) * 3 + ((0, 1, 0, 0, 0, 0),), low=low, high=high, initial=2 ** (low + high) - 2
    )
    result = _observe(source, samples)
    interval = result.intervals[0]
    assert interval.rising_edges == 4 and interval.observed_wrap_changes == 1 and interval.observed_unit_changes == 3
    assert interval.held_edges == 1 and interval.observed_modular_delta == 3
    assert interval.unit_increments is None
    assert len(result.states) == 3 and result.states[0].random_initialization_offset == 20
    assert len(result.transitions) == len(samples)
    assert result.getter_width == low + high
    assert "unit_increment_meaning_for_general_state_updates" in result.unknowns


def test_actual_control_state_is_simultaneous_not_candidate_input_substitution():
    source = _source()
    cycles = ((0, 0, 1, 1, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0))
    samples = _samples(cycles, initial=2)
    result = _observe(source, samples)
    assert tuple(row.outputs[0] for row in samples[1::2]) == (3, 3, 3, 4)
    assert result.intervals[0].observed_unit_changes == 2 and result.intervals[0].held_edges == 2
    assert result.transitions[1].before == (2, 0, 0) and result.transitions[1].after == (3, 0, 1)
    wrong = replace(samples[1], states=(samples[1].states[0], samples[1].states[1], 0))
    with pytest.raises(ValueError, match="actual source transitions"):
        _observe(source, (*samples[:1], wrong, *samples[2:]))


def test_source_low_reset_mux_high_primitive_reset_priority_and_nonunit_writes():
    source = _source()
    samples = _samples(((0, 0, 0, 0, 1, 7), (1, 0, 1, 1, 1, 20), (0, 0, 0, 0, 0, 0)), initial=1)
    result = _observe(source, samples)
    interval = result.intervals[0]
    assert interval.reset_edges == 1 and interval.nonunit_changes == 1 and interval.observed_unit_changes == 1
    assert interval.unit_increments is None
    assert result.transitions[3].reset_registers == (result.states[1].ordinal, result.states[2].ordinal)
    assert result.transitions[3].after == (0, 0, 0)


def test_numeric_unit_write_cannot_issue_unit_event_meaning():
    result = _observe(_source(), _samples(((0, 0, 0, 0, 1, 4),), initial=3))
    assert result.intervals[0].observed_unit_changes == 1
    assert result.intervals[0].unit_increments is None


def test_complete_secondary_clock_history_can_hold_original_control_state():
    source = _source(secondary=True)
    samples = _samples(
        ((0, 0, 1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 0)), secondary=True, initial=2
    )
    result = _observe(source, samples)
    assert tuple(row.outputs[0] for row in samples[1::2]) == (3, 4, 4)
    assert result.states[2].ordinal not in result.transitions[1].rising_registers
    assert result.states[2].ordinal in result.transitions[3].rising_registers


def test_overlapping_intervals_retain_reset_and_each_original_endpoint():
    samples = _samples(((0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)))
    result = _observe(_source(), samples, intervals=(CounterInterval(0, 5), CounterInterval(3, 5)))
    assert result.intervals[0].reset_edges == 1 and result.intervals[1].reset_edges == 0
    assert all(row.unit_increments is None for row in result.intervals)


def test_nested_getter_derives_every_full_register_in_actual_order():
    source = _source()
    source = source.replace(
        '"hw.output"(%value, %blocked, %carry) : (i5, i1, i1)',
        '%composed = "comb.concat"(%value, %blocked) : (i5, i1) -> i6\n'
        '        "hw.output"(%composed, %blocked, %carry) : (i6, i1, i1)',
    ).replace("output visible : i5", "output visible : i6")
    selection = _selection(source)
    parsed = parse_generic_hw(source)
    module = next(op for op in parsed.walk() if _name(op) == "hw.module")
    third = next(
        ordinal
        for ordinal, op in enumerate(module.regions[0].block.ops)
        if _name(op) == "seq.firreg" and op.results[0].name_hint == "blocked"
    )
    selection = replace(selection, register_ordinals=(*selection.register_ordinals, third))
    rows = _samples(((0, 0, 1, 1, 0, 0), (0, 0, 0, 0, 0, 0)), initial=2)
    rows = tuple(replace(row, outputs=((row.outputs[0] << 1) | row.outputs[1], *row.outputs[1:])) for row in rows)
    observation = _observe(source, rows, selection=selection)
    assert observation.getter_width == 6
    assert observation.intervals[0].unit_increments is None


def test_unused_unsupported_original_expression_cannot_be_removed_from_the_roster():
    source = _source().replace('"hw.output"(%value', '%unused = "unknown.body"(%lo) : (i2) -> i2\n "hw.output"(%value')
    with pytest.raises(ValueError, match="expression semantics"):
        _observe(source, _samples(((0, 0, 0, 0, 0, 0),)))


@pytest.mark.parametrize("attribute", ["-1 : ui64", "20 : i64", "true"])
def test_original_random_initialization_metadata_is_typed_and_not_a_known_initial_state(attribute):
    source = _source().replace("20 : ui64", attribute)
    selection = replace(_selection(_source()), source_sha256=hashlib.sha256(source.encode()).hexdigest())
    with pytest.raises(ValueError):
        _observe(source, _samples(((0, 0, 0, 0, 0, 0),)), selection=selection)


@pytest.mark.parametrize(
    "mutation", ["input", "state", "output", "bool", "gap", "clock", "secondary", "unstable", "partial", "negative"]
)
def test_changed_partial_or_unstable_original_timeline_refuses(mutation):
    source = _source(secondary=True)
    rows = list(_samples(((0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1)), secondary=True))
    row = rows[1]
    if mutation == "input":
        row = replace(row, inputs=row.inputs[:-1])
    elif mutation == "state":
        row = replace(row, states=row.states[:-1])
    elif mutation == "output":
        row = replace(row, outputs=(row.outputs[0], 1, row.outputs[2]))
    elif mutation == "bool":
        row = replace(row, states=(True, *row.states[1:]))
    elif mutation == "gap":
        row = replace(row, ordinal=2)
    elif mutation == "clock":
        row = replace(row, inputs=(0, *row.inputs[1:]))
    elif mutation == "secondary":
        rows[2] = replace(rows[2], inputs=(*rows[2].inputs[:-1], 1))
        row = replace(row, inputs=(*row.inputs[:-1], 0))
    elif mutation == "unstable":
        row = replace(row, inputs=(row.inputs[0], 1, *row.inputs[2:]))
    elif mutation == "partial":
        rows = rows[:-1]
    else:
        row = replace(row, states=(-1, *row.states[1:]))
    rows[1] = row
    with pytest.raises(ValueError):
        _observe(source, tuple(rows), expected=4)


@pytest.mark.parametrize(
    "field,value",
    [("module", "Other"), ("getter_output", "control"), ("clock_input", "clear"), ("register_ordinals", (0, 1))],
)
def test_changed_source_endpoints_refuse(field, value):
    source = _source()
    with pytest.raises(ValueError):
        _observe(source, _samples(((0, 0, 0, 0, 0, 0),)), selection=replace(_selection(source), **{field: value}))


def test_source_digest_and_resigned_update_changes_stay_distinct():
    source = _source()
    changed = source.replace('"comb.and"(%carry, %not_inhibit)', '"comb.or"(%carry, %not_inhibit)')
    samples = _samples(((0, 0, 0, 0, 0, 0),), initial=0)
    with pytest.raises(ValueError, match="source bytes"):
        _observe(changed, samples, selection=_selection(source))
    with pytest.raises(ValueError, match="actual source transitions"):
        _observe(changed, samples)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "reverse", "loss", "gate", "async", "preset", "unknown", "opaque", "reset_value", "state_cycle"],
)
def test_unsupported_original_getter_clock_and_state_forms_refuse(mutation):
    source = _source()
    if mutation == "duplicate":
        source = source.replace(
            '"comb.concat"(%hi, %lo) : (i3, i2) -> i5', '"comb.concat"(%hi, %lo, %lo) : (i3, i2, i2) -> i7'
        )
    elif mutation == "reverse":
        source = source.replace('"comb.concat"(%hi, %lo) : (i3, i2) -> i5', '"comb.concat"(%lo, %hi) : (i2, i3) -> i5')
    elif mutation == "loss":
        source = source.replace(
            '"comb.concat"(%hi, %lo) : (i3, i2) -> i5', '"comb.extract"(%hi) {lowBit = 0 : i32} : (i3) -> i1'
        )
    elif mutation == "gate":
        source = source.replace(
            '%tick = "seq.to_clock"(%c)', '%g = "comb.and"(%c, %stall) : (i1, i1) -> i1\n %tick = "seq.to_clock"(%g)'
        )
    elif mutation in {"async", "preset", "unknown"}:
        extra = {"async": ", isAsync", "preset": ", preset = 0 : i3", "unknown": ", unsupported = true"}[mutation]
        source = source.replace('name = "opaque_b"', 'name = "opaque_b"' + extra)
    elif mutation == "opaque":
        source = source.replace('"comb.add"(%hi, %one_high)', '"unknown.update"(%hi, %one_high)')
    elif mutation == "reset_value":
        source = source.replace(
            '%hi = "seq.firreg"(%high_write, %tick, %reset, %zero_high)',
            '%hi = "seq.firreg"(%high_write, %tick, %reset, %hi)',
        )
    else:
        source = source.replace('"comb.add"(%hi, %one_high)', '"comb.add"(%high_sum, %one_high)')
    with pytest.raises(ValueError):
        _observe(
            source,
            _samples(((0, 0, 0, 0, 0, 0),)),
            selection=replace(_selection(_source()), source_sha256=hashlib.sha256(source.encode()).hexdigest()),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("registers", 2),
        ("intervals", 1),
        ("nodes", 8),
        ("source_bytes", 32),
        ("bit_work", 512),
        ("cases", 1),
        ("scalar_bits", 4),
    ],
)
def test_complete_preallocation_limits_refuse(field, value):
    limits = (
        replace(LIMITS, **{field: value})
        if field in {"registers", "intervals"}
        else replace(LIMITS, expressions=replace(LIMITS.expressions, **{field: value}))
    )
    samples = _samples(((0, 0, 0, 0, 0, 0),))
    intervals = (CounterInterval(0, 1), CounterInterval(0, 1))
    with pytest.raises(ValueError):
        _observe(_source(), samples, limits=limits, intervals=intervals)


@pytest.mark.parametrize("value", [True, 0, -1, 1 << 63])
def test_invalid_closed_metadata_refuses(value):
    with pytest.raises(ValueError):
        StateTimelineLimits(LIMITS.expressions, value, 1)
    if value != 0 or type(value) is not int:
        with pytest.raises(ValueError):
            StateGetterSelection("0" * 64, "x", (value,), "y", "z")


def test_actual_zero_ordinal_is_legal_metadata_not_an_admission():
    assert StateGetterSelection("0" * 64, "x", (0,), "y", "z").register_ordinals == (0,)


def test_error_does_not_publish_unparsed_source_excerpt():
    source = "builtin.module { private_unparsed_payload }"
    selection = StateGetterSelection(hashlib.sha256(source.encode()).hexdigest(), "x", (0,), "y", "z")
    with pytest.raises(ValueError) as error:
        _observe(source, _samples(((0, 0, 0, 0, 0, 0),)), selection=selection)
    assert "private_unparsed_payload" not in str(error.value)
    assert str(error.value) == "State timeline source grammar is unsupported."


def test_observations_are_data_only_with_original_unknown_denominator():
    record = asdict(_observe(_source(), _samples(((0, 0, 0, 0, 0, 0),))))
    assert len(record["unknowns"]) == 9
    assert "physical_clock_units_frequency_and_loaded_image" in record["unknowns"]
    assert record["intervals"][0]["unit_increments"] is None
    assert not {"qualified", "cold", "warm", "cycles", "frequency"} & set(record)


def _metadata_source(
    *, visibility='"private"', locations='[loc(unknown), loc("origin":2:3), loc("carry")]', fragments="[]"
):
    return _source().replace(
        'sym_name = "StateCell",',
        f'sym_name = "StateCell", sym_visibility = {visibility}, '
        f"result_locs = {locations}, emit.fragments = {fragments},",
    )


@pytest.mark.parametrize("visibility", ["public", "private", "nested"])
def test_exact_typed_module_locations_and_visibility_are_retained(visibility):
    result = _observe(_metadata_source(visibility=f'"{visibility}"'), _samples(((0, 0, 0, 0, 0, 0),)))
    record = asdict(result)["module_metadata"]
    assert record["symbol_visibility"] == visibility
    assert record["emission_fragments"] == ()
    assert tuple(row["kind"] for row in record["result_locations"]) == (
        "unknown_loc",
        "file_line_loc",
        "builtin.name_loc",
    )
    assert record["result_locations"][1]["assembly"] == 'loc("origin":2:3)'
    assert len(result.output_ports) == len(record["result_locations"]) == 3
    assert result.intervals[0].unit_increments is None


def test_absent_metadata_and_explicit_empty_emission_roster_stay_distinct():
    record = _observe(_source(), _samples(((0, 0, 0, 0, 0, 0),))).module_metadata
    assert record.symbol_visibility is None and record.result_locations is None and record.emission_fragments is None


def test_shared_location_dag_is_bounded_before_expanded_printing():
    aliases = ["#where0 = loc(unknown)"]
    for index in range(1, 10):
        aliases.append(f"#where{index} = loc(callsite(#where{index - 1} at #where{index - 1}))")
    source = "\n".join(aliases) + "\n" + _metadata_source(locations="[#where9, loc(unknown), loc(unknown)]")
    with pytest.raises(ValueError, match="location metadata exceeds its node budget"):
        _observe(source, _samples(((0, 0, 0, 0, 0, 0),)))


def test_shared_fused_location_metadata_cannot_hide_expansion_in_a_dictionary():
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FusedLoc, StringAttr, UnknownLoc

    from merlin.targetgen.rtl.hw_counter_state_timelines import _module_metadata

    # The selected parser does not implement fused-location metadata syntax.
    # Exercise the typed metadata helper directly; this does not admit a source.
    parsed = parse_generic_hw(_metadata_source())
    module = next(op for op in parsed.walk() if _name(op) == "hw.module")
    payload = DictionaryAttr({"tag": StringAttr("debug")})
    for _ in range(9):
        payload = DictionaryAttr({"left": payload, "right": payload})
    module.attributes["result_locs"] = ArrayAttr([FusedLoc([UnknownLoc()], payload), UnknownLoc(), UnknownLoc()])
    with pytest.raises(ValueError, match="location metadata exceeds its node budget"):
        _module_metadata(module, 3, LIMITS.expressions)


def test_duplicate_original_metadata_ownership_refuses_even_identical_values():
    source = _metadata_source().replace('"hw.module"()', '"hw.module"() <{sym_visibility = "private"}>')
    with pytest.raises(ValueError, match="module semantics"):
        _observe(source, _samples(((0, 0, 0, 0, 0, 0),)))


@pytest.mark.parametrize("visibility", ['"unknown"', "true", "1 : i64"])
def test_unknown_or_wrong_typed_visibility_refuses(visibility):
    with pytest.raises(ValueError, match="symbol visibility"):
        _observe(_metadata_source(visibility=visibility), _samples(((0, 0, 0, 0, 0, 0),)))


@pytest.mark.parametrize("locations", ["[]", "[loc(unknown)]", "[loc(unknown), loc(unknown), true]", '"locations"'])
def test_complete_original_location_membership_and_type_are_required(locations):
    with pytest.raises(ValueError, match="result locations"):
        _observe(_metadata_source(locations=locations), _samples(((0, 0, 0, 0, 0, 0),)))


@pytest.mark.parametrize("fragments", ['"body"', "[true]", "[@body::@nested]", "[@body, @body]"])
def test_emission_dependency_metadata_is_typed_and_not_an_allowlist(fragments):
    with pytest.raises(ValueError, match="emission dependency metadata"):
        _observe(_metadata_source(fragments=fragments), _samples(((0, 0, 0, 0, 0, 0),)))


@pytest.mark.parametrize("resolved", [False, True])
def test_even_resolved_unused_emission_body_is_a_required_semantic_dependency(resolved):
    source = _metadata_source(fragments="[@body]")
    if resolved:
        source = source.replace(
            "builtin.module {",
            'builtin.module {\n "emit.fragment"() ({\n'
            ' "sv.verbatim"() {text = "opaque emission"} : () -> ()\n'
            ' }) {sym_name = "body"} : () -> ()\n',
            1,
        )
    with pytest.raises(ValueError, match="emission dependencies are unresolved"):
        _observe(source, _samples(((0, 0, 0, 0, 0, 0),)))


@pytest.mark.parametrize("direct", [False, True])
def test_original_root_clock_cast_replication_and_unsigned_overshift(direct):
    source = _source(direct_clock=direct)
    clock = "%c" if direct else "%tick"
    source = source.replace(
        '"hw.output"(%value, %blocked, %carry) : (i5, i1, i1)',
        '%repeated = "comb.replicate"(%blocked) : (i1) -> i8\n'
        '%amount = "hw.constant"() {value = 255 : i8} : () -> i8\n'
        '%shifted = "comb.shru"(%repeated, %amount) : (i8, i8) -> i8\n'
        f'%level = "seq.from_clock"({clock}) : (!seq.clock) -> i1\n'
        '"hw.output"(%value, %blocked, %carry, %repeated, %shifted, %level) : (i5, i1, i1, i8, i8, i1)',
    ).replace("output carry : i1>", "output carry : i1, output repeated : i8, output shifted : i8, output level : i1>")
    samples = _samples(((0, 0, 1, 1, 0, 0), (0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)))
    samples = tuple(replace(row, outputs=(*row.outputs, 255 * row.states[2], 0, row.inputs[0])) for row in samples)
    result = _observe(source, samples)
    assert len(result.output_ports) == 6 and len(result.states) == 3
    assert result.samples == samples and result.intervals[0].unit_increments is None
    wrong = replace(samples[1], outputs=(*samples[1].outputs[:-1], 0))
    with pytest.raises(ValueError, match="actual source transitions"):
        _observe(source, (samples[0], wrong, *samples[2:]))


@pytest.mark.parametrize("mutation", ["integer", "wrong_result", "attribute", "opaque", "other_clock"])
def test_clock_value_cast_requires_original_typed_clock_and_complete_outputs(mutation):
    source = _source().replace(
        '"hw.output"(%value', '%level = "seq.from_clock"(%tick) : (!seq.clock) -> i1\n "hw.output"(%value'
    )
    if mutation == "integer":
        source = source.replace('"seq.from_clock"(%tick) : (!seq.clock)', '"seq.from_clock"(%c) : (i1)')
    elif mutation == "wrong_result":
        source = source.replace(
            '"seq.from_clock"(%tick) : (!seq.clock) -> i1', '"seq.from_clock"(%tick) : (!seq.clock) -> i2'
        )
    elif mutation == "attribute":
        source = source.replace('"seq.from_clock"(%tick)', '"seq.from_clock"(%tick) {unsupported}')
    elif mutation == "opaque":
        source = source.replace("%level =", '%opaque = "unknown.clock"() : () -> !seq.clock\n %level =').replace(
            '"seq.from_clock"(%tick)', '"seq.from_clock"(%opaque)'
        )
    else:
        source = source.replace('"seq.from_clock"(%tick)', '"seq.from_clock"(%reset)')
    with pytest.raises(ValueError):
        _observe(
            source,
            _samples(((0, 0, 0, 0, 0, 0),)),
            selection=replace(_selection(_source()), source_sha256=hashlib.sha256(source.encode()).hexdigest()),
        )


@pytest.mark.parametrize(
    "effect",
    [
        '"sim.fatal"(%reset) : (i1) -> ()',
        '"sv.fwrite"(%lo) {formatString = "%d"} : (i2) -> ()',
        '"sv.if"(%reset) ({ "sim.fatal"() : () -> () }) : (i1) -> ()',
        '%macro = "sv.macro.ref"() {macro = @condition} : () -> i1',
    ],
)
def test_unused_original_sv_sim_and_macro_effects_cannot_be_ignored(effect):
    source = _metadata_source().replace('"hw.output"(%value', effect + '\n "hw.output"(%value')
    with pytest.raises(ValueError, match="expression semantics"):
        _observe(source, _samples(((0, 0, 0, 0, 0, 0),)))


@pytest.mark.parametrize("count", [17, 116])
def test_complete_large_two_clock_roster_retains_every_state_output_and_unused_effect(count):
    statements = ['%one = "hw.constant"() {value = 1 : i4} : () -> i4']
    for index in range(count):
        clock = "%first" if index % 2 == 0 else "%second"
        statements.extend(
            [
                f'%next{index} = "comb.add"(%r{index}, %one) : (i4, i4) -> i4',
                f'%r{index} = "seq.firreg"(%next{index}, {clock}) {{name = "state{index}"}} : (i4, !seq.clock) -> i4',
            ]
        )
    statements.append('%joined = "comb.concat"(%r0, %r2) : (i4, i4) -> i8')
    values = ", ".join(f"%r{index}" for index in range(count))
    types = ", ".join("i4" for _ in range(count))
    statements.append(f'"hw.output"(%joined, {values}) : (i8, {types}) -> ()')
    ports = ", ".join(f"output state{index} : i4" for index in range(count))
    locations = ", ".join("loc(unknown)" for _ in range(count + 1))
    source = (
        'builtin.module { "hw.module"() ({\n ^bb0(%first: !seq.clock, %second: !seq.clock):\n'
        + "\n".join(statements)
        + '\n }) {sym_name = "WholeState", parameters = [], '
        + f'sym_visibility = "private", result_locs = [{locations}], '
        + f"module_type = !hw.modty<input first : !seq.clock, input second : !seq.clock, output joined : i8, {ports}>"
        + "} : () -> () }"
    )
    parsed = parse_generic_hw(source)
    module = next(op for op in parsed.walk() if _name(op) == "hw.module")
    ordinals = {
        op.results[0].name_hint: ordinal
        for ordinal, op in enumerate(module.regions[0].block.ops)
        if _name(op) == "seq.firreg"
    }
    selection = StateGetterSelection(
        hashlib.sha256(source.encode()).hexdigest(), "WholeState", (ordinals["r0"], ordinals["r2"]), "joined", "first"
    )
    states = tuple(index % 16 for index in range(count))
    rows = []
    for ordinal in range(6):
        first = ordinal % 2
        second = int(ordinal == 3)
        if first:
            states = tuple((value + int(index % 2 == 0 or second)) % 16 for index, value in enumerate(states))
        outputs = (states[0] * 16 + states[2], *states)
        rows.append(StatePhaseSample(ordinal, (first, second), states, outputs))
    limits = StateTimelineLimits(EvaluationLimits(65536, 2048, 64, 16, 4_000_000), 128, 1)
    kwargs = {
        "selection": selection,
        "limits": limits,
        "expected_phases": 6,
        "samples": tuple(rows),
        "intervals": (CounterInterval(0, 5),),
    }
    result = observe_state_getter_timeline(source, **kwargs)
    assert len(result.states) == count and len(result.output_ports) == count + 1
    assert len(result.module_metadata.result_locations) == count + 1
    assert tuple(state.clock_input for state in result.states) == tuple(
        "first" if index % 2 == 0 else "second" for index in range(count)
    )
    incomplete = tuple(replace(row, states=row.states[:-1]) for row in rows)
    with pytest.raises(ValueError, match="complete original typed roster"):
        observe_state_getter_timeline(source, **(kwargs | {"samples": incomplete}))
    changed = source.replace('"hw.output"', '"sim.fatal"() : () -> ()\n "hw.output"')
    changed_selection = replace(selection, source_sha256=hashlib.sha256(changed.encode()).hexdigest())
    with pytest.raises(ValueError, match="expression semantics"):
        observe_state_getter_timeline(changed, **(kwargs | {"selection": changed_selection}))
