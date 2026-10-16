"""Complete source-local counter timelines do not qualify physical timing."""

import hashlib
from dataclasses import asdict, replace

import pytest

from merlin.targetgen.rtl.hw_combinational import EvaluationLimits
from merlin.targetgen.rtl.hw_counter_intervals import (
    CounterEndpointSelection,
    CounterInterval,
    CounterIntervalLimits,
    CounterPhaseSample,
    observe_counter_intervals,
)

LIMITS = CounterIntervalLimits(EvaluationLimits(32768, 128, 64, 1024, 2_000_000), 128)


def _source(width=4, *, alternate=False, reset=True):
    reset_operands = ", %r, %zero" if reset else ""
    reset_types = f", i1, i{width}" if reset else ""
    # The renamed body has a different width and output/control order. Selection
    # must follow those original declarations rather than port-name conventions.
    output_values = "%value, %masked" if alternate else "%masked, %value"
    output_types = f"i{width}, i1" if alternate else f"i1, i{width}"
    output_names = (
        f"output visible : i{width}, output blocked : i1"
        if alternate
        else f"output blocked : i1, output visible : i{width}"
    )
    return f"""builtin.module {{
      "hw.module"() ({{
      ^bb0(%c: i1, %r: i1, %a: i1, %b: i1):
        %clk = "seq.to_clock"(%c) : (i1) -> !seq.clock
        %one = "hw.constant"() {{value = 1 : i{width}}} : () -> i{width}
        %zero = "hw.constant"() {{value = 0 : i{width}}} : () -> i{width}
        %masked = "comb.or"(%a, %b) {{twoState}} : (i1, i1) -> i1
        %add = "comb.add"(%value, %one) : (i{width}, i{width}) -> i{width}
        %next = "comb.mux"(%masked, %value, %add) {{twoState}} : (i1, i{width}, i{width}) -> i{width}
        %value = "seq.firreg"(%next, %clk{reset_operands}) {{name = "opaque_storage"}}
          : (i{width}, !seq.clock{reset_types}) -> i{width}
        "hw.output"({output_values}) : ({output_types}) -> ()
      }}) {{sym_name = "SampleCell", module_type = !hw.modty<input root : i1,
        input clear : i1, input first_control : i1, input second_control : i1,
        {output_names}>, parameters = []}} : () -> ()
    }}"""


def _selection(source):
    return CounterEndpointSelection(hashlib.sha256(source.encode()).hexdigest(), "SampleCell", 6, "visible", "root")


def _samples(cycles, *, initial=14, width=4, alternate=False, reset=True):
    # Independent integer expectation, not produced by the consumer under test.
    value = initial
    result = []
    for clear, first, second in cycles:
        for phase in (0, 1):
            if phase:
                if clear and reset:
                    value = 0
                elif not (first or second):
                    value = (value + 1) % (2**width)
            blocked = int(bool(first or second))
            outputs = (value, blocked) if alternate else (blocked, value)
            result.append(CounterPhaseSample(len(result), (phase, clear, first, second), outputs))
    return tuple(result)


def _observe(source, samples, *, intervals=None, selection=None, limits=LIMITS, expected=None):
    return observe_counter_intervals(
        source,
        selection=_selection(source) if selection is None else selection,
        limits=limits,
        expected_phases=len(samples) if expected is None else expected,
        samples=samples,
        intervals=(CounterInterval(0, len(samples) - 1),) if intervals is None else intervals,
    )


@pytest.mark.parametrize("width,alternate", [(1, False), (4, False), (7, True), (64, True)])
def test_source_guarded_updates_reset_inhibit_stall_and_wrap(width, alternate):
    source = _source(width, alternate=alternate)
    samples = _samples(
        ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0)), initial=2**width - 2, width=width, alternate=alternate
    )
    result = _observe(source, samples)
    interval = result.intervals[0]
    assert interval.rising_edges == 5
    assert interval.held_edges == 2
    assert interval.unit_increments == 3
    assert interval.wraps == 1
    assert interval.observed_modular_delta == 3 % (2**width)
    assert result.samples == samples
    assert result.counter_width == width
    assert len(result.input_ports) == 4 and len(result.output_ports) == 2
    assert "physical_clock_units_frequency_and_loaded_image" in result.unknowns
    assert "complete_stage_composition_and_cold_warm_reuse" in result.unknowns
    assert not any(word in asdict(result) for word in ("cycles", "cold", "warm", "qualified"))


def test_reset_has_primitive_priority_and_reset_crossing_interval_stays_unknown():
    samples = _samples(((0, 0, 0), (1, 1, 1), (0, 0, 0), (0, 0, 0)), initial=14)
    result = _observe(_source(), samples, intervals=(CounterInterval(0, 7), CounterInterval(3, 7)))
    assert result.transitions[3].reset and not result.transitions[3].increment
    assert result.transitions[3].after == 0
    assert result.intervals[0].unit_increments is None
    assert result.intervals[0].reset_edges == 1
    assert result.intervals[1].unit_increments == 2
    assert result.intervals[1].reset_edges == 0
    assert "initial_state_reachability_and_external_events" in result.unknowns


def test_absent_reset_retains_unproved_boundary_state():
    source = _source(reset=False)
    result = _observe(source, _samples(((1, 0, 0), (1, 0, 0)), initial=8, reset=False))
    assert not result.reset_present
    assert result.intervals[0].unit_increments == 2
    assert result.transitions[0].before == 8
    assert result.intervals[0].reset_edges == 0


@pytest.mark.parametrize(
    "defect",
    [
        "increment_during_inhibit",
        "increment_during_stall",
        "ignore_reset",
        "no_wrap",
        "falling_edge_change",
        "changed_other_output",
    ],
)
def test_actual_complete_numeric_source_defects_refuse(defect):
    source = _source()
    samples = list(_samples(((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 0, 0)), initial=15))
    phase, outputs = {
        "increment_during_inhibit": (3, (1, 1)),
        "increment_during_stall": (5, (1, 1)),
        "ignore_reset": (7, (0, 1)),
        "no_wrap": (1, (0, 15)),
        "falling_edge_change": (2, (1, 1)),
        "changed_other_output": (0, (1, 15)),
    }[defect]
    samples[phase] = replace(samples[phase], outputs=outputs)
    with pytest.raises(ValueError, match="actual source transition"):
        _observe(source, tuple(samples))


@pytest.mark.parametrize(
    "defect",
    ["gap", "missing_pair", "clock", "unstable", "missing_input", "missing_output", "bool", "unsigned", "iterator"],
)
def test_complete_phase_roster_is_checked_before_interval_evaluation(defect):
    source = _source()
    samples = _samples(((0, 0, 0), (0, 0, 0)))
    expected = 4
    if defect == "gap":
        samples = (samples[0], replace(samples[1], ordinal=3), *samples[2:])
    elif defect == "missing_pair":
        samples = samples[:2]
    elif defect == "clock":
        samples = (replace(samples[0], inputs=(1, 0, 0, 0)), *samples[1:])
    elif defect == "unstable":
        samples = (samples[0], replace(samples[1], inputs=(1, 0, 1, 0)), *samples[2:])
    elif defect == "missing_input":
        samples = (replace(samples[0], inputs=(0, 0, 0)), *samples[1:])
    elif defect == "missing_output":
        samples = (replace(samples[0], outputs=(14,)), *samples[1:])
    elif defect == "bool":
        samples = (replace(samples[0], ordinal=False), *samples[1:])
    elif defect == "unsigned":
        samples = (replace(samples[0], outputs=(0, -1)), *samples[1:])
    elif defect == "iterator":
        samples = iter(samples)
    with pytest.raises(ValueError):
        _observe(source, samples, expected=expected, intervals=(CounterInterval(0, 3),))


@pytest.mark.parametrize(
    "field,value",
    [("getter_output", "blocked"), ("clock_input", "clear"), ("register_ordinal", 5), ("module", "Absent")],
)
def test_changed_getter_clock_register_or_module_endpoint_refuses(field, value):
    source = _source()
    with pytest.raises(ValueError, match="relation|membership"):
        _observe(source, _samples(((0, 0, 0),)), selection=replace(_selection(source), **{field: value}))


def test_changed_source_pin_is_not_re_signed_implicitly():
    source = _source()
    changed = source.replace('%masked = "comb.or"', '%masked = "comb.and"')
    samples = _samples(((0, 1, 0),))
    with pytest.raises(ValueError, match="source bytes"):
        _observe(changed, samples, selection=_selection(source))
    # Even a new source identity cannot accept original rows with different
    # actual control semantics.
    with pytest.raises(ValueError, match="actual source transition"):
        _observe(changed, samples)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace('name = "opaque_storage"', 'name = "opaque_storage", isAsync'),
        lambda s: s.replace('name = "opaque_storage"', 'name = "opaque_storage", preset = 0 : i4'),
        lambda s: s.replace('"comb.add"', '"comb.mul"'),
        lambda s: s.replace("value = 1 : i4", "value = 2 : i4"),
        lambda s: s.replace("%masked, %value, %add", "%masked, %zero, %add"),
        lambda s: s.replace('"seq.firreg"', '"seq.compreg"'),
        lambda s: s.replace("%c) : (i1) -> !seq.clock", "%masked) : (i1) -> !seq.clock"),
        lambda s: s.replace("parameters = []", 'parameters = [#hw.param.decl<"P" = 4 : i32> : i32]'),
        lambda s: s.replace("parameters = []", "parameters = [], unknown = true"),
        lambda s: s.replace(
            '"hw.output"(%masked, %value)', '%opaque = "unknown.op"() : () -> i4\n "hw.output"(%masked, %value)'
        ),
    ],
)
def test_unsupported_or_non_unit_state_body_refuses(change):
    source = change(_source())
    with pytest.raises(ValueError):
        _observe(source, _samples(((0, 0, 0),)))


@pytest.mark.parametrize(
    "interval",
    [
        CounterInterval(-1, 1),
        CounterInterval(0, 4),
        CounterInterval(2, 1),
        CounterInterval(1, 1),
        CounterInterval(False, 1),
    ],
)
def test_changed_or_unavailable_interval_endpoints_refuse(interval):
    with pytest.raises(ValueError, match="endpoints"):
        _observe(_source(), _samples(((0, 0, 0), (0, 0, 0))), intervals=(interval,))


@pytest.mark.parametrize(
    "field,value", [("source_bytes", 64), ("nodes", 8), ("scalar_bits", 3), ("cases", 2), ("bit_work", 20)]
)
def test_complete_preallocation_budget_refuses(field, value):
    limits = replace(LIMITS, expressions=replace(LIMITS.expressions, **{field: value}))
    with pytest.raises(ValueError):
        _observe(_source(), _samples(((0, 0, 0), (0, 0, 0))), limits=limits)


def test_limits_and_selection_metadata_are_bounded_and_exactly_typed():
    for value in (True, 0, -1, 2**63):
        with pytest.raises(ValueError):
            replace(LIMITS, intervals=value)
    with pytest.raises(ValueError):
        replace(LIMITS, expressions=replace(LIMITS.expressions, scalar_bits=10**9))
    with pytest.raises(ValueError):
        replace(_selection(_source()), register_ordinal=True)
    with pytest.raises(ValueError):
        replace(_selection(_source()), source_sha256="unknown")


def test_interval_budget_precedes_source_parse_and_results():
    limits = replace(LIMITS, intervals=1)
    with pytest.raises(ValueError, match="roster"):
        _observe("not parsed", (), limits=limits, expected=2, intervals=(CounterInterval(0, 1), CounterInterval(0, 1)))


def test_reset_active_without_rising_edge_does_not_change_synchronous_state():
    samples = _samples(((0, 0, 0), (1, 0, 0)), initial=8)
    assert samples[2].outputs == (0, 9)
    result = _observe(_source(), samples)
    assert not result.transitions[2].reset
    assert result.transitions[3].reset
    broken = (*samples[:2], replace(samples[2], outputs=(0, 0)), samples[3])
    with pytest.raises(ValueError, match="actual source transition"):
        _observe(_source(), broken)


def test_nonconstant_or_state_dependent_reset_is_unsupported():
    source = _source().replace("%next, %clk, %r, %zero", "%next, %clk, %r, %value")
    with pytest.raises(ValueError, match="reset value"):
        _observe(source, _samples(((1, 0, 0),)))
    source = _source(1).replace("%next, %clk, %r, %zero", "%next, %clk, %value, %zero")
    with pytest.raises(ValueError, match="reset dependencies"):
        _observe(source, _samples(((1, 0, 0),), initial=0, width=1))


def test_input_as_next_cannot_substitute_a_source_counter():
    source = _source(1).replace("%masked, %value, %add", "%masked, %a, %add")
    with pytest.raises(ValueError, match="unit increment"):
        _observe(source, _samples(((0, 0, 0),), initial=0, width=1))


def test_overlapping_intervals_do_not_infer_independent_cost_ownership():
    samples = _samples(tuple((0, 0, 0) for _ in range(35)), initial=14)
    result = _observe(
        _source(), samples, intervals=(CounterInterval(0, 69), CounterInterval(1, 33), CounterInterval(20, 23))
    )
    assert [row.unit_increments for row in result.intervals] == [35, 16, 2]
    assert result.intervals[0].wraps == 3
    assert result.intervals[0].observed_modular_delta == 3
    assert "complete_stage_composition_and_cold_warm_reuse" in result.unknowns


@pytest.mark.parametrize("count,accepted", [(48, True), (65, False)])
def test_shared_update_graph_has_bounded_dependency_depth(count, accepted):
    # Each new mux shares both branches, so memoization must retain a DAG rather
    # than expanding the same subtree exponentially.
    statements, previous = [], "%next"
    for index in range(count):
        result = f"%nested{index}"
        statements.append(f'{result} = "comb.mux"(%a, {previous}, {previous}) {{twoState}} : (i1, i4, i4) -> i4')
        previous = result
    source = _source().replace(
        '%value = "seq.firreg"(%next', "\n".join(statements) + f'\n %value = "seq.firreg"({previous}'
    )
    selection = replace(_selection(source), register_ordinal=6 + count)
    samples = _samples(((0, 0, 0), (0, 1, 0)))
    if accepted:
        result = _observe(source, samples, selection=selection)
        assert result.intervals[0].unit_increments == 1
        assert result.intervals[0].held_edges == 1
    else:
        with pytest.raises(ValueError, match="depth"):
            _observe(source, samples, selection=selection)


def test_changed_actual_getter_body_cannot_echo_a_selected_endpoint():
    source = _source().replace('"hw.output"(%masked, %value)', '"hw.output"(%masked, %add)')
    with pytest.raises(ValueError, match="getter relation"):
        _observe(source, _samples(((0, 0, 0),)))


def test_unknown_late_clock_phase_and_partial_interval_rosters_refuse():
    samples = _samples(((0, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError, match="phase or interval"):
        _observe(_source(), samples[:3], expected=3)
    with pytest.raises(ValueError, match="phase or interval"):
        _observe(_source(), samples, intervals=())
    with pytest.raises(ValueError, match="phase or interval"):
        _observe(_source(), samples, intervals=iter((CounterInterval(0, 3),)))


def test_malformed_source_error_does_not_publish_source_text():
    marker = "excluded_source_annotation"
    source = f'builtin.module {{ "unknown.{marker}"('
    with pytest.raises(ValueError) as failure:
        _observe(source, _samples(((0, 0, 0),)))
    assert str(failure.value) == "Counter source grammar is unsupported."
    assert marker not in str(failure.value)


def test_malformed_unsigned_literal_has_a_bounded_source_refusal():
    source = _source().replace("value = 1 : i4", "value = -1 : ui64")
    with pytest.raises(ValueError, match="Counter source grammar is unsupported") as error:
        _observe(source, _samples(((0, 0, 0),)))
    assert "ui64" not in str(error.value)
