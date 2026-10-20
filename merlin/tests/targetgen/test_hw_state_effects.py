"""Explicit macro premises retain complete source-local conditional effects."""

import hashlib
import json
from dataclasses import replace

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

LIMITS = StateTimelineLimits(EvaluationLimits(65536, 512, 128, 32, 100000), 8, 8)
MACROS = ("SYNTHESIS", "PRINT_GATE", "PRINT_GATE_", "STOP_GATE", "STOP_GATE_")


def _fragment(symbol, macro):
    return f'''"emit.fragment"() <{{sym_name = "{symbol}"}}> ({{
      "sv.verbatim"() {{format_string = "// source-local comment", symbols = []}} : () -> ()
      "sv.ifdef"() ({{^bb0:}}, {{
        "sv.ifdef"() ({{
          "sv.macro.def"() {{macroName = @{macro}_, format_string = "(`{macro})", symbols = []}} : () -> ()
        }}, {{
          "sv.macro.def"() {{macroName = @{macro}_, format_string = "1", symbols = []}} : () -> ()
        }}) {{cond = #sv<macro.ident @{macro}>}} : () -> ()
      }}) {{cond = #sv<macro.ident @{macro}_>}} : () -> ()
    }}) : () -> ()'''


def _source():
    declarations = "\n".join(f'"sv.macro.decl"() {{sym_name = "{symbol}"}} : () -> ()' for symbol in MACROS)
    return f"""builtin.module {{
      {declarations}
      {_fragment("PrintFragment", "PRINT_GATE")}
      {_fragment("StopFragment", "STOP_GATE")}
      "hw.module"() ({{
      ^bb0(%pulse:i1, %reset:i1, %fail_a:i1, %fail_b:i1):
        %clock = "seq.to_clock"(%pulse) : (i1) -> !seq.clock
        %clockbit = "seq.from_clock"(%clock) : (!seq.clock) -> i1
        %zero = "hw.constant"() {{value = 0 : i4}} : () -> i4
        %one = "hw.constant"() {{value = 1 : i4}} : () -> i4
        %fd = "hw.constant"() {{value = 37 : i32}} : () -> i32
        %sum = "comb.add"(%state,%one) : (i4,i4) -> i4
        %state = "seq.firreg"(%sum,%clock,%reset,%zero) {{name = "state"}} : (i4,!seq.clock,i1,i4) -> i4
        %at_one = "comb.icmp"(%state,%one) {{predicate = 0 : i64}} : (i4,i4) -> i1
        %fault_a = "comb.and"(%fail_a,%at_one) : (i1,i1) -> i1
        %fault_b = "comb.and"(%fail_b,%at_one) : (i1,i1) -> i1
        %stop = "sv.macro.ref"() {{macroName = @STOP_GATE_}} : () -> i1
        %stop_a = "comb.and"(%stop,%fault_a) : (i1,i1) -> i1
        "sim.fatal"(%clock,%stop_a) : (!seq.clock,i1) -> ()
        "sv.ifdef"() ({{^bb0:}}, {{
          "sv.always"(%clockbit) ({{
            %print = "sv.macro.ref"() {{macroName = @PRINT_GATE_}} : () -> i1
            %print_a = "comb.and"(%print,%fault_a) : (i1,i1) -> i1
            "sv.if"(%print_a) ({{"sv.fwrite"(%fd) {{format_string = "first"}} : (i32) -> ()}}, {{}}) : (i1) -> ()
            %print_b = "comb.and"(%print,%fault_b) : (i1,i1) -> i1
            "sv.if"(%print_b) ({{"sv.fwrite"(%fd) {{format_string = "second"}} : (i32) -> ()}}, {{}}) : (i1) -> ()
          }}) {{events = [0 : i32]}} : (i1) -> ()
        }}) {{cond = #sv<macro.ident @SYNTHESIS>}} : () -> ()
        %stop_b = "comb.and"(%stop,%fault_b) : (i1,i1) -> i1
        "sim.fatal"(%clock,%stop_b) : (!seq.clock,i1) -> ()
        "hw.output"(%state) : (i4) -> ()
      }}) {{sym_name = "Cell", parameters = [], emit.fragments = [@PrintFragment,@StopFragment],
            module_type = !hw.modty<input pulse:i1,input reset:i1,input fault_a:i1,
              input fault_b:i1,output value:i4>}} : () -> ()
    }}"""


def _selection(source):
    module = next(op for op in parse_generic_hw(source).walk() if _name(op) == "hw.module")
    ordinal = next(index for index, op in enumerate(module.regions[0].block.ops) if _name(op) == "seq.firreg")
    return StateGetterSelection(hashlib.sha256(source.encode()).hexdigest(), "Cell", (ordinal,), "value", "pulse")


def _environment(source, settings=None):
    settings = settings or {}
    return json.dumps(
        {
            "schema": "merlin.source_macro_environment.v1",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "macros": [{"symbol": name, "defined": name in settings, "value": settings.get(name)} for name in MACROS],
        }
    )


def _samples(cycles):
    result, state = [], 0
    for reset, a, b in cycles:
        for phase in (0, 1):
            if phase:
                state = 0 if reset else (state + 1) % 16
            result.append(StatePhaseSample(len(result), (phase, reset, a, b), (state,), (state,)))
    return tuple(result)


def _observe(source, samples, *, environment=None, limits=LIMITS):
    return observe_state_getter_timeline(
        source,
        selection=_selection(source),
        limits=limits,
        samples=samples,
        intervals=(CounterInterval(0, len(samples) - 1),),
        expected_phases=len(samples),
        macro_environment=environment,
    )


def test_default_nonempty_emission_refusal_is_unchanged():
    with pytest.raises(ValueError, match="emission dependencies are unresolved"):
        _observe(_source(), _samples(((0, 0, 0),)))


def test_full_source_effect_roster_preedge_predicates_and_last_phase_termination():
    source = _source()
    result = _observe(source, _samples(((0, 0, 0), (0, 1, 1))), environment=_environment(source))
    emission = result.source_emission
    assert emission.original_fragment_symbols == ("PrintFragment", "StopFragment")
    assert len(emission.original_macro_premises) == 5 and len(emission.original_definition_ordinals) == 7
    assert len(emission.fragment_operations) == 12
    assert len(emission.original_effects) == 4 and len(emission.phases) == 16
    assert all(row.status == "inactive" for row in emission.phases[:12])
    # The source predicate observes pre-edge state1, while the completed sample
    # records state2. A post-edge predicate evaluation would miss these effects.
    assert [row.status for row in emission.phases[-4:]] == ["triggered"] * 4
    assert [row.file_descriptor for row in emission.phases[-4:]] == [None, 37, 37, None]
    assert [row.format_string for row in emission.original_effects] == [None, "first", "second", None]
    assert all(row.clock_input == 0 for row in emission.original_effects)
    assert emission.compiler_runtime_environment_correspondence == "UNKNOWN"
    assert emission.source_effect_scheduling_correspondence == "UNKNOWN"
    assert result.intervals[0].unit_increments is None


def test_triggered_termination_cannot_be_followed_by_complete_later_phases():
    source = _source()
    with pytest.raises(ValueError, match="continues after a triggered termination"):
        _observe(source, _samples(((0, 0, 0), (0, 1, 0), (0, 0, 0))), environment=_environment(source))


@pytest.mark.parametrize("settings", [{"SYNTHESIS": 0}, {"SYNTHESIS": 1}])
def test_explicit_macro_definedness_controls_all_original_effects_even_value_zero(settings):
    source = _source()
    result = _observe(source, _samples(((0, 0, 0), (0, 1, 1), (0, 0, 0))), environment=_environment(source, settings))
    assert len(result.source_emission.original_effects) == 4
    assert all(row.status == "inactive" for row in result.source_emission.phases)
    assert all(not row.compile_enabled for row in result.source_emission.original_effects)


def test_external_alias_and_direct_override_values_remain_explicit():
    source = _source()
    env = _environment(source, {"PRINT_GATE": 0, "STOP_GATE_": 0})
    result = _observe(source, _samples(((0, 0, 0), (0, 1, 1), (0, 0, 0))), environment=env)
    assert all(row.status == "inactive" for row in result.source_emission.phases)
    assert {row.symbol: row.value for row in result.source_emission.resolved_macros}["PRINT_GATE_"] == 0
    assert result.source_emission.environment_sha256 == hashlib.sha256(env.encode()).hexdigest()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "reordered",
        "duplicate",
        "unknown_field",
        "bool_value",
        "undefined_value",
        "stale_source",
        "duplicate_key",
    ],
)
def test_macro_premise_complete_identity_and_closed_schema(change):
    source = _source()
    data = json.loads(_environment(source))
    if change == "missing":
        data["macros"].pop()
    elif change == "reordered":
        data["macros"].reverse()
    elif change == "duplicate":
        data["macros"][1] = data["macros"][0]
    elif change == "unknown_field":
        data["authority"] = True
    elif change == "bool_value":
        data["macros"][0].update(defined=True, value=True)
    elif change == "undefined_value":
        data["macros"][0]["value"] = 0
    elif change == "stale_source":
        data["source_sha256"] = "0" * 64
    env = json.dumps(data)
    if change == "duplicate_key":
        env = '{"schema":"ignored",' + env[1:]
    with pytest.raises(ValueError, match="macro premise"):
        _observe(source, _samples(((0, 0, 0),)), environment=env)


@pytest.mark.parametrize(
    "old,new",
    [
        ('format_string = "// source-local comment"', 'format_string = "assign hidden = 1;"'),
        ('format_string = "1"', 'format_string = "$random"'),
        ("macroName = @STOP_GATE_}", "macroName = @Absent}"),
        ("emit.fragments = [@PrintFragment,@StopFragment]", "emit.fragments = [@PrintFragment,@Absent]"),
        ('sym_name = "PrintFragment"', 'sym_name = "StopFragment"'),
        ("events = [0 : i32]", "events = [1 : i32]"),
        ('"sv.always"(%clockbit)', '"sv.always"(%fail_a)'),
        ('format_string = "first"', 'format_string = "%d"'),
        ('"sv.fwrite"(%fd)', '"sv.finish"(%fd)'),
    ],
)
def test_unsupported_or_changed_dependencies_effects_and_clocks_refuse_even_inactive(old, new):
    source = _source().replace(old, new)
    with pytest.raises(ValueError):
        _observe(source, _samples(((0, 0, 0),)), environment=_environment(source, {"SYNTHESIS": 1}))


def test_changed_typed_source_getter_output_cannot_reuse_correct_samples():
    source = _source().replace('"hw.output"(%state)', '"hw.output"(%sum)')
    with pytest.raises(ValueError, match="complete original register bits"):
        _observe(source, _samples(((0, 0, 0),)), environment=_environment(source))


def test_changed_count_or_output_roster_and_effect_budget_refuse():
    source = _source()
    samples = _samples(((0, 0, 0), (0, 0, 0)))
    for damaged in (samples[:-1], (*samples[:-1], replace(samples[-1], outputs=(3,)))):
        with pytest.raises(ValueError):
            _observe(source, damaged, environment=_environment(source))
    limits = replace(LIMITS, expressions=replace(LIMITS.expressions, nodes=15))
    with pytest.raises(ValueError, match="node budget"):
        _observe(source, samples, environment=_environment(source), limits=limits)


def test_external_zero_stop_gate_does_not_hide_actual_output_effects():
    source = _source()
    result = _observe(
        source, _samples(((0, 0, 0), (0, 1, 1), (0, 0, 0))), environment=_environment(source, {"STOP_GATE_": 0})
    )
    effects = result.source_emission.phases[12:16]
    assert [row.status for row in effects] == ["inactive", "triggered", "triggered", "inactive"]
    assert result.samples[-1].outputs == (3,)


def test_unreferenced_fragment_is_retained_and_an_opaque_body_still_refuses():
    source = _source().replace("emit.fragments = [@PrintFragment,@StopFragment]", "emit.fragments = [@StopFragment]")
    env = _environment(source, {"PRINT_GATE_": 0})
    result = _observe(source, _samples(((0, 0, 0),)), environment=env)
    assert result.source_emission.original_fragment_symbols == ("StopFragment",)
    assert len(result.source_emission.fragment_operations) == 12
    damaged = source.replace('format_string = "// source-local comment"', 'format_string = "$finish;"', 1)
    with pytest.raises(ValueError, match="opaque text semantics"):
        _observe(damaged, _samples(((0, 0, 0),)), environment=_environment(damaged, {"PRINT_GATE_": 0}))


def test_nested_sibling_ssa_cannot_substitute_a_condition_or_source_value():
    source = _source().replace('%stop_b = "comb.and"(%stop,%fault_b)', '%stop_b = "comb.and"(%print,%fault_b)')
    selection = replace(_selection(_source()), source_sha256=hashlib.sha256(source.encode()).hexdigest())
    samples = _samples(((0, 0, 0),))
    with pytest.raises(ValueError, match="source grammar"):
        observe_state_getter_timeline(
            source,
            selection=selection,
            limits=LIMITS,
            samples=samples,
            intervals=(CounterInterval(0, 1),),
            expected_phases=2,
            macro_environment=_environment(source),
        )


def test_typed_effect_else_branch_and_original_whole_input_values():
    source = _source().replace(
        'format_string = "first"} : (i32) -> ()}, {})',
        'format_string = "first"} : (i32) -> ()}, {"sv.fwrite"(%fd) {format_string = "otherwise"} : (i32) -> ()})',
    )
    result = _observe(source, _samples(((0, 0, 0),)), environment=_environment(source))
    assert len(result.source_emission.original_effects) == 5
    assert [(row.operation, row.status) for row in result.source_emission.phases[5:]] == [
        ("sim.fatal", "inactive"),
        ("sv.fwrite", "inactive"),
        ("sv.fwrite", "triggered"),
        ("sv.fwrite", "inactive"),
        ("sim.fatal", "inactive"),
    ]
    assert result.source_emission.original_effects[2].branch_predicate_endpoints[0][1] is False
