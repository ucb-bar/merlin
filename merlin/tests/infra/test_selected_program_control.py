"""Actual control flow, physical aliases and launch state under selected bounds."""

import json
from dataclasses import replace

import pytest
from test_private_copy_elimination import command, fixture
from test_selected_program_contracts import (
    QUALIFICATION,
    compile_case,
    finalize,
    inputs,
    native,  # noqa: F401
)

from merlin.compile.scheduling.program_limits import BranchRule, Continuation, finalize_program
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.sched.ir import Kernel


@pytest.mark.parametrize("direction,word_format", [("forward", "<BBBQQQ"), ("backward", ">BBBIII")])
def test_actual_relative_branches_stay_closed_and_execute_complete_outputs(tmp_path, native, direction, word_format):  # noqa: F811
    count = 3 if direction == "forward" else 2
    model = inputs(tmp_path, branch=True, count=count, word_format=word_format)
    calls = []
    output = bytearray([91] * (2 * model[7] + 8))
    expected = bytearray(output)
    for region in range(3):
        counter, destination = 2 * region, 2 * region + 1
        output[counter] = 1 if direction == "forward" else 3
        expected[counter] = 0
        expected[destination] = 92 if direction == "forward" else 95
        increment = command("increment", destination, destination, 1, 3, 3)
        branch = command("branch", 1 if direction == "forward" else -2, counter, 0)
        calls += ([branch, command("copy", 0, destination, 1, 2, 3), increment]
                  if direction == "forward" else [increment, branch])
    result = finalize(model, calls)
    assert len(result.segments) == 3 and result.report["instruction_count"] == count * 3
    assert result.report["branch_count"] == 3
    data = bytearray(range(len(output)))
    observed = []
    for segments in ((b"".join(result.segments),), result.segments):
        current = bytearray(output)
        counts, _mode = native(segments, bytearray(data), current,
                               width=8 if word_format.endswith("Q") else 4, big=word_format[0] == ">")
        assert current == expected
        observed.append(counts)
    assert observed[0] == observed[1]
    assert observed[1][0] == (6 if direction == "forward" else 24)


def test_alias_projection_enables_legal_cuts_and_false_alias_claim_fails_execution(tmp_path, native):  # noqa: F811
    model = inputs(tmp_path, count=2)
    calls = [command("copy", 0, 16, 1, 2, 0), command("increment", 2, 2, 1, 3, 3),
             command("copy", 1, 16, 1, 2, 0), command("copy", 16, 1, 1, 1, 3)]
    result = finalize(model, calls)
    assert len(result.segments) == 2
    selected, _kernel, _objects, _adapter, codec, _controls, _word, _length = model
    policy, decoder, continuation, _binding = selected.program.bind(
        selected, None, codec.program_decoder, None, QUALIFICATION)
    with pytest.raises(ValueError, match="no completion-safe"):
        finalize_program(b"".join(result.segments), policy, decoder, continuation=continuation,
                         outputs=selected.program.published())
    data = bytearray(range(2 * model[7] + 8))
    data[:2] = bytes([7, 9])
    expected = bytearray([91] * len(data))
    expected[1], expected[2] = 9, 92
    for false_claim in (False, True):
        output = bytearray([91] * len(data))
        counts, _mode = native(result.segments, bytearray(data), output, separate=false_claim)
        assert counts == (4, 4, 4)
        assert (output == expected) is (not false_claim)


def test_live_alias_and_partial_overwrite_cannot_cross_a_volatile_launch(tmp_path):
    model = inputs(tmp_path, count=2)
    for calls in (
        [command("copy", 0, 16, 2, 2, 0), command("increment", 2, 2, 1, 3, 3),
         command("copy", 16, 0, 2, 1, 3)],
        [command("copy", 0, 16, 2, 2, 0), command("increment", 2, 2, 1, 3, 3),
         command("copy", 1, 16, 1, 2, 0), command("copy", 16, 0, 2, 1, 3)],
    ):
        with pytest.raises(ValueError, match="no completion-safe"):
            finalize(model, calls)


def test_selected_asynchronous_copy_requires_actual_global_drain_before_launch(tmp_path, native):  # noqa: F811
    model = inputs(tmp_path, count=2)
    calls = []
    for offset in range(3):
        calls += [command("async_copy", offset, offset, 1, 2, 3), command("wait", 0, 0, 1)]
    result = finalize(model, calls)
    assert list(map(len, result.segments)) == [2 * model[6].size] * 3
    data = bytearray(range(2 * model[7] + 8))
    output = bytearray([91] * len(data))
    assert native(result.segments, bytearray(data), output)[0] == (6, 3, 3)
    assert output == data[:3] + bytes([91] * (len(data) - 3))
    with pytest.raises(ValueError, match="completion"):
        finalize(model, calls[:-1])
    narrower = replace(model[5], max_bytes=model[6].size, max_instructions=1)
    image = b"".join(result.segments)
    with pytest.raises(ValueError, match="completion-safe"):
        finalize_program(image, narrower, model[4].program_decoder, selected_contracts=model[0],
                         continuation_qualification_identity=QUALIFICATION)


@pytest.mark.parametrize("repeat", [1, 2])
def test_a_branch_that_can_skip_a_drain_does_not_establish_completion(tmp_path, repeat):
    model = inputs(tmp_path, branch=True, count=4)
    region = [command("async_copy", 1, 1, 1, 2, 3), command("branch", 1, 0, 0),
              command("wait", 0, 0, 1), command("increment", 2, 2, 1, 3, 3)]
    with pytest.raises(ValueError, match="control-flow path ending before explicit device completion"):
        finalize(model, region * repeat)


def test_complete_branch_paths_can_drain_and_segment_without_expanding_iterations(tmp_path, native):  # noqa: F811
    model = inputs(tmp_path, branch=True, count=4)
    region = [command("async_copy", 1, 1, 1, 2, 3), command("branch", 0, 2, 0),
              command("wait", 0, 0, 1), command("increment", 0, 0, 1, 3, 3)]
    result = finalize(model, region * 2)
    assert len(result.segments) == 2
    observed = []
    for segments in ((b"".join(result.segments),), result.segments):
        data = bytearray(range(2 * model[7] + 8))
        output = bytearray([91] * len(data))
        output[2] = 1
        counts, _mode = native(segments, data, output)
        assert output[:3] == bytes([93, 1, 0])
        observed.append(counts)
    assert observed[0] == observed[1]


def test_configuration_persistence_is_checked_from_actual_words_and_runtime(tmp_path, native):  # noqa: F811
    def preserve(facts, software, program):
        program["continuation"]["preserved_configuration"] = {"mode": [{"fact": ["facts", "state_index"]}]}
    model = inputs(tmp_path, count=2, configuration=True, mutate=preserve)
    calls = [command("set_mode", 0, 0, 7), command("increment", 0, 0, 1, 3, 3),
             command("increment", 1, 1, 1, 3, 3)]
    result = finalize(model, calls)
    assert len(result.segments) == 2
    data = bytearray(range(2 * model[7] + 8))
    outputs = []
    for preserve_state in (True, False):
        output = bytearray([91] * len(data))
        _counts, state = native(result.segments, bytearray(data), output, preserve_configuration=preserve_state)
        assert output == bytes([92, 92]) + bytes([91] * (len(data) - 2))
        outputs.append(state)
    assert outputs == [7, 0]  # Same declared continuation; false runtime qualification loses complete state.
    directory = tmp_path / "missing"
    directory.mkdir()
    with pytest.raises(ValueError, match="configuration-field persistence"):
        finalize(inputs(directory, count=2, configuration=True), calls)


def test_unknown_execution_state_needs_a_separate_continuation_contract(tmp_path):
    def mutate(facts, software, program):
        software["isa"]["resources"].append(dict(name="engine", kind="execution", unit="slots"))
        software["isa"]["instructions"][1]["effects"]["reads"].append("engine")
    model = inputs(tmp_path, count=2, mutate=mutate)
    with pytest.raises(ValueError, match="nonmemory execution resources"):
        finalize(model, [command("increment", 0, 0, 1, 3, 3)] * 3)


@pytest.mark.parametrize("fault", ["qualification", "completion", "entry", "launch_budget"])
def test_missing_launch_capability_preserves_a_fitting_route_or_diagnoses_growth(tmp_path, fault):
    def mutate(facts, software, program):
        if fault in ("completion", "entry"):
            program["continuation"]["sequential_completion" if fault == "completion" else "starts_at_zero"] = False
        elif fault == "launch_budget":
            program["continuation"]["max_segments"] = 1
    model = inputs(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="budget|continuation"):
        compile_case(model, qualification=None if fault == "qualification" else QUALIFICATION)
    directory = tmp_path / "fits"
    directory.mkdir()
    result = compile_case(inputs(directory, count=64, mutate=mutate),
                          qualification=None if fault == "qualification" else QUALIFICATION)
    assert len(result.program.segments) == 1


def test_budget_controls_cannot_expand_hardware_branch_or_address_rules(tmp_path):
    model = inputs(tmp_path, branch=True)
    program = json.loads(model[0].program.declarations_json)
    controls = model[5]
    result = compile_case(model, limits=controls)
    assert max(result.program.report["segment_bytes"]) <= program["limits"]["max_bytes"]
    assert result.program.report["limits"]["max_instructions"] == 6
    bad = replace(controls, branches=(BranchRule("relative", -128, 127, 1, "instruction"),))
    with pytest.raises(ValueError, match="branch control"):
        compile_case(model, limits=bad)
    bad = replace(controls, windows=(replace(controls.windows[0], start=0, size=192), *controls.windows[1:]))
    with pytest.raises(ValueError, match="address control"):
        compile_case(model, limits=bad)


def test_caller_cannot_omit_published_windows_or_expand_persistence(tmp_path):
    model = inputs(tmp_path, count=2)
    continuation = Continuation(QUALIFICATION, ("input", "output", "bytes", "typed_view"), True, True, 16)
    with pytest.raises(ValueError, match="expand selected"):
        finalize(model, [command("increment", 0, 0, 1, 3, 3)] * 3, continuation=continuation)
    def observe(facts, software, program):
        for name in ("bytes", "typed_view"):
            software["scheduling"]["storage"]["banks"][name]["ownership"] = "unknown"
            program["windows"][name]["visibility"] = "unknown"
    directory = tmp_path / "observable"
    directory.mkdir()
    opaque = inputs(directory, count=2, mutate=observe)
    with pytest.raises(ValueError, match="no completion-safe"):
        finalize(opaque, [command("copy", 0, 16, 1, 2, 0)] +
                 [command("increment", 0, 0, 1, 3, 3)] * 3, outputs=())


@pytest.mark.parametrize("mutation", ["missing_branch", "wrong_address", "segment_effect", "changed_inputs"])
def test_provider_projection_cannot_substitute_for_actual_final_words(tmp_path, mutation):
    model = inputs(tmp_path, count=2, branch=True)
    selected, _kernel, _objects, _adapter, codec, _limits, word, _length = model
    calls = [command("branch", 0, 0, 0), command("increment", 1, 1, 1, 3, 3)] * 2
    image = codec.encode(Kernel("final", (), tuple(calls)))
    original = codec.program_decoder.decode
    def decode(data):
        values = original(data)
        if mutation == "missing_branch":
            return tuple(replace(v, branches=()) for v in values)
        if mutation == "wrong_address" or mutation == "segment_effect" and len(data) < len(image):
            return (replace(values[0], accesses=()), *values[1:])
        if mutation == "changed_inputs":
            selected.software_path.write_bytes(selected.software_bytes + b"\n")
        return values
    decoder = replace(codec.program_decoder, decode=decode)
    with pytest.raises(ValueError, match="actual selected ISA|changed; reselect"):
        finalize_program(image, None, decoder, selected_contracts=selected,
                         continuation_qualification_identity=QUALIFICATION)


def test_actual_final_count_is_after_ordinary_copy_rewriting(tmp_path, native):  # noqa: F811
    model = inputs(tmp_path, count=3)
    kernel, objects, _banks, _ownership = fixture(width=7)
    model = (model[0], kernel, objects, *model[3:])
    with pytest.raises(ValueError, match="no completion-safe"):
        compile_case(model)
    result = compile_case(model, qualification=None, budgets=ScheduleBudgets(eliminate_copies=True))
    assert result.program.report["instruction_count"] == 3
    assert result.program.report["encoded_bytes"] == 3 * model[6].size
    assert len(result.program.segments) == 1
    data = bytearray(range(2 * model[7] + 8))
    output = bytearray([91] * len(data))
    assert native(result.program.segments, data, output)[0] == (3, 21, 21)
    assert output == bytes(range(1, 8)) + bytes([91] * (len(data) - 7))


@pytest.mark.parametrize("field", ["max_bytes", "max_instructions", "instruction_alignment"])
def test_unknown_hardware_limits_are_explicit_and_budgets_are_not_hardware_evidence(tmp_path, field):
    def mutate(facts, software, program):
        program["limits"][field] = None
    model = inputs(tmp_path, mutate=mutate)
    if field == "instruction_alignment":
        with pytest.raises(ValueError, match="alignment is unknown"):
            compile_case(model, limits=model[5])
        return
    if field == "max_bytes":
        with pytest.raises(ValueError, match="byte capacity is unknown"):
            compile_case(model)
        result = compile_case(model, limits=model[5])
        assert result.program.report["selected_contracts"]["program"]["hardware_byte_capacity"] == "unknown"
    else:
        result = compile_case(model)
        assert result.program.report["instruction_budget"] == "unknown"
        assert result.program.report["instruction_count"] == 18


@pytest.mark.parametrize("fault", ["size", "origin", "alignment", "alias", "private_bound", "visibility", "effect",
                                  "control", "persistence", "state_index"])
def test_invalid_selected_program_contracts_reject_before_decoding(tmp_path, fault):
    def mutate(facts, software, program):
        if fault == "size":
            program["limits"]["max_bytes"] = 4096
        elif fault == "origin":
            program["branches"]["relative"]["origin"] = "guessed"
        elif fault == "alignment":
            facts["instruction_alignment"] = 0
        elif fault == "alias":
            program["windows"]["typed_view"]["backing"] = "guessed"
        elif fault == "private_bound":
            program["windows"]["bytes"]["start"] = program["windows"]["output"]["start"]
            program["windows"]["bytes"]["size"] = {"fact": ["facts", "capacities", 0]}
        elif fault == "visibility":
            program["windows"]["bytes"]["visibility"] = "external"
        elif fault == "effect":
            del software["scheduling"]["storage"]["instructions"]["increment"]
        elif fault == "control":
            program["instructions"]["branch"]["control_flow"] = "fallthrough"
        elif fault == "persistence":
            program["continuation"]["preserved_backings"].append("guessed")
        else:
            facts["state_index"] = 2
            program["continuation"]["preserved_configuration"] = {"mode": [{"fact": ["facts", "state_index"]}]}
    with pytest.raises(ValueError):
        inputs(tmp_path, branch=True, configuration=True, mutate=mutate)


def test_runtime_alias_preservation_cannot_be_contradictory(tmp_path):
    def mutate(facts, software, program):
        program["continuation"]["preserved_backings"].append("first")
    model = inputs(tmp_path, mutate=mutate)
    broken = Continuation(QUALIFICATION, ("input", "output", "bytes"), True, True, 16)
    with pytest.raises(ValueError, match="all views"):
        finalize(model, [command("increment", 0, 0, 1, 3, 3)], continuation=broken)


def test_branch_reach_targets_and_unknown_origin_come_from_actual_selected_words(tmp_path):
    model = inputs(tmp_path, branch=True)
    with pytest.raises(ValueError, match="instruction boundary"):
        finalize(model, [command("branch", 1, 0, 0)])
    selected, _kernel, _objects, _adapter, codec, _limits, _word, _length = model
    limits, _decoder, _continuation, _binding = selected.program.bind(
        selected, None, codec.program_decoder, None, QUALIFICATION)
    limits = replace(limits, branches=(replace(limits.branches[0], minimum=0, maximum=0),))
    image = codec.encode(Kernel("reach", (), tuple([command("branch", 1, 0, 0)] +
                                                  [command("increment", 0, 0, 1, 3, 3)] * 2)))
    with pytest.raises(ValueError, match="declared reach"):
        finalize_program(image, limits, codec.program_decoder, selected_contracts=selected)
    def unknown(facts, software, program):
        program["branches"]["relative"]["origin"] = "unknown"
    directory = tmp_path / "unknown"
    directory.mkdir()
    opaque = inputs(directory, branch=True, mutate=unknown)
    with pytest.raises(ValueError, match="undeclared encoding"):
        finalize(opaque, [command("branch", 0, 0, 0), command("increment", 0, 0, 1, 3, 3)])


def test_unknown_operations_do_not_acquire_effects_from_the_provider(tmp_path):
    def mutate(facts, software, program):
        program["instructions"].pop("increment")
    model = inputs(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="unknown selected control/effect"):
        compile_case(model)


def test_final_word_addresses_are_checked_against_current_windows(tmp_path):
    model = inputs(tmp_path)
    with pytest.raises(ValueError, match="outside declared address windows"):
        finalize(model, [command("copy", 0, 0, 1, 2, 0)])  # private view begins at byte8


def test_changed_resolver_inside_final_decode_is_stale(tmp_path, monkeypatch):
    from merlin.compile.scheduling import selected_contracts
    model = inputs(tmp_path)
    selected, _kernel, _objects, _adapter, codec, _limits, _word, _length = model
    original = selected_contracts._resolver_sources
    changed = False
    def sources():
        return (*original(), ("mutation", "changed")) if changed else original()
    monkeypatch.setattr(selected_contracts, "_resolver_sources", sources)
    def decode(data):
        nonlocal changed
        result = codec.program_decoder.decode(data)
        changed = True
        return result
    image = codec.encode(Kernel("stale", (), (command("increment", 0, 0, 1, 3, 3),)))
    with pytest.raises(ValueError, match="resolver changed"):
        finalize_program(image, None, replace(codec.program_decoder, decode=decode), selected_contracts=selected)


@pytest.mark.parametrize("field", ["limits", "decoder", "continuation", "outputs"])
def test_current_final_controls_cannot_change_inside_decoder_callbacks(tmp_path, field):
    model = inputs(tmp_path)
    selected, _kernel, _objects, _adapter, codec, limits, _word, _length = model
    continuation = Continuation(QUALIFICATION, ("input", "output"), True, True, 16)
    from merlin.compile.scheduling.program_limits import EncodedAccess
    outputs = (EncodedAccess("output", 0, 1),)
    def decode(data):
        result = codec.program_decoder.decode(data)
        if field == "limits":
            object.__setattr__(limits, "max_bytes", limits.max_bytes + 1)
        elif field == "decoder":
            object.__setattr__(decoder, "qualification_identity", "different-evidence")
        elif field == "continuation":
            object.__setattr__(continuation, "max_segments", 17)
        else:
            object.__setattr__(outputs[0], "size", 2)
        return result
    decoder = replace(codec.program_decoder, decode=decode)
    image = codec.encode(Kernel("changed", (), (command("increment", 0, 0, 1, 3, 3),)))
    with pytest.raises(ValueError, match="changed during checking"):
        finalize_program(image, limits, decoder, continuation=continuation, outputs=outputs,
                         selected_contracts=selected)


def test_final_checker_implementation_cannot_change_during_decode(tmp_path, monkeypatch):
    from merlin.compile.scheduling import program_limits
    model = inputs(tmp_path)
    original = program_limits._library_sources
    changed = False
    def sources():
        return {**original(), "mutation": "changed"} if changed else original()
    monkeypatch.setattr(program_limits, "_library_sources", sources)
    def decode(data):
        nonlocal changed
        result = model[4].program_decoder.decode(data)
        changed = True
        return result
    image = model[4].encode(Kernel("changed", (), (command("increment", 0, 0, 1, 3, 3),)))
    with pytest.raises(ValueError, match="checker changed"):
        finalize_program(image, model[5], replace(model[4].program_decoder, decode=decode))
