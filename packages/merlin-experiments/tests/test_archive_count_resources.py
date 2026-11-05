"""Explicit resource budgets preserve the frozen count command and model."""

import json
from dataclasses import replace

import pytest
from merlin_experiments.phase2 import archive_spike_runtime as R
from merlin_experiments.phase2.contracts import StageGateError
from test_ordinary_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_archive_screen import source as source

from merlin.perf import dynamic_census as C


def select(runtime, protocol):
    return R.freeze_runtime(
        review_id=runtime.review_id, functional_argv=runtime.functional_argv,
        observer=runtime.observer, disassembler=runtime.disassembler,
        model_path=runtime.model_path, source_manifest=runtime.source_manifest,
        libraries=runtime.libraries, dependencies=runtime.dependencies,
        entry_symbol=runtime.entry_symbol, completion_symbol=runtime.completion_symbol,
        timing_selection=json.loads(runtime.timing_json), observation_resource_protocol=protocol,
    )


def test_default_resource_record_and_thirty_second_command_cap_are_preserved(source):
    runtime = source["runtime"]
    assert runtime.record()["schema"] == "merlin.phase2.archive_spike_runtime.v1"
    assert "observation_resources" not in runtime.record()
    assert runtime.observation_limits(2400).wall_s == 30
    assert runtime.observation_limits(7).wall_s == 7


def test_explicit_120_second_resource_selection_keeps_both_runtime_commands_and_model(source):
    runtime = source["runtime"]
    selected = select(runtime, R.COUNT_OBSERVATION_120S_PROTOCOL)
    assert selected.sha256 != runtime.sha256
    assert selected.model_json == runtime.model_json and selected.files == runtime.files
    assert selected.record()["observation_resources"] == dict(
        protocol=R.COUNT_OBSERVATION_120S_PROTOCOL, per_command_wall_seconds=120, parent_deadline_preserved=True,
    )
    assert selected.observation_limits(2400).wall_s == 120
    assert selected.observation_limits(11).wall_s == 11
    scope = source["root"] / "resource_scope.json"
    scope.write_text(json.dumps({"input_sha256": "a" * 64}))
    args = source["root"] / "program.elf", source["root"] / "report.json", scope
    assert selected.commands(*args) == runtime.commands(*args)


@pytest.mark.parametrize("protocol", [True, {}, "", "120", "unbounded"])
def test_unknown_or_unbounded_resource_protocol_refuses_at_runtime_freeze(source, protocol):
    with pytest.raises(StageGateError, match="bounded observation resource protocol"):
        select(source["runtime"], protocol)


def test_changed_selected_resource_protocol_refuses_before_count_commands(source):
    runtime = select(source["runtime"], R.COUNT_OBSERVATION_120S_PROTOCOL)
    with pytest.raises(StageGateError, match="bounded observation resource protocol"):
        replace(runtime, observation_resource_protocol="unbounded").observation_limits(2400)


def test_explicit_600_second_resource_class_preserves_parent_and_all_other_bounds(source):
    runtime = select(source["runtime"], R.COUNT_OBSERVATION_600S_PROTOCOL)
    limits = runtime.observation_limits(2400)
    assert limits.wall_s == 600 and limits.wall_resource_protocol == C.WALL_600S_PROTOCOL
    assert runtime.record()["schema"] == "merlin.phase2.archive_spike_runtime.v3"
    assert runtime.record()["observation_resources"]["per_command_wall_seconds"] == 600
    assert runtime.observation_limits(17).wall_s == 17
    assert runtime.model_json == source["runtime"].model_json
    assert replace(limits, wall_s=30, wall_resource_protocol=None) == C.CensusLimits()
    assert C.unknown("resource", limits=limits)["limits"]["wall_resource_protocol"] == C.WALL_600S_PROTOCOL
    assert "wall_resource_protocol" not in C.unknown("default", limits=C.CensusLimits())["limits"]


@pytest.mark.parametrize("protocol,wall", [
    (None, 121), (None, 600), (True, 600), ({}, 600), ("unbounded", 600), (C.WALL_600S_PROTOCOL, 601),
])
def test_extended_wall_cannot_be_obtained_by_unselected_or_forged_limits(protocol, wall):
    with pytest.raises(C.CensusUnavailable):
        C.CensusLimits(wall_s=wall, wall_resource_protocol=protocol).verify()


def test_default_census_constructor_still_caps_at_thirty_seconds():
    assert C.CensusLimits.for_timeout(2400).wall_s == 30
    assert C.CensusLimits.for_timeout(2400, wall_resource_protocol=C.WALL_600S_PROTOCOL).wall_s == 600
    assert C.CensusLimits.for_timeout(9, wall_resource_protocol=C.WALL_600S_PROTOCOL).wall_s == 9


def test_explicit_900_second_resource_class_preserves_commands_model_and_parent_deadline(source):
    original = source["runtime"]
    runtime = select(original, R.COUNT_OBSERVATION_900S_PROTOCOL)
    assert runtime.record()["schema"] == "merlin.phase2.archive_spike_runtime.v4"
    assert runtime.record()["observation_resources"]["per_command_wall_seconds"] == 900
    limits = runtime.observation_limits(2400)
    assert limits.wall_s == 900 and limits.wall_resource_protocol == C.WALL_900S_PROTOCOL
    assert runtime.observation_limits(29).wall_s == 29
    assert replace(limits, wall_s=30, wall_resource_protocol=None) == C.CensusLimits()
    assert runtime.model_json == original.model_json and runtime.files == original.files
    scope = source["root"] / "scope900.json"
    scope.write_text(json.dumps({"input_sha256": "a" * 64}))
    args = source["root"] / "program.elf", source["root"] / "report.json", scope
    assert runtime.commands(*args) == original.commands(*args)


@pytest.mark.parametrize("protocol,wall", [
    (None, 900), (C.WALL_600S_PROTOCOL, 900), (C.WALL_900S_PROTOCOL, 901), ("wall900", 900),
])
def test_nine_hundred_second_class_does_not_broaden_previous_or_unknown_resource_classes(protocol, wall):
    with pytest.raises(C.CensusUnavailable):
        C.CensusLimits(wall_s=wall, wall_resource_protocol=protocol).verify()
