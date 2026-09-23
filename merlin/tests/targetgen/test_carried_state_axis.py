"""A stage that is pure configuration has to be followed by a command that does not want it."""

from __future__ import annotations

import copy

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.runtime.commandbuffer import CONFIGURATION_ONLY_STAGES, EPILOGUE_STAGES, STIMULUS_RANGE_KEY
from merlin.targetgen import conformance as CF
from merlin.targetgen import corpus_synth as CS


def _capsule(tmp_path, name, stages_per_command, *, stimulus=None, label="public"):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    capsule = {
        "name": name,
        "label": label,
        "operation": {
            "op": "resident_reuse",
            "attributes": {"matmuls": [{"epilogue": list(stages)} for stages in stages_per_command]},
        },
    }
    if stimulus is not None:
        capsule[STIMULUS_RANGE_KEY] = list(stimulus)
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule), encoding="utf-8")


_REQUIRED = [{"stage": "relu", "evidenced_by": ["synthetic"]}]


def test_configuration_only_stages_are_stages_of_the_abi() -> None:
    assert set(CONFIGURATION_ONLY_STAGES) <= set(EPILOGUE_STAGES)


def test_the_stage_has_to_come_first_on_a_stimulus_that_goes_negative(tmp_path) -> None:
    _capsule(tmp_path, "stage_last", [[], ["relu"]], stimulus=(-128, 127))
    _capsule(tmp_path, "unsigned", [["relu"], []])
    gap = CF._carried_state_gap(_REQUIRED, [tmp_path])
    # Both carry the stage in a two-command program and neither can show it leaking.
    assert gap["uncovered"] == ["relu"] and gap["cannot_fail"] == {"relu": ["stage_last", "unsigned"]}

    _capsule(tmp_path, "shows_it", [["relu"], []], stimulus=(-128, 127))
    gap = CF._carried_state_gap(_REQUIRED, [tmp_path])
    assert gap["uncovered"] == [] and gap["demanded_by"] == {"relu": ["shows_it"]}


def test_a_spec_without_the_axis_is_not_measured_and_synthesizes_nothing_new() -> None:
    root = merlin_dir() / "contract/capsules/conformance"
    specs = sorted(root.glob("*.yaml"))
    if not specs:
        pytest.skip("no tracked conformance spec")
    doc = yaml.safe_load(specs[0].read_text(encoding="utf-8")) or {}
    if "carried_state" in doc:
        pytest.skip("the tracked spec already carries the axis")
    names = {entry["name"] for entry in CS.synthesize(copy.deepcopy(doc))["capsules"]}
    assert not [name for name in names if "_carried_" in name]


def test_the_synthesized_member_runs_the_stage_first_and_is_signed() -> None:
    root = merlin_dir() / "contract/capsules/conformance"
    specs = sorted(root.glob("*.yaml"))
    if not specs:
        pytest.skip("no tracked conformance spec")
    doc = yaml.safe_load(specs[0].read_text(encoding="utf-8")) or {}
    doc["carried_state"] = {"required": _REQUIRED}
    members = [e for e in CS.synthesize(doc)["capsules"] if "_carried_" in e["name"]]
    if not members:
        pytest.skip("this spec admits no dtype to build the member at")
    (member,) = members
    first, second = member["matmuls"]
    assert (member["op"], first["epilogue"], second["epilogue"]) == ("resident_reuse", ["relu"], [])
    assert member["stimulus_range"][0] < 0
    assert member["generalization"] == {"generalization_axis": "carried_state"}
