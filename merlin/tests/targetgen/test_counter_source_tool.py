"""Installed operator tooling exercises the actual complete-row readers."""

import hashlib
import json
from dataclasses import asdict

import pytest
import test_hw_counter_intervals as unit
import test_hw_counter_state_timelines as state
import test_hw_state_effects as effects

from merlin.targetgen.tool_cli import main


def request(tmp_path, kind):
    fixture = unit if kind == "unit_counter" else state
    source = fixture._source()
    cycles = ((0, 0, 0), (0, 1, 0)) if kind == "unit_counter" else ((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 7))
    samples = fixture._samples(cycles)
    value = {
        "schema": "merlin.counter_source_request.v1",
        "kind": kind,
        "selection": asdict(fixture._selection(source)),
        "limits": asdict(fixture.LIMITS),
        "expected_phases": len(samples),
        "samples": [asdict(sample) for sample in samples],
        "intervals": [{"start": 0, "end": len(samples) - 1}],
    }
    source_path, request_path, output = (tmp_path / name for name in ("source.mlir", "request.json", "result.json"))
    source_path.write_text(source)
    request_path.write_text(json.dumps(value))
    argv = [
        "counter-source-observation",
        "--source",
        str(source_path),
        "--request",
        str(request_path),
        "--max-request-bytes",
        "65536",
        "--out",
        str(output),
    ]
    return json.loads(request_path.read_text()), request_path, source_path, output, argv


@pytest.mark.parametrize("kind", ["unit_counter", "state_getter"])
def test_operator_command_evaluates_complete_rows_and_retains_unknowns(tmp_path, kind):
    value, request_path, source_path, output, argv = request(tmp_path, kind)
    assert main(argv) == 0
    result = json.loads(output.read_text())
    assert result["request_sha256"] == hashlib.sha256(request_path.read_bytes()).hexdigest()
    assert result["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    observed = result["observation"]
    assert observed["samples"] == value["samples"]
    assert len(observed["transitions"]) == value["expected_phases"]
    assert "physical_clock_units_frequency_and_loaded_image" in observed["unknowns"]
    assert "independent_held_group_predictive_qualification" in observed["unknowns"]
    assert not {"qualified", "cycles", "cold", "warm"} & set(result)
    assert result["schema"] == "merlin.counter_source_observation.v1"
    assert "macro_environment_sha256" not in result and "source_emission" not in observed
    if kind == "state_getter":
        assert set(observed) == {
            "source_sha256",
            "selection",
            "input_ports",
            "output_ports",
            "states",
            "getter_width",
            "samples",
            "transitions",
            "intervals",
            "unknowns",
        }
        assert observed["intervals"][0]["unit_increments"] is None


@pytest.mark.parametrize("mutation", ["field", "sample", "result", "limit", "selection", "kind", "duplicate", "bytes"])
def test_operator_request_refuses_changed_incomplete_or_unbounded_inputs(tmp_path, mutation):
    value, request_path, source_path, output, argv = request(tmp_path, "unit_counter")
    if mutation == "field":
        value["trusted_timer"] = True
    elif mutation == "sample":
        value["samples"].pop()
    elif mutation == "result":
        value["samples"][1]["outputs"] = [1, 0]
    elif mutation == "limit":
        value["limits"]["expressions"]["cases"] = True
    elif mutation == "selection":
        source_path.write_text(source_path.read_text() + "\n")
    elif mutation == "kind":
        value["kind"] = "qualified_timer"
    elif mutation == "bytes":
        argv[argv.index("--max-request-bytes") + 1] = "1"
    request_path.write_text(json.dumps(value))
    if mutation == "duplicate":
        request_path.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ValueError):
        main(argv)
    assert not output.exists()


def test_operator_output_requires_a_fresh_destination(tmp_path):
    _, _, _, output, argv = request(tmp_path, "unit_counter")
    output.write_text("original retained observation")
    with pytest.raises(FileExistsError):
        main(argv)
    assert output.read_text() == "original retained observation"


def effect_request(tmp_path, *, settings=None, cycles=((0, 0, 0), (0, 1, 1))):
    source = effects._source()
    samples = effects._samples(cycles)
    value = {
        "schema": "merlin.counter_source_request.v2",
        "kind": "state_getter",
        "selection": asdict(effects._selection(source)),
        "limits": asdict(effects.LIMITS),
        "expected_phases": len(samples),
        "samples": [asdict(sample) for sample in samples],
        "intervals": [{"start": 0, "end": len(samples) - 1}],
        "macro_environment": effects._environment(source, settings),
    }
    source_path, request_path, output = (tmp_path / name for name in ("source.mlir", "request.json", "result.json"))
    source_path.write_text(source)
    request_path.write_text(json.dumps(value))
    argv = [
        "counter-source-observation",
        "--source",
        str(source_path),
        "--request",
        str(request_path),
        "--max-request-bytes",
        "65536",
        "--out",
        str(output),
    ]
    return json.loads(request_path.read_text()), request_path, source_path, output, argv


def test_explicit_operator_macro_premise_retains_complete_effects_and_exact_bytes(tmp_path):
    value, request_path, source_path, output, argv = effect_request(tmp_path)
    assert main(argv) == 0
    result = json.loads(output.read_text())
    assert result["schema"] == "merlin.counter_source_observation.v2"
    assert result["request_sha256"] == hashlib.sha256(request_path.read_bytes()).hexdigest()
    assert result["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert result["macro_environment_sha256"] == hashlib.sha256(value["macro_environment"].encode()).hexdigest()
    observed = result["observation"]
    assert observed["samples"] == value["samples"]
    emission = observed["source_emission"]
    assert len(emission["original_effects"]) == 4 and len(emission["phases"]) == 16
    assert [row["status"] for row in emission["phases"][-4:]] == ["triggered"] * 4
    assert emission["original_fragment_symbols"] == ["PrintFragment", "StopFragment"]
    assert emission["compiler_runtime_environment_correspondence"] == "UNKNOWN"
    assert emission["source_effect_scheduling_correspondence"] == "UNKNOWN"
    assert "independent_held_group_predictive_qualification" in observed["unknowns"]
    assert not {"qualified", "cycles", "cold", "warm"} & set(result)


@pytest.mark.parametrize("setting", [{"SYNTHESIS": 0}, {"PRINT_GATE": 0, "STOP_GATE_": 0}])
def test_explicit_operator_inactive_effects_require_actual_supplied_premise(tmp_path, setting):
    _, _, _, output, argv = effect_request(tmp_path, settings=setting, cycles=((0, 0, 0), (0, 1, 1), (0, 0, 0)))
    assert main(argv) == 0
    emission = json.loads(output.read_text())["observation"]["source_emission"]
    assert len(emission["original_effects"]) == 4
    assert len(emission["phases"]) == 24 and all(row["status"] == "inactive" for row in emission["phases"])


@pytest.mark.parametrize("kind", ["unit_counter", "state_getter"])
def test_explicit_request_handles_both_readers_without_changing_old_values(tmp_path, kind):
    value, request_path, source_path, output, argv = request(tmp_path, kind)
    value["schema"] = "merlin.counter_source_request.v2"
    value["macro_environment"] = (
        None
        if kind == "unit_counter"
        else json.dumps(
            {
                "schema": "merlin.source_macro_environment.v1",
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "macros": [],
            }
        )
    )
    request_path.write_text(json.dumps(value))
    assert main(argv) == 0
    result = json.loads(output.read_text())
    assert result["schema"] == "merlin.counter_source_observation.v2"
    assert result["observation"]["samples"] == value["samples"]
    if kind == "unit_counter":
        assert result["macro_environment_sha256"] is None
    else:
        assert "module_metadata" in result["observation"]
        assert result["observation"]["source_emission"]["original_effects"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "null",
        "mapping",
        "legacy_field",
        "legacy_default",
        "unknown_field",
        "stale_macro_source",
        "missing_macro",
        "duplicate_macro_key",
        "phase",
        "states",
        "outputs",
        "limit",
        "termination",
        "source",
    ],
)
def test_explicit_operator_request_refuses_unresolved_or_changed_full_source_rows(tmp_path, mutation):
    value, request_path, source_path, output, argv = effect_request(tmp_path)
    if mutation == "missing":
        del value["macro_environment"]
    elif mutation == "null":
        value["macro_environment"] = None
    elif mutation == "mapping":
        value["macro_environment"] = json.loads(value["macro_environment"])
    elif mutation == "legacy_field":
        value["schema"] = "merlin.counter_source_request.v1"
    elif mutation == "legacy_default":
        value["schema"] = "merlin.counter_source_request.v1"
        del value["macro_environment"]
    elif mutation == "unknown_field":
        value["qualified"] = True
    elif mutation in {"stale_macro_source", "missing_macro"}:
        premise = json.loads(value["macro_environment"])
        if mutation == "stale_macro_source":
            premise["source_sha256"] = "0" * 64
        else:
            premise["macros"].pop()
        value["macro_environment"] = json.dumps(premise)
    elif mutation == "duplicate_macro_key":
        value["macro_environment"] = '{"schema":0,' + value["macro_environment"][1:]
    elif mutation == "phase":
        value["samples"].pop()
    elif mutation == "states":
        value["samples"][-1]["states"] = [7]
    elif mutation == "outputs":
        value["samples"][-1]["outputs"] = [7]
    elif mutation == "limit":
        value["limits"]["expressions"]["nodes"] = 1
    elif mutation == "termination":
        samples = effects._samples(((0, 0, 0), (0, 1, 1), (0, 0, 0)))
        value["samples"] = [asdict(sample) for sample in samples]
        value["expected_phases"] = len(samples)
        value["intervals"][0]["end"] = len(samples) - 1
    elif mutation == "source":
        source_path.write_text(source_path.read_text() + "\n")
    request_path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        main(argv)
    assert not output.exists()


@pytest.mark.parametrize("premise", ["{}", False, 0, []])
def test_unit_reader_does_not_claim_emission_semantics_from_new_request(tmp_path, premise):
    value, request_path, _, output, argv = request(tmp_path, "unit_counter")
    value.update(schema="merlin.counter_source_request.v2", macro_environment=premise)
    request_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="macro premise is unsupported for this kind"):
        main(argv)
    assert not output.exists()


def test_changed_explicit_macro_bytes_change_identity_without_mutating_source_or_request(tmp_path):
    value, request_path, source_path, output, argv = effect_request(tmp_path, settings={"SYNTHESIS": 0})
    before_source, before_request = source_path.read_bytes(), request_path.read_bytes()
    assert main(argv) == 0
    first = json.loads(output.read_text())
    assert source_path.read_bytes() == before_source and request_path.read_bytes() == before_request
    premise = json.loads(value["macro_environment"])
    premise["macros"][0]["value"] = 1
    value["macro_environment"] = json.dumps(premise)
    request_path.write_text(json.dumps(value))
    second_request = request_path.read_bytes()
    argv[-1] = str(tmp_path / "second-result.json")
    assert main(argv) == 0
    second = json.loads((tmp_path / "second-result.json").read_text())
    assert source_path.read_bytes() == before_source and request_path.read_bytes() == second_request
    assert first["source_sha256"] == second["source_sha256"]
    assert first["request_sha256"] != second["request_sha256"]
    assert first["macro_environment_sha256"] != second["macro_environment_sha256"]
    assert all(row["status"] == "inactive" for row in second["observation"]["source_emission"]["phases"])
