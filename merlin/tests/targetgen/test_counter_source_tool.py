"""Installed operator tooling exercises the actual complete-row readers."""

import hashlib
import json
from dataclasses import asdict

import pytest
import test_hw_counter_intervals as unit
import test_hw_counter_state_timelines as state

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
    if kind == "state_getter":
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
