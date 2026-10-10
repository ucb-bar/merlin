"""The installed operator route consumes complete original source identities."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import test_hw_value_bindings as fixture

from merlin.targetgen.rtl import hw_value_bindings as V
from merlin.targetgen.tool_cli import main


def request(tmp_path):
    text = fixture.source()
    source, selection, output = (tmp_path / name for name in ("source.mlir", "request.json", "result.json"))
    source.write_text(text)
    doc = {
        "schema": "merlin.value_binding_request.v1",
        "root": "Top",
        "selections": [asdict(fixture.selection(text, ordinal=2, typ="i16")), asdict(fixture.selection(text))],
        "limits": asdict(fixture.LIMITS),
    }
    selection.write_text(json.dumps(doc))
    argv = [
        "value-binding-observation",
        "--source",
        str(source),
        "--request",
        str(selection),
        "--max-request-bytes",
        "65536",
        "--max-output-bytes",
        "65536",
        "--out",
        str(output),
    ]
    return json.loads(selection.read_text()), source, selection, output, argv


def test_actual_operator_route_exports_complete_typed_memberships_and_explicit_cuts(tmp_path):
    doc, source, selection, output, argv = request(tmp_path)
    before = (source.read_bytes(), selection.read_bytes())
    assert main(argv) == 0
    result = json.loads(output.read_text())
    assert result["schema"] == "merlin.value_binding_observation.v1"
    assert result["source_sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert result["request_sha256"] == hashlib.sha256(before[1]).hexdigest()
    observed = result["observation"]
    assert [row["selection"] for row in observed["selections"]] == doc["selections"]
    assert [row["path"] for row in observed["frames"]] == [["Top"], ["Top", "first"], ["Top", "second"]]
    assert {row["kind"] for row in observed["expressions"]} >= {"root_input", "state_result", "instance_input_binding"}
    nodes = {row["id"]: row for row in observed["expressions"]}
    word = nodes[observed["selections"][0]["value"]]
    assert word["expression"] == "comb.concat"
    assert [nodes[n]["parameter"] for n in word["operands"]] == [8, 0]
    cell = next(row for row in observed["visited_definition_members"] if row["module"] == "Cell")
    assert len(cell["complete_operation_members"]) == 6
    assert len(cell["complete_operation_members"][4]["nested_operations"]) == 2
    assert "getter_return_sample_event_custody" in observed["unknowns"]
    assert observed["admission_authority"] is False
    assert not {"cycles", "qualified", "cold", "warm"} & set(result)
    assert before == (source.read_bytes(), selection.read_bytes())


@pytest.mark.parametrize(
    "change",
    [
        "field",
        "schema",
        "root",
        "empty",
        "duplicate_selection",
        "field_selection",
        "occurrence",
        "module",
        "kind",
        "ordinal",
        "slot",
        "type",
        "identity",
        "limit_field",
        "bool_limit",
        "operation_bound",
        "metadata_bound",
        "dense",
        "encoding",
        "duplicate_key",
    ],
)
def test_real_command_refuses_incomplete_changed_or_unsupported_source_request(tmp_path, change):
    doc, source, selection, output, argv = request(tmp_path)
    row = doc["selections"][0]
    if change == "field":
        doc["qualified"] = True
    elif change == "schema":
        doc["schema"] += "changed"
    elif change == "root":
        doc["root"] = "Unavailable"
    elif change == "empty":
        doc["selections"] = []
    elif change == "duplicate_selection":
        doc["selections"].append(row)
    elif change == "field_selection":
        row["role"] = "trusted_counter"
    elif change == "occurrence":
        row["occurrence"] = "Top"
    elif change == "module":
        row["module"] = "Cell"
    elif change == "kind":
        row["kind"] = "trusted_counter"
    elif change == "ordinal":
        row["ordinal"] = True
    elif change == "slot":
        row["slot"] = True
    elif change == "type":
        row["type"] = "i15"
    elif change == "identity":
        source.write_text(source.read_text() + "\n")
    elif change == "limit_field":
        doc["limits"]["unused"] = 1
    elif change == "bool_limit":
        doc["limits"]["modules"] = True
    elif change == "operation_bound":
        doc["limits"]["operations"] = 2
    elif change == "metadata_bound":
        doc["limits"]["metadata_bytes"] = 1
    elif change == "dense":
        source.write_text('builtin.module { "fixture"() {value = dense<0> : tensor<2xi8>} : () -> () }')
    elif change == "encoding":
        source.write_bytes(b"\xff")
    selection.write_text(json.dumps(doc))
    if change == "duplicate_key":
        selection.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ValueError):
        main(argv)
    assert not output.exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-request-bytes", "0"),
        ("--max-request-bytes", "1"),
        ("--max-request-bytes", "-1"),
        ("--max-output-bytes", "0"),
        ("--max-output-bytes", "1"),
        ("--max-output-bytes", "-1"),
    ],
)
def test_actual_operator_resource_bounds_refuse_before_output(tmp_path, flag, value):
    _, _, _, output, argv = request(tmp_path)
    argv[argv.index(flag) + 1] = value
    with pytest.raises(ValueError):
        main(argv)
    assert not output.exists()


@pytest.mark.parametrize("member", ["source", "request", "output", "output_parent"])
def test_actual_alias_refuses_and_preserves_all_original_bytes(tmp_path, member):
    _, source, selection, output, argv = request(tmp_path)
    before = (source.read_bytes(), selection.read_bytes())
    if member == "output_parent":
        real = tmp_path / "real"
        real.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(real, target_is_directory=True)
        argv[argv.index("--out") + 1] = str(alias / "result.json")
    else:
        target = {"source": source, "request": selection, "output": output}[member]
        alias = tmp_path / "alias"
        alias.symlink_to(source if member == "output" else target)
        argv[argv.index({"source": "--source", "request": "--request", "output": "--out"}[member]) + 1] = str(alias)
    with pytest.raises(ValueError):
        main(argv)
    assert before == (source.read_bytes(), selection.read_bytes())
    assert not output.exists()


@pytest.mark.parametrize("member", ["source", "request"])
def test_actual_missing_or_nonregular_inputs_refuse(tmp_path, member):
    _, source, selection, output, argv = request(tmp_path)
    selected = source if member == "source" else selection
    selected.unlink()
    with pytest.raises(ValueError):
        main(argv)
    selected.mkdir()
    with pytest.raises(ValueError):
        main(argv)
    assert not output.exists()


@pytest.mark.parametrize("member", ["source", "request"])
def test_actual_growth_at_input_open_refuses(tmp_path, monkeypatch, member):
    _, source, selection, output, argv = request(tmp_path)
    selected = source if member == "source" else selection
    original = Path.open

    def growth(path, *args, **kwargs):
        if path == selected:
            with original(path, "ab") as stream:
                stream.write(b"growth")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", growth)
    with pytest.raises(ValueError):
        main(argv)
    assert not output.exists()


@pytest.mark.parametrize("member", ["source", "request"])
def test_changed_inputs_after_actual_export_refuse(tmp_path, monkeypatch, member):
    _, source, selection, output, argv = request(tmp_path)
    original = V.prepare_value_bindings

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        selected = source if member == "source" else selection
        with selected.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(V, "prepare_value_bindings", changed)
    with pytest.raises(ValueError):
        main(argv)
    assert not output.exists()


def test_changed_inputs_after_output_are_not_returned_as_success(tmp_path, monkeypatch):
    _, source, _, output, argv = request(tmp_path)
    original = Path.open

    def changed(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        if path == output:
            with original(source, "ab") as stream:
                stream.write(b"changed")
        return handle

    monkeypatch.setattr(Path, "open", changed)
    with pytest.raises(ValueError):
        main(argv)
    assert output.exists()  # Retained refused product; no successful command result.


def test_exclusive_actual_output_preserves_existing_observation(tmp_path):
    _, _, _, output, argv = request(tmp_path)
    output.write_bytes(b"retained original")
    with pytest.raises(FileExistsError):
        main(argv)
    assert output.read_bytes() == b"retained original"


def test_no_implicit_selection_or_extra_option_route(tmp_path):
    _, _, _, output, argv = request(tmp_path)
    with pytest.raises(SystemExit):
        main(argv + ["--trusted-role", "counter"])
    for flag in ("--request", "--max-output-bytes"):
        index = argv.index(flag)
        with pytest.raises(SystemExit):
            main(argv[:index] + argv[index + 2 :])
    assert not output.exists()
