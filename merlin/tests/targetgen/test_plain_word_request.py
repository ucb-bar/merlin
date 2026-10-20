"""Actual closed source-word request consumption and publication refusals."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_plain_word_relation import _inputs

from merlin.targetgen.rtl import plain_word_request as R


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if type(value) is dict:
        return {name: _plain(item) for name, item in value.items()}
    if type(value) in (tuple, list):
        return [_plain(item) for item in value]
    return value


def _fixture(tmp_path):
    selections = _inputs(tmp_path)
    raw = {name: _plain(asdict(value)) for name, value in selections.items()}
    raw["schema"] = R.REQUEST_SCHEMA
    request, out = tmp_path / "request.json", tmp_path / "observation.json"
    request.write_text(json.dumps(raw))
    args = {"request": request, "out": out, "max_request_bytes": 16384, "max_output_bytes": 32768}
    return raw, args


def test_actual_request_retains_full_relation_and_all_unknowns(tmp_path):
    raw, args = _fixture(tmp_path)
    actual = R.write_plain_word_observation(**args)
    assert json.loads(args["out"].read_bytes()) == actual
    assert actual["word_bits"] == 9 and [row["width"] for row in actual["fields"]] == [5, 3, 1]
    assert actual["selection"]["declaration"] == raw["declaration"]
    assert actual["selection"]["cast"] == raw["cast"]
    assert actual["selection"]["semantics"] == raw["semantics"]
    assert actual["request"]["sha256"] == hashlib.sha256(args["request"].read_bytes()).hexdigest()
    assert actual["status"] == "conditional_source_relation" and actual["capabilities_issued"] == 0
    assert "original_hw_word_and_field_occurrence_correspondence" in actual["required_unknowns"]
    assert "complete_prohibited_source_role_policy" in actual["required_unknowns"]


@pytest.mark.parametrize("name", ["schema", "declaration", "cast", "semantics", "limits"])
def test_missing_original_top_level_roster_refuses_before_output(tmp_path, name):
    raw, args = _fixture(tmp_path)
    del raw[name]
    args["request"].write_text(json.dumps(raw))
    with pytest.raises(R.W.WordRelationError):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize("name", ["accepted", "opcode", "policy", "runtime_authority", "factory"])
def test_unselected_authority_or_factory_fields_refuse(tmp_path, name):
    raw, args = _fixture(tmp_path)
    raw[name] = True
    args["request"].write_text(json.dumps(raw))
    with pytest.raises(R.W.WordRelationError):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize(
    "path",
    [
        ("declaration",),
        ("declaration", "source"),
        ("cast",),
        ("semantics",),
        ("semantics", "primitives", 0),
        ("semantics", "primitives", 0, "width_source"),
        ("semantics", "premises", 0),
        ("limits",),
    ],
)
def test_nested_extra_or_missing_selected_fields_refuse(tmp_path, path):
    raw, args = _fixture(tmp_path)
    target = raw
    for name in path:
        target = target[name]
    target["unselected"] = 1
    args["request"].write_text(json.dumps(raw))
    with pytest.raises(R.W.WordRelationError):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize("body", ['{"schema":1,"schema":2}', '{"limits":NaN}', "[" * 1500 + "]" * 1500])
def test_duplicate_nonfinite_and_deep_requests_refuse(tmp_path, body):
    _, args = _fixture(tmp_path)
    args["request"].write_text(body)
    with pytest.raises(R.W.WordRelationError):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize("name", ["max_request_bytes", "max_output_bytes"])
@pytest.mark.parametrize("value", [True, 1.0, 0, -1])
def test_invalid_explicit_command_bounds_refuse(tmp_path, name, value):
    _, args = _fixture(tmp_path)
    args[name] = value
    with pytest.raises(R.W.WordRelationError):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize("name", ["max_request_bytes", "max_output_bytes"])
def test_actual_complete_input_and_output_byte_bounds_refuse(tmp_path, name):
    _, args = _fixture(tmp_path)
    args[name] = 1
    with pytest.raises(R.W.WordRelationError, match="byte bound"):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize("role", ["request", "source", "out", "parent"])
def test_linked_paths_refuse_without_overwriting_owned_files(tmp_path, role):
    raw, args = _fixture(tmp_path)
    if role == "request":
        alias = tmp_path / "linked-request.json"
        alias.symlink_to(args["request"])
        args["request"] = alias
    elif role == "source":
        alias = tmp_path / "linked-source.scala"
        alias.symlink_to(raw["declaration"]["source"]["path"])
        raw["declaration"]["source"]["path"] = str(alias)
        args["request"].write_text(json.dumps(raw))
    elif role == "out":
        args["out"].symlink_to(args["request"])
    else:
        alias = tmp_path / "linked-parent"
        alias.symlink_to(tmp_path, target_is_directory=True)
        args["out"] = alias / "observation.json"
    before = args["request"].read_bytes()
    with pytest.raises(R.W.WordRelationError):
        R.write_plain_word_observation(**args)
    assert args["request"].read_bytes() == before


def test_existing_source_destination_never_overwrites_original(tmp_path):
    raw, args = _fixture(tmp_path)
    args["out"] = Path(raw["declaration"]["source"]["path"])
    before = args["out"].read_bytes()
    with pytest.raises(R.W.WordRelationError, match="fresh"):
        R.write_plain_word_observation(**args)
    assert args["out"].read_bytes() == before


@pytest.mark.parametrize("role", ["request", "declaration", "packing"])
def test_actual_selected_bytes_mutation_after_observation_refuses(tmp_path, monkeypatch, role):
    raw, args = _fixture(tmp_path)
    original = R.W.observe_plain_word_relation

    def changed(**selected):
        result = original(**selected)
        if role == "request":
            path = args["request"]
        elif role == "declaration":
            path = Path(raw["declaration"]["source"]["path"])
        else:
            path = Path(raw["semantics"]["premises"][0]["source"]["path"])
        path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(R.W, "observe_plain_word_relation", changed)
    with pytest.raises(R.W.WordRelationError, match="changed"):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()


def test_relative_selected_source_cannot_acquire_cwd_identity(tmp_path):
    raw, args = _fixture(tmp_path)
    raw["declaration"]["source"]["path"] = "declaration.scala"
    args["request"].write_text(json.dumps(raw))
    with pytest.raises(R.W.WordRelationError):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()


@pytest.mark.parametrize("role", ["request", "source", "out"])
def test_actual_bytes_mutation_during_publication_is_never_success(tmp_path, monkeypatch, role):
    raw, args = _fixture(tmp_path)
    original, changed = R._path, False

    def during_publication(value):
        nonlocal changed
        path = original(value)
        if path == args["out"] and path.exists() and not changed:
            changed = True
            if role == "request":
                destination = args["request"]
            elif role == "source":
                destination = Path(raw["cast"]["source"]["path"])
            else:
                destination = args["out"]
            destination.write_bytes(destination.read_bytes() + b" ")
        return path

    monkeypatch.setattr(R, "_path", during_publication)
    with pytest.raises(R.W.WordRelationError, match="changed"):
        R.write_plain_word_observation(**args)
    assert changed and args["out"].exists()


def test_unrepresentable_request_read_is_explicitly_unavailable(tmp_path):
    _, args = _fixture(tmp_path)
    args["max_request_bytes"] = 1 << 100
    with pytest.raises(R.W.WordRelationError, match="unavailable"):
        R.write_plain_word_observation(**args)
    assert not args["out"].exists()
