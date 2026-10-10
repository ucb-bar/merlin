"""Actual operator dispatch preserves the source reader's conditional boundary."""

import hashlib
import json
from pathlib import Path

import pytest
from test_plain_word_request import _fixture

from merlin.targetgen.rtl import plain_word_relation as W
from merlin.targetgen.tool_cli import main


def request(tmp_path):
    raw, args = _fixture(tmp_path)
    argv = [
        "plain-word-relation",
        "--request",
        str(args["request"]),
        "--out",
        str(args["out"]),
        "--max-request-bytes",
        str(args["max_request_bytes"]),
        "--max-output-bytes",
        str(args["max_output_bytes"]),
    ]
    return raw, args, argv


def test_actual_operator_dispatch_retains_complete_original_source_relation(tmp_path):
    raw, args, argv = request(tmp_path)
    before = args["request"].read_bytes()
    assert main(argv) == 0
    observed = json.loads(args["out"].read_bytes())
    assert observed["word_bits"] == 9
    assert [(row["ordinal"], row["width"], row["low_bit"]) for row in observed["fields"]] == [
        (0, 5, 4),
        (1, 3, 1),
        (2, 1, 0),
    ]
    assert observed["selection"]["declaration"] == raw["declaration"]
    assert observed["request"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert observed["status"] == "conditional_source_relation" and observed["capabilities_issued"] == 0
    assert len(observed["source_membership"]["files"]) == 4
    assert "original_hw_word_and_field_occurrence_correspondence" in observed["required_unknowns"]
    assert "observer_physical_resource_and_timing" in observed["required_unknowns"]
    assert args["request"].read_bytes() == before


@pytest.mark.parametrize("flag", ["--request", "--out", "--max-request-bytes", "--max-output-bytes"])
def test_actual_operator_has_no_implicit_inputs_or_bounds(tmp_path, flag):
    _, args, argv = request(tmp_path)
    index = argv.index(flag)
    with pytest.raises(SystemExit):
        main(argv[:index] + argv[index + 2 :])
    assert not args["out"].exists()


def test_actual_operator_has_no_unselected_role_option(tmp_path):
    _, args, argv = request(tmp_path)
    with pytest.raises(SystemExit):
        main(argv + ["--qualified-role", "instruction"])
    assert not args["out"].exists()


def test_actual_operator_refuses_changed_original_source(tmp_path):
    raw, args, argv = request(tmp_path)
    path = Path(raw["cast"]["source"]["path"])
    path.write_text(path.read_text() + "changed source")
    with pytest.raises(W.WordRelationError):
        main(argv)
    assert not args["out"].exists()


def test_actual_operator_refuses_unselected_request_fields(tmp_path):
    raw, args, argv = request(tmp_path)
    raw["admission"] = True
    args["request"].write_text(json.dumps(raw))
    with pytest.raises(W.WordRelationError):
        main(argv)
    assert not args["out"].exists()


@pytest.mark.parametrize("flag", ["--max-request-bytes", "--max-output-bytes"])
def test_actual_operator_retains_bounded_input_output_refusals(tmp_path, flag):
    _, args, argv = request(tmp_path)
    argv[argv.index(flag) + 1] = "1"
    with pytest.raises(W.WordRelationError):
        main(argv)
    assert not args["out"].exists()


def test_actual_operator_output_alias_does_not_overwrite_original(tmp_path):
    _, args, argv = request(tmp_path)
    before = args["request"].read_bytes()
    args["out"].symlink_to(args["request"])
    with pytest.raises(W.WordRelationError):
        main(argv)
    assert args["request"].read_bytes() == before


def test_actual_operator_preserves_existing_output(tmp_path):
    _, args, argv = request(tmp_path)
    args["out"].write_bytes(b"existing observation")
    with pytest.raises(W.WordRelationError):
        main(argv)
    assert args["out"].read_bytes() == b"existing observation"


def test_actual_operator_refuses_source_change_after_actual_observation(tmp_path, monkeypatch):
    raw, args, argv = request(tmp_path)
    original = W.observe_plain_word_relation

    def changed(**selected):
        observation = original(**selected)
        path = Path(raw["declaration"]["source"]["path"])
        path.write_text(path.read_text() + "changed source")
        return observation

    monkeypatch.setattr(W, "observe_plain_word_relation", changed)
    with pytest.raises(W.WordRelationError):
        main(argv)
    assert not args["out"].exists()
