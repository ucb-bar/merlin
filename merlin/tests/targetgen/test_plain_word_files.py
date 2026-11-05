"""Selected nonregular files and unrepresentable reader budgets refuse before open."""

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_plain_word_relation import _inputs
from test_plain_word_request import _fixture

from merlin.targetgen.rtl import plain_word_relation as W
from merlin.targetgen.rtl import plain_word_request as R


def _fifo(path):
    path.unlink(missing_ok=True)
    os.mkfifo(path)


def _no_open(monkeypatch):
    calls = []

    def opened(path, *args, **kwargs):
        calls.append(path)
        raise RuntimeError("selected file was opened before nonregular refusal")

    monkeypatch.setattr(Path, "open", opened)
    return calls


def test_declaration_fifo_refuses_without_any_open(tmp_path, monkeypatch):
    selected = _inputs(tmp_path)
    _fifo(selected["declaration"].source.path)
    calls = _no_open(monkeypatch)
    with pytest.raises(W.WordRelationError, match="regular"):
        W.observe_plain_word_relation(**selected)
    assert calls == []


@pytest.mark.parametrize("role", ["request", "out"])
def test_request_and_output_fifo_refuse_without_any_open(tmp_path, monkeypatch, role):
    _, args = _fixture(tmp_path)
    _fifo(args[role])
    calls = _no_open(monkeypatch)
    with pytest.raises(W.WordRelationError, match="regular"):
        R.write_plain_word_observation(**args)
    assert calls == []


@pytest.mark.parametrize("role", ["request", "out"])
def test_directories_are_not_selected_file_members(tmp_path, monkeypatch, role):
    _, args = _fixture(tmp_path)
    directory = tmp_path / "directory"
    directory.mkdir()
    args[role] = directory
    calls = _no_open(monkeypatch)
    with pytest.raises(W.WordRelationError, match="regular"):
        R.write_plain_word_observation(**args)
    assert calls == []


@pytest.mark.parametrize("phase", ["parser", "publication"])
def test_source_replaced_with_fifo_refuses_without_opening_fifo(tmp_path, monkeypatch, phase):
    selected = _inputs(tmp_path)
    path = selected["declaration"].source.path
    original_open, original_parse = Path.open, W._declaration
    fifo_opens = []

    def opened(member, *args, **kwargs):
        if member == path and path.exists() and not path.is_file():
            fifo_opens.append(member)
            raise RuntimeError("source replacement FIFO was opened")
        return original_open(member, *args, **kwargs)

    def parsed(*args):
        value = original_parse(*args)
        _fifo(path)
        return value

    monkeypatch.setattr(Path, "open", opened)
    if phase == "parser":
        monkeypatch.setattr(W, "_declaration", parsed)
        with pytest.raises(W.WordRelationError, match="regular"):
            W.observe_plain_word_relation(**selected)
    else:
        _, args = _fixture(tmp_path)
        original_observe = R.W.observe_plain_word_relation

        def observed(**selected):
            result = original_observe(**selected)
            _fifo(path)
            return result

        monkeypatch.setattr(R.W, "observe_plain_word_relation", observed)
        with pytest.raises(W.WordRelationError, match="regular"):
            R.write_plain_word_observation(**args)
        assert not args["out"].exists()
    assert fifo_opens == []


@pytest.mark.parametrize("name", ["source_bytes", "aggregate_source_bytes", "tokens", "fields", "word_bits", "nesting"])
def test_unrepresentable_source_reader_budgets_refuse_before_open(tmp_path, monkeypatch, name):
    selected = _inputs(tmp_path)
    selected["limits"] = replace(selected["limits"], **{name: 1 << 20000})
    calls = _no_open(monkeypatch)
    with pytest.raises(W.WordRelationError, match="representable"):
        W.observe_plain_word_relation(**selected)
    assert calls == []


@pytest.mark.parametrize("name", ["max_request_bytes", "max_output_bytes"])
def test_unrepresentable_command_budgets_refuse_before_open(tmp_path, monkeypatch, name):
    _, args = _fixture(tmp_path)
    args[name] = sys.maxsize
    calls = _no_open(monkeypatch)
    with pytest.raises(W.WordRelationError, match="unavailable"):
        R.write_plain_word_observation(**args)
    assert calls == []


def test_representable_budget_boundary_is_reader_scope_only():
    limits = W.WordRelationLimits(*(sys.maxsize - 1 for _ in range(6)))
    limits.verify()
    R._bound(sys.maxsize - 1)
