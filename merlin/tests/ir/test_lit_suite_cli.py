"""Explicit core lit-suite selection, independent of installed native tools or recipes."""

import json
from pathlib import Path

import pytest

from merlin.targetgen import lit_check_compiler, lit_suite


@pytest.mark.parametrize("argv", [[], ["--all"], ["--target", "fixture", "--all"], ["--tar", "fixture"]])
def test_target_selection_is_required_and_never_implicit(argv, monkeypatch):
    monkeypatch.setattr(lit_suite, "emit", lambda *args, **kwargs: pytest.fail("invalid selection emitted a suite"))
    with pytest.raises(SystemExit) as raised:
        lit_suite.main(argv)
    assert raised.value.code == 2


def test_explicit_targets_preserve_order_without_recipe_discovery(monkeypatch, capsys):
    seen = []

    def emit(target, *, write):
        seen.append((target, write))
        return {"target": target}

    monkeypatch.setattr(lit_suite, "emit", emit)
    monkeypatch.setattr(Path, "glob", lambda *args, **kwargs: pytest.fail("core CLI discovered recipe paths"))
    assert lit_suite.main(["--target", "second-device", "--target", "first-device", "--json"]) == 0
    assert seen == [("second-device", False), ("first-device", False)]
    assert json.loads(capsys.readouterr().out) == [{"target": "second-device"}, {"target": "first-device"}]
    assert not hasattr(lit_suite, "known_targets")


@pytest.mark.parametrize("write,expected", [(False, None), (True, True)])
def test_run_and_verdict_observations_are_unchanged(monkeypatch, capsys, write, expected):
    events = []
    compiled = object()
    monkeypatch.setattr(lit_suite, "emit", lambda target, **kwargs: {"target": target})
    monkeypatch.setattr(lit_check_compiler, "compile_checks", lambda target: compiled)

    def run(target):
        events.append(("run", target))
        return True

    def record(target, actual, *, lit_passed):
        assert actual is compiled
        events.append(("record", target, lit_passed))

    monkeypatch.setattr(lit_suite, "run_suite", run)
    monkeypatch.setattr(lit_suite, "record_verdicts", record)
    argv = ["--target", "fixture", "--run", "--json", *(["--write"] if write else [])]
    assert lit_suite.main(argv) == 0
    assert events == ([("run", "fixture")] if write else []) + [("record", "fixture", expected)]
    assert json.loads(capsys.readouterr().out)[0]["suite_passed"] is expected
