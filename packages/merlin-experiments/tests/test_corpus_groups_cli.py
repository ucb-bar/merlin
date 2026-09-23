"""Canonical and legacy group entrypoints share parsing and execution, never review."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.cli import main

from merlin.targetgen import group_capsules


def test_canonical_group_command_dispatches_the_legacy_parsed_arguments(monkeypatch, tmp_path):
    observed = []

    def run(args):
        observed.append(vars(args))
        return 19

    monkeypatch.setattr(group_capsules, "run_from_args", run)
    argv = [
        "--target",
        "fixture",
        "--capture",
        "capture.mlir",
        "--manifest",
        "weights.json",
        "--model",
        "fixture-model",
        "--out",
        str(tmp_path / "unused"),
        "--package",
        "candidate",
        "--only",
        "first",
        "--only",
        "second",
        "--timeout",
        "17",
        "--max-tier",
        "L2",
        "--promote",
    ]
    assert group_capsules.main(argv) == 19
    assert main(["corpus", "groups", *argv]) == 19
    canonical = {key: value for key, value in observed[1].items() if key not in {"catalog", "verb", "operation"}}
    assert canonical == observed[0]
    assert canonical["only"] == ["first", "second"]
    assert canonical["promote"] is True
    assert not list(tmp_path.iterdir())


def test_canonical_groups_uses_the_shared_parser(monkeypatch, capsys):
    actual = group_capsules.configure_parser
    seen = []

    def configure(parser):
        assert isinstance(parser, argparse.ArgumentParser)
        seen.append(parser.prog)
        actual(parser)

    monkeypatch.setattr(group_capsules, "configure_parser", configure)
    with pytest.raises(SystemExit) as result:
        main(["corpus", "groups", "--help"])
    assert result.value.code == 0
    assert seen == ["merlin experiment corpus groups"]
    assert "--max-tier" in capsys.readouterr().out


@pytest.mark.parametrize("legacy", [False, True])
def test_group_help_and_unknown_options_never_execute(monkeypatch, tmp_path, capsys, legacy):
    def forbidden(_args):
        pytest.fail("help or invalid options executed the corpus handler")

    monkeypatch.setattr(group_capsules, "run_from_args", forbidden)
    invoke = group_capsules.main if legacy else lambda argv: main(["corpus", "groups", *argv])
    with pytest.raises(SystemExit) as help_result:
        invoke(["--help"])
    assert help_result.value.code == 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as invalid:
        invoke(["--target", "fixture", "--capture", "missing.mlir", "--out", str(tmp_path / "unused"), "--timout", "7"])
    assert invalid.value.code == 2
    assert "unrecognized arguments: --timout 7" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_public_command_help_is_import_light_and_does_not_load_sealing(tmp_path):
    script = """
import sys
from merlin_experiments.cli import main
try:
    main(['corpus', 'groups', '--help'])
except SystemExit as result:
    assert result.code == 0
else:
    raise AssertionError('help did not terminate parsing')
assert not any(name.startswith(('merlin.targetgen.capsule_runner',
    'merlin.targetgen.capsule_golden', 'merlin_experiments.corpus.release',
    'merlin.common.mlir_query', 'torch', 'aet')) for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "merlin experiment corpus groups" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "legacy",
    [False, True],
)
def test_real_public_and_legacy_module_help_work_without_mutations(tmp_path, legacy):
    command = (
        [sys.executable, "-B", "-m", "merlin.targetgen.group_capsules"]
        if legacy
        else [str(Path(sys.executable).with_name("merlin")), "experiment", "corpus", "groups"]
    )
    result = subprocess.run(
        [*command, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--capture" in result.stdout
    assert "--promote" in result.stdout
    assert "--max-tier" in result.stdout
    assert not list(tmp_path.iterdir())
