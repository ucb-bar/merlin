"""The functional compiler CLI has one parser without target initialization."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import pytest
from merlin_experiments.phase1.options import RunOptions, build_parser, parse_options


def test_parser_fields_have_exactly_one_typed_owner():
    values = vars(build_parser(environ={}).parse_args(["--run-id", "probe"]))
    assert set(values) == {field.name for field in dataclasses.fields(RunOptions)}
    assert dataclasses.asdict(parse_options(["--run-id", "probe"], environ={})) == values


def test_environment_defaults_are_invocation_inputs_not_import_state(monkeypatch):
    explicit = {"AWS_REGION": "region-a", "AWS_PROFILE": "profile-a", "CLAUDE_CONFIG_DIR": "account-a"}
    args = parse_options(["--run-id", "probe"], environ=explicit)
    assert (args.aws_region, args.aws_profile, args.account_config_dir) == ("region-a", "profile-a", "account-a")
    monkeypatch.setenv("AWS_REGION", "region-b")
    assert parse_options(["--run-id", "probe"]).aws_region == "region-b"
    assert parse_options(["--run-id", "probe", "--aws-region", "region-c"], environ=explicit).aws_region == "region-c"
    assert explicit == {"AWS_REGION": "region-a", "AWS_PROFILE": "profile-a", "CLAUDE_CONFIG_DIR": "account-a"}


def test_repeated_tools_are_ordered_and_do_not_leak_between_invocations():
    args = parse_options(["--run-id", "probe", "--with-tool", "one", "--with-tool", "two", "--without-tool", "three"])
    assert args.with_tool == ["one", "two"]
    assert args.without_tool == ["three"]
    assert parse_options(["--run-id", "other"]).with_tool == []
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.schedule = "rounds"


@pytest.mark.parametrize(
    "args", [[], ["--run-id", "probe", "--schedule", "unknown"], ["--run-id", "probe", "--qa-timeout", "bad"]]
)
def test_invalid_arguments_fail_before_initializing_an_experiment(args):
    with pytest.raises(SystemExit) as exc:
        parse_options(args, environ={})
    assert exc.value.code == 2


def test_help_does_not_discover_target_or_run_git(tmp_path):
    # This uses the real installed/editable package in a fresh process, not the native
    # launcher, which retains its historical checkout bootstrap during migration.
    program = """
import os, subprocess, sys
before = dict(os.environ)
def forbidden(*args, **kwargs): raise AssertionError('process launch during help')
subprocess.run = subprocess.Popen = forbidden
from merlin_experiments.phase1.options import build_parser
try:
    build_parser().parse_args(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert dict(os.environ) == before
assert not any(name in sys.modules for name in ('_common', 'run_baseline_qa_loop', 'qa_check'))
assert 'merlin_experiments.phase1.context' not in sys.modules
"""
    environment = dict(os.environ, MERLIN_TARGET_EXPERIMENT=str(tmp_path / "nonexistent.yaml"))
    result = subprocess.run(
        [sys.executable, "-I", "-c", program], cwd=tmp_path, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--schedule" in result.stdout
    assert "--seal-current" in result.stdout
