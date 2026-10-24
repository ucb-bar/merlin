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


def test_validation_scope_is_explicit_closed_and_legacy_full_by_default():
    assert parse_options(["--run-id", "probe"], environ={}).validation_scope == "full"
    selected = parse_options(["--run-id", "probe", "--validation-scope", "capsules"], environ={})
    assert selected.validation_scope == "capsules"
    with pytest.raises(SystemExit):
        parse_options(["--run-id", "probe", "--validation-scope", "none"], environ={})


def test_readback_policy_is_explicit_and_closed():
    assert parse_options(["--run-id", "probe"], environ={"MERLIN_READBACK_POLICY": "out_b64_v1"}).readback_policy == ""
    assert parse_options(["--run-id", "probe", "--readback-policy", "out_b64_v1"]).readback_policy == "out_b64_v1"
    with pytest.raises(SystemExit):
        parse_options(["--run-id", "probe", "--readback-policy", "digest_only"])


def test_public_object_build_probe_is_explicit_and_bounded():
    assert parse_options(["--run-id", "probe"], environ={}).public_object_build_budget_s == 0
    assert (
        parse_options(["--run-id", "probe", "--public-object-build-budget-s", "15"]).public_object_build_budget_s == 15
    )
    for bad in ("-1", "121", "abc"):
        with pytest.raises(SystemExit):
            parse_options(["--run-id", "probe", "--public-object-build-budget-s", bad], environ={})


def test_unpaid_qualification_selects_a_submission_explicitly():
    assert parse_options(["--run-id", "probe"], environ={}).qualify_submission == ""
    assert (
        parse_options(
            ["--run-id", "probe", "--qualify-submission", "/operator/candidate"], environ={}
        ).qualify_submission
        == "/operator/candidate"
    )


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


def test_device_coverage_objective_requires_explicit_bounded_schedule():
    assert parse_options(["--run-id", "probe"], environ={}).objective == "correctness"
    good = [
        "--run-id",
        "probe",
        "--objective",
        "device-coverage",
        "--schedule",
        "continuous",
        "--max-wall-s",
        "30",
        "--plateau-rounds",
        "0",
    ]
    assert parse_options(good, environ={}).objective == "device-coverage"
    for args in (
        ["--run-id", "probe", "--objective", "device-coverage"],
        good + ["--continuous"],
        good + ["--plateau-rounds", "2"],
        good + ["--max-wall-s", "0"],
        good + ["--objective", "unknown"],
    ):
        with pytest.raises(SystemExit):
            parse_options(args, environ={})


def test_correctness_truth_and_coverage_termination_are_independent():
    from merlin_experiments.phase1.feedback.certification import _authoring_completion, _formal_completion

    assert _authoring_completion(True, True)
    assert not _authoring_completion(False, True)
    assert not _authoring_completion(True, False)
    for correct in (False, True):
        for conformant in (False, True):
            assert not _authoring_completion(correct, conformant, objective="device-coverage")
    assert _formal_completion(True, True, True, True)
    with pytest.raises(ValueError):
        _authoring_completion(True, True, objective="unknown")
