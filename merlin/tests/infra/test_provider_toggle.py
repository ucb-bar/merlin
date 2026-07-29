"""WS-D: the experiments-only provider toggle (subscription <-> Bedrock).

The toggle must be a clean no-op for the default (subscription keeps the machine's ~/.claude creds) and,
under bedrock, both thread the AWS routing flags to the driver AND cause the sandbox to bind ~/.aws — but
ONLY when Bedrock is actually active, so a subscription run never exposes AWS creds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


@pytest.fixture()
def _launcher(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT",
                       str(_HARNESS.parent / "targets/atlas/target_experiment.yaml"))
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import launch_ab_batch as LB
    return LB


class _Args:
    model = "m"; effort = "high"; max_rounds = 2; max_rate_limit_waits = 8; round_timeout = 100
    experiment = "full"; skip_hidden = False; sandbox = "bwrap"
    provider = "subscription"; aws_region = "us-east-1"; aws_profile = ""


def test_subscription_provider_is_a_noop_flag(_launcher):
    cmd = _launcher._arm_cmd("merlin_rtlchecks", "rid", _Args())
    assert "--provider" not in cmd            # subscription = unchanged claude --print path


def test_bedrock_provider_threads_aws_routing(_launcher):
    a = _Args(); a.provider = "bedrock"; a.aws_profile = "merlin-bedrock"
    cmd = _launcher._arm_cmd("merlin_rtlchecks", "rid", a)
    assert cmd[cmd.index("--provider") + 1] == "bedrock"
    assert cmd[cmd.index("--aws-region") + 1] == "us-east-1"
    assert cmd[cmd.index("--aws-profile") + 1] == "merlin-bedrock"


def test_sandbox_binds_aws_only_under_bedrock(monkeypatch):
    from merlin.targetgen.sandbox import bwrap as BW
    aws = Path(os.path.expanduser("~/.aws"))
    # subscription (no Bedrock env): never bind ~/.aws
    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    assert str(aws) not in BW.claude_runtime_binds()
    # bedrock active: bind ~/.aws iff it exists on this host
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    binds = BW.claude_runtime_binds()
    if aws.exists():
        assert str(aws) in binds
    else:
        assert str(aws) not in binds          # nothing to bind, but must not crash
