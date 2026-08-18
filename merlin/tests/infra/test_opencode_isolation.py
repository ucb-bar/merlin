"""The opencode driver must run under the same sandbox the other drivers do.

Two things stood between opencode and `--sandbox bwrap`. Its launcher resolves (the `~/.nvm` bind covers
it) but its state lives under `~/.local/share/opencode`, which nothing binds — so inside the box it
starts with no data dir. And binding the real one would expose `~/.local/share/opencode/storage`, every
prior opencode session on the host, for the same reason `~/.codex/sessions` is an answer-leak surface.

Running the open-weight models at `--sandbox none` while the Codex arm ran under bwrap would also make
the comparison asymmetric in its isolation strength, which is a caveat on every number that follows.
So the driver gets a per-run isolated data home, exactly as the Codex driver does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

HARNESS = repo_root() / "merlin/experiments/capsule_bench/harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))


@pytest.fixture()
def binds(tmp_path):
    sys.argv = ["x"]
    import opencode_agent as OC
    return OC.opencode_runtime_binds(tmp_path / "home"), tmp_path / "home"


def test_the_data_home_is_writable_and_relocated(binds):
    args, home = binds
    assert home.is_dir(), "the isolated data home must exist before bwrap binds it"
    assert "--bind" in args, "the data home must be WRITABLE (opencode writes state there)"
    assert args[args.index("--setenv") + 1] == "XDG_DATA_HOME"
    assert args[args.index("--setenv") + 2] == str(home)


def test_the_real_data_dir_is_never_bound(binds):
    """The agent must not receive the host's opencode state — that is where prior sessions live."""
    args, home = binds
    real = str(Path.home() / ".local" / "share" / "opencode")
    assert not any(a == real or a.startswith(real + "/") for a in args), (
        f"the real opencode data dir {real} is bound into the sandbox; prior sessions would be readable")


def test_each_round_gets_its_own_home(tmp_path):
    """Round-scoped, like the Codex home: state must not leak between rounds or runs."""
    sys.argv = ["x"]
    import opencode_agent as OC
    a = OC.opencode_runtime_binds(tmp_path / "run_r00")
    b = OC.opencode_runtime_binds(tmp_path / "run_r01")
    assert a != b and (tmp_path / "run_r00").is_dir() and (tmp_path / "run_r01").is_dir()
