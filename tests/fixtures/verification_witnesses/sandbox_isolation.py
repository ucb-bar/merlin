from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import sim_toolchain

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface, answer_surfaces, audit_tokens, weight_files
from merlin.targetgen.sandbox.preflight import probe_sandbox

_SANDBOX = probe_sandbox()
_NO_SANDBOX = None if _SANDBOX.usable else _SANDBOX.describe()


def remove_bundle_snapshot(ws: Path) -> None:
    """Tear down this test's immutable scratch snapshot without chmodding shared blobs."""
    root = BW.bundle_snapshot_root(ws)
    if root.is_symlink():
        raise RuntimeError(f"refusing to remove symlinked bundle snapshot: {root}")
    if root.exists():
        BW._make_snapshot_writable(root)
        shutil.rmtree(root)


@pytest.fixture(autouse=True)
def _sim_family_policy_without_its_checkout(monkeypatch, tmp_path_factory):
    """Every check here replays a mount table or a probe list; none launches the RTL-sim tools.

    A roster target's sim family refuses to resolve without its configured checkout, which a hosted
    runner never has, so an absent one is stood in by an empty root (see ``sim_toolchain``)."""
    sim_toolchain.stand_in_absent_chipyard(monkeypatch, tmp_path_factory.mktemp("absent-chipyard"))


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "AnswerSurface",
    "BW",
    "_NO_SANDBOX",
    "_SANDBOX",
    "_sim_family_policy_without_its_checkout",
    "answer_surfaces",
    "audit_tokens",
    "probe_sandbox",
    "remove_bundle_snapshot",
    "pytest",
    "sim_toolchain",
    "subprocess",
    "weight_files",
]
