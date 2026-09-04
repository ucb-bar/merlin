"""An arm whose deny list CONTAINS its workspace must still be able to enter it.

`base_argv` binds the workspace after the deny layer precisely because a deny can cover it: the
unassisted arm is withheld from `merlin/`, and every run's workspace lives at
`merlin/experiments/.../_qa_ws/<run>/workspace`. When the frozen-snapshot layer began re-appending the
bundle mounts to reassert grants over later toolchain binds, it re-appended the DENIES too -- putting
the tmpfs back on top of the workspace bind.

bwrap then died with `Can't chdir to <ws>: No such file or directory` before the agent ran a single
turn. The loop did not report a setup failure: it graded the empty submission, scored 0, and repeated
for five rounds in under three minutes, which reads exactly like an arm that cannot do the task.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap


@pytest.fixture
def no_snapshot(monkeypatch):
    """Bypass snapshot verification: this is about MOUNT ORDER, not about frozen bytes."""
    def _grants(ws, bundle, repo):
        rows = [(str(e["path"]), Path(repo) / e["path"], Path(repo) / e["path"])
                for e in bundle.get("allowed", [])]
        return {}, rows
    monkeypatch.setattr(bwrap, "_snapshot_grants", _grants)


def _order(argv, ws):
    w = str(ws)
    last_mask = max((i for i, a in enumerate(argv)
                     if a == "--tmpfs" and w.startswith(argv[i + 1].rstrip("/"))), default=-1)
    last_bind = max((i for i, a in enumerate(argv)
                     if a == "--bind" and argv[i + 1] == w), default=-1)
    return last_mask, last_bind


def test_a_deny_containing_the_workspace_does_not_mask_it(no_snapshot, tmp_path):
    repo = tmp_path
    ws = repo / "merlin" / "experiments" / "x" / "_qa_ws" / "run" / "workspace"
    ws.mkdir(parents=True)
    (repo / "merlin" / "contract").mkdir(parents=True)
    bundle = {"allowed": [{"path": "merlin/contract", "mode": "ro"}],
              "denied": [{"path": "merlin", "reason": "unassisted arm"}]}
    argv = bwrap.reapply_bundle_snapshot(["seed"], ws, bundle, repo=repo)
    mask, bind = _order(argv, ws)
    assert mask >= 0, "fixture no longer produces a deny covering the workspace; the test proves nothing"
    assert bind > mask, "the workspace bind must come AFTER any deny that covers it (bwrap is ordered)"


def test_denies_still_win_over_their_own_grants(no_snapshot, tmp_path):
    """Not vacuous: re-asserting the workspace must not weaken the deny layer itself."""
    repo = tmp_path
    ws = repo / "ws"
    ws.mkdir()
    (repo / "pkg" / "secret").mkdir(parents=True)
    bundle = {"allowed": [{"path": "pkg", "mode": "ro"}],
              "denied": [{"path": "pkg/secret", "reason": "answer surface"}]}
    argv = bwrap.reapply_bundle_snapshot(["seed"], ws, bundle, repo=repo)
    grant = max(i for i, a in enumerate(argv) if a == "--ro-bind" and argv[i + 2] == str(repo / "pkg"))
    deny = max(i for i, a in enumerate(argv) if a == "--tmpfs" and argv[i + 1] == str(repo / "pkg/secret"))
    assert deny > grant, "a denied subpath must still be applied after the grant it sits inside"


def test_a_broad_deny_does_not_eat_a_more_specific_grant(no_snapshot, tmp_path):
    """The same ordering bug, one level out: a deny must not swallow an allow it merely CONTAINS.

    Measured on the atlas launch of 2026-09-04. The unassisted arm grants the shared hardware spec
    (the contract dir, the ISA definition, the green card) under `merlin/experiments/...` and denies
    `merlin/` as "internals". Emitting every allow and then every deny let that broad deny tmpfs-wipe
    all three: inside the box the workspace symlinks for the ISA dangled, `find . -type f` returned
    five files, and the agent -- whose task file says to derive every encoding from those files and
    never invent one -- invented one instead, over 29 builds in 2h08m. No gate saw it: the grant
    resolves perfectly on the host, which is the question the other checks ask.
    """
    repo = tmp_path
    ws = repo / "ws"
    ws.mkdir()
    isa = repo / "merlin" / "experiments" / "spec" / "isa_definition.py"
    isa.parent.mkdir(parents=True)
    isa.write_text("opcode = 0b1110111\n", encoding="utf-8")
    (repo / "merlin" / "python").mkdir(parents=True)
    bundle = {"allowed": [{"path": "merlin/experiments/spec", "mode": "ro"}],
              "denied": [{"path": "merlin", "reason": "internals"}]}
    argv = bwrap.reapply_bundle_snapshot(["seed"], ws, bundle, repo=repo)
    deny = max(i for i, a in enumerate(argv)
               if a == "--tmpfs" and argv[i + 1] == str(repo / "merlin"))
    grant = max(i for i, a in enumerate(argv)
                if a == "--ro-bind" and argv[i + 2] == str(repo / "merlin" / "experiments" / "spec"))
    assert grant > deny, (
        "a grant DEEPER than a deny must be applied after it (longest-prefix-wins); "
        "otherwise the broad deny silently withholds the spec the arm was promised")
