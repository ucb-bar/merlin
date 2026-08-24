"""The holdout-leak check must not pass when it cannot see the holdouts.

The check derives every held-out capsule name by walking `hidden/*/capsule.yaml`, then looks for those
names inside the tree the bundles grant read-only. A preflighted run answer-locks the holdout store
(chmod 000) — correctly, since the agent must not read it — but that also makes the walk return nothing,
and "no names" was treated as "nothing to check" and reported PASS.

So the check went green precisely when it ran, and its success was indistinguishable from its blindness.
That matters more here than almost anywhere: this is the check that exists to stop a held-out spec
reaching the granted tree, and a real leak of exactly that kind was found in this corpus — a
generalization result measured against readable specs had to be withdrawn.
"""
from __future__ import annotations

import os

import pytest

from merlin.common.paths import repo_root

vnc = pytest.importorskip("verify_no_cheat")


def test_an_unreadable_holdout_store_is_not_a_pass(tmp_path, monkeypatch):
    """Existence survives chmod 000 even though listing does not, which is what tells the cases apart."""
    caps = tmp_path / "merlin" / "contract" / "capsules"
    hidden = caps / "radiance" / "hidden" / "RH0"
    hidden.mkdir(parents=True)
    (hidden / "capsule.yaml").write_text("name: RH0\n", encoding="utf-8")
    monkeypatch.setattr(vnc, "REPO", tmp_path, raising=False)

    store = caps / "radiance" / "hidden"
    ok_before, _ = vnc.check_holdout_not_specified()
    os.chmod(store, 0o000)
    try:
        ok_locked, detail = vnc.check_holdout_not_specified()
    finally:
        os.chmod(store, 0o755)

    assert not ok_locked, "an unreadable holdout store must fail closed, not report nothing to check"
    assert any("could not be read" in d for d in detail)


def test_a_genuinely_absent_holdout_store_still_passes(tmp_path, monkeypatch):
    """Fail-closed must not mean fail-always: a checkout with no holdouts is a legitimate state, and is
    what a public clone looks like."""
    caps = tmp_path / "merlin" / "contract" / "capsules" / "radiance" / "isa"
    caps.mkdir(parents=True)
    monkeypatch.setattr(vnc, "REPO", tmp_path, raising=False)
    ok, detail = vnc.check_holdout_not_specified()
    assert ok
    assert any("nothing to check" in d for d in detail)


def test_the_real_repo_reports_a_definite_answer():
    """Whatever the live tree's state, the check must not be silently vacuous on it."""
    ok, detail = vnc.check_holdout_not_specified()
    assert isinstance(ok, bool) and detail, "the check must say something about what it examined"
    if not ok:
        assert any("could not be read" in d or "holdout" in d.lower() for d in detail)
