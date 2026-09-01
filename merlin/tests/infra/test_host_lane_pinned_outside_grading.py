"""The host compiler an UNGRADED compile uses must be the one the registry pins.

`compile_cli.default_package` resolves the currently certified champion, which is right for tuning and
wrong for reproducing a graded result. The hole was not hypothetical-in-principle but
hypothetical-in-timing: no fp32 package declares `publication.champion`, so the fp32 choice falls out
of `publish._rank_key`'s newest-wins tie-break over two dozen candidates. Minting one more fp32 package
silently redirects every unpinned compile and nothing prints a word. These tests hold the two halves
that close it -- the champion must agree with the pin, and every compile must record which lane it used.
"""

from __future__ import annotations

import pytest

from merlin.common.provenance import load_artifacts, verify_artifact


def _lane_artifacts() -> dict:
    return {n: a for n, a in load_artifacts().items() if n.startswith("rvv_host_lane_")}


def test_the_host_lanes_are_declared_as_tree_artifacts():
    """A compiler package is identified by ALL of its bytes -- manifest, schedule and knobs -- not by
    one file, so it is declared as a tree and hashed with the same hasher the descriptor's host_lane
    pin uses. Two hashers for one question is the drift this registry exists to prevent."""
    lanes = _lane_artifacts()
    assert lanes, "no host lane is pinned; an ungraded compile's host compiler is unreproducible"
    for name, art in lanes.items():
        assert art.kind == "tree", f"{name} must be a tree artifact, got {art.kind!r}"
        assert art.digest, f"{name} declares no digest; an artifact that certifies itself is not a pin"
        assert art.repo_relative, (
            f"{name} must resolve against the repo root; without it the same declaration verifies from "
            f"the repo root and reports 'no directory' from anywhere else")


@pytest.mark.parametrize("name", sorted(_lane_artifacts()))
def test_each_pinned_lane_is_present_and_matches_its_digest(name):
    check = verify_artifact(name)
    assert check.present, f"{name}: {'; '.join(check.gaps)}"
    assert check.matches is True, f"{name}: {'; '.join(check.gaps)}"


@pytest.mark.parametrize("dtype", ["fp32", "int8"])
def test_the_certified_champion_agrees_with_the_pinned_lane(dtype):
    """The refusal path, exercised on the real registry: if these ever disagree, default_package raises
    rather than compiling against a host lane the registry does not name."""
    from pathlib import Path

    from merlin.compile_cli import _DTYPE_STRATEGY, default_package, host_lane_pin_name

    pkg = Path(default_package(dtype))            # raises (SystemExit) on drift
    pinned = load_artifacts()[host_lane_pin_name(_DTYPE_STRATEGY[dtype])].resolve()
    assert pkg.resolve() == Path(pinned).resolve()


def test_drift_between_champion_and_pin_is_refused_not_warned():
    """Refuse rather than warn, because the failure mode is that nothing is printed at all."""
    import merlin.compile_cli as cc

    class _Sel:
        package_dir = "/nowhere/some_other_package"

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("merlin.targetgen.publish.select_champion", lambda *a, **k: _Sel())
        with pytest.raises(SystemExit) as excinfo:
            cc.default_package("fp32")
        assert "DRIFTED" in str(excinfo.value)
    finally:
        monkey.undo()


@pytest.mark.parametrize("dtype", ["fp32", "int8"])
def test_every_compile_can_record_which_lane_it_used(dtype):
    """The record a graded run and an ordinary one must share, so a result built against an unpinned
    lane is detectable afterwards rather than indistinguishable."""
    from merlin.compile_cli import default_package, host_lane_identity

    ident = host_lane_identity(default_package(dtype))
    for key in ("package", "package_sha256", "n_files", "dtype_strategy", "pinned_as"):
        assert key in ident, f"host_lane record is missing {key!r}"
    assert ident["package_sha256"], "a lane record without a digest identifies nothing"
    assert ident["pinned_as"], "this lane resolves to no declared artifact, so it is unpinned"


def test_an_unpinned_lane_warns_rather_than_refusing():
    """The registry is opt-in per lane. Requiring a pin nobody has written yet would break every dtype
    that has one package and no declaration, so an undeclared lane is a warning -- a DIFFERENT thing
    from a lane that is pinned and has drifted, which is refused."""
    import merlin.compile_cli as cc

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(cc, "host_lane_pin_name", lambda strategy: "rvv_host_lane_not_declared_at_all")
        pkg = cc.default_package("fp32")
        assert pkg, "an unpinned lane must still resolve"
    finally:
        monkey.undo()
