"""An empty graded cohort must never read as a pass.

`capsule_runner.main` scored the suite with `npass == len(results)`, which is vacuously TRUE at zero:
a `--capsules-root` that matched nothing printed "0/0 pass" and exited 0, byte-identical to a suite
that ran and passed everything. That is the same fail-open shape as a self-check that skips and
reports success -- the run looks green while nothing was measured.

The concrete way in: `discover_capsules` walks with `Path.rglob`, which does NOT descend into
symlinked directories. A cohort root assembled out of symlinks to real capsule dirs -- the obvious
way to build an ad-hoc subset -- therefore discovers zero capsules and, before this, exited 0.
"""
from __future__ import annotations

from pathlib import Path

from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.capsule_runner import main as runner_main


def test_rglob_does_not_follow_symlinked_capsule_dirs(tmp_path: Path):
    """Pins the mechanism, so the reason an empty cohort happens stays documented and detected."""
    real = tmp_path / "real" / "C0_probe"
    real.mkdir(parents=True)
    (real / "capsule.yaml").write_text("name: C0_probe\n", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "C0_probe").symlink_to(real, target_is_directory=True)

    # the capsule is reachable through the symlink, but the recursive walk does not descend into it
    assert (linked / "C0_probe" / "capsule.yaml").is_file()
    assert list(linked.rglob("capsule.yaml")) == []
    assert discover_capsules(linked, labels=None) == []


def test_empty_cohort_refuses_a_verdict(tmp_path: Path, capsys):
    """A root that matches nothing exits non-zero and says so, instead of reporting 0/0 pass."""
    empty = tmp_path / "empty_root"
    empty.mkdir()
    rc = runner_main(["--package", str(tmp_path / "unused_pkg"),
                      "--capsules-root", str(empty),
                      "--contract", "merlin/contract",
                      "--target", "gemmini"])
    assert rc != 0, "an empty cohort must not exit 0"
    err = capsys.readouterr().err
    assert "empty cohort" in err
    # the message has to name the symlink mechanism, because that is how this is reached in practice
    assert "symlink" in err


def test_neither_capsule_nor_root_is_rejected(tmp_path: Path):
    """Omitting both selectors used to reach `Path(None)`; argparse should reject it up front."""
    import pytest
    with pytest.raises(SystemExit) as e:
        runner_main(["--package", str(tmp_path / "unused_pkg"), "--target", "gemmini"])
    assert e.value.code != 0


def test_empty_outputs_are_not_an_exact_match():
    """Two empty output sets must not compare equal.

    `all(...)` over an empty mapping is vacuously True, so a kernel that stored NOTHING, graded
    against a reference that recorded nothing, read as bit-exact. Same failure shape as the empty
    cohort above: absence of evidence presenting itself as evidence.
    """
    from merlin.targetgen.capsule_runner import _exact_match

    assert _exact_match({}, {}) is False
    assert _exact_match({}, {"Y": [1, 2]}) is False
    assert _exact_match({"Y": [1, 2]}, {}) is False
    # and it still matches what it should
    assert _exact_match({"Y": [1, 2]}, {"Y": [1, 2]}) is True
    assert _exact_match({"Y": [1, 2]}, {"Y": [1, 3]}) is False
