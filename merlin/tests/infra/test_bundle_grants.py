"""A bundle GRANT must name something the workspace can DELIVER — and tracked bytes must stay portable.

The bundle manifest is the claim about what an arm can read; the stager is the delivery. When they
disagree nothing fails: the run succeeds with the arm credited for a tool it never carried, and the
number it produces gets cited. Two spellings of the same defect were live before this gate:

  * a grant naming a path that is not there (measured: five of six targets were assembled without the
    RTL-facts grant that DEFINES the CIRCT rung, and four targets named at least one further dead grant,
    one of them a target's ISA headers);
  * a grant that resolves only through a TRACKED symlink storing an absolute machine path. It delivers
    on the machine that made the link and dangles on every clone after — and a dangling entry is skipped
    exactly as silently as an absent one, so the check "the path exists" passes for the author forever.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import bundle_grants as BG
from merlin.targetgen.target_experiment import load_target_experiment

GATE = repo_root() / "build_tools" / "scripts" / "check_bundle_grants.py"


def _descriptors() -> list[Path]:
    return sorted(merlin_dir().glob("experiments/*/targets/*/target_experiment.yaml"))


def test_every_grant_of_every_arm_of_every_target_resolves():
    """The whole point. A grant is OK, or it is a generated artifact the tooling produces on demand —
    nothing else."""
    dead: list[tuple[str, str, str]] = []
    for d in _descriptors():
        te = load_target_experiment(d)
        for bundle_id, entries in BG.audit(te).items():
            dead += [(te.target, bundle_id, p) for p, status in entries if status == BG.MISSING]
    assert not dead, ("grants that deliver nothing (the manifest claims a tool the arm cannot read):\n"
                      + "\n".join(f"  {t}/{b}: {p}" for t, b, p in dead))


def test_the_audit_actually_looked_at_something():
    """A check that reviewed nothing reports success too. Pin the coverage, not just the verdict."""
    assert _descriptors(), "no target descriptors found — the audit above proved nothing"
    total = sum(BG.grant_count(load_target_experiment(d)) for d in _descriptors())
    assert total > 100, f"only {total} grants reviewed; the audit is not covering the ladder"


def test_no_tracked_symlink_stores_a_machine_path():
    """Committed bytes must not name one machine's filesystem. Content that genuinely lives outside the
    repo is reached through ``third_party/ext/<name>``, whose own target comes from ``.env``."""
    bad = BG.tracked_escaping_links()
    assert not bad, ("tracked symlinks storing a path outside the repo:\n"
                     + "\n".join(f"  {k}\n      -> {v}" for k, v in sorted(bad.items()))
                     + "\n  Point them at third_party/ext/<name>/... and set MERLIN_EXT_<NAME> in .env.")


def test_a_portable_link_through_the_externals_root_is_not_flagged():
    """The distinction the gate turns on. A repo-relative link is portable even when what it finally
    reaches is an external checkout — that is a SETUP fact, not a defect in the tree. Judging by
    ``resolve()`` instead would condemn the correct spelling exactly as hard as the broken one."""
    root = repo_root()
    inside = root / "third_party" / "ext" / "some-name"
    # a relative link that lands inside the repo, even on a path that is itself an external mount point
    probe = root / ".pytest-portable-link-probe"
    try:
        probe.symlink_to(os.path.relpath(inside, root))
        assert not BG._link_leaves_repo(probe)
    finally:
        probe.unlink(missing_ok=True)
    # ... and the absolute spelling of the very same destination IS flagged
    probe2 = root / ".pytest-machine-link-probe"
    try:
        probe2.symlink_to(inside)
        assert BG._link_leaves_repo(probe2)
    finally:
        probe2.unlink(missing_ok=True)


def test_the_gate_fails_on_a_dead_grant():
    """The gate has to actually catch the regression it exists for. Asserted by feeding it one, rather
    than by trusting that it would."""
    te = load_target_experiment(_descriptors()[0])
    real = BG.audit(te)
    assert all(s != BG.MISSING for v in real.values() for _, s in v), "fixture target is already clean"

    status, where = BG.resolve("merlin/no/such/place/at/all/", derive=False)
    assert (status, where) == (BG.MISSING, None)

    # and the classifier must not launder it into the benign bucket: it is not a shape the tooling
    # produces, so there is nothing to derive and the only honest answer is "this delivers nothing".
    assert not BG.is_generated_shape("merlin/no/such/place/at/all/")
    assert BG.derive_grant("merlin/no/such/place/at/all/") is None


def test_a_generated_grant_is_classified_generated_not_missing():
    """RTL facts are DERIVED from the target's RTL and gitignored on purpose; reporting their absence as
    a dead grant would push the reader toward committing them, which is the wrong fix."""
    te = load_target_experiment(_descriptors()[0])
    status, _ = BG.resolve(te.rtl_facts_pin, derive=False)
    assert status in (BG.OK, BG.DERIVED)
    assert BG.is_generated_shape(te.rtl_facts_pin)
    assert not BG.is_generated_shape("merlin/contract/")


@pytest.mark.skipif(not GATE.is_file(), reason="gate script absent")
def test_the_gate_script_passes_on_this_tree():
    r = subprocess.run([sys.executable, str(GATE)], capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "0 undeliverable" in r.stdout
