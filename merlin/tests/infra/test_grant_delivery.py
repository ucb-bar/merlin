"""A granted path that does not reach the workspace must be REPORTED, never silently dropped.

``assemble_workspace`` decides what an arm can actually read. It skipped any grant whose path was not
on disk, with no record anywhere, so a manifest could state that the CIRCT arm was granted its
RTL-extracted facts while the workspace it ran in contained no such entry — and the arm would then be
credited with a tool it never carried. Measured across the declared targets, five of six had no
rtl_facts directory on disk, and four targets additionally named at least one other grant that does
not exist (one of them a target's ISA headers, which every arm is supposed to share).

The directory is gitignored on purpose: RTL facts are DERIVED from the target's RTL, not committed. So
the fix is to derive a missing generated grant rather than to commit one, and to surface anything that
still cannot be delivered.

This function had no test at all, which is how the silent skip survived.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


@pytest.fixture
def rae():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("run_agent_experiment",
                                                  HARNESS / "run_agent_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"run_agent_experiment not importable here: {type(e).__name__}: {e}")
    return mod


def test_a_grant_that_exists_lands_in_the_workspace(rae, tmp_path):
    bundle = {"arm": "merlin_rtlchecks",
              "allowed": [{"path": "merlin/python/merlin/common/"}], "denied": []}
    ws = tmp_path / "ws"
    rae.assemble_workspace(bundle, ws)
    assert (ws / "common").exists()


def test_an_undeliverable_grant_is_reported_not_silently_skipped(rae, tmp_path, capsys):
    bundle = {"arm": "merlin_rtlchecks",
              "allowed": [{"path": "merlin/does/not/exist/anywhere/"}], "denied": []}
    ws = tmp_path / "ws"
    rae.assemble_workspace(bundle, ws)
    err = capsys.readouterr().err
    assert "merlin/does/not/exist/anywhere/" in err, "an undelivered grant left no trace"
    assert "NOT in the workspace" in err
    assert "merlin_rtlchecks" in err, "the report must name the arm that was credited with it"


def test_a_missing_rtl_facts_grant_is_derived_rather_than_skipped(rae, tmp_path, capsys):
    """The generated-not-committed case. Skips only where derivation genuinely cannot run here, and
    says so — it never passes by treating an underivable artifact as success."""
    from merlin.common.paths import repo_root
    from merlin.targetgen import target_experiment as TE

    root = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    pins = []
    for d in sorted(root.iterdir()):
        p = d / "target_experiment.yaml"
        if p.is_file():
            pins.append(TE.load_target_experiment(p).rtl_facts_pin)
    if not pins:
        pytest.skip("no target descriptors in this checkout")
    from merlin.common.paths import targets_dir

    before = {p.name for p in targets_dir().iterdir()} if targets_dir().is_dir() else set()
    delivered, created = [], []
    for pin in pins:
        existed = (repo_root() / pin).is_dir()
        ws = tmp_path / pin.replace("/", "_")
        rae.assemble_workspace({"arm": "merlin_rtlchecks", "allowed": [{"path": pin}],
                                "denied": []}, ws)
        delivered.append((ws / "rtl_facts").exists())
        if not existed and (repo_root() / pin).is_dir():
            created.append(pin)
    assert not created, (
        f"staging a workspace populated {created} in the tree. Doing so re-points every other "
        f"consumer: _committed_facts_path prefers a populated pin over the purgeable cache, so this "
        f"silently changes what manifest derivation reads repo-wide.")

    # And it must not create the PARENT either. facts.target_base returns merlin/targets/<t> whenever
    # that directory exists and the generated artifacts/targets/<t> home otherwise, so an empty
    # merlin/targets/<t> left behind by an mkdir shadows the generated home and every residual, plan
    # and contract under it vanishes. Measured: four empty directories took 27 tests red across the
    # manifest, routing, onboarding and capacity suites, from nothing but staging a sandbox.
    after = {p.name for p in targets_dir().iterdir()} if targets_dir().is_dir() else set()
    assert after == before, (
        f"staging created {sorted(after - before)} under {targets_dir()}; an empty target directory "
        f"shadows the generated target home")
    if not any(delivered):
        pytest.skip("no target's RTL facts are derivable in this environment")
    # Not every target's facts are derivable everywhere -- some need an RTL checkout this machine may
    # not have. The invariant is not "all delivered"; it is that a grant is EITHER delivered OR named
    # in the report. Silence is the failure.
    err = capsys.readouterr().err
    for pin, got in zip(pins, delivered):
        assert got or pin in err, f"{pin} was neither delivered nor reported"


def test_the_deriver_only_claims_paths_it_understands(rae, tmp_path):
    """It must not try to produce arbitrary missing grants — only the generated shape it knows."""
    assert rae._derive_missing_grant(tmp_path / "some" / "random" / "path") is None
    assert rae._derive_missing_grant(tmp_path / "contracts" / "not_rtl_facts") is None
