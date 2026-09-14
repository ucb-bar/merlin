"""The VCS oracle resolves a target's simv from that target's own facts.

It used to read a fixed list of two targets' variables for EVERY target and fall back to one SoC's simv,
so any target -- including one with no VCS build at all -- reported VCS as available on a host that had
that one binary. The variable is now ``MERLIN_<TARGET>_SIMV`` and the default is the chipyard VCS build of
the design the target itself declares.
"""
from __future__ import annotations

from merlin.targetgen import heavy_oracles as HO
from merlin.targetgen import runtime_build as RB


def test_the_simv_variable_is_derived_from_the_target():
    assert HO.simv_env_name("acme") == "MERLIN_ACME_SIMV"


def test_a_target_is_never_handed_another_targets_simv(tmp_path, monkeypatch):
    other = tmp_path / "other_simv"
    other.write_text("")
    monkeypatch.setenv(HO.simv_env_name("other_target"), str(other))
    monkeypatch.delenv(HO.simv_env_name("acme"), raising=False)
    monkeypatch.setattr(RB, "rtl_sim_config", lambda target: None)
    assert HO.vcs_simv("acme") is None
    assert HO.vcs_available("acme") is False
    assert HO.vcs_simv("other_target") == other


def test_the_default_simv_is_the_design_the_target_declares(tmp_path, monkeypatch):
    monkeypatch.delenv(HO.simv_env_name("acme"), raising=False)
    monkeypatch.setattr(RB, "rtl_sim_config",
                        lambda target: "AcmeSoCConfig" if target == "acme" else None)
    monkeypatch.setenv("MERLIN_EXT_CHIPYARD", str(tmp_path))
    assert HO.vcs_simv("acme") is None, "declared but not built"
    simv = tmp_path / "sims" / "vcs" / "simv-chipyard.harness-AcmeSoCConfig"
    simv.parent.mkdir(parents=True)
    simv.write_text("")
    assert HO.vcs_simv("acme") == simv
    assert HO.vcs_simv("a_target_declaring_nothing") is None
