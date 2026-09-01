"""Generalization proof: a SECOND SIMT target plugs into the onboarding seams with ZERO edits to shared
merlin code — only the public registration APIs + a synthesized contract. This extends the RG4 proof
(``test_muon_runtime_abi`` — a perturbed runtime_abi fact flowing into the preamble/harness) to the three
seams generalized for the atlas+radiance run readiness:

  * G2 — the SIMT fact-bundle registry (``mlc_bridge._SIMT_INTROSPECTS`` / ``register_simt_introspect``);
  * G1 — the contract-derived ``RunnerConfig`` (``runner_config_from_manifest``);
  * G3 — the sim-engine oracle registry (``capsule_runner._SIM_ORACLES`` / ``register_sim_oracle``),
          including the exclusive-precedence rule (a self-hosted SIMT core's own sim wins over the
          ``external_backend`` program-oracle default).

A second SIMT core would arrive as: a descriptor + a residual + a registered introspect (its RTL facts)
+ a registered sim oracle (its simulator). None of the assertions below touch shared dispatch code — they
register through the public seams and observe that resolution tracks the synthetic target.
"""
from __future__ import annotations

import types

from merlin.targetgen.rtl import mlc_bridge as MB
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.runner_config import runner_config_from_manifest
from merlin.targetgen.target_experiment import CapabilityManifest

FAKE = "warpcore2"


def _fake_simt_introspect():
    m = types.ModuleType("fake_simt_introspect")
    m.TARGET = FAKE
    # facts.facts.<name> is a FLAT dict (the real muon introspect shape: _simt_fact_bundle reads
    # fields[name]['value'] = facts['facts'][name] directly, no extra wrapper).
    m.build_facts = lambda: {
        "generator": {"name": "fake", "method": "synthetic SIMT introspect"},
        "inputs": {"rtl_present": True},
        "facts": {
            # `state` is REQUIRED for a block to count as derived. The bridge used to infer that from
            # key presence, which credited a block that had merely been written -- including one whose
            # numbers were `cfg.get(name, default)` fallbacks over a config file that was never opened.
            # A fixture without it is now correctly worth nothing, so this one declares it.
            "simt": {"lanes_per_warp": 8, "warps_per_core": 4, "cores": 1,
                     "state": "derived", "evidence": "synthetic"},
            "isa": {"encoding_bits": 32, "instruction_classes": ["FOO_INVOKE", "OP"],
                    "state": "derived", "evidence": "synthetic"},
        },
    }
    return m


def test_g2_second_simt_fact_bundle_resolves_via_registry(monkeypatch):
    """G2: register a 2nd SIMT introspect; ``_simt_fact_bundle`` resolves it by its declared identity."""
    monkeypatch.setitem(MB._SIMT_INTROSPECTS, FAKE, _fake_simt_introspect())
    monkeypatch.setattr(MB, "_arc_target", lambda t: FAKE if t == FAKE else t)
    b = MB._simt_fact_bundle(FAKE)
    assert b["kind"] == "simt" and b["n_derived"] >= 1
    assert b["fields"]["simt"]["value"]["lanes_per_warp"] == 8
    # a target no registered introspect serves is still honestly empty (fail-closed, never mis-attributed)
    monkeypatch.setattr(MB, "_arc_target", lambda t: t)
    assert MB._simt_fact_bundle("nonexistent_simt")["n_derived"] == 0


def test_g1_second_simt_runner_config_tracks_the_contract():
    """G1: a perturbed SIMT contract yields a RunnerConfig reflecting ITS knobs, not a muon literal."""
    m = CapabilityManifest(
        target=FAKE, kind="simt", endpoint_kind="external_backend",
        suite=f"{FAKE}-capsule-bench", dtype="fp8", fourth_output_name=None,
        tier_sim={"L2": "wc_sim", "L3": "wc_rtl"}, rtl_tiers=("L3",),
        perf_fields=("flops",), trace_gate=None,
        force_match_policy={"compare": "float", "atol": 1e-2},
        encoding_required=False, encoding={}, contract={})
    cfg = runner_config_from_manifest(m)
    assert cfg.dtype == "fp8"
    assert cfg.tier_sim == {"L2": "wc_sim", "L3": "wc_rtl"}
    assert cfg.oracle_tiers == ("L2", "L3")            # sorted tier_sim keys, no hardcode
    assert cfg.force_match_policy["atol"] == 1e-2
    assert cfg.fourth_output_name == "kernel.cpp"      # external_backend endpoint default


def test_g3_second_simt_oracle_routes_to_its_registered_sim(monkeypatch):
    """G3: an EXCLUSIVE bespoke sim registered under its engine name wins over the external_backend
    program-oracle default — a 2nd self-hosted SIMT core is graded on its own kernel ELF, no dispatch edit."""
    sentinel = {"L2": (lambda *a, **k: None), "L3": (lambda *a, **k: None)}
    CR.register_sim_oracle("wc_sim", adapters=lambda t: sentinel,
                           available=lambda t: (True, f"{t}: wc_sim ready"), exclusive=True)
    try:
        monkeypatch.setattr(CR, "_bespoke_sim_via", lambda t: "wc_sim")
        monkeypatch.setattr(CR, "_endpoint_of", lambda t: ("external_backend", None))
        ad = CR.oracle_adapters(FAKE)
        assert ad is sentinel                          # exclusive sim replaced the program-oracle default
        ok, why = CR.oracle_available(FAKE)
        assert ok and "wc_sim" in why
    finally:
        CR._SIM_ORACLES.pop("wc_sim", None)
