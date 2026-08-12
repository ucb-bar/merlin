"""P4 — the headline cross-target proof: the 4-arm agentic ladder generalizes beyond gemmini.

Real, committed target descriptors drive this — nothing gemmini-specific and nothing pre-baked:

  * ATLAS    — an NPU with a systolic MXU. A ready mlc arc-matmul target: its ISA/DIM/memory-map are
               DERIVED from RTL via mlc discovery, and its RTL oracle is the mlc arc MXU model
               (arc_available=True). This is the real 2nd target the plan calls for.
  * RADIANCE — the RadianceMuon / Muon SIMT config. A real, matmul-capable target (SIMT tensor-core +
               an embedded gemmini-mx MX PE; derived contract materialized at
               out/artifacts/targets/radiance) that exercises a DIFFERENT oracle path than an arc
               MXU: its RTL oracle is the bespoke cyclotron perf model (muon_oracles.cyclotron_adapter,
               toolchain.sim_via=cyclotron), and its facts come from muon_introspect, NOT the arc
               decoder. So arc_available=False is by design (radiance is not an mlc arc-matmul target),
               not an absence of a model. This target guards against silently assuming every target is
               an arc-graded MXU while still being a genuine, validated 3rd target.
  * MX_GEMMINI — GemminiMxFPConfigs.defaultMxFPConfig: a config variant of gemmini (same RoCC custom3
               ISA, same 16x16 WS systolic mesh) whose PEs are microscaling block-scaled FP. It is the
               systolic MX PE radiance embeds. Structural profile == gemmini's; graded via the chipyard
               sim (no mlc arc model for the MX config yet). Guards that a same-generator variant plugs
               in from its RTL config without gemmini-specific harness code.

What must generalize (and is asserted below): the bundle GENERATION (the four rungs' agnostic tool
blocks are byte-identical across targets — only the small target block differs) and the increasing-help
gradient. What is legitimately per-target (the RTL oracle) degrades honestly rather than crashing.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.targetgen.generate_bundles import generate_bundles
from merlin.targetgen import rtl_backend as RB
from merlin.targetgen.rtl import mlc_bridge as B

RUNGS = ["raw_baseline", "cpp_merlininfra", "merlin_assisted", "merlin_assisted_rtlchecks"]
# The C++ scaffold generators that arm-3 (xDSL) legitimately RETIRES when it swaps modality.
CPP_SCAFFOLD = {
    "merlin/python/merlin/targetgen/generate/llvm_plan.py",
    "merlin/python/merlin/targetgen/generate/mlir_scaffold.py",
    "merlin/python/merlin/targetgen/generate/target_repo.py",
}


def _te(target):
    d = repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target / "target_experiment.yaml"
    return load_target_experiment(d)


def _paths(manifest):
    return {e["path"] for e in (manifest.get("allowed") or [])}


def _bundles(target):
    return generate_bundles(_te(target))


@pytest.mark.parametrize("target", ["atlas", "radiance", "mx_gemmini"])
def test_four_arms_generated_for_any_target(target):
    b = _bundles(target)
    assert set(b) == {f"{r}_hwbringup_v0" for r in RUNGS}
    assert b["merlin_assisted_hwbringup_v0"]["task"] == f"{target}-mlir-oot-capsule"


@pytest.mark.parametrize("target", ["atlas", "radiance", "mx_gemmini"])
def test_increasing_help_gradient(target):
    """arm1 ⊂ arm2 ⊂ arm4 additively; arm2→arm3 is the one modality swap (C++ scaffold → xDSL spine)."""
    b = _bundles(target)
    p = {r: _paths(b[f"{r}_hwbringup_v0"]) for r in RUNGS}
    # monotonic tool count across the ladder
    counts = [len(p[r]) for r in RUNGS]
    assert counts == sorted(counts) and len(set(counts)) == 4, counts
    # arm1 -> arm2: purely additive (raw scaffold gains the merlin C++ generators)
    assert p["raw_baseline"] <= p["cpp_merlininfra"]
    # arm2 -> arm3: swaps ONLY the C++ scaffold generators for the xDSL+CCA spine
    assert p["cpp_merlininfra"] - p["merlin_assisted"] == CPP_SCAFFOLD
    assert p["merlin_assisted"] - p["cpp_merlininfra"], "arm-3 must add the xDSL/CCA spine"
    # arm3 -> arm4: purely additive (gains the CIRCT/RTL-check tools)
    assert p["merlin_assisted"] <= p["merlin_assisted_rtlchecks"]


def test_tool_blocks_are_target_agnostic():
    """The AGNOSTIC tool core (the merlin/python/... paths) of every rung is byte-identical across
    targets — the compiler help a rung grants does not vary by target. Only the small target block
    (the experiment-dir-relative scripts/task/corpus paths) legitimately differs."""
    a, r = _bundles("atlas"), _bundles("radiance")
    tools = lambda m: {p for p in _paths(m) if p.startswith("merlin/python/")}
    for rung in RUNGS:
        assert tools(a[f"{rung}_hwbringup_v0"]) == tools(r[f"{rung}_hwbringup_v0"]), rung


def test_target_fact_bundle_is_derived_and_honest():
    """The static fact bundle (legal opcodes + DIM + memory map + capacities) is DERIVED per target with
    per-field provenance, and honestly reports what it cannot ground — no guessed facts. It is the
    non-labeling extraction handed to both the agent and the FileCheck compiler."""
    if not (B.mlc_available()[0] and B.arc_available("gemmini")):
        pytest.skip("mlc/gemmini not present in this checkout")
    gem = B.target_fact_bundle("gemmini")
    # legal_opcodes reads the HW dialect directly (no cache) -> stable + provenance-stamped.
    assert gem["fields"]["legal_opcodes"]["derived"] and gem["fields"]["legal_opcodes"]["source"]
    assert len(gem["fields"]["legal_opcodes"]["value"]) >= 20
    # every field carries provenance; a derived mesh_dim (cache-dependent) must be the real 16, never a guess.
    assert all({"value", "source", "derived"} <= set(f) for f in gem["fields"].values())
    md = gem["fields"]["mesh_dim"]
    assert md["value"] == 16 if md["derived"] else md["value"] is None
    # a SIMT/prototype target with no HW dialect yields an all-unavailable bundle, not fabricated facts.
    rad = B.target_fact_bundle("radiance")
    assert rad["n_derived"] == 0
    assert all(not f["derived"] and f["value"] in (None, [], 0) for f in rad["fields"].values())


def test_atlas_is_a_ready_arc_matmul_target():
    """The plan's real 2nd target: ISA/DIM DERIVED from RTL, arc MXU model is the RTL oracle."""
    if not (B.mlc_available()[0] and B.arc_available("atlas")):
        pytest.skip("mlc/atlas arc model not present in this checkout")
    prof = RB.target_profile("atlas")
    assert prof.legal_opcodes, "atlas opcodes must be discovered, not hand-listed"
    assert prof.dim == 32
    assert "spatial.dataflow" in RB.derived_levers(prof)  # a mesh target earns the dataflow lever
    assert _te("atlas").sim_via == ""                     # arc-only; no bespoke sim declared


def test_radiance_uses_the_simt_cyclotron_oracle_not_arc_mxu():
    """Radiance is a real matmul-capable SIMT target whose PERF oracle is cyclotron, with the muon
    RadianceCluster arc as its bit-exact functional tier (reached via the residual's arc_target alias).

    The bit-exact model IS registered, so ``arc_available`` is True — but radiance is SIMT, not a
    systolic MXU, so the systolic-MXU DISCOVERY leg still yields nothing on its own profile: no RoCC
    opcodes, no mesh DIM, no fabricated spatial levers. The embedding cluster's systolic geometry is
    NOT impersonated onto radiance (the arc_target alias is oracle-scoped, not applied to structural
    discovery). radiance ships a committed OOT contract (matmul-capable) + a cyclotron adapter.
    """
    # bit-exact muon-arc tier is available; the systolic-MXU discovery leg yields nothing (SIMT target).
    assert B.arc_available("radiance") is True
    prof = RB.target_profile("radiance")
    assert not prof.legal_opcodes and prof.dim is None    # SIMT: no RoCC systolic facts on its profile
    assert not RB.derived_levers(prof)                    # nothing fabricated on the arc systolic leg

    # But radiance has a real oracle: the committed cyclotron adapter (fail-closed via MuonUnavailable).
    from merlin.runtime.backends.base import get_backend
    muon_oracles = get_backend("muon").muon_oracles
    assert callable(muon_oracles.cyclotron_adapter())
    assert muon_oracles.default_adapters()                # radiance ships real L-tier oracle adapters
    assert _te("radiance").sim_via == "cyclotron"

    # And it is genuinely matmul-capable per its DERIVED contract (not degraded to nothing). The
    # target_contract.yaml is gitignored (regenerable from the residual + RTL facts), so derive it via
    # the manifest deriver rather than reading a committed generated file.
    from merlin.targetgen import capability_manifests as cm
    manifest = cm.manifest_for("radiance")
    assert "matmul" in manifest["capabilities"]["ops"]


def test_mx_gemmini_is_a_systolic_gemmini_variant_graded_by_chipyard():
    """mx-gemmini is a config variant of gemmini (same RoCC ISA + 16x16 WS mesh), differing only in the
    MX block-scaled numeric datapath. Its structural profile is gemmini's; its oracle is the chipyard
    sim (an mlc arc model for the MX config is not yet registered — honest, not a gap)."""
    te = _te("mx_gemmini")
    assert te.sim_via == "chipyard"                       # elaborates through chipyard like gemmini
    assert B.arc_available("mx_gemmini") is False         # no mlc arc model for the MX config yet
    # mx_gemmini shares gemmini's structural facts (DIM=16, RoCC custom3), but those are RTL-DERIVED, not
    # pinned in the descriptor — the descriptor must NOT carry them (cf. test_target_experiment's forbidden
    # set + test_encoding_manifest: a baked dim/isa reads as authoritative but is ignored, the overfit smell).
    import yaml
    spec = yaml.safe_load(_te("mx_gemmini").path.read_text())["hardware_spec"]
    assert "dim" not in spec and "isa" not in spec and "rtl_config" not in spec


# The full roster the cross-target proof is built on. atlas/radiance/mx_gemmini are out-of-tree targets
# discovered via MERLIN_TARGET_PATH; gemmini is the in-tree reference.
ROSTER = ("gemmini", "radiance", "atlas", "mx_gemmini")


def test_four_target_roster_loads_through_the_capability_spine(monkeypatch, tmp_path):
    """Headline spine guarantee: EVERY roster target resolves a CapabilityManifest with a compute-unit
    ``kind`` derived from its contract — no fabricated kind, no missing contract. This pins the 4/4 load
    claim (atlas + mx_gemmini were previously missing/misnamed through the spine).

    Hermetic: the OOT contracts are gitignored/regenerable, so we regenerate them from the generator
    (the source of truth) into a tmp dir rather than depend on a dev working tree. gemmini is the in-tree
    reference and resolves independent of MERLIN_TARGET_PATH."""
    from merlin.targetgen.target_experiment import load_capability_manifest
    from merlin.targetgen import families, capability_manifests as cm
    for name in ("radiance", "atlas", "mx_gemmini"):
        cm.write_oot_target(name, tmp_path / name)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(tmp_path))
    kinds = {}
    for t in ROSTER:
        m = load_capability_manifest(t)
        assert m.target == t
        assert m.kind in families.known_kinds()
        assert m.endpoint_kind in families.ENDPOINT_KINDS
        kinds[t] = m.kind
    # the derived kinds the roster is built to prove: 3 systolic MXUs + 1 SIMT tensor core
    assert kinds == {"gemmini": "systolic", "atlas": "systolic",
                     "mx_gemmini": "systolic", "radiance": "simt"}
