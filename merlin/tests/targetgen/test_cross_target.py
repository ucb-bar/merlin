"""P4 — the headline cross-target proof: the 4-arm agentic ladder generalizes beyond gemmini.

Two real, committed target descriptors drive this — nothing gemmini-specific and nothing pre-baked:

  * ATLAS    — an NPU with a systolic MXU. A ready mlc arc-matmul target: its ISA/DIM/memory-map are
               DERIVED from RTL via mlc discovery, and its RTL oracle is the mlc arc MXU model
               (arc_available=True). This is the real 2nd target the plan calls for.
  * RADIANCE — the RadianceMuon / Muon SIMT config. A real, matmul-capable target (SIMT tensor-core +
               an embedded gemmini-mx MX PE; committed OOT contract at
               out/artifacts/targets/radiance_oot) that exercises a DIFFERENT oracle path than an arc
               MXU: its RTL oracle is the bespoke cyclotron perf model (muon_oracles.cyclotron_adapter,
               toolchain.sim_via=cyclotron), and its facts come from muon_introspect, NOT the arc
               decoder. So arc_available=False is by design (radiance is not an mlc arc-matmul target),
               not an absence of a model. This target guards against silently assuming every target is
               an arc-graded MXU while still being a genuine, validated 3rd target.

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
    d = repo_root() / "merlin" / "experiments" / f"{target}_capsule_bench_v0" / "target_experiment.yaml"
    return load_target_experiment(d)


def _paths(manifest):
    return {e["path"] for e in (manifest.get("allowed") or [])}


def _bundles(target):
    return generate_bundles(_te(target))


@pytest.mark.parametrize("target", ["atlas", "radiance"])
def test_four_arms_generated_for_any_target(target):
    b = _bundles(target)
    assert set(b) == {f"{r}_hwbringup_v0" for r in RUNGS}
    assert b["merlin_assisted_hwbringup_v0"]["task"] == f"{target}-mlir-oot-capsule"


@pytest.mark.parametrize("target", ["atlas", "radiance"])
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
    """Radiance is a real matmul-capable SIMT target, but graded via cyclotron, not the arc MXU model.

    The arc-MXU path is correctly unavailable (radiance is SIMT, not a systolic MXU) — that is by
    design, NOT a missing model: radiance ships a committed OOT contract (matmul-capable) and a
    dedicated cyclotron oracle adapter with the same signature as the arc/gemmini path.
    """
    # arc-MXU is N/A by design; the arc discovery leg yields nothing for a SIMT target.
    assert B.arc_available("radiance") is False
    prof = RB.target_profile("radiance")
    assert not prof.legal_opcodes and prof.dim is None    # facts come from muon_introspect, not arc
    assert not RB.derived_levers(prof)                    # nothing fabricated on the arc leg

    # But radiance has a real oracle: the committed cyclotron adapter (fail-closed via MuonUnavailable).
    from merlin.targetgen import muon_oracles
    assert callable(muon_oracles.cyclotron_adapter())
    assert muon_oracles.default_adapters()                # radiance ships real L-tier oracle adapters
    assert _te("radiance").sim_via == "cyclotron"

    # And it is genuinely matmul-capable per its committed OOT contract (not degraded to nothing).
    import yaml
    contract = yaml.safe_load(
        (repo_root() / "out/artifacts/targets/radiance_oot/contracts/target_contract.yaml").read_text())
    assert "matmul" in contract["capabilities"]["ops"]
