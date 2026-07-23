"""P1: the per-target bundle generator emits the 4-arm ladder from a descriptor — target-agnostic tool
blocks + descriptor-derived target-specific paths — and reproduces the hand-authored gemmini bundles."""
from __future__ import annotations

import yaml
from pathlib import Path

from merlin.targetgen.target_experiment import load_target_experiment
from merlin.targetgen.generate_bundles import generate_bundles, _CPP_ALLOW, _XDSL_ALLOW
from merlin.common.paths import repo_root


def _te():
    return load_target_experiment(
        repo_root() / "merlin/experiments/gemmini_capsule_bench_v0/target_experiment.yaml")


def _sets(m):
    return ({e["path"] for e in m.get("allowed", [])}, {e["path"] for e in m.get("denied", [])})


def test_generates_the_four_arms():
    gen = generate_bundles(_te())
    assert set(gen) == {"raw_baseline_hwbringup_v0", "cpp_merlininfra_hwbringup_v0",
                        "merlin_assisted_hwbringup_v0", "merlin_assisted_rtlchecks_hwbringup_v0"}


def _norm(paths):
    # normalize the stale bare 'artifacts/' (hand-authored) vs the correct 'out/artifacts/' (generated)
    return {p[4:] if p.startswith("out/artifacts/") else p for p in paths}


def test_reproduces_hand_authored_gemmini_bundles():
    """Parity: the generated allow-set matches the hand-authored one, and the hand deny-set is a subset
    of the generated (the generator may add a safe extra prior-backend deny; the answer-surface prefix
    is normalized since the generator fixes the hand-authored stale 'artifacts/' -> 'out/artifacts/')."""
    gen = generate_bundles(_te())
    B = repo_root() / "merlin/experiments/gemmini_capsule_bench_v0/input_bundles"
    for bid, gm in gen.items():
        hand = yaml.safe_load((B / bid / "input_bundle_manifest.yaml").read_text())
        ga, gd = _sets(gm)
        ha, hd = _sets(hand)
        assert ga == ha, f"{bid} allow drift: gen-only={ga-ha} hand-only={ha-ga}"
        assert _norm(hd) <= _norm(gd), f"{bid} deny missing from generated: {_norm(hd)-_norm(gd)}"


def test_agnostic_tool_blocks_have_no_target_name():
    """The per-rung tool blocks are literal merlin/python paths — no target name, for any target."""
    for p in _CPP_ALLOW + _XDSL_ALLOW:
        assert p.startswith("merlin/python/merlin/") and "gemmini" not in p


def test_increasing_help_gradient():
    """arm1 ⊂ arm2/arm3 tools; arm4 = arm3 tools + the CIRCT rtl generators + the rtl_facts pin."""
    gen = generate_bundles(_te())
    raw_a, _ = _sets(gen["raw_baseline_hwbringup_v0"])
    cpp_a, _ = _sets(gen["cpp_merlininfra_hwbringup_v0"])
    mer_a, _ = _sets(gen["merlin_assisted_hwbringup_v0"])
    rtl_a, _ = _sets(gen["merlin_assisted_rtlchecks_hwbringup_v0"])
    py = "merlin/python/merlin/"
    assert not any(p.startswith(py) for p in raw_a)                    # arm1: no merlin tools
    assert f"{py}targetgen/generate/mlir_scaffold.py" in cpp_a         # arm2: C++ generators
    assert f"{py}kernels/cca_contract.py" in mer_a and f"{py}targetgen/rtl_backend.py" in mer_a  # arm3: CCA spine
    assert f"{py}targetgen/rtl/" in rtl_a and rtl_a > mer_a            # arm4: + CIRCT rtl, superset of arm3


def test_target_specific_paths_come_from_descriptor():
    te = _te()
    gen = generate_bundles(te)
    mer_a, _ = _sets(gen["merlin_assisted_hwbringup_v0"])
    assert te.corpus_rel() in mer_a and all(h in mer_a for h in te.isa_headers)
    _, rtl_d = _sets(gen["merlin_assisted_hwbringup_v0"])
    assert te.rtl_facts_pin in rtl_d and f"merlin/targets/{te.target}/" in te.rtl_facts_pin
