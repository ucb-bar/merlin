"""A capsule counts as passed only when its cycle-accurate cert tier passes.

The screen may eliminate; it may never certify. That was the stated design, and the corpus
materializer quietly broke it: `public_capsules_for` capped every capsule's `required_oracle_tiers` to
the per-round LOOP tier (the cheapest one), so on a target whose capsules declare verilator the
materialized corpus asked only for spike. Two consequences, neither visible in any score:

  * a capsule that passed the functional sim was recorded `pass` with the RTL tier never run;
  * `_cycle_accurate_checkpoint_enabled` asks whether any cert tier is MANDATORY in the pilot corpus,
    found none, and skipped the end-of-run RTL barrier entirely -- so the elaborated RTL never ran at
    all, in the loop or after it.

The ceiling exists to avoid REQUIRING a tier the endpoint cannot reach. These pin it to exactly that.
"""
from __future__ import annotations

import sys

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.contract.materialize import _cap_required

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))


def test_capping_keeps_a_reachable_cert_tier():
    kept, unreachable = _cap_required(["L0", "L1", "L2", "L3"], {"L0", "L1", "L2", "L3"})
    assert kept == ["L0", "L1", "L2", "L3"] and unreachable == []


def test_capping_drops_only_what_cannot_be_reached():
    """An UNREACHABLE tier is still dropped -- that is what the ceiling is for."""
    kept, unreachable = _cap_required(["L0", "L1", "L2", "L5"], {"L0", "L1", "L2", "L3"})
    assert kept == ["L0", "L1", "L2"] and unreachable == ["L5"]


def test_capping_never_substitutes_a_cheaper_tier():
    """When nothing survives, the result is EMPTY -- never back-filled with the ceiling."""
    kept, unreachable = _cap_required(["L4", "L5"], {"L0", "L1", "L2"})
    assert kept == [] and unreachable == ["L4", "L5"]


@pytest.mark.parametrize("target", ["gemmini", "atlas"])
def test_the_pilot_corpus_demands_a_cycle_accurate_tier(target, monkeypatch):
    """End-to-end on the real descriptors: the materialized pilot set each run grades must still ask for
    a tier above the screen, and the RTL barrier must therefore arm."""
    desc = repo_root() / f"merlin/experiments/capsule_bench/targets/{target}/target_experiment.yaml"
    if not desc.is_file():
        pytest.skip(f"no descriptor for {target}")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(desc))
    for mod in [m for m in list(sys.modules) if m in ("_common", "run_baseline_qa_loop")]:
        del sys.modules[mod]
    monkeypatch.setattr(sys, "argv", ["x"])
    import run_baseline_qa_loop as L

    caps = list((L._pilot_subset()).rglob("capsule.yaml")) if hasattr(L._pilot_subset(), "rglob") \
        else list(__import__("pathlib").Path(L._pilot_subset()).rglob("capsule.yaml"))
    assert caps, "pilot subset is empty"
    declared = set()
    for cf in caps:
        declared |= set((yaml.safe_load(cf.read_text()) or {}).get("required_oracle_tiers") or [])
    assert declared - {"L0", "L1", "L2"}, (
        f"{target}: the pilot corpus demands nothing above the screen tier ({sorted(declared)}) -- a "
        f"functional-sim pass would count as done and the RTL barrier would not arm")

    armed, why = L._cycle_accurate_checkpoint_enabled()
    assert armed, f"{target}: cycle-accurate barrier disarmed: {why}"
