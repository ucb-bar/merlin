"""C2: any target goes on the shared bench spine via its declarative descriptor — no per-target wiring.
The gemmini experiment (as an example) builds a BenchTargetSpec from its descriptor + the capsule runner,
and the shared discovery loop finds its corpus."""
from __future__ import annotations

from merlin.benchharness.from_descriptor import spec_from_experiment
from merlin.benchharness.spec import BenchTargetSpec
from merlin.targetgen import capsule_runner
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.common.paths import repo_root


def _gemmini_te():
    return load_target_experiment(
        repo_root() / "merlin/experiments/gemmini_capsule_bench_v0/target_experiment.yaml")


def test_spec_built_from_descriptor():
    spec = spec_from_experiment(_gemmini_te(), capsule_runner)
    assert isinstance(spec, BenchTargetSpec)
    assert spec.name == "Gemmini" and spec.perf_tier == "L2"
    assert spec.corpus_root.name == "isa"                    # from the descriptor's capsule_corpus
    assert spec.perf_fields({"cycles": 241}) == {"cycles": 241}


def test_shared_discovery_finds_the_corpus():
    """The shared spine's discover() (not a gemmini-specific path) enumerates the target's capsules."""
    spec = spec_from_experiment(_gemmini_te(), capsule_runner)
    caps = spec.discover()
    assert caps and any(c.get("name", "").startswith(("A", "B")) for c in caps)  # ISA capsules present
