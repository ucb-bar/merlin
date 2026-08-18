"""The sandbox public-capsule view must stay derivable from the single source (no hand drift).

The capsule-bench sandbox reads ``scripts/full_public_capsules/``. That directory is NOT a
hand-maintained copy — it is materialized from ``merlin/contract/capsules/`` (public label, oracle
tiers capped at L2) by ``merlin.targetgen.contract.materialize``. This test asserts the committed
copy still equals a fresh materialization, so a hand-edit or a contract change that isn't
re-materialized fails CI instead of silently drifting.
"""
from __future__ import annotations

import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen.contract.materialize import materialize_public_capsules

PUB = (merlin_dir() / "experiments" / "capsule_bench" / "harness" / "full_public_capsules")
_CAPSULE_FILES = ("capsule.yaml", "capsule.interface.mlir", "golden.yaml",
                  "expected_instruction_coverage.yaml", "README.md")


def _load(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_public_capsules_match_materializer(tmp_path):
    fresh = materialize_public_capsules(tmp_path, tier_ceiling="L2")
    committed = sorted(d.name for d in PUB.iterdir() if d.is_dir())
    assert fresh == committed, (
        f"public capsule SET drifted from the contract — missing: {set(fresh)-set(committed)}, "
        f"stale: {set(committed)-set(fresh)}. Re-run: "
        f"python -m merlin.targetgen.contract.materialize {PUB}")

    for name in fresh:
        for f in _CAPSULE_FILES:
            a, b = tmp_path / name / f, PUB / name / f
            assert b.is_file(), f"committed public capsule missing {name}/{f}"
            if f in ("capsule.yaml", "golden.yaml", "expected_instruction_coverage.yaml"):
                assert _load(a) == _load(b), (
                    f"{name}/{f} drifted from the materialized view — re-materialize.")
            else:
                assert a.read_text() == b.read_text(), f"{name}/{f} drifted — re-materialize."


def test_materializer_caps_tiers_below_ceiling(tmp_path):
    materialize_public_capsules(tmp_path, tier_ceiling="L2")
    for cap_yaml in tmp_path.rglob("capsule.yaml"):
        tiers = _load(cap_yaml).get("required_oracle_tiers", [])
        assert all(t in ("L0", "L1", "L2") for t in tiers), (
            f"{cap_yaml.parent.name} requires an unreachable tier in the sandbox: {tiers}")


def test_public_capsules_for_is_target_aware_and_gemmini_parity():
    """The graded public set is DERIVED per-target from the descriptor's capsule_corpus (the target-aware
    replacement for the committed gemmini set the loop used to hardcode). gemmini must reproduce its exact
    committed 20-capsule set; atlas must yield its OWN fp8/bf16 set (disjoint names) — proving no gemmini
    leak into another target's grade."""
    from merlin.common.paths import repo_root
    from merlin.targetgen.contract.materialize import public_capsules_for
    from merlin.targetgen.target_experiment import load_target_experiment
    root = repo_root()

    te_g = load_target_experiment(root / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    gem = sorted(p.name for p in public_capsules_for(te_g).iterdir() if p.is_dir())
    committed = sorted(d.name for d in PUB.iterdir() if d.is_dir())
    assert gem == committed, f"gemmini derived set drifted from the committed set: {set(gem) ^ set(committed)}"

    te_a = load_target_experiment(root / "merlin/experiments/capsule_bench/targets/atlas/target_experiment.yaml")
    atlas = sorted(p.name for p in public_capsules_for(te_a).iterdir() if p.is_dir())
    assert atlas and not (set(atlas) & set(committed)), (
        f"atlas graded set must be disjoint from gemmini's (no leak); got overlap {set(atlas) & set(committed)}")


def test_public_capsules_for_is_concurrency_safe():
    """Many A/B arms materialize the SAME target's public set at once. The publish must be atomic (build a
    unique versioned dir, then repoint a per-target symlink) so no arm rmtrees another's half-built cache
    mid-read: every concurrent caller must see a COMPLETE corpus (equal, non-zero capsule count)."""
    from concurrent.futures import ThreadPoolExecutor
    from merlin.common.paths import repo_root
    from merlin.targetgen.contract.materialize import public_capsules_for
    from merlin.targetgen.target_experiment import load_target_experiment
    te = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    assert public_capsules_for(te).is_symlink()          # atomic-publish handle, not an in-place dir

    def worker(_):
        d = public_capsules_for(te)
        return sum(1 for _ in d.rglob("capsule.yaml"))   # full traversal blows up on a half-deleted tree
    with ThreadPoolExecutor(max_workers=8) as ex:
        counts = list(ex.map(worker, range(16)))
    assert len(set(counts)) == 1 and counts[0] > 0, f"racey materialization corrupted the corpus: {counts}"
