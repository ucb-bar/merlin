"""The sandbox public-capsule view stays derivable from the contract and descriptor."""
from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen.contract.materialize import materialize_public_capsules

PUB = (merlin_dir() / "experiments" / "capsule_bench" / "harness" / "full_public_capsules")
_LEGACY_MIRROR_FILES = ("capsule.yaml", "capsule.interface.mlir", "README.md")


def _load(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _name_digest(names):
    return hashlib.sha256(
        json.dumps(sorted(names), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_legacy_public_mirror_is_a_valid_named_smoke_subset(tmp_path):
    fresh = materialize_public_capsules(tmp_path, tier_ceiling="L2")
    committed = sorted(d.name for d in PUB.iterdir() if d.is_dir())
    # ``full_public_capsules`` is a legacy checked-in smoke fixture, not the formal cohort and not a
    # byte-for-byte mirror: its capsule semantics intentionally lag the live contract.  Preserve the
    # honest invariant it still provides (valid, correctly named public rows drawn from the source) and
    # enforce exact active-formal parity separately below.
    assert set(committed).issubset(fresh)

    for name in committed:
        for f in _LEGACY_MIRROR_FILES:
            b = PUB / name / f
            assert b.is_file(), f"committed public capsule missing {name}/{f}"
        cap = _load(PUB / name / "capsule.yaml")
        assert cap.get("name") == name and cap.get("label") == "public"


def test_materializer_caps_tiers_below_ceiling(tmp_path):
    materialize_public_capsules(tmp_path, tier_ceiling="L2")
    for cap_yaml in tmp_path.rglob("capsule.yaml"):
        tiers = _load(cap_yaml).get("required_oracle_tiers", [])
        assert all(t in ("L0", "L1", "L2") for t in tiers), (
            f"{cap_yaml.parent.name} requires an unreachable tier in the sandbox: {tiers}")


def test_materializer_copies_whole_model_compile_inputs(tmp_path):
    """A model sandbox bundle carries the exact source + weights its interface names.

    These files are optional for ordinary direct-MLIR capsules, but dropping them from a model leaves
    a plausible-looking capsule that cannot reproduce the end-to-end compile.  The golden remains a
    separate answer file; this test only establishes byte-for-byte materialization of every artifact.
    """
    source = tmp_path / "source"
    capsule = source / "M_model"
    capsule.mkdir(parents=True)
    files = {
        "capsule.interface.mlir": b'builtin.module attributes {prov.weights_file = "capsule.weights.safetensors"}\n',
        "capsule.pytorch.py": b"def get_model_and_inputs():\n    return object(), ()\n",
        "capsule.linalg.mlir": b"module { func.func @forward() { return } }\n",
        "capsule.weights.safetensors": b"exact-model-weights\x00\xff",
        "golden.yaml": b"outputs: {Y0: [1]}\n",
        "expected_instruction_coverage.yaml": b"instruction_classes: [MVIN, MVOUT]\n",
        "README.md": b"model fixture\n",
    }
    (capsule / "capsule.yaml").write_text(
        "name: M_model\nkind: model\nlabel: public\nrequired_oracle_tiers: [L2]\n",
        encoding="utf-8",
    )
    for name, payload in files.items():
        (capsule / name).write_bytes(payload)

    assert materialize_public_capsules(
        tmp_path / "materialized", tier_ceiling="L2", corpus_roots=[source]
    ) == ["M_model"]
    for name, payload in files.items():
        assert (tmp_path / "materialized" / "M_model" / name).read_bytes() == payload


def test_public_capsules_for_is_target_aware_and_gemmini_parity():
    """The graded public set is DERIVED per-target from the descriptor's capsule_corpus (the target-aware
    replacement for the committed gemmini set the loop used to hardcode). gemmini must reproduce exactly
    the cohort ITS DESCRIPTOR declares; atlas must yield its OWN fp8/bf16 set (disjoint names) — proving
    no gemmini leak into another target's grade.

    THE EXPECTED CARDINALITIES ARE READ FROM THE DESCRIPTOR, never repeated here. They used to be the
    literals 48 and 34, which went stale two pool changes ago and then failed as an arithmetic mismatch
    that said nothing about which of the two numbers was wrong. The descriptor is the single frozen
    declaration; this test's job is to prove the materializer AGREES with it, not to hold a second copy
    that can drift independently."""
    from merlin.common.paths import repo_root
    from merlin.targetgen.contract.materialize import public_capsules_for
    from merlin.targetgen.target_experiment import load_target_experiment
    root = repo_root()

    te_g = load_target_experiment(root / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    gem_root = public_capsules_for(te_g, tier_ceiling="L3")
    gem = sorted(p.name for p in gem_root.iterdir() if p.is_dir())
    source = sorted(cap["name"] for cap in __import__(
        "merlin.targetgen.capsule_runner", fromlist=["discover_capsules"]
    ).discover_capsules(te_g.graded_roots(), labels={"public", "dev"},
                        contract=str(root / "merlin/contract")))
    n_source = te_g.graded_expected_source_capsules
    n_admitted = te_g.graded_expected_admitted_capsules
    n_capability = len(te_g.graded_capability_exclude)
    n_resource = len(te_g.graded_resource_exclude)
    assert set(gem) == set(source) - set(te_g.graded_exclude)
    assert len(source) == n_source and len(gem) == n_admitted
    record = json.loads((gem_root / ".cohort_admission.json").read_text(encoding="utf-8"))
    # The policy NAME tracks which classes the record accounts for, so it is derived from the descriptor
    # rather than spelled out here: a descriptor that also declares a PHASE partition records a
    # three-class policy, and pinning the two-class name would make adding the third read as corruption.
    assert record["policy"] == ("descriptor_capability_resource_and_phase_v1"
                               if te_g.graded_phase is not None
                               else "descriptor_capability_and_resource_v1")
    assert (record["n_source_capsules"], record["n_admitted_capsules"],
            record["n_capability_excluded"], record["n_resource_excluded"]) == (
                n_source, n_admitted, n_capability, n_resource)
    if te_g.graded_phase is not None:
        # A phase partition is RECORDED and not subtracted, so it must leave the denominator alone --
        # this is the assertion that would catch someone later "applying" it and shrinking the cohort.
        assert record["n_phase_excluded"] == len(te_g.graded_phase_exclude) == 0
        assert record["n_phase2_only"] == len(te_g.graded_phase2_only) > 0
        assert set(te_g.graded_phase2_only) <= set(gem)
        assert record["phase_budget_s"] == te_g.graded_phase_budget_s
    assert record["required_admitted_models"] == sorted(te_g.graded_required_models)
    assert record["descriptor_sha256"] == te_g.descriptor_sha256
    assert record["excluded_name_set_sha256"] == _name_digest(te_g.graded_exclude)
    assert record["admitted_name_set_sha256"] == _name_digest(gem)

    te_a = load_target_experiment(root / "merlin/experiments/capsule_bench/targets/atlas/target_experiment.yaml")
    atlas = sorted(p.name for p in public_capsules_for(te_a, tier_ceiling="L3").iterdir() if p.is_dir())
    assert atlas and set(atlas) != set(gem)

    # THE LEAK GUARD IS ABOUT PROVENANCE, NOT SPELLING. It used to assert the two name sets were
    # disjoint, which stopped meaning "no leak" once the roster synthesizer began naming capsules by the
    # ROLE they fill rather than by the target they were written for: two targets legitimately each own
    # a capsule called after the same role, in their own corpus, with their own dtypes. A shared name is
    # a leak only if it resolves to the SAME directory, so that is what is checked.
    def _origin(te, name):
        from pathlib import Path as _Path
        for r in te.graded_roots():
            cand = _Path(r) / name
            if (cand / "capsule.yaml").is_file():
                return cand.resolve()
        raise AssertionError(f"{name} materialized but is under none of the declared roots")

    for shared in sorted(set(atlas) & set(gem)):
        a, g = _origin(te_a, shared), _origin(te_g, shared)
        assert a != g, (f"{shared} materialized into BOTH targets' grades from the same directory "
                        f"{g} — that is a leak, not a per-target synthesis")


def test_materialized_cohort_rejects_descriptor_drift(tmp_path):
    from merlin.targetgen.contract.materialize import (
        public_capsules_for, validate_materialized_cohort,
    )
    from merlin.targetgen.target_experiment import load_target_experiment

    corpus = tmp_path / "corpus" / "isa"
    capsule = corpus / "A0"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(
        "name: A0\nkind: isa\nlabel: public\nrequired_oracle_tiers: [L2]\n",
        encoding="utf-8",
    )
    (capsule / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(
        "target: cohort_drift_fixture\n"
        f"capsule_corpus: {corpus}\n"
        "rtl: {via: mlc}\n"
        "toolchain: {sim_via: chipyard}\n"
        "grading:\n"
        "  expected_cohort: {source_capsules: 1, admitted_capsules: 1}\n",
        encoding="utf-8",
    )
    te = load_target_experiment(descriptor)
    materialized = public_capsules_for(te, tier_ceiling="L2").resolve()
    assert validate_materialized_cohort(materialized, te)["n_admitted_capsules"] == 1

    descriptor.write_text(descriptor.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after it was loaded"):
        validate_materialized_cohort(materialized, te)


def test_descriptor_rejects_cohort_count_arithmetic_drift(tmp_path):
    from merlin.common.paths import repo_root
    from merlin.targetgen.target_experiment import load_target_experiment

    source = (repo_root()
              / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    doc = yaml.safe_load(source.read_text(encoding="utf-8"))
    doc["grading"]["expected_cohort"]["admitted_capsules"] = 33
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="arithmetic does not match"):
        load_target_experiment(descriptor)


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
    assert public_capsules_for(te, tier_ceiling="L3").is_symlink()

    def worker(_):
        d = public_capsules_for(te, tier_ceiling="L3")
        return sum(1 for _ in d.rglob("capsule.yaml"))   # full traversal blows up on a half-deleted tree
    with ThreadPoolExecutor(max_workers=4) as ex:
        counts = list(ex.map(worker, range(4)))
    assert len(set(counts)) == 1 and counts[0] > 0, f"racey materialization corrupted the corpus: {counts}"
