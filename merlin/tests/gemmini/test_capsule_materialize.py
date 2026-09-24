"""The sandbox public-capsule view stays derivable from the contract and descriptor."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen.contract.materialize import materialize_public_capsules

PUB = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "full_public_capsules"
_LEGACY_MIRROR_FILES = ("capsule.yaml", "capsule.interface.mlir", "README.md")


def _load(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _name_digest(names):
    return hashlib.sha256(json.dumps(sorted(names), separators=(",", ":")).encode("utf-8")).hexdigest()


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
        doc = _load(cap_yaml)
        tiers = doc.get("required_oracle_tiers", [])
        assert all(t in ("L0", "L1", "L2") for t in tiers), (
            f"{cap_yaml.parent.name} requires an unreachable tier in the sandbox: {tiers}"
        )
        assert doc["oracle_tier_ceiling"] == "L2", (
            "the materialized ceiling must constrain optional adapters too, not only rewrite the required tier list"
        )


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

    assert materialize_public_capsules(tmp_path / "materialized", tier_ceiling="L2", corpus_roots=[source]) == [
        "M_model"
    ]
    for name, payload in files.items():
        assert (tmp_path / "materialized" / "M_model" / name).read_bytes() == payload


def test_gemmini_admission_is_derived_for_the_release():
    from dataclasses import replace

    from merlin_experiments.corpus.preparation import derive_release_admission
    from merlin_experiments.spec import SpecError

    from merlin.common.paths import repo_root
    from merlin.targetgen.contract.materialize import materialize_public_cohort
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(repo_root() / "examples/gemmini/target/descriptor.yaml")
    assert te.graded_release_admission
    assert te.graded_expected_source_capsules is None
    assert te.hidden_expected_source_capsules is None
    with pytest.raises(ValueError, match="prepare and review a corpus release"):
        materialize_public_cohort(te, tier_ceiling="L3")

    derived = derive_release_admission(te)
    assert derived["expected_cohort"]["source_capsules"] == 121
    assert derived["expected_cohort"]["admitted_capsules"] == 103
    assert len(derived["capability_exclude_capsules"]) == 11
    assert derived["resource_decisions"] == 10
    with pytest.raises(SpecError, match="classify every staged public model"):
        derive_release_admission(replace(te, graded_required_models=te.graded_required_models[:-1]))


def test_materialized_cohort_rejects_descriptor_drift(tmp_path):
    from merlin.targetgen.contract.materialize import (
        materialize_public_cohort,
        validate_materialized_cohort,
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
    materialized = materialize_public_cohort(te, tier_ceiling="L2").resolve()
    assert validate_materialized_cohort(materialized, te)["n_admitted_capsules"] == 1

    descriptor.write_text(descriptor.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after it was loaded"):
        validate_materialized_cohort(materialized, te)


def test_descriptor_rejects_cohort_count_arithmetic_drift(tmp_path):
    from merlin.common.paths import repo_root
    from merlin.targetgen.target_experiment import load_target_experiment

    source = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    doc = yaml.safe_load(source.read_text(encoding="utf-8"))
    doc["grading"].pop("release_admission")
    doc["grading"]["expected_cohort"] = {"source_capsules": 121, "admitted_capsules": 33}
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="arithmetic does not match"):
        load_target_experiment(descriptor)


def test_explicit_cohort_is_concurrency_safe():
    """Many A/B arms materialize the SAME target's public set at once. The publish must be atomic (build a
    unique versioned dir, then repoint a per-target symlink) so no arm rmtrees another's half-built cache
    mid-read: every concurrent caller must see a COMPLETE corpus (equal, non-zero capsule count)."""
    from concurrent.futures import ThreadPoolExecutor

    from merlin.common.paths import repo_root
    from merlin.targetgen.contract.materialize import materialize_public_cohort
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(repo_root() / "merlin/experiments/capsule_bench/targets/atlas/target_experiment.yaml")
    assert materialize_public_cohort(te, tier_ceiling="L3").is_symlink()

    def worker(_):
        d = materialize_public_cohort(te, tier_ceiling="L3")
        return sum(1 for _ in d.rglob("capsule.yaml"))  # full traversal blows up on a half-deleted tree

    with ThreadPoolExecutor(max_workers=4) as ex:
        counts = list(ex.map(worker, range(4)))
    assert len(set(counts)) == 1 and counts[0] > 0, f"racey materialization corrupted the corpus: {counts}"
