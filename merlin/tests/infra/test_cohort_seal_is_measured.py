"""A frozen cohort cardinality must equal what the corpus actually contains — for every descriptor.

WHY. The gemmini descriptor freezes its source/admitted counts so the graded denominator cannot move
without someone noticing, and the materializer fails closed when discovery disagrees. That guard works;
what was missing is anything that notices the disagreement BEFORE a run tries to start. A capsule was
added to the shared pool, no descriptor was re-sealed, and the first thing to find out was
``public_capsules_for`` raising mid-launch.

These tests are cardinality-only on purpose, and so is the seal they check: the hidden names stay
unpublished. That means neither can see a SWAP — one row added and one removed leaves the count intact —
which is why the last test asserts the materializer records a name-set digest, the thing that can.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.contract.materialize import _public_capsule_dirs_in
from merlin.targetgen.target_experiment import load_target_experiment

DESCRIPTORS = sorted((merlin_dir() / "experiments/capsule_bench/targets").glob(
    "*/target_experiment.yaml"))


def _sealed():
    """Descriptors that actually declare a frozen cardinality, with their loaded experiment."""
    out = []
    for path in DESCRIPTORS:
        te = load_target_experiment(path)
        if getattr(te, "graded_expected_source_capsules", None) is not None:
            out.append((path, te))
    return out


def _discovered(te):
    root = Path(repo_root())
    roots = ([te.capsule_corpus] if te.capsule_corpus else [])
    roots += [root / rel.rstrip("/") for rel in te.corpus_siblings()]
    return sorted(p.name for p in _public_capsule_dirs_in(roots))


def test_at_least_one_descriptor_is_sealed():
    """The sweep below is only worth anything if it has something to sweep.

    A parametrized test over an empty list is green and proves nothing — the shape that has burned this
    repo before (a check that could not run and reported success).
    """
    assert _sealed(), "no capsule-bench descriptor declares expected_cohort; this suite is vacuous"


@pytest.mark.parametrize("path,te", _sealed(), ids=lambda x: getattr(x, "name", ""))
def test_the_frozen_source_count_is_what_the_corpus_holds(path, te):
    names = _discovered(te)
    assert len(names) == te.graded_expected_source_capsules, (
        f"{path}: declares {te.graded_expected_source_capsules} source capsules, corpus holds "
        f"{len(names)}. Re-seal it only after proving the moved rows belong on the side they moved to; "
        "bumping the integer to match is how an unreviewed row enters a formal denominator.")


@pytest.mark.parametrize("path,te", _sealed(), ids=lambda x: getattr(x, "name", ""))
def test_the_frozen_admitted_count_is_source_minus_the_declared_exclusions(path, te):
    names = set(_discovered(te))
    excluded = set(te.graded_exclude)
    assert not (excluded - names), (
        f"{path}: excludes {sorted(excluded - names)}, which is in no corpus root. An exclusion that "
        "matches nothing silently GROWS the graded set.")
    assert len(names - excluded) == te.graded_expected_admitted_capsules, (
        f"{path}: declares {te.graded_expected_admitted_capsules} admitted, discovery gives "
        f"{len(names - excluded)}")


@pytest.mark.parametrize("path,te", _sealed(), ids=lambda x: getattr(x, "name", ""))
def test_the_admission_record_seals_the_NAMES_the_counts_cannot(path, te):
    """The counts freeze a size; only the name digest freezes a cohort. Both must exist."""
    from merlin.targetgen.contract.materialize import public_capsules_for
    import json

    root = public_capsules_for(te)
    record = json.loads((root / ".cohort_admission.json").read_text(encoding="utf-8"))
    for key in ("admitted_name_set_sha256", "excluded_name_set_sha256", "descriptor_sha256"):
        assert len(str(record.get(key, ""))) == 64, (
            f"{path}: the admission record has no {key}, so a one-in-one-out swap of the cohort would "
            "leave every declared cardinality intact and nothing would notice")
