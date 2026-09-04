"""The convolution family must expand to distinct members, and its record must not lie.

Two failures this pins, both of which actually happened:

* the family was recorded ``blocked_unimplemented`` on the claim that the integer reference engine
  has no convolution definition. It does. The claim was written from a failure caused by a library
  regression that had removed the definition, and a wrong "blocked" record is worse than none
  because it reads as a settled capability finding and gets cited.
* a corpus regeneration once collapsed a family's members onto ONE workload -- two capsules that
  should have differed in contraction depth emitted byte-identical shapes -- and nothing noticed,
  because every check counted members rather than comparing them. A fit over points that are
  secretly the same point is not a fit.
"""
from __future__ import annotations

import importlib
import sys

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.runtime import reference

PROFILE = repo_root() / "merlin" / "contract" / "capsules" / "profiles" / "_perf.yaml"
#: The generator is a script beside the corpus it writes, not an installed module. Importing it needs
#: its directory on the path -- and it must IMPORT, not skip: a skipped corruption guard is exactly
#: the check that cannot run and reports success.
GENERATOR_DIR = repo_root() / "merlin" / "contract" / "capsules"
CONV = "conv2d"
TILE_DIM = 16


def _document():
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def _conv_sweep():
    for sweep in _document().get("sweeps") or []:
        if ((sweep.get("base") or {}).get("op")) == CONV:
            return sweep
    return None


def test_the_reference_engine_models_convolution():
    """The premise the family's old blocked record denied. Assert it, so it cannot be re-asserted."""
    opcode = reference.OPCODE_CONV2D if hasattr(reference, "OPCODE_CONV2D") else "CONV2D"
    assert opcode in reference.MODELED_OPCODES


def test_no_blocked_family_claims_convolution_is_unmodelled():
    """A blocked record naming an op the engine models is a false capability finding."""
    for entry in _document().get("blocked_unimplemented") or []:
        reason = str(entry.get("reason") or "")
        for opcode in reference.MODELED_OPCODES:
            assert opcode not in reason or "no definition" not in reason, (
                f"family {entry.get('family')} is blocked on {opcode}, which the reference engine "
                f"models; the record is false")


def test_the_conv_family_is_live_and_fits_over_four_distinct_points():
    sweep = _conv_sweep()
    assert sweep is not None, "the convolution family must be a live sweep, not a blocked record"
    fit_axes = sweep.get("fit_axes") or []
    assert len(fit_axes) == 1, "one varied axis, so the fitted rate is against one quantity"
    points = (sweep.get("axes") or {}).get(fit_axes[0])
    assert len({repr(p) for p in points}) >= 4, (
        "two distinct points fit a line exactly and cannot refute one; a law needs slack")


def test_the_fitted_axis_is_derived_from_the_array_not_an_absolute_extent():
    """An absolute extent pins the cohort to one array width and degrades to a FALSE refutation on a
    wider one -- the family reads as violating its own law when only the machine changed."""
    sweep = _conv_sweep()
    points = (sweep.get("axes") or {})[(sweep.get("fit_axes") or [])[0]]
    assert all(isinstance(p, str) and "tile" in p for p in points), (
        f"every fitted point must be a tile multiple; got {points!r}")


def test_the_conv_members_expand_to_distinct_workloads():
    """The corruption guard: distinct members must emit distinct shapes, not merely distinct names."""
    if str(GENERATOR_DIR) not in sys.path:
        sys.path.insert(0, str(GENERATOR_DIR))
    generate_corpus = importlib.import_module("generate_corpus")
    from merlin.targetgen import corpus_spec

    sweep = _conv_sweep()
    binding = corpus_spec.CorpusBinding(
        target="gemmini", tile_dim=TILE_DIM, operand_dtype="int8", accum_dtype="int32",
        integer=True, tiers=["L2", "L3"], compare="exact")
    rows = generate_corpus.expand_sweeps({"sweeps": [sweep]}, binding)
    assert len(rows) >= 4

    shapes = set()
    for row in rows:
        _document_out, mlir = corpus_spec.build_conv2d(dict(row), binding)
        weight = [line for line in mlir.splitlines() if "resident_pack" in line]
        assert weight, f"member {row.get('name')} declares no packed weight"
        shapes.add(weight[0])
    assert len(shapes) == len(rows), (
        f"{len(rows)} members collapsed onto {len(shapes)} distinct workload(s); a fit over "
        f"repeated points is not a fit")
