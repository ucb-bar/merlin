"""The capsules that exist because a DERIVED requirement demanded them — and still land where it asked.

``check_conformance_coverage.py`` measures a corpus on three axes that are derived from the target's own
manifest and from what real captured models contain: which ``(family, dtype, alignment)`` cells the
corpus computes, which composition shapes it assembles, and which memory-mapping regime it puts the
target's operand store in. Three of those demands had no capsule at all:

* ``fits_single`` and ``spills`` — 12.1% and 42.5% of the contraction regions in the 20 real captures
  land there, and every one of the 37 public capsules fit the store TWICE over (the largest used 2.34%
  of it). So no capsule could detect a memory-mapping failure of any kind, and the one time it mattered
  a graded backend addressed all ``kt*nt`` weight tiles as simultaneously resident, asked for 16384 rows
  against 16384, and the simulator aborted three layers away in a range check.
* ``A->H->A`` — reported covered, but covered INCIDENTALLY: the only capsules containing the seam were
  the four whole models, every one of which classifies as ``routing``.

The capsules that close those are sized by NUMBERS, and a number written into a profile goes stale the
moment the target it was derived from changes. So this file re-derives every one of them from the
target's own store and its own capability map and asserts the capsule still lands where the requirement
asked — rather than asserting the extents, which would only restate the profile.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root

TARGET = "gemmini"          # target-ok: this file is ABOUT one target's derived requirement

#: The capsules authored to discharge a requirement, and the regime each was sized for.
_MEMORY_CAPSULES = {
    "layers/GM0_deep_k_fits_single_i8": "fits_single",
    "layers/GM1_deep_k_spills_i8": "spills",
}
_SEAM_CAPSULE = "model/M3_host_island_seam_gemmini"


def _capsule_dir(rel: str):
    return repo_root() / "merlin" / "contract" / "capsules" / rel


def _capsule(rel: str) -> dict:
    p = _capsule_dir(rel) / "capsule.yaml"
    if not p.is_file():
        pytest.skip(f"{rel} is not generated in this checkout")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------------------------------
# memory-mapping regimes
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rel,want", sorted(_MEMORY_CAPSULES.items()))
def test_the_capsule_lands_in_the_regime_it_was_sized_for(rel, want):
    """Re-derive the regime from the TARGET'S OWN operand store, never from the number in the profile.

    A capsule sized against a 16384-row store and then read back against a different one is a capsule
    that silently stopped testing what it was written to test. Asserting the derived regime rather than
    the extents is what makes that fail loudly.
    """
    from merlin.targetgen import memory_regime as MR

    _cap = _capsule(rel)
    got = MR.capsule_regime(_capsule_dir(rel), TARGET)
    if got["regime"] == MR.UNKNOWN:
        pytest.skip(f"no operand-store capacity derivable for {TARGET!r}: {got.get('why')}")
    assert got["regime"] == want, (
        f"{rel} was sized for {want!r} but occupies {got['rows']} of {got['capacity_rows']} rows "
        f"({100.0 * got['fraction_of_capacity']:.2f}% of capacity), which is {got['regime']!r}")


def test_the_two_regimes_the_corpus_lacked_are_now_reached_by_some_capsule():
    """The gate's question, asked directly: does ANY graded capsule reach each regime?

    Parameterised per capsule above, this one is the corpus-level claim -- it would still pass if the two
    capsules were renamed or replaced, which is the point. A regime real models occupy that no capsule
    reaches means the corpus cannot detect a memory-mapping failure of that kind.
    """
    from merlin.targetgen import memory_regime as MR
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets"
                                / TARGET / "target_experiment.yaml")
    corpus = MR.corpus_regimes(list(te.graded_roots()), TARGET, labels={"public", "dev"},
                               exclude=set(getattr(te, "graded_exclude", ()) or ()))
    if not corpus.get("capacity_rows"):
        pytest.skip(f"no operand-store capacity derivable for {TARGET!r}")
    reached = set(corpus["by_regime"])
    assert {MR.FITS_SINGLE, MR.SPILLS} <= reached, (
        f"the corpus reaches only {sorted(reached)}; largest working set "
        f"{corpus['largest_working_set']}")


def test_the_spilling_capsule_actually_violates_the_capacity_obligation():
    """``spills`` is not an adjective: the contract predicate the interface already names must say NO.

    ``capacity_fit`` counts the operands a resident lowering keeps on chip. If the capsule the corpus
    calls a spill still satisfies it, the capsule proves nothing -- a backend that loads every weight up
    front passes it, which is the failure the whole regime axis exists to catch. Counted in ELEMENTS here
    and in ROWS by ``memory_regime``; both must convict, because a shape that spills only by row padding
    would let element-accounting call it a fit.
    """
    from merlin.compile_cli import capacity_fit

    cap = _capsule("layers/GM1_deep_k_spills_i8")
    attrs = cap["operation"]["attributes"]
    shapes = {t["name"]: t["shape"] for t in cap["inputs"]}
    m, k = shapes[attrs["lhs"]]
    _k, n = shapes[attrs["weight"]]
    tile = _tile_edge()
    v = capacity_fit(TARGET, m, k, n, cap["inputs"][0]["dtype"], tile,
                     cap["numeric_policy"]["dtype"])
    if v["operands_hold"] is None:
        pytest.skip(f"{TARGET!r} declares no operand-store capacity")
    assert v["operands_hold"] is False, (
        f"the capsule the corpus calls a spill satisfies capacity_fit: {v}")


def test_the_fitting_capsule_does_not_also_spill():
    """The pair is only informative if they are on opposite sides of the boundary.

    Two spilling capsules would leave ``fits_single`` -- the regime in which staging is IMPOSSIBLE and
    serialising is correct -- as uncovered as it was, while looking covered on the corpus tally.
    """
    from merlin.compile_cli import capacity_fit

    cap = _capsule("layers/GM0_deep_k_fits_single_i8")
    attrs = cap["operation"]["attributes"]
    shapes = {t["name"]: t["shape"] for t in cap["inputs"]}
    m, k = shapes[attrs["lhs"]]
    _k, n = shapes[attrs["weight"]]
    v = capacity_fit(TARGET, m, k, n, cap["inputs"][0]["dtype"], _tile_edge(),
                     cap["numeric_policy"]["dtype"])
    if v["operands_hold"] is None:
        pytest.skip(f"{TARGET!r} declares no operand-store capacity")
    assert v["operands_hold"] is True, v


def _tile_edge() -> int:
    """The target's tile edge, derived. Never a literal: the extents above are relative to it."""
    from merlin.targetgen import conformance as CF

    edge = (CF.spec(TARGET, {}).get("boundaries") or {}).get("tile_edge")
    if not edge:
        pytest.skip(f"no tile edge derivable for {TARGET!r}")
    return int(edge)


# ---------------------------------------------------------------------------------------------------
# the A->H->A seam
# ---------------------------------------------------------------------------------------------------
def test_the_seam_capsule_is_named_by_the_shape_it_proves():
    """PRIMARY ``A->H->A``, not merely containing it.

    ``uncovered_boundaries`` reports a shape as covered only INCIDENTALLY when every capsule containing
    it is named by a different shape, and that is exactly what the four whole models did: each contains
    the seam and each classifies as ``routing``. One region more on either side of this capsule -- a
    dequantize after the last GEMM, a cast before the first -- makes a second host run and retitles it
    ``routing`` too, so this assertion is load-bearing rather than decorative.
    """
    from merlin.targetgen import boundary as B

    if not (_capsule_dir(_SEAM_CAPSULE) / "capsule.yaml").is_file():
        pytest.skip(f"{_SEAM_CAPSULE} is not generated in this checkout")
    prof = B.profile_capsule(_capsule_dir(_SEAM_CAPSULE), TARGET)
    assert prof.kind == B.A_H_A, f"{_SEAM_CAPSULE} classifies as {prof.kind!r}: {prof.to_dict()}"
    assert prof.accel_segments == 2 and prof.host_segments == 1, prof.to_dict()


def test_the_island_is_host_work_because_the_hardware_cannot_do_it():
    """The island must be a family the target declares NO capability for, not one an author chose.

    A host island made of work the accelerator COULD have run tests the wrong thing: it would charge a
    submission for a placement decision the capsule itself got wrong. Normalization is the honest choice
    here -- real captures contain 249 such regions and the manifest declares nothing for the family --
    and this asserts that fact rather than trusting the prose.
    """
    from merlin.targetgen.eligibility import capability_map_for_target
    from merlin.targetgen.model_coverage import load_module, regions_from_module

    d = _capsule_dir(_SEAM_CAPSULE)
    if not (d / "capsule.interface.mlir").is_file():
        pytest.skip(f"{_SEAM_CAPSULE} is not generated in this checkout")
    cap_map = capability_map_for_target(TARGET)
    families = {r.resolved_family() for r in regions_from_module(load_module(d / "capsule.interface.mlir"))}
    assert "normalization" in families, f"the island vanished; families present: {sorted(map(str, families))}"
    assert "normalization" not in cap_map, (
        "the target now declares a normalization capability, so this island is no longer honest host "
        "work -- pick a family the manifest still does not admit, or the capsule charges a submission "
        "for declining work the hardware can do")


def test_both_sides_of_the_seam_are_accelerator_work_the_target_admits():
    """And the outer regions must be a cell the manifest DOES admit, or the seam has only one side."""
    from merlin.targetgen.eligibility import capability_map_for_target, is_eligible
    from merlin.targetgen.model_coverage import load_module, regions_from_module

    d = _capsule_dir(_SEAM_CAPSULE)
    if not (d / "capsule.interface.mlir").is_file():
        pytest.skip(f"{_SEAM_CAPSULE} is not generated in this checkout")
    cap_map = capability_map_for_target(TARGET)
    regions = list(regions_from_module(load_module(d / "capsule.interface.mlir")))
    accel = [r for r in regions
             if r.resolved_family() in cap_map and is_eligible(r, cap_map).eligible]
    contractions = [r for r in accel if r.resolved_family() == "contraction"]
    assert len(contractions) >= 2, (
        f"the seam needs an admitted contraction on BOTH sides; found {len(contractions)} "
        f"among {len(regions)} region(s)")


def test_the_seam_capsule_demands_its_accelerator_regions_actually_accelerate():
    """Without ``must_accelerate`` the capsule passes on a submission that ran the whole thing on the CPU.

    That is not hypothetical for this corpus: the gemmini profile added the corpus-wide default precisely
    because every coverage certificate had been passing VACUOUSLY with no capsule asserting it.
    """
    cap = _capsule(_SEAM_CAPSULE)
    sem = cap.get("semantic") or {}
    assert sem.get("must_accelerate") is True, (
        f"{_SEAM_CAPSULE} does not require its eligible regions to run on the accelerator: {sem}")
    assert cap.get("required_oracle_tiers"), "a capsule with no required tier is graded by nothing"


def test_the_seam_capsule_is_small_enough_to_be_worth_running_at_the_cycle_accurate_tier():
    """The reason the whole models cannot prove the seam is size; a replacement that grew to their size
    would inherit the same problem, so the budget is asserted rather than assumed.

    Measured against the corpus's own existing whole-model capsules: this one must stay well under the
    smallest of them, not under an absolute number invented here.
    """
    d = _capsule_dir(_SEAM_CAPSULE)
    if not (d / "capsule.interface.mlir").is_file():
        pytest.skip(f"{_SEAM_CAPSULE} is not generated in this checkout")
    mine = (d / "capsule.interface.mlir").stat().st_size
    others = [p.stat().st_size
              for p in (repo_root() / "merlin" / "contract" / "capsules" / "model").glob(
                  "*/capsule.interface.mlir")
              if p.parent.name != d.name]
    if not others:
        pytest.skip("no other whole-model capsule to compare against")
    assert mine < min(others), (
        f"the seam capsule's interface is {mine} bytes against the smallest other whole model's "
        f"{min(others)}; it was written to be affordable where they are not")
