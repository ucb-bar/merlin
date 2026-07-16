"""WS-C C3: the compiler-region registry (all 8 regions as explicit, registrable edit-points)."""
from __future__ import annotations

from merlin.kernels import cca_contract, regions as R


def test_check_regions_clean():
    # every region names real modules + has edit-points; every lever axis governed by exactly one region
    assert R.check_regions() == []


def test_all_eight_regions_present():
    assert set(R.REGIONS) == {
        "quantization", "global-passes", "dispatch-gen", "tiling-instsel-fusion",
        "heuristics", "dispatch-scheduling", "asm-emission", "runtime-hooks"}


def test_every_lever_axis_maps_to_one_region():
    for ax in cca_contract.leverable_axes("rvv"):
        r = R.region_for_axis(ax)
        assert r is not None, f"lever axis {ax} governed by no region"


def test_edit_point_files_exist():
    from merlin.common.paths import repo_root
    root = repo_root()
    for _key, ep in R.all_edit_points():
        # only the first path token (before any parenthetical) is the file
        assert (root / ep.file).is_file(), ep.file


def test_honest_gaps_are_marked_not_hidden():
    # regions without a clean seam yet are flagged forkable_now=False, not silently omitted
    gaps = {k for k, ep in R.all_edit_points() if not ep.forkable_now}
    assert {"quantization", "dispatch-gen", "dispatch-scheduling", "runtime-hooks"} <= gaps


def test_quantization_axes_live_in_quantization_region():
    # the dtype-datapath axes are governed by the quantization region (even though they currently
    # ROUTE to schedule:dtype_strategy — the mis-routing C3 fixes next)
    for ax in ("compute.widening", "compute.accumulator_dtype", "vector.sew"):
        assert R.region_for_axis(ax).key == "quantization"
