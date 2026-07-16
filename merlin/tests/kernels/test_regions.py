"""WS-C C3: the compiler-region registry (all 8 regions as explicit, registrable edit-points)."""
from __future__ import annotations

from merlin.kernels import cca_contract, regions as R


def test_check_regions_clean():
    # every region names real modules + has edit-points; every lever axis governed by exactly one region
    assert R.check_regions() == []


def test_two_level_taxonomy_phases_and_fine_regions():
    # phase-grouped (compilation stages) AND a fine per-concern registry (the point)
    assert R.phases() == ("frontend", "global", "dispatch", "kernel-codegen", "memory",
                          "emission", "runtime", "cross-cutting", "target-gen")
    # every region has a valid phase; the fine concerns the user named are DISTINCT regions
    assert all(r.phase in R.phases() for r in R.REGIONS.values())
    fine = {"data-tiling", "vectorization", "instruction-selection", "instruction-scheduling",
            "inner-loop", "fusion", "accumulation"}
    assert fine <= set(R.REGIONS)                         # no longer lumped into one region
    # the newly-surfaced groups are first-class
    assert {"graph-ingest", "numerics-precision", "bufferization-memplan", "layout-packing",
            "target-lowering", "cost-model-capabilities", "target-dialect-gen"} <= set(R.REGIONS)
    assert len(R.REGIONS) >= 20                            # the larger registry


def test_kernel_codegen_phase_has_the_fine_concerns():
    keys = {r.key for r in R.regions_by_phase("kernel-codegen")}
    assert keys == {"data-tiling", "vectorization", "instruction-selection",
                    "instruction-scheduling", "inner-loop", "fusion", "accumulation"}


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
    assert {"quantization", "dispatch-gen", "hw-sync", "fusion"} <= gaps


def test_quantization_axes_live_in_quantization_region():
    # the dtype-datapath axes are governed by the quantization region (even though they currently
    # ROUTE to schedule:dtype_strategy — the mis-routing C3 fixes next)
    for ax in ("compute.widening", "compute.accumulator_dtype", "vector.sew"):
        assert R.region_for_axis(ax).key == "quantization"
