# Presentation slide candidates (all)

> Show these, not the descriptive QA plots. Each slide = one DSE claim + its evidence + caveat. Structural only; no speedup/area/energy.

## Slide 1 — What Merlin does

- **claim:** flat capture -> workload contract -> DSE search-space axes (it enumerates the search space; it does not choose a design)
- **show:** `dse_search_space_knobs`
- **caveat:** no performance claimed

## Slide 2 — Primitive set is a DSE axis

- **claim:** one primitive worst-cov 0.13; gemv_lane_64+tile_8x16 -> 1.00
- **show:** `primitive_set_frontier`
- **caveat:** set-union coverage, structural

## Slide 3 — Residency is a loop/rate abstraction

- **claim:** weight bytes moved grows with K under reload; flat if resident; threshold set by dtype
- **show:** `decision_weight_residency + decision_capacity_dtype`
- **caveat:** K is configured/reference

## Slide 4 — Inter-op parallelism is low

- **claim:** work/span ~1.1-1.6 -> pushes DSE to intra-op sharding / pipelining / specialized units
- **show:** `critical_path_parallelism`
- **caveat:** flattened capture may erase loop/pipeline parallelism

## Slide 5 — HW/SW boundary placement is a search axis

- **claim:** abstraction necessity: 4 necessary / 7 blocked -> DSE searches WHERE state/loops/layout/sync/reductions live
- **show:** `boundary_necessity_matrix`
- **caveat:** categorical, not a score

## Slide 6 — Capture fidelity is the limiting factor

- **claim:** the flat capture erases K-loop / KV / packed-layout the loop & residency claims need
- **show:** `capture_fidelity_matrix`
- **caveat:** the central next-step result, not a side note

