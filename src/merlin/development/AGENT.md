# src/merlin/development

`testgen/` owns test and corpus generation from selected software/hardware inputs.
`backend/` owns the combined correctness/performance authoring engine.
`backend/optimization/` retains specialized measured optimization mechanisms.
Experiment definitions, catalogs and process orchestration remain in `merlin.experiments`.

This package retains the operator-only role of the former experiment engine packages.
Preserve numerical references, budgets, source ownership, withheld inputs and independent
acceptance checks. Historical schema IDs and frozen source inventories remain unchanged.
