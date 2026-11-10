# src/merlin/evaluation/capsule

`runner.py` exposes the established lifecycle API. `operation.py` orchestrates
one operation; `model.py` prepares model execution and `model_assessment.py`
assesses its returned evidence. `ladder.py` owns required-tier ordering and
certificate reuse. `oracle_call.py` supervises a single adapter invocation;
`oracle_result.py` interprets its returned measurements. Keep those boundaries
explicit so execution cannot manufacture an acceptance decision.

Preserve exception order, numerical tolerances, source custody, worker lifetimes,
not-run status and required execution evidence. Old `targetgen.capsule_runner`
is an exact import identity for real consumers and historical pickle lookup;
new internal callers use the actual defining owner. Grading-source inventories
must include the entire implementation package, including its worker scripts.
