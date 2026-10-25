# Independent frontend byte-context source reproduction

Passed on 2026-10-23 UTC. This planner-run reproduction uses the sealed byte-context packet
from `/tmp/merlin-pv-frontend-byte-compat`, through implementation commit `705494248`.
It leaves the worker sources, original frozen build and main checkout code unchanged.

- Rebuilt all **111 imported project modules** from hash-checked source copies into new
  objects. Zero worker project objects were copied.
- Fresh imported-closure audit: **5,214 theorems, zero violations**. The complete-program
  root, both original numerical-law roots and all-encoding byte-conversion root use only
  the standard three axioms; their freshly elaborated statements match the qualified receipt.
- Reexecuted **32 retained checker invocations**, with **22 acceptances and 10 refusals**.
  Every complete JSON answer matches its qualified record. This covers 23 byte-mode exported
  pairs and nine recorded receipt boundaries; it is not a new 32-test suite or model count.
- Bound **2,198 actual imported objects**, including all 111 project modules from the new
  output, and checked their identities afterward. All **50 pinned IEEE source/object records**,
  original project sources and original checker/auditor binaries remain unchanged.

Exact inputs, commands and identities are in manifest.json, processes.json, audit.json,
cases.json, imports.json, ieee-core.json, SUMMARY.json and POSTCHECK.json. The executed
rebuild and postcheck sources are retained alongside them. Sessions 83517 and 87863 are
terminal and successful; do not restart them.

This independently checks source compilation, proof dependencies and checker execution on
retained exports. It reuses pinned external packages on the same host. It is not a
clean-environment reproduction, new production-pass/exporter execution, fresh receipt-protocol
replay, native numerical proof or combined-build qualification. The unsigned source-interface
review, selected-capture binding, native NaN differences and broader W8A8 obligations remain
explicit. Do not add these counts to overlapping historical audits or tests.

From main, reproduce in a new directory with:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u \
  out/artifacts/verification/merlin-passes/progress/planner/eight-day-paper/integration/reproduce_frontend.py \
  --byte --name frontend-byte-reproduction-v2
```

Then run postcheck_frontend.py with that new evidence directory. Relocation needs a new
explicit environment/dependency recipe and identities rather than edited receipts.
