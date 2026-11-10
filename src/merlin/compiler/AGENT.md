# src/merlin/compiler

Public request, artifact, backend and pass contracts live here. Keep them
independent of optional experiments, framework imports and target-specific
implementations. Backend implementation claims do not establish independent
numerical verification or hardware qualification. Preserve compatibility of
existing caller paths while moving their actual implementations to these owners.

`inspect.py` owns compile-debugging requests, pipeline stage discovery, dumps and
stop reporting. Its lazy driver imports discover stages declared beside actual
execution; lower-level passes and IR contracts must not import this inspection
owner. Recording and occurrence identities remain in `common.compile_trace`.

Shared source preprocessing is composed in `frontend/preprocess.py`; the exact
historical `llvmlower.passes_xdsl` module alias preserves selected-stage patches.
Its real algorithms live in `quantization/linalg.py`, `transforms/matmul.py`,
`transforms/dead.py`, `lower/scalar/casts.py` and `ir/provenance.py`. Literal legacy
syntax repair lives in `frontend/repairs.py`. Keep numerical and effect contracts
there; preprocessing observations are source accounting, not semantic proof.
`frontend.preprocess.implementation_paths()` pins the complete former source
envelope and actual selected stage callbacks for receipts. SDK26 exports those
actual owners; earlier releases must use their immutable source archives.
