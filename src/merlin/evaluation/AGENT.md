# src/merlin/evaluation

Independent execution assessment lives here, outside compiler implementations.
Candidates cannot import this package or change acceptance thresholds. Keep
numerical references, execution evidence and performance observations distinct.
Relocations must preserve historical access denials and fingerprint every actual
implementation used by grading, certificates and child-process admission.

Capsule lifecycle and its CLI start in `capsule/runner.py`; operation grading
lives in `capsule/operations/`, model grading in `capsule/models/`, and certificates,
lanes and source custody in `capsule/evidence/`.

`model/bundles/` owns sealed inputs and references; `model/kernels/` owns kernel
replay and diagnostics; `model/whole_model/` owns linked-model assessment;
`model/binding/` binds machine, tools and artifacts. The independent source-IR
interpreter lives in `model/linalg_evaluator.py`, with scalar numerical semantics
in `model/linalg_semantics.py`. Neither may grade lowering by invoking that same
lowering as its reference.

`route_admission.py` checks explicit route coverage and factual refusal claims.
`host_compute_accounting.py` checks tensor-data arithmetic in host code. Keep
those claims separate from numerical correctness and observed target execution.
