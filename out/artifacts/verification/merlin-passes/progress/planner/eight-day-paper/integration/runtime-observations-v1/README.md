# Unresolved host-runtime observations

Observed 2026-10-23. Copies and original identities preserve two W8A8 binary-integrity
mismatches, the bounded instrumented read/audit diagnostic, frontend's initial CPython
GC crash, and the isolated SmolVLA test result. The subsequent matching reads and
isolated passing test do not explain or supersede the original failures. No common
cause has been established and no checker comparison has been weakened.

The bounded kernel-log query returned no matching entries in its recorded time window.
That does not establish that hardware, filesystem, memory pressure or runtime faults
were absent. The usual memory.events path is unavailable here; no zero OOM count is
inferred. The recorded Python runtime is3.14.7. No runtime packages, system settings,
source checkers or frozen artifacts were changed by this observation.

These observations are separate from successful theorem checks, measured test runs,
receipt outcomes and human approval. They do not qualify a new build or establish the
soundness of deployed execution. Keep them visible in the final reproduction packet.
