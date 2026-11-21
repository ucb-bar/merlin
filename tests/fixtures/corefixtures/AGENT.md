# Compiler test fixtures

Own the small authored programs, synthetic provider adapters, encoders and independent
native reference machines used by the compiler tests. Collect test bodies from the
existing `infra`, `ir`, `runtime` and `targetgen` buckets; do not put tests here or import
legacy test modules, including from generated child scripts.

These fixtures exercise software contracts. They do not supply approved hardware facts,
production verification algorithms, acceptance thresholds or accelerator qualification.
