# Ceiling measurement controllers

Owns `run_expert_gemm` and `multishape_compare`, retaining their historical imports.
Driver C/header resources are core-owned and resolved through that package, not this directory.
Use the analysis `mining` extra for generated workload comparisons. Keep source identities,
measurement scopes, failures and numeric checks unchanged. Importing a controller must not run
a compiler, simulator or board job; validation here uses fixture/subprocess stubs.
