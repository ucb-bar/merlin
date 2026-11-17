---
title: "Explicit execution allocation budgets"
kind: design
status: current
owner: core
last_verified: 2026-11-17
related: [execution_owned_results, execution_call_views]
code_refs: [src/merlin/frontends/execution_result_signature.py, src/merlin/frontends/derived_compile_inputs.py, src/merlin/frontends/compile_inputs.py, src/merlin/llvmlower/owned_results.py, src/merlin/llvmlower/owned_result_driver.py, merlin/runtime/abi/owned_results.c, merlin/tests/ir/test_execution_resource_policy.py]
---

# Explicit execution allocation budgets

The execution-owned interface accounts for live charged bytes, cumulative
allocation work and complete output bytes separately. A bounded model may use
little live memory but perform many small allocations. Preserving its original
algorithm does not justify deleting those checks or changing the default.

Derived receipt v2 can explicitly select `execution_allocation_policy` with
`schema: merlin.owned-result-limits.v2` and exactly `max_bytes`,
`max_allocations`, `max_output_bytes`, and `index_bits`. All three limits are
positive signed-i64 integers; booleans and floating values are invalid. The
current explicit transport retains 64-bit indices. Limits are operator resource
choices, without kernel names, evaluator values or inferred output dimensions.

Independent staging admits this selection only on an execution-owned derived
v2 receipt. It validates the complete policy before publishing signature v6.
The selected policy remains bound to the input receipt and signature; original
input restrictions, complete parent/kernel reconstruction and evaluator exclusion
remain active. Other receipt kinds refuse the new field. Without an explicit
selection, the original exact v1 limits remain: 64 MiB live/output bytes and
4,096 total allocations.
Changing those constants while retaining the v1 schema refuses. Older signature
readers reject the explicit v2 policy.

The existing host compiler, saved program, generated C driver and serial target
executor consume these exact selected limits. Their allocation, alignment,
overflow, initialization, lifetime and result-descriptor checks remain enabled.
A larger work budget neither expands the byte budget nor establishes source,
numeric, effect or target correctness. Qualification must bind the same policy
as compilation; changing it cannot reuse an existing observed execution claim.

Protocol tests check source-only staging without evaluator reads, legacy default
compatibility and malformed/unknown profiles. Separate producer integrations
execute packed reads and content-sized scan/scatter with every model result;
their host/target measurements do not imply arbitrary-kernel or accelerator
performance coverage.
