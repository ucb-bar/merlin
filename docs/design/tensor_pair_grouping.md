---
title: Private tensor pair grouping
kind: design
status: current
owner: core
last_verified: 2026-10-18
---

# Private tensor pair grouping

`merlin.xdsl_dialects.lowering.tensor_chains` finds disjoint producer/consumer
pairs in typed tensor IR. The target supplies the operations it supports;
the shared analysis checks a single direct input use, a shared parent block,
and legality of delaying the producer to the consumer. It does not mutate IR,
remove arithmetic, select a numerical approximation, or assert profitability.

By default every crossed operation must be demonstrably pure. Structured
linalg operations require tensor outputs, scalar or tensor inputs, no external
body captures, and pure scalar arithmetic. Unknown effects, memrefs, nested
regions and observable intermediate results prevent selection.

An explicit caller can additionally allow `destination_only_tensor_call`.
It checks an external function declaration with read-only tensor inputs and
one write-only destination, a fresh private `tensor.empty` or `tensor.splat`,
and a matching tensor result. The declaration's input-access promises are a
precondition. This is not an instruction-level effect or alias proof for an
arbitrary external implementation. The delayed pure arithmetic cannot observe
the call's external device state. Unknown calls and calls that may mutate an
input remain barriers.

Targets retain responsibility for arithmetic recognition, layout and capacity
checks, numerical equivalence to the selected materialized implementation,
code generation, and cost measurement. A two-node section can use the existing
connected-dispatch planner and its exact realization check. Selecting an
unmeasured legal candidate is an explicit experiment; production profitability
still requires positive cost evidence.

Tests in `merlin/tests/ir/test_tensor_chains.py` cover private chains, fanout,
live outputs, unknown effects, and read/write destination contracts. These
rules do not establish whole-model correctness or a hardware timing result.
