---
title: Closed row domains and private partition plans
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [architecture, lowering_pipeline]
code_refs:
  - src/merlin/llvmlower/closed_row_domain.py
  - src/merlin/llvmlower/row_partition_storage.py
---

# Closed row domains and private partition plans

These optional analysis APIs describe a typed tensor region whose observations
can be partitioned along one exact parallel dimension. They retain every other
axis and the complete original scalar bodies, reduction dimensions, seeds,
lane order and observation boundary. They do not rewrite IR or enable routing.

`analyze_closed_row_domain` first closes the original tensor observation and
checks the complete function's tensor/scalar effects. Every row path must agree
on one exact parallel dimension. A row cannot enter a reduction or an affine
expression on another axis. An invariant operand cannot be indexed by the row
dimension unless its uniform value is separately established. Dynamic views,
opaque effects, external escapes and unsupported operations refuse.
Scalar index observations also refuse: preserving an original index after
stripmining requires an independently proved block offset.
Nested scalar regions require a separate source proof. An empty tensor supplies
no uniform-value evidence, including through casts or views; an output seed is
admitted only when its original scalar argument is unused or its values have a
separate uniform proof.

The proof revalidates original source/use/context witnesses before use. A static
partition must cover the complete original extent once, in order, including
its tail. Shape matching or a source identity does not grant legality.

`plan_dense_row_scatter` describes logical copies from packed private blocks to
one complete dense private result. Prefix axes remain in each block; suffix
axes retain their original ordering. It grants no permission for early public
writes. `plan_partitioned_consumer_leases` explicitly expands an immutable RHS
owner's public use quota into ordered internal uses and refuses quota overflow.

Before replacement, the caller must independently close physical allocations,
nonaliasing and lifetimes; all source and downstream numeric observations;
complete current product imports for every internal shape and tail; synchronous
completion; and the retained original fallback. A product compiled for the
public shape cannot silently accept an internal shape. Failure after partial
private execution must publish no public fragment or reusable partial result.

## Compiler ownership and cost

Typed row legality, private host representations, source packing, immutable
epochs, original column certificates and deferred publication belong in Merlin.
Target dialects own physical product implementations, instruction/resource
facts and target calling conventions. Model names, provenance and golden
outputs do not select transformations; experiment drivers may bind explicit
source witnesses while a general matcher and cost selection remain pending.

Price every internal call, transfer, source preparation, proof/decoder operation,
private scatter and final drain. Partitioning may repeat invariant work and
increase device traffic. A smaller working set establishes neither profitability
nor a whole-model cycle target. Functional instruction counts, simulator wall
time, device cycles and whole-model hardware cycles remain separate evidence.

## Qualification and automatic-loop requirements

A qualification must include the complete source consumer and every internal
product import, plus independent shape/tail cases and failure after private
partial execution. Reducing the row block can increase callback count and repeat
source gathering or column certificates. Reusing an original prepared value or
certificate requires a separate complete-use, immutable-epoch and effect proof;
partition legality alone does not authorize it. Keep measured journeys and
point-in-time receipts under the configured artifact root.

Phase 0 should expose closed parallel domains, complete observation and source
effect witnesses, invariant source values and required tail shapes. Phase 1
should materialize private representations and current product imports, with
explicit epochs, quotas, compiled layout queries and original fallback. Phase 2
should allow general row schedules and prepared certificate reuse, using joint
host/device costs and complete correctness gates. Agents need access to these
legal shared transformations and physical backend schedules; they must not
patch a frozen workload leaf to bypass the same compiler contracts.

Useful tooling includes immutable source/package receipts, actual exclusive PC
costs with verified debug companions, operation/call/transfer inventories and
complete-run admission budgets derived from setup and validation as well as the
ROI. Exact per-topic token billing was unavailable for this prototype; reported
scope counts must not be presented as measured token usage.
