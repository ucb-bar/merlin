---
title: Source-derived small execution admission
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-08
code_refs:
  - src/merlin_experiments/phase0/component_execution_budget.py
  - src/merlin_experiments/phase0/component_coverage_plan.py
  - src/merlin_experiments/phase0/generation.py
  - packages/merlin-experiments/tests/test_component_execution_budget.py
---

# Small execution admission

An externally selected reviewed `merlin.component_coverage_plan.v2` requires an
explicit `execution_budget` with schema `merlin.component_execution_budget.v1`.
The protected campaign freezes these selected bytes before authoring. No budget
limit is inferred from a model, validation shape, winning program or timing.
The historical v1 plan and report remain readable without a bounded-execution
claim. Fresh qualification must select v2 when requiring this admission.

All limits are required positive integers:

- `max_reference_work`
- `max_materialized_elements`
- `max_tensor_payload_bytes`
- `max_scalar_bits`
- `max_total_reference_work`
- `max_total_materialized_elements`
- `max_total_tensor_payload_bytes`

The first three apply to each member; the total limits apply to the complete
generation roster, including ordinary development sweeps and private guard and
transfer cohorts. Generation order is deterministic and part of the frozen
source selection. An unavailable case does not consume a successful execution
allowance, but remains requested and unavailable. Policy and complete admission
roster hashes enter the generation identity before capsule stamps are issued.

## Source derivation and staging

The current admitted numerical paths are direct integer `component_program`
DAGs and integer movement. The normal generator checks their actual declared
typed semantics using the canonical source analyzer and the selected per-entry
datapath. It never calls a builder, materializes a palette, captures a frontend,
allocates a leaf or evaluates an oracle to estimate cost. A supplied work count,
performance cost or candidate schedule cannot replace this derivation.

Unknown numerical engines, operations or frontend routes are unavailable.
They require a new independently derived cost owner before admission; they do
not use a zero or guessed cost. Unsupported or unknown costs and exceeded
limits are recorded as generation failures. Every mandatory obligation remains
in the private coverage denominator. This path does not convert a large failed
execution into a compilation success. Large compilation evidence needs a
separate future source-through-link qualification mode.

Budget admission runs before sealed capture selection and capsule staging.
Refused entries are excluded from capture selection and stopped before the
ordinary writer. Admitted entries use the original writer, independent
arithmetic and complete output goldens. The private v2 receipt reopens actual
capsule declarations and rederives counts before freeze or downstream use.
The public summary exposes commitments and cohort counts, without private
costs, extents, program selectors or unavailable reasons.

Generated integer contractions initialize their result using a scalar zero and
standard `linalg.fill` into `tensor.empty`. A tensor `dense<0>` splat can expand
into one host value per element during structural parsing even when no execution
or reference is requested. The symbolic initialization keeps source construction
and type inspection independent of tensor volume. A regression parses the actual
large source in a subprocess with explicit address-space, CPU and wall-time
limits, and checks the original argument/result types and contraction body.
Ordinary bounded generation retains its independent complete integer outputs.
This source representation change grants no large-shape lowering, linking,
index/resource legality, numerical or physical execution proof.

## Counts and limits of the evidence

Counts are conservative logical reference quantities, with this fixed v1
definition. They do not estimate device cycles or charge a hardware schedule.

| Source action | Reference work | Materialized elements | Typed tensor payload |
| --- | --- | --- | --- |
| DAG input with E elements | 2E, construction and range pass | E | E times declared storage bytes |
| Matmul result with E outputs and K reduction | 2EK + 2E, worst-case multiply/add plus initialization and projection | 2E | Raw result and projected result |
| Other supported DAG node with E outputs | 3E, source computation/copy and projection allowance | 2E | Raw result and projected result |
| Integer movement input with E elements | E, construction | E | Declared input storage |
| Complete published output with E elements | E, publication | E | Declared output storage |

Alias nodes allocate in the independent reference, so their copies are charged.
Every published output is charged even when several publications select the
same value. Palettes charge an additional complete input realization and their
selected scalar alphabet. Matmul never discounts zero-skipping. Long thin
reductions can fit; no dimension has an invented small upper limit.

Payload is the sum of logical tensor allocations, including temporary and
published values. Signed storage bytes round up from actual width. Raw sums
also charge a conservative width derived from operand widths and reduction
extent; additions charge their carry bit. `max_scalar_bits` prevents a small
shape with an enormous scalar format from allocating huge Python integers.

These are reference admission counts, not measured peak process heap or total
Python instructions. Python object/serialization overhead, repeated witness
passes, native compilation time and simulator runtime require separate resource
and timeout controls. Bounded references do not certify compiler completeness,
large-input arithmetic, physical effects, device timing or whole models.
