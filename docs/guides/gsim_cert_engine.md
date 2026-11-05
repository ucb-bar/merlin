---
title: Select and qualify a gSIM certification engine
kind: guide
status: current
owner: runtime
last_verified: 2026-10-12
related: [gsim_worker_setup, simulator_selection, neutral_runtime_tooling, phase0_phase1_setup]
code_refs: [src/merlin/targetgen/gsim_emulator.py, src/merlin/runtime/backends/chipyard_rocc.py, src/merlin/targetgen/phase_policy.py]
---

# Select and qualify a gSIM certification engine

A simulator executable, source-bound build receipt and deployment qualification
serve different purposes. The executable runs a model; the receipt identifies
its selected build; independent controls establish which execution and timing
claims that model can support.

## Select the exact configuration

Use one reviewed contract, RTL source/configuration, extracted facts and engine
roster through preparation and grading. For Chipyard RoCC execution, select the
neutral data contract and facts explicitly with `MERLIN_TARGET_CONTRACT` and
`MERLIN_RTL_FACTS`. See [worker setup](gsim_worker_setup.md) for the build process
and [Phase 0 and Phase 1 setup](phase0_phase1_setup.md) for contract preparation.

A binary's filename or location cannot establish its hardware identity. Recheck
its digest, actual observed build stages and selected FIRRTL. An adopted model
cannot acquire missing elaboration or tool provenance by changing a status field.
Any changed source, tool, configuration or binary needs fresh qualification.

## Qualify the selected use

Run finite independent controls on the actual worker. A positive control checks
complete nonzero outputs; numerical mutations and prohibited instructions must
be rejected under the original gates. Retain source, ELF, console, engine and
receipt identities. Probe success does not establish whole-model compilation,
all-domain correctness or final performance.

The engine selected by the grading path must enforce the same receipt and
availability requirements as its probe. An unavailable engine reports an
explicit refusal. It cannot silently substitute another engine and retain the
original timing identity.

## Interpret timing and activity

Retain the actual engine discriminator with every timing observation and cost
sample. Different simulators and configurations can have different evaluation
costs; source equality alone cannot establish equivalent cycle semantics.

Activity columns need declared units and meanings. Nested counters that cover
one execution unit must not be counted as independent overlapping units.
Unobserved units remain unmeasured, so an idle estimate based on partial activity
is a bound. Compare instruments only after establishing their matching coverage,
clock semantics and measurement boundaries.

Functional instruction counts, source hashes and output agreement do not supply
physical traffic, cache behavior or target timing. Use qualified cycle windows
for performance claims and keep model predictions separate from observations.

## Keep evidence reproducible

Retain exact source/tool selections and build receipts outside ephemeral cache
entries. Record every qualification boundary and unknown dependency. Run the
ordinary startup checks and fresh-client isolation check before authoring; final
whole-program correctness and hardware performance remain independent gates.
