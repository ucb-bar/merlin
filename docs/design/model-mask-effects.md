---
title: Explicit effects for closed-mask model lowering
kind: design
status: current
owner: core
last_verified: 2026-11-09
related: [lowering_pipeline, architecture]
code_refs:
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/masked_contraction.py
  - src/merlin/runtime/backends/spike_model.py
  - src/merlin/runtime/backends/zephyr_model.py
---

# Explicit effects for closed-mask model lowering

The ordinary model APIs accept `masked_contraction_effects`, the existing
immutable `MaskEffectContract`. The text/file lowering APIs and bare-metal and
Zephyr builders forward it unchanged to the shared upstream pipeline.

This contract explicitly permits nontrapping arithmetic with unobserved
floating flags. It allows the selected closed-mask pass to omit only reductions
whose complete source consumer discards their results. The pass still proves
the source masks, views and complete use graph; active and partial tiles retain
the original arithmetic and increasing reduction order.

Callers separately select `scalar_contraction_masked_observation` and one of
its supported scalar schedules. Missing/invalid effects or effects supplied
without that policy refuse. Supplying a contract never enables a feature or
derives permission from a target, workload or accuracy result. Default `None`
retains the existing behavior and original numerical policy.

The normal lowering recipe binds the actual runner permission and source
selection report. Final object/ELF identity, original output gates and complete
hardware cost remain separate qualification obligations. These permissions do
not establish profitable masking or permit changing an observed result.

`masked_contraction` owns the source-use proof and feature composition. The
model builders transport the selected effects into ordinary lowering; they do
not derive those permissions from a simulator success or the selected backend.
