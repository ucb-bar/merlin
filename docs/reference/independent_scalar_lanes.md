---
title: "Independent scalar lane schedules"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [prepared_operand_owners]
code_refs:
  - src/merlin/llvmlower/independent_lane_schedule.py
  - src/merlin/llvmlower/radius_stage_schedule.py
  - src/merlin/llvmlower/source_attention_frontier.py
  - merlin/runtime/c/separable_fma_radius.h
---

# Independent scalar lane schedules

`independent_lane_schedule` validates a typed scalar SSA body and emits either
lane-major or stage-major ordering. It retains each lane's operand dependencies,
precision and rounding. Complete effects must establish independent immutable
inputs, stable rounding, nontrapping execution, no memory operations in the
scalar DAG and no intermediate exception observations. Final sticky flags can
remain observed because the same nontrapping operations run in either order.
This is an ordering contract, not arithmetic reassociation or a target ISA.

The regular source attention emitter accepts `radius_stage_effects` explicitly.
It also requires the existing admitted separable-radius producer and a typed
exact outward-conversion capability. The selected private coordinate consumer
stages eight independent columns; scalar tails, original bounds and checked
refusal remain intact. Runtime disjointness checks protect outputs from source
metadata, centers and one another. The caller binds private output ownership and
immutable metadata; named functions do not supply effect proof. The default
`None` produces the original C bytes. Target conversion/rounding hooks remain
provider obligations outside core.

Native scalar-versus-staged tests compare every lower/upper word and sticky flags,
source ordered-FMA containment, zero/subnormal/cancellation and overflow refusal,
independent column counts and tails. Unsupported effects or changed generated
consumer syntax refuse. This regular compiler alternative retains historical
experimental timing evidence separately; no automatic profitable width or
whole-model speedup is implied by choosing a schedule.
