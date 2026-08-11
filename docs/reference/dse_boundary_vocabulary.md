---
title: DSE boundary-placement vocabulary
kind: reference
status: current
owner: dse
last_verified: 2026-08-10
related: [dse_guidance, contracts, design_pressure]
code_refs: [merlin/python/merlin/dse_guidance/boundary_placement.py, merlin/python/merlin/dse_guidance/compiler_proof.py, merlin/python/merlin/dse_guidance/operator_geometry.py, merlin/python/merlin/dse_guidance/case_study.py]
---

# DSE boundary-placement vocabulary

The boundary-placement phase answers one question per workload-implied abstraction: **not** "should
we build this", but "at which of six levels could this responsibility sit, and what would each
placement cost in compiler proof, runtime support, ISA surface, and metadata". It emits the search
space; a DSE tool later chooses.

This page defines the field and status vocabulary that appears in the emitted artifacts. Source of
truth is `merlin/python/merlin/dse_guidance/boundary_placement.py`.

## Which list of abstractions?

Two distinct catalogs use the word "abstraction" and their names partly overlap. They are not the
same list:

| Catalog | Where | What it holds |
|---|---|---|
| **Boundary catalog** (27 entries) | `boundary_placement.py` — `_BOUNDARY_CATALOG`, exported as `ABSTRACTIONS` | each abstraction × where it could live |
| **DSE-axis map** (smaller) | `contract.py` — `ABSTRACTION_MAP` | DSE axis → the `system_abstraction` it implies |

Everything below describes the **boundary catalog**.

## Per-abstraction fields

Each catalog entry is
`{sources, region_roles, support, cp_axis, erased, metadata, knobs[(name, reason)], risk, levels{level: status}}`.

| Field | Definition |
|---|---|
| `sources` | Which **upstream analysis phases** produced the evidence that this abstraction is implied at all — artifact/table names from earlier pipeline phases (`operator_shape_table`, `memory_hierarchy_envelope`, `state_lifetime`, `command_graph`, `numerical_contract`, `pipeline_envelope`, `sharding_opportunities`, …). Provenance only; nothing enters the catalog without a source phase. |
| `region_roles` | **Where in the execution structure** the enabling structure appears: `repeated_head` (runs K× per replan — the denoising/action head), `backbone_once` (runs 1× per replan — the VLM/vision backbone), `prefix_builder` (builds prefix/KV state), `unknown`. Assigned per-op in `operator_geometry.py` from attribution / `prov_fqn`. This is the field that encodes **execution rate**. |
| `support` | The **gating predicate** a workload must satisfy before it is listed as supporting this abstraction. Keys: `all`, `k_loop` (K > 1), `control_loop`, `decode` (autoregressive class), `dense` (a squareish GEMM exists), `gemv` (skinny/GEMV shapes exist), `epilogue` (bias present), `lowbit`. Resolved by `_supporting()` against the `ev_ctx` dict built in `case_study.py`. Note `lowbit` is treated the same as `all` — every f32 workload is a low-bit *opportunity*. |
| `cp_axis` | The **compiler-proof-matrix axis** whose fact a compiler must prove before the abstraction is legally targetable (see below). `None` means no compiler proof is required — those entries are pure hardware-sizing questions. |
| `erased` | The **capture-honesty flag**: the structure this abstraction needs was destroyed by how the workload was captured (the recaptures are dequantized f32, so packed weights and scales are gone). The consequence is mechanical: every non-compiler level is forced to `blocked` (`_ERASED_LEVELS`), the pressure score takes a penalty, and every knob's evidence becomes `unavailable`. |
| `kv` | Sibling of `erased` for structure that was never visible rather than merely erased — attention lowered into matmul projections. Levels come from `_KV_LEVELS`: `not_applicable` at the compiler level, `unavailable` everywhere else. |
| `metadata` | What would have to **cross the HW/SW boundary** if the abstraction is placed above the compiler level — the fields of the handle/descriptor, i.e. an ABI sketch. Reported only where the placement actually crosses (level is not the compiler level and status is not `not_applicable`/`unavailable`). |
| `knobs` | The **DSE search-space dimensions this placement creates** (`resident_capacity`, `tile_M`, `pack_format`, `loop_bound_K`, …). Each becomes a row in `boundary_dse_knobs.yaml` carrying its abstraction, strongest candidate level, reason, and evidence (`recovered_from_ir` vs `unavailable`). Dimensions to sweep — **not** recommendations. |
| `risk` | The **overfit / anti-pattern note**: the specific way committing to this placement could be wrong. Two recurring flavors — hardware over-serving ("a square matrix engine over-serves the skinny/GEMV decode shapes") and hardware hiding what the compiler already knew ("a hardware cache would hide the semantic lifetime the compiler already knows"). `"; highest overfit risk at a fixed/native level"` is appended when ISA or datapath placement is a live candidate. |
| `levels` | Per-level plausibility — the matrix itself (below). `levels=None` together with `erased`/`kv` means the levels are filled from the corresponding template. |

## Boundary levels

`LEVELS`, in order of increasing hardening:

| Level | Software manages | Hardware manages |
|---|---|---|
| `compiler_transform` | compiler rewrites the workload; hardware sees ordinary ops | none (generic ops) |
| `runtime_hal_object` | runtime tracks the object's lifetime + layout | exploits the declared lifetime / layout |
| `command_buffer_or_command_isa` | submits higher-level commands instead of ops | command processor executes / loops / tracks deps |
| `accelerator_isa` | compiler targets the semantic ISA op | datapath executes the semantic instruction |
| `device_microcode_or_controller` | submits a bounded region; device iterates | controller owns the loop / deps / prefetch / state |
| `fixed_hardware_datapath` | none (absorbed into the unit) | hardwired unit |

`_LEVEL_ROLE` also records, per level, the runtime support, ISA semantics, and hardware support that
placement would require — that is what fills the per-level rows of the emitted report.

## Placement status

The distinction between the last three values is the important part:

| Status | Meaning |
|---|---|
| `strong_candidate` | plausible, widest evidence |
| `possible` | plausible |
| `weak_candidate` | plausible, thin evidence |
| `not_applicable` | structurally meaningless at that level — not a gap |
| `blocked` | would be plausible, but requires evidence we do not have (a low-bit capture + accuracy numbers) |
| `unavailable` | the structure is not in the capture at all, so the question cannot be asked |

## Compiler-proof axes

`cp_axis` points into the matrix built by `compiler_proof.py`, which attaches an honest status:
`proven_for_workload` (the capture/topology establishes it), `assumed` (rests on an assumed
reference, e.g. K), `unknown` (the capture erased what the proof needs), or `unavailable` (no matrix
entry). Status is taken as the **weakest** observed across workloads, so the matrix never
over-claims.

The seven axes: `resident_action_head_weights`, `autonomous_K_loop`, `command_batching`,
`async_chunk_overlap`, `fused_requant_epilogue`, `resident_packed_lowbit_weights`,
`decode_kv_cache_path`.

Seven abstractions carry `cp_axis: None` — `matrix_engine`, `skinny_gemm_or_gemv_engine`,
`dma_engine`, `multi_stream_dma_descriptor`, `prefetch_descriptor`, `partial_sum_object`,
`accumulator_merge`. No compiler proof gates them; they are hardware-sizing questions.

## What is blocked today, and why

Seven of the 27 are `blocked` or `unavailable` **by construction**, not by oversight:

- `erased` (dequantized f32 capture removed packed weights + scales) — `packed_lowbit_tensor`,
  `scale_object`, `resident_packed_weight_object`, `fused_dequant_matmul`, `native_lowbit_matmul`.
  Missing evidence: a low-bit capture (packed weights + scales) plus per-format accuracy.
- `kv` (attention lowered into matmul projections) — `prefix_kv_object`, `kv_cache_object`.
  Missing evidence: a loop-preserving capture that retains attention/KV structure.

For an analysis asking *which values are recovered from the IR versus currently assumed*, these two
flags plus the `cp_axis` status are the answer, and each knob's `evidence` field
(`recovered_from_ir` vs `unavailable`) is the machine-readable form of it.

## `boundary_pressure_score` is evidence breadth

Not performance, not priority, not benefit. It is the sum of `pressure_components`:
`n_supporting_workloads`, `n_region_roles`, `crosses_rate_boundary`, `in_repeated_loop`,
`compiler_provable`, minus `overfit_penalty` and `missing_evidence_penalty`. A high score means
"many phases point at this", never "this would be fast". Every certificate also carries an explicit
`what_is_not_claimed` field: no speedup, cycles, area, or energy claim, and no chosen design.

## The responsibility-split matrix is separate

`responsibility_rows()` emits a coarser cross-cutting view that is **not** part of the 27: 17
functions (`region_partitioning`, `layout_selection`, `dtype_selection`, `weight_packing`,
`scale_metadata_management`, `resident_object_lifetime`, `K_loop_iteration`,
`loop_carried_state_update`, `command_dependency_tracking`, `event_synchronization`, `DMA_prefetch`,
`buffer_allocation`, `sharding_split`, `partial_sum_merge`, `epilogue_requant`,
`deadline_enforcement`, `safety_action_commit`) × six columns (`compiler`, `runtime_hal`,
`command_processor`, `accelerator_isa`, `device_microcode`, `datapath`), with cells drawn from
`owns` / `assists` / `declares` / `consumes` / `not_applicable` / `unknown`. Functions whose
evidence a model-forward capture does not carry (`deadline_enforcement`, `safety_action_commit`,
`scale_metadata_management`) carry an explicit note rather than an invented cell.

## Emitted artifacts

Written by the case study under its output directory:

| Artifact | Contents |
|---|---|
| `hw_sw_boundary_matrix.csv` | abstraction × level status, supporting workloads, pressure score |
| `boundary_candidate_contracts.yaml` | the full per-abstraction certificates |
| `boundary_placement_report.md` | narrative: strong placements, genuine design axes, blocked list |
| `responsibility_split_matrix.csv` | the 17 × 6 responsibility matrix |
| `boundary_dse_knobs.yaml` | every knob with abstraction, level, reason, evidence |
| `runtime_object_candidates.yaml` · `command_isa_candidates.yaml` · `isa_candidate_primitives.yaml` | per-level candidate lists |
| `interface_contract_sketches.md` | HAL / command / ISA field sketches |
