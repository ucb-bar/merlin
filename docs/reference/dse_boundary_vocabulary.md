---
title: DSE boundary-placement vocabulary
kind: reference
status: current
owner: dse
last_verified: 2026-08-11
related: [dse_guidance, contracts, design_pressure]
code_refs: [merlin/python/merlin/dse_guidance/boundary_placement.py, merlin/python/merlin/dse_guidance/compiler_proof.py, merlin/python/merlin/dse_guidance/operator_geometry.py, merlin/python/merlin/dse_guidance/case_study.py]
---

# DSE boundary-placement vocabulary

The boundary-placement phase asks one question per workload-implied abstraction: not whether to build
it, but **at which of six levels the responsibility could sit**, and what each placement would cost in
compiler proof, runtime support, ISA surface, and metadata. It emits the search space; a DSE tool
chooses within it.

This page is the reference definition of that vocabulary — the 27 abstractions in the catalog, the
per-abstraction fields, and the level and status values that appear in the emitted artifacts. Source
of truth is `merlin/python/merlin/dse_guidance/boundary_placement.py`.

## Two catalogs named "abstraction"

Two distinct catalogs use the word, and their entry names partly overlap. They are not the same list:

| Catalog | Where | What it holds |
|---|---|---|
| **Boundary catalog** (27 entries) | `boundary_placement.py` — `_BOUNDARY_CATALOG`, exported as `ABSTRACTIONS` | each abstraction × where it could live |
| **DSE-axis map** (smaller) | `contract.py` — `ABSTRACTION_MAP` | DSE axis → the `system_abstraction` it implies |

Everything below describes the **boundary catalog**.

## What an entry is

An entry is a **named unit of responsibility** the workload evidence implies someone must own — a
thing that would appear as an object, handle, descriptor, or instruction in some layer's interface.
An entry is not itself an optimization; the optimization is implied by where it is placed. The same
`resident_weight_object` is a compiler decision, a runtime/HAL object, a command-buffer handle, or a
hardwired buffer depending on the level chosen, and the catalog exists to keep those alternatives
readable side by side.

## The 27 abstractions

The third column lists the levels marked `strong_candidate` (see [Placement status](#placement-status)).
An entry with no strong level is either genuinely spread across levels or currently blocked; see
[What is blocked, and why](#what-is-blocked-and-why).

### Resident state and data objects — what stays put, and for how long

| Abstraction | What it is | Strong candidate levels |
|---|---|---|
| `resident_weight_object` | Weights held in local memory across the whole K-loop instead of re-fetched per step, with an explicitly declared lifetime (the compiler has proven them loop-invariant). | runtime · command |
| `resident_packed_weight_object` | The same, for weights kept in *packed low-bit* form while resident, carrying a handle to their scales. | none (`blocked`) |
| `packed_lowbit_tensor` | The sub-byte tensor as a first-class object: storage dtype, packing layout, quant group size, scale handle. | none (`blocked`) |
| `scale_object` | Quantization scales and zero-points as their own object, separate from the tensor, with a declared granularity (per-tensor / per-channel / per-group). | none (`blocked`) |
| `loop_carried_state_handle` | State updated on *every* K iteration and read by the next one (a denoiser's evolving action estimate) — what makes the loop sequential. Distinct from resident weights, which do not change. | command · microcode |
| `prefix_kv_object` | Prefix KV state produced once by the backbone and consumed repeatedly afterwards. | none (`unavailable`) |
| `kv_cache_object` | The growing per-token KV cache of an autoregressive decode. | none (`unavailable`) |

### Command and control-flow objects — what drives the loop

| Abstraction | What it is | Strong candidate levels |
|---|---|---|
| `bounded_loop_command` | A single submitted command meaning "run this region K times", carrying trip count, body handle, and which state is loop-carried versus invariant — instead of the host issuing K separate dispatches. | command · microcode |
| `region_level_dispatch` | Submitting a whole region (a subgraph) as one unit with a dependency list, rather than one dispatch per operator. The granularity control for host↔device traffic. | command · microcode |
| `persistent_command_buffer` | A command stream recorded once and replayed across replans instead of re-encoded each replan. Requires a dependency graph that is static at submit time. | command · microcode |
| `decode_loop_controller` | The autoregressive counterpart of `bounded_loop_command`: a device-side controller owning the token loop, its trip bound, and the KV update. | command · microcode |

### Asynchrony and rate decoupling — what waits for what

These four derive from `pipeline_envelope` and share the `async_chunk_overlap` proof axis. They
concern overlap and coordination rather than compute, which is why they are strong at the runtime/HAL
and command levels and are never candidates at the datapath.

| Abstraction | What it is | Strong candidate levels |
|---|---|---|
| `event_token` | A handle representing "this work has finished", with a producer and a consumer — the synchronization primitive that lets a submitter stop blocking. The other three build on it. Knob: `event_depth` (how many may be in flight). | runtime · command |
| `async_queue` | A submission channel with a depth, so work is enqueued and the submitter returns immediately instead of blocking per operation. Generic — it applies to every workload. Knob: `queue_depth`. | runtime · command |
| `producer_consumer_queue` | An `async_queue` specialized to two stages running at **different rates** (hence `support: control_loop`): a model produces action chunks while a control loop consumes them at a fixed control rate. Its `pc_queue_depth` knob is the slack between those rates, so it is only meaningful where a real rate split exists. | runtime · command |
| `double_buffered_action_chunk` | Two or more chunk buffers, so the next action chunk is computed while the current one is still being consumed at the control rate. Knob: `buffer_count`. | runtime · command |

### Data movement — how bytes arrive

| Abstraction | What it is | Strong candidate levels |
|---|---|---|
| `dma_engine` | Explicit asynchronous bulk transfer under software control, as opposed to a demand-fetch cache. Recorded `risk`: a cache would rediscover reuse the compiler can already express. | runtime · command |
| `multi_stream_dma_descriptor` | Independent, separately described streams for weights / activations / outputs, so one descriptor batches what would otherwise be per-tile software-issued transfers. | runtime · command |
| `prefetch_descriptor` | A declared "fetch this, this far ahead, on this trigger" object — lookahead expressed as data rather than inferred by hardware. | runtime · command · microcode |

### Compute units — what the datapath is shaped like

All three carry `cp_axis: None`: no compiler proof gates them, they are hardware-sizing questions.

| Abstraction | What it is | Strong candidate levels |
|---|---|---|
| `matrix_engine` | A tiled MAC array for squareish GEMM, parameterized by tile shape and accumulator dtype. Recorded `risk`: it over-serves skinny/GEMV decode shapes. | isa · datapath |
| `skinny_gemm_or_gemv_engine` | A lane-oriented unit for GEMV / tall-skinny / wide-skinny shapes, with the mirror-image risk of under-serving large dense GEMM. The pairing with `matrix_engine` exists so that the "one general unit or two specialized ones" question stays explicit rather than assumed. | isa · datapath |
| `epilogue_requant_unit` | Hardware for the post-matmul tail — bias, activation, requantization — so an accumulator output is finished without a separate pass over memory. | isa · datapath |

### Reduction and accumulators — how partial results combine

| Abstraction | What it is | Strong candidate levels |
|---|---|---|
| `partial_sum_object` | A partial result of a K-split reduction as a named object (wide dtype, shard count, M×N tile), so the split stays visible to the compiler instead of being hidden inside the unit. | isa · microcode · datapath |
| `accumulator_merge` | The combining step itself — a reduction tree with a radix and an accumulator width. | isa · microcode · datapath |
| `accumulator_commit` | Narrowing the wide accumulator to the output dtype and writing it out; a knob because where this happens (in the epilogue or as a separate pass) is a design choice. | isa · datapath |

### Numerical fusion primitives — what a single instruction covers

| Abstraction | What it is | Strong candidate levels |
|---|---|---|
| `fused_requant_epilogue` | Folding requantization into the matmul epilogue rather than a separate pass. The bias slot is proven; the requant numerics are unmeasured. | none — `possible` at every level except runtime |
| `fused_dequant_matmul` | Unpacking/dequantizing weights on the load path into a higher-precision compute, so packed weights never materialize. | none (`blocked`) |
| `native_lowbit_matmul` | A datapath multiplying packed low-bit values directly and accumulating wide — no dequantization at all. The end state the previous two approximate. | none (`blocked`) |

## Per-abstraction fields

Each catalog entry is
`{sources, region_roles, support, cp_axis, erased, metadata, knobs[(name, reason)], risk, levels{level: status}}`.

| Field | Definition |
|---|---|
| `sources` | Which **upstream analysis phases** produced the evidence that this abstraction is implied at all — artifact/table names from earlier pipeline phases (`operator_shape_table`, `memory_hierarchy_envelope`, `state_lifetime`, `command_graph`, `numerical_contract`, `pipeline_envelope`, `sharding_opportunities`, …). Provenance only; nothing enters the catalog without a source phase. |
| `region_roles` | **Where in the execution structure** the enabling structure appears: `repeated_head` (runs K× per replan — the denoising/action head), `backbone_once` (runs 1× per replan — the VLM/vision backbone), `prefix_builder` (builds prefix/KV state), `unknown`. Assigned per-op in `operator_geometry.py` from attribution / `prov_fqn`. This is the field that encodes **execution rate**. |
| `support` | The **gating predicate** a workload must satisfy before it is listed as supporting this abstraction. Keys: `all`, `k_loop` (K > 1), `control_loop`, `decode` (autoregressive class), `dense` (a squareish GEMM exists), `gemv` (skinny/GEMV shapes exist), `epilogue` (bias present), `lowbit`. Resolved by `_supporting()` against the `ev_ctx` dict built in `case_study.py`. Note `lowbit` is treated the same as `all` — every f32 workload is a low-bit *opportunity*. |
| `cp_axis` | The **compiler-proof-matrix axis** whose fact a compiler must prove before the abstraction is legally targetable (see [Compiler-proof axes](#compiler-proof-axes)). `None` means no compiler proof is required — those entries are pure hardware-sizing questions. |
| `erased` | The **capture-honesty flag**: the structure this abstraction needs was destroyed by how the workload was captured (the recaptures are dequantized f32, so packed weights and scales are gone). The consequence is mechanical — every non-compiler level is forced to `blocked` (`_ERASED_LEVELS`), the pressure score takes a penalty, and every knob's evidence becomes `unavailable`. |
| `kv` | Sibling of `erased` for structure that was never visible rather than merely erased — attention lowered into matmul projections. Levels come from `_KV_LEVELS`: `not_applicable` at the compiler level, `unavailable` everywhere else. |
| `metadata` | What would have to **cross the HW/SW boundary** if the abstraction is placed above the compiler level — the fields of the handle/descriptor, i.e. an ABI sketch. Reported only where the placement actually crosses (the level is not the compiler level and the status is not `not_applicable`/`unavailable`). |
| `knobs` | The **DSE search-space dimensions the placement creates** (`resident_capacity`, `tile_M`, `pack_format`, `loop_bound_K`, …). Each becomes a row in `boundary_dse_knobs.yaml` carrying its abstraction, strongest candidate level, reason, and evidence (`recovered_from_ir` vs `unavailable`). Dimensions to sweep — not recommendations. |
| `risk` | The **overfit / anti-pattern note**: the specific way committing to this placement could be wrong. Two recurring flavors — hardware over-serving ("a square matrix engine over-serves the skinny/GEMV decode shapes") and hardware hiding what the compiler already knew ("a hardware cache would hide the semantic lifetime the compiler already knows"). `"; highest overfit risk at a fixed/native level"` is appended when ISA or datapath placement is a live candidate. |
| `levels` | Per-level plausibility — the matrix itself. `levels=None` together with `erased`/`kv` means the levels are filled from the corresponding template. |

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

`_LEVEL_ROLE` also records, per level, the runtime support, ISA semantics, and hardware support a
placement would require; that is what fills the per-level rows of the emitted report.

## Placement status

The last three values are distinct and not interchangeable:

| Status | Meaning |
|---|---|
| `strong_candidate` | plausible, widest evidence |
| `possible` | plausible |
| `weak_candidate` | plausible, thin evidence |
| `not_applicable` | structurally meaningless at that level — not a gap |
| `blocked` | would be plausible, but requires evidence the current captures do not provide (a low-bit capture plus accuracy numbers) |
| `unavailable` | the structure is not in the capture at all, so the question cannot be asked |

## Compiler-proof axes

`cp_axis` points into the matrix built by `compiler_proof.py`, which attaches a status:
`proven_for_workload` (the capture/topology establishes it), `assumed` (rests on an assumed
reference, e.g. K), `unknown` (the capture erased what the proof needs), or `unavailable` (no matrix
entry). The status is the **weakest** observed across workloads, so the matrix never over-claims.

The seven axes: `resident_action_head_weights`, `autonomous_K_loop`, `command_batching`,
`async_chunk_overlap`, `fused_requant_epilogue`, `resident_packed_lowbit_weights`,
`decode_kv_cache_path`.

Seven abstractions carry `cp_axis: None` — `matrix_engine`, `skinny_gemm_or_gemv_engine`,
`dma_engine`, `multi_stream_dma_descriptor`, `prefetch_descriptor`, `partial_sum_object`,
`accumulator_merge`. No compiler proof gates them; they are hardware-sizing questions.

## What is blocked, and why

Seven of the 27 are `blocked` or `unavailable` **by construction**, not by oversight:

- `erased` (the dequantized f32 capture removed packed weights and scales) —
  `packed_lowbit_tensor`, `scale_object`, `resident_packed_weight_object`, `fused_dequant_matmul`,
  `native_lowbit_matmul`. Missing evidence: a low-bit capture (packed weights + scales) plus
  per-format accuracy.
- `kv` (attention lowered into matmul projections) — `prefix_kv_object`, `kv_cache_object`.
  Missing evidence: a loop-preserving capture that retains attention/KV structure.

These two flags plus the `cp_axis` status are what separates a value recovered from the IR from one
that rests on an assumption. Each knob's `evidence` field (`recovered_from_ir` vs `unavailable`) is
the machine-readable form of the same distinction.

## `boundary_pressure_score` is evidence breadth

Not performance, not priority, not benefit. It is the sum of `pressure_components`:
`n_supporting_workloads`, `n_region_roles`, `crosses_rate_boundary`, `in_repeated_loop`,
`compiler_provable`, minus `overfit_penalty` and `missing_evidence_penalty`. A high score means many
phases point at the abstraction, never that it would be fast. Every certificate also carries an
explicit `what_is_not_claimed` field: no speedup, cycles, area, or energy claim, and no chosen design.

## The responsibility-split matrix is separate

`responsibility_rows()` emits a coarser cross-cutting view that is **not** part of the 27: 17
functions (`region_partitioning`, `layout_selection`, `dtype_selection`, `weight_packing`,
`scale_metadata_management`, `resident_object_lifetime`, `K_loop_iteration`,
`loop_carried_state_update`, `command_dependency_tracking`, `event_synchronization`, `DMA_prefetch`,
`buffer_allocation`, `sharding_split`, `partial_sum_merge`, `epilogue_requant`,
`deadline_enforcement`, `safety_action_commit`) × six columns (`compiler`, `runtime_hal`,
`command_processor`, `accelerator_isa`, `device_microcode`, `datapath`), with cells drawn from
`owns` / `assists` / `declares` / `consumes` / `not_applicable` / `unknown`. Functions whose evidence
a model-forward capture does not carry (`deadline_enforcement`, `safety_action_commit`,
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
