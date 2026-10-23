# From a fresh backend to competitive generated programs

The objective is a compiler whose generated programs can achieve strong
whole-workload performance through reusable transformations and target-derived
scheduling. Matching another executable byte for byte is not the objective:
instruction sequences, unrolling, layout and allocation can differ while
preserving semantics and obtaining comparable performance.

Use this with [the compiler change map](compiler_change_map.md) and
[the construction rehearsal](construction_rehearsal.md). A successful small
kernel is a first executable milestone, not evidence of complete-model reach.

## Two evidence channels

The operator can privately inspect external programs to identify missing
mechanisms and compare final results. Keep their source, binaries, preferred
parameters, schedules, weights and performance scores outside author inputs.
An archive's label is not proof of matching hardware, quantization, preparation,
cache state, timing boundaries or complete-output validation.

Author inputs remain the selected public workload IR, numerical contracts,
hardware/ISA facts, editable generic library and its own execution feedback.
Any new mechanism must be expressed generically and checked on independently
constructed public cases. Derive target-specific implementations inside the fresh
backend. Do not transplant an external kernel into a pass or turn its schedule
into a required policy. Keep inspection-derived study changes explicit in the
operator record so the experiment's treatment can be reconstructed.

## Mechanisms and evidence

| Mechanism | Representation and compiler boundary | Public witness and measurement |
| --- | --- | --- |
| Preserve activation or weight panels across work | Source operand identity, `dense_source`, compact loops, shared-input groups and target storage adapter | Same input consumed by multiple output blocks or projections. Check all outputs, actual repeated transfers, live storage and complete-call cycles. |
| Prepare weights/layouts once | Shared preparation/host codegen, explicit ownership and invalidation keys | Repeated invocation with unchanged weights, then changed weights/layout. Separate initial preparation from recurring work; verify invalidation. |
| Keep reductions in local accumulators | Checked compact schedule and target accumulation/readout semantics | Reduction spanning several panels with tails and signed values. Check initialization once, continuation, overflow semantics and final stores. |
| Overlap movement, compute and readout | Scheduling IR, dependency/effect contracts, queue capacities and lifetime-aware allocation | Public serialized and legally overlapped alternatives. Check emitted effects and completion; measure queue stalls, traffic and complete calls. A two-buffer allocation alone proves no overlap. |
| Reuse configuration | Configuration state, decoded reads/writes/clobbers and loop invariants | Repeated compatible commands plus intervening clobber/control cases. Count actual configuration operations and CPU issue work. |
| Keep host control compact | Counted IR, LLVM/C emission, address expressions and loop policy | Small and large shapes through the same semantic route. Measure emitted code size, dynamic host work and compilation time; consider instruction-fetch costs before extensive unrolling. |
| Share inputs across outputs | Multi-output graph boundaries, independent epilogues, shared-input schedule | Multiple projections with distinct output shapes/scales and an externally observed input. Compare outputs, preparation, transfers and full-region time. |
| Compose producer/consumer stages | Typed region bindings, all live-outs, allocation and target launch ABI | Dense → pointwise plus an additional consumer of the intermediate. Retain the unfused path; check every observed value and real boundary work. |
| Fuse arithmetic and readout | Source scalar semantics, output-fusion rules, exact device arithmetic | Bias/activation/requantization in different legal and illegal orders, including saturation and rounding boundaries. A similar result format is insufficient. |
| Generate efficient convolution movement | Source index maps, panel/gather planning and target transfer adapter | Interior, padding, stride and channel-tail cases. Measure materialized temporary bytes, repeated gathering and device/CPU placement. |
| Cover non-contraction operators | Semantic capability and numerical admission, placement and host/runtime lowering | Pooling, residual arithmetic, conversion and unsupported nonlinear seams. Distinguish hardware unsupported from compiler unimplemented; check complete chains. |
| Preserve whole-chain state and storage | Connected compilation, prepared buffers, stream/state bindings and shared host runtime | Multiple distinct inputs and recurrent/retained state when required. Measure orchestration, waits, allocation and complete outputs, not just stage sums. |

Not all mechanisms are currently connected end to end. The source-bound LLVM
bridge supports the selected synchronous integer dense domain. Finite copy
pipelining and staged descriptor scheduling have their own effect/ownership
contracts; they are not automatically legal asynchronous compact-loop schedules.
Mixed-region products now preserve bindings, but a target guest launch adapter
and complete-output execution still need to be demonstrated. These are concrete
compiler development tasks, not reasons to mark a hardware capability unsupported.

### First asynchronous compact-loop extension to investigate

The existing C emitter already accepts explicit target `issue_on`, `on_await`
and `on_produce` hooks; the LLVM index bridge deliberately refuses annotations
that it cannot preserve. No additional dialect is needed merely to name these
events. Current token checks visit each loop body once, however, and do not prove
dynamic token generations or all cross-iteration memory lifetimes. Per-call
placement checks are also insufficient for outstanding accesses.

Start with a bounded periodic block whose pending work is empty at entry and
exit. Reuse split/peel/rotation and the existing actual-effect storage checker.
An admission must cover the full affine iteration domain, configuration
invariants, reduction state, resource bounds, final output coverage and separate
tails. Testing two iterations alone cannot prove arbitrary repetition. Initially
retain accesses until actual completion; do not infer earlier release events.

The target must independently project emitted commands back to those effects.
A neutral delayed-completion execution should accept legal independent-slot
overlap and reject early overwrite, missing waits, incorrect tails and incomplete
drains. Only then qualify the target adapter through ordinary numerical and
physical checks. This extension is proposed, not implemented by the current
synchronous dense emitter. Continuous pipelines with work pending across block
boundaries require a further recurrence and lifetime representation.

## Progression through the ordinary authoring workflow

1. Expose the complete public corpus and semantic families. Build one compiler
   across frontend, numerical/layout, scheduling/storage, target lowering and
   host execution. Keep the backend initially empty.
2. Establish one small executable path using the normal emission, linking,
   independent comparison and simulator tools. Start its medium/large variants
   and other ready families while authoring continues on immutable snapshots.
3. For a missing mechanism, first try the applicable existing representation and
   pass. Extend its public interface or algorithm when a concrete mismatch is
   observed. Record the consumed product and the owning compiler layer.
4. Compare a retained correct baseline with a small number of legal alternatives.
   Bind both to the same numerical, hardware, input and measurement identities.
   Count genuine emitted changes, including unsuccessful/no-op alternatives.
   Correlated policy changes establish a policy-bundle result, not attribution
   to each sub-knob.
5. Use affected small cases for correctness and required physical checks. Put
   primary performance work into scalable and connected cases. Use labeled
   provisional estimates until the whole-SoC model is qualified; public physical
   evidence may improve calibration, while private reference scores do not.
6. Validate improvements on independent shapes and source regions. Integrate
   reusable pass changes or semantic/resource predicates into the same compiler.
   Passing one workload must not introduce workload-name dispatch.
7. Compile complete models and preserve all input preparation and host/device
   boundaries in the measurement record. Perform final independent physical
   validation with the selected unchanged hardware configuration.

Offline construction uses the same editable grant, action client, immutable
evaluation and independent numerical checks as automatic authoring. An
operator-only emitter, reference implementation or privileged runtime shortcut
does not demonstrate that the automatic loop can reach the same result.

## What to retain for the research question

Record time and aggregate tokens to executable capability, compiler/IR revisions,
pass and policy choices, actual emitted artifacts, failed legal/numerical
alternatives, relevant simulation cost and large/connected performance. Separate
host pass time, harness admission, target compilation, simulation and feedback.
Record author-visible source changes between experiment versions.

The key comparison is whether the representations and transformation library
reduce duplicated compiler work and enable better legal schedules under the
same workload and resource budget. Available APIs, imported dialects or a
manifest declaring support cannot answer that question on their own.
