---
title: Shared-input contraction scheduling
kind: reference
status: current
owner: core
last_verified: 2026-10-20
related: [shared_pass_phase2, architecture]
code_refs: [src/merlin/compile/scheduling/shared_input_groups.py, src/merlin/compile/scheduling/shared_input_schedule.py, src/merlin/compile/scheduling/shared_input_verification.py, merlin/tests/infra/test_shared_input_schedule.py]
---

# Shared-input contraction scheduling

`find_shared_input_groups` recognizes consecutive, independent signed integer
contractions that read the **same immutable tensor SSA value**. It reuses the
existing exact matmul and one-use zero-initializer checks. Equal tensor shapes,
similar value names and model identities do not establish shared input.

Only checked scalar-zero, tensor-splat and contraction operations occur inside
each proposed group. An unknown operation or an intervening result consumer
splits the region. Every original operation, SSA use and result stays intact.
Different output widths are supported; each result keeps its original later
epilogue and its own source-operation identity. The current family recognizes
ordinary unencoded `i8` inputs and exact signed modular `i32` accumulation only.

`schedule_shared_input` retains every reduction tile of one input M panel in
explicitly assigned private storage across **all** sibling output waves. RHS
panels and output accumulators occupy bounded reusable regions. Output waves
may cross sibling boundaries, but each load/store still names its original
RHS/result; no concatenated tensor, weight copy or packed host buffer is needed.
K remains increasing for each independent output. Live output accumulators
remain owned until complete readout. Weights may stay in the logical array
across M tiles within a wave.

Primitive M/K/N dimensions, usable resource bytes, physical row widths,
alignments and backing intervals are selected inputs. Capacities are after
other reservations. Padded rows and aliases between differently named views
are included in admission. No literal hardware capacity, address scheme,
instruction encoding, model name or provenance selector chooses a schedule.
If one full-K input stripe cannot fit, this family declines. Existing compact
dense and bounded convolution planners cover other dataflows.

The bounded search exposes M/K/N panel choices through `selected_block` for
an external tuner. Its default orders choices by requested logical operand
bytes, command count and storage use. It reports no cycle estimate. The input
is read once per group, rather than at least once per separately scheduled
contraction. RHS replication still depends on the chosen M panel. This does
not imply the same reduction in complete runtime.

`verify_shared_input` independently reconstructs source row intervals from
actual loads, array ownership from preload/compute effects, and accumulator
ownership from each output's next required K index. It checks complete,
nonduplicated output intervals for every original result. Row-interval
induction proves all products without expanding M*K*N individual products.
Tests additionally execute actual byte addresses and compare every result
with independent matrix arithmetic, including tails, different rectangular
geometries, padded rows, retained weights and modular overflow. Corrupted
source bindings, addresses, continuations, output identities and effects fail.

These are synchronous **logical** schedules, emitted as bounded finite effect
tuples. They provide neither compact machine loops nor asynchronous overlap.
An independently qualified target consumer must bind current typed inputs,
physical external-buffer ownership, configuration, actual instruction effects,
retained-array support, arithmetic, completion and readout. It must project its
actual emitted source bindings/addresses into the checker; copying planned
effects into a report is insufficient. Unknown physical semantics retain the
original route. These APIs do not generate target instructions or admit FSM
or hardware-loop instructions.

New source owners require a fresh explicitly reviewed shared-library release.
An existing frozen library or successful functional compiler stays unchanged.
Target integration is authored through the normal performance phase starting
from that immutable functional compiler; these helpers are not a handwritten
target consumer or a measured accelerator improvement.
