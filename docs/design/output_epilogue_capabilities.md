---
title: "Design note: output-epilogue capabilities are derived, and grouped only when exact"
kind: design
status: current
owner: targetgen
last_verified: 2026-10-17
related: [accumulator_load_order, capsule_phase_split]
code_refs: [src/merlin/targetgen/rtl/output_epilogue.py, src/merlin/targetgen/capability_manifests.py, src/merlin/targetgen/readout_facet.py, src/merlin/xdsl_dialects/lowering/compute_groups.py, src/merlin/xdsl_dialects/lowering/group_exact_float.py, src/merlin/targetgen/conformance.py, src/merlin/targetgen/corpus_synth.py, packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_full_models.py, merlin/schemas/target_contract.schema.yaml]
---

# Output-epilogue capabilities are derived, and grouped only when exact

A quantized model follows every integer contraction with the same chain: convert the
accumulator, scale it, add a bias, rectify, requantize, and sometimes add a residual or pool a
window. Many accelerators do this as they move the accumulator out. Whether a compiler is asked to
use that hardware depends on two things: what the target *declares* its output path does, and
whether the chain in the source can be computed *exactly* that way.

## What is derived

`merlin.targetgen.rtl.output_epilogue` produces an `output_epilogue_v1` record from the target's own
sources. Each capability is `derived` or `UNKNOWN` with a reason, never a default:

| capability | evidence (strongest rung) |
| --- | --- |
| `scale` | the scale's IEEE format, read from how the elaborated readout splits the scale into sign, exponent and fraction fields (looked up in the format registry by field widths); the instance that combines it with the accumulator; register fields with the `scale` role; datapath formats; rounding and clamp from the scalar readout contract |
| `activation` | a node `mux(eq(<activation field>, c), max(x, 0), x)` makes code `c` a rectifier; whether it precedes the scale is read from the fan-in. Other compared codes are listed, never claimed |
| `accumulate` | the `accumulate_on_load` behavioural fact (`rtl/accumulate_fact.py`) |
| `pool` | the elaborated pooling build gate in the facts plus register fields with the `pool` role; the reduction comes from the gate's own feature-name tokens in a closed vocabulary |

The module and signal names the FIRRTL reading needs are target data, declared under
`rtl_extraction.output_epilogue: {module, value, scale, activation}`. Without that probe, the
FIRRTL-only capabilities stay UNKNOWN. Nothing in the module names a target, an opcode or a register.

## Where it is declared

`derive_manifest` writes the record into every derived contract. A selected contract that is not
derived (an operator-built JSON) is declared with `python -m merlin.targetgen.rtl.output_epilogue
--contract C --facts F --probe '{...}' --out NEW`, from the facts it is selected with. The command
writes a new file and never overwrites one. `capability_manifests.validate` refuses a malformed
record instead of reading it as absent.

## How grouping uses it

`readout_facet` folds the record into a contraction unit's facet as `derived_stages` (with rung
`rtl_derived`). `TargetReadout.applies_stage` then admits a derived stage even where no readout is
declared, and otherwise answers as before. Admission stays conservative:

- **Exactness before absorption.** A group that takes a converted accumulator through a stored float
  bias and a quantize becomes an integer readout only when `group_exact_float` proves, over the
  stored weights and the full input range, that one integer bias and one scale round exactly as the
  source's f32 chain does. The private full-model gate now runs that proof and re-forms any unproved
  group with its float chain on the host. So the gate never asks for an accelerator epilogue that
  computes something else.
- **Granularity is the readout's.** A scale must vary at a granularity the readout was derived to
  hold. A per-command scale register holds one scale per tensor, so a per-output-channel weight scale
  keeps the epilogue on the host (`scale_granularity`, OG0).
- **Residuals are refused by name.** When the output path is derived to accumulate on load, a
  residual add is refused as `accumulate_exactness_unproved` (OG6): the hardware evidence exists and
  the integer-sum proof does not. Before this change it was refused as a missing declaration.
- **Structural stages only.** A rectifier written as compare-against-zero plus select is classified
  as `relu` only for ordered float and signed integer predicates, with the arms in max order.

The host-compute audit of built device artifacts does not change. Group output extents, which bound
its budget, are still read from the source.

## What Phase 0 demands

A capture can close an epilogue only where its numbers fit the readout. A model quantized per output
channel closes none, so a group axis built only from captures would ask for no fused epilogue.
`conformance._group_axis` therefore also demands, for each contraction format the target declares,
the derived chain: a contraction-scoped bias route, then scale, then rectifier, and the same chain
closed by the derived pool. Only stages the epilogue axis requires are chained, so the capsule
builder can write every demand. Extents come from the builder, not from a workload.

## Known limits

- Finer scale granularity (one scale per output channel through split stores) is not derived. It
  needs a fact about how much of the accumulator one readout command drains.
- A residual needs an exactness proof over pairs of accumulator and residual values before it can
  join a group.
- A selected contract carries the record only once it is declared from its facts. Until then grouping
  behaves exactly as before.
