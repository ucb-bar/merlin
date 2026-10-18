---
title: "Design note: output-epilogue capabilities are derived, and fused only when exact"
kind: design
status: current
owner: targetgen
last_verified: 2026-10-18
related: [accumulator_load_order, capsule_phase_split]
code_refs: [src/merlin/targetgen/rtl/output_epilogue.py, src/merlin/targetgen/output_path.py, src/merlin/xdsl_dialects/lowering/output_fusion.py, src/merlin/targetgen/capability_manifests.py, src/merlin/targetgen/readout_facet.py, src/merlin/xdsl_dialects/lowering/compute_groups.py, src/merlin/xdsl_dialects/lowering/group_exact_float.py, src/merlin/llvmlower/source_compute_groups.py, src/merlin/targetgen/conformance.py, src/merlin/targetgen/corpus_synth.py, packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_full_models.py, merlin/schemas/target_contract.schema.yaml]
---

# Output-epilogue capabilities are derived, and fused only when exact

A quantized model follows every integer contraction with the same chain: convert the
accumulator, scale it, add a bias, rectify, requantize, and sometimes add a residual or pool a
window. Many accelerators do part of this as they move the accumulator out. Three questions decide
how much a compiler is asked to use that hardware: what the design's output path does, which part of
the source chain it can compute *exactly*, and what Phase 0 demands. Each is answered from the target's
own sources, and nowhere by a target name.

## What is derived

`merlin.targetgen.rtl.output_epilogue` produces an `output_epilogue_v1` record from the target's
facts and its elaborated FIRRTL. Each capability is `derived` or `UNKNOWN` with a reason, never a
default:

| capability | evidence (structural rung) |
| --- | --- |
| `scale` | the scale's IEEE format, read from how the readout splits the scale into sign, exponent and fraction (looked up by field widths); the instance that multiplies it with the accumulator; `columns_per_scale`, the element count of the vector the one scale is carried beside; rounding and clamp from the scalar readout contract |
| `activation` | `mux(eq(<activation field>, c), max(x, 0), x)` makes code `c` a rectifier; whether it precedes the scale is read from the fan-in. Other compared codes are listed, never claimed |
| `accumulate` | the `accumulate_on_load` behavioural fact (`rtl/accumulate_fact.py`) |
| `bias_preload` | the accumulator store writes the write's own data when its accumulate bit is clear, and an instance on the load path copies accumulator-width lanes with no arithmetic: a full-width value is loaded unchanged (`dtype`, `bits`, `range`, `rounding: none`) |
| `full_width_readout` | the accumulator value is copied whole into a field of its own type, and an output field of that type is driven with no arithmetic: the raw accumulator can leave the unit |
| `pool` | the elaborated pooling build gate plus register fields with the `pool` role; the reduction comes from the gate's feature-name tokens in a closed vocabulary |

## The output path: one target-neutral sequence

`merlin.targetgen.output_path` restates the record as `output_path_v1`, the only thing grouping
reads:

- `accumulator_stages`: integer stages that change the accumulator before readout. These are
  `preload` (with its range) and `accumulate` (an operand at unit scale).
- `readout`, in order: `activation` (domain `accumulator` or `scaled`), `convert` (to a format, with
  rounding), `scale` (format, rounding, granularity `per_tensor` / `per_command` / `per_channel`,
  `columns_per_scale`), `requant` (format, rounding, clamp), then `pool` (reductions, padding).
- `exits`: `accumulator` (raw, full width) and `requant`.

The validator refuses a path out of order (a scale before its convert, a stage after requant other
than a pool) or out of vocabulary.

## One fusion pass

`merlin.xdsl_dialects.lowering.output_fusion` reads a contraction's source chain by structure. It
reads convert, scales (scalar or per channel, following the channel axis through views and
transposes), bias, residual, rectifier, pad, pool and quantize. It then tries the cuts longest-first:

1. `pool`: the whole chain leaves pooled and requantized;
2. `requant`: the whole chain up to the quantize;
3. `accumulator`: the bias is preloaded and the residual accumulated; the raw accumulator leaves and
   the host runs the rest as written, without those adds;
4. `bare`: the contraction alone.

A cut is absorbed only with a complete proof. Every stage is monotone for positive, finite scales.
So the source and device functions are equal over the whole accumulator range exactly when every
output level is first reached at the same accumulator value, channel by channel. The range comes from
K, the stored weights and the operand formats. One integer offset (the preload) must align every
level. With a residual, it must align them for every value the residual can take. Order (relu
before or after the scale), formats, rounding and clamp are modelled exactly as the path declares
them. A per-channel scale is foldable only where it is uniform within each group of
`columns_per_scale` columns. The proof then records how many readout commands and scale
reconfigurations that costs, and it never issues one command per column.

`compute_groups.form_groups(fusion=...)` applies the decisions. It records on each group the cut, the
absorbed stages, the proof and every longer cut's refusal. The private full-model gate runs the pass
whenever the selected contract declares an output path. Otherwise the earlier rules, including the
`group_exact_float` proof, still apply. The host-compute audit of built artifacts is unchanged.

## What makes the pass refuse

| code | gap | meaning |
| --- | --- | --- |
| `stage_undeclared` | OG1 | a source stage no declared stage performs (no rectifier, no preload for a bias, no accumulate for a residual, a pool on a cut that does not pool, a zero point) |
| `exit_undeclared` | OG1 | the path declares no exit at this cut |
| `pool_padding_unknown` | OG1 | a padded pool where the path's padding value is not derived |
| `scale_granularity` | OG0 | the device scale cannot vary as the source scale does |
| `inexact` | OG0 | no integer offset reproduces every output level. Examples: a float bias off the accumulator grid, a residual off the accumulator scale, a narrower float format |
| `source_unread` | OG7 | the chain's numbers, channel axis or closing quantize could not be read |
| `format_unmodeled` | OG6 | a format or rounding the proof has no exact model of |

## How a second target declares its output path

Nothing in shared code changes for a new target. Its selected contract names, under
`rtl_extraction.output_epilogue: {module, value, scale, activation}`, the elaborated readout module and,
inside it, the accumulator element, the scale operand and the activation selector. Its facts already
carry the datapaths, the register layouts and the elaborated feature gates. Its support provider's
`semantic_probes.yaml` declares the `accumulate_on_load` store and load names. A design that pools,
accumulates or preloads in another way states it through the same structures. Any reading that fails
leaves the capability UNKNOWN, and the pass then refuses whatever needed it.
`derive_manifest` writes the record and its `output_path`. A selected contract built some other way is
declared with `python -m merlin.targetgen.rtl.output_epilogue --contract C --facts F --probe '{...}'
--semantic-facts S --out NEW`, which writes a new file and never overwrites one.
`merlin/tests/ir/test_output_fusion.py` drives the pass over synthetic paths: no activation,
activation after the scale, per-channel scale, another mesh width, no preload, no accumulate, no
pool, and other scale formats.

## What Phase 0 demands

When a path is declared, the epilogue axis requires each stage it performs (`derived_output_path`
evidence), unless a declared readout refuses the stage. The group axis demands, for each contraction
format, the requantizing chain with the preloaded bias, and the same chain closed by the pool.
Capabilities no capsule stage can state yet, an accumulated residual and a column-group scale, are
listed as unstated demands with the reason.

## Known limits

- An accumulated residual needs its operand as a dequantized integer tensor at the accumulator's
  scale; a float residual from the host is not proved.
- A decision is per group, all channels or none, so a few exact channels do not split a group.
- A selected contract carries the path only once it is declared from its facts. Until then grouping
  behaves as before.
