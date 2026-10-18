---
title: "Design note: a target's quantization methodology, applied to every model"
kind: design
status: current
owner: targetgen
last_verified: 2026-10-18
related: [output_epilogue_capabilities, accumulator_load_order]
code_refs: [src/merlin/targetgen/quant_methodology.py, src/merlin/targetgen/_integer_epilogue.py, src/merlin/targetgen/_recipe_quantizer.py, src/merlin/targetgen/quantization_spec.py, src/merlin/targetgen/output_path.py, src/merlin/xdsl_dialects/lowering/output_fusion.py, packages/merlin-experiments/src/merlin_experiments/phase0/capture_recipe.py, packages/merlin-experiments/src/merlin_experiments/phase0/requirements.py, examples/gemmini/target/software-spec.yaml]
---

# A target's quantization methodology, applied to every model

A recipe says which number formats a unit computes. That does not quantize a model the way the unit
computes it. The output path (`output_path_v1`, see the
[output-epilogue note](output_epilogue_capabilities.md)) fixes much more: an i32 value preloaded into
the accumulator, a rectifier before or after the scale, one f32 scale per readout command covering
`columns_per_scale` columns, round half to even, a clamp, a window maximum. A model quantized without
those facts computes a different function, and the fusion pass's exactness proof then fails on
numbers rather than on hardware:

- a per-channel weight scale that varies inside a 16-column group: `scale_granularity`;
- a float bias off the accumulator grid: `inexact`;
- three float roundings (`* s_x`, `* s_w`, `* 1/s_out`) where the unit rounds once: `inexact`.

So each target declares ONE methodology, and every model's recipe is derived from it automatically.

## Where it is declared

In the target's software spec, under `quantization.methodology`
(`examples/<target>/target/software-spec.yaml`). It is validated with the rest of the spec
(`software_spec.load_software_spec` → `quantization_spec.validate_quantization_declarations` →
`quant_methodology.validate`). The vocabulary is closed:

| field | values | meaning |
| --- | --- | --- |
| `scale_granularity` | `from_output_path`, `per_tensor` | weight scales per readout command (`channels_per_scale = columns_per_scale`), per column, or per tensor, as the path's scale stage says |
| `bias` | `accumulator_preload_int32`, `host_float` | a bias is the i32 code `round_half_even(b / (s_x * s_w))`, refused outside i32, where the path preloads; otherwise a host add |
| `epilogue` | `integer_domain` | the quantized graph computes the readout's own arithmetic, in the path's order |
| `rounding`, `clamp` | `from_output_path` | the requant rounding and integer range; a rounding the graph cannot spell exactly makes the recipe underivable |
| `calibration` | `activation: mse \| percentile \| minmax`, `weight: minmax` | `mse` is TorchAO's histogram observer (it minimizes the L2 quantization error); `percentile` states its percentile |
| `recovery` | `cross_layer_equalization`, `bias_correction` (both stated) | applied generically before observers and before conversion |
| `float_island` | `dtype: fp32`, `ops` from `softmax, layer_norm, rms_norm, group_norm, gelu, silu, rotary` | operations that stay fp32 on Merlin's host; a narrower value (bf16, f16) there is refused, and the model's own wider arithmetic (an f64 table) is recorded |
| `model_variants` | reviewed rows only | see below |

Every hardware value is the word `from_output_path`; no number about the hardware is written in the
spec.

## How a recipe is derived

`quant_methodology.derive(base, methodology, output_path, model=None)` takes the hardware-derived
base recipe (formats, families, software admission) and the selected contract's derived output path
(`output_epilogue.output_path`) and returns a new `quant_recipe_v1`:

- `weight.channels_per_scale` from the scale stage, the activation range from the requant clamp,
  observers from the calibration methods;
- `bias_domain: accumulator` and `epilogue.bias.mode: accumulator_preload` when the path preloads an
  i32 value;
- an `epilogue` block (`integer_epilogue_v1`): the readout order, the rectifier and its domain, the
  convert, scale, requant and pool stages, the bias mode and the open exits;
- `recovery` and `float_island` blocks, and a `methodology` block naming the methodology's and the
  output path's digests, the model, and any variant applied.

A base's own variant blocks (`dynamic`, `static_planes`, GPTQ rounding) are dropped. A fact the path
does not establish (no path, no `columns_per_scale`, a non-RNE rounding, a non-f32 scale, a narrower
requant format) makes the recipe `underivable` with the reason. It is never filled with a default, and
the hardware-only recipe is never used in its place.

## The one generic quantizer

The recipe crosses to the capture interpreter as JSON. There `_recipe_quantizer.build_quantizer`
builds a TorchAO PT2E `Quantizer`; when the recipe has an `epilogue` block, its sibling
`_integer_epilogue.Epilogue` does three things.

**Annotation (one pattern per contraction).** A contraction (`conv2d`, `linear`) and the stages after it
that the path performs (a `relu`, then an unpadded max pool when the path pools) are annotated as ONE
pattern. The bias is a `DerivedQuantizationSpec`: int32, scale `s_x * s_w`, per channel when the
weight is. The output is quantized only at the requantizing exit. A pattern takes that exit when every
reader requantizes the value anyway (another quantized contraction, possibly through shape-only views
or a host max pool that shares the grid). Otherwise it leaves through the full-width accumulator exit
when the path has one, and the host applies the rest as written. A padded pool, whose padding value
the path does not state, stays on the host and reads the requantized integers unchanged.

**Rewrite (after `convert_pt2e`).** Each pattern is restated as the unit computes it:

```text
acc  = int8(x) @ int8(w)            dequantize of scale 1: the integerizer emits an exact i32 contraction
acc += k                            the i32 bias code, carried exactly in f32
acc  = relu(acc)                    in the accumulator domain when the path says so
y    = f32(acc) * M                 M = f32(s_x * s_w / s_out), one value per column group
q    = clamp(rne(y), lo, hi)        quantize of scale 1: round half to even, the path's clamp
```

At the full-width exit `M` is `f32(s_x * s_w)` and the host continues in float. The fusion pass reads
back exactly this arithmetic. Its offset proof therefore holds by construction: the device multiplier
is the graph's own `M`, and the bias offset is the graph's own `k`. The PT2E integer reference
reproduces the integerized graph bit for bit. A pattern the rewrite cannot express stays in PT2E form
and is listed with the reason (`quantization_stats.integer_epilogue.kept`).

The portable graph carries the accumulator and bias code in fp32. Before treating this as an
int32 preload, the rewrite bounds each output channel by the activation code range, the sum of
absolute frozen weight codes, and the absolute bias code. It requires that complete bound to fit
the exact fp32 integer range. Checking the bias code alone is insufficient: rounding an accumulator
above that range before adding its bias can differ from adding the bias in int32 first. A site
without this proof remains in PT2E form and carries an explicit refusal.

**Recovery.** Cross-layer equalization (Nagel et al., 2019) runs in `transform_for_annotation`, after
Conv+BN folding and before any observer. It covers contraction pairs joined only by positively
homogeneous steps: a relu, a max pool, or a product with a tensor not computed from the first
contraction. Empirical bias correction, `b -= (Q(W) - W) E[x]`, uses each site's input mean, recorded
by hooks on its activation observer during the calibration pass. It runs before conversion, so the
DerivedQuantizationSpec quantizes the corrected bias. The float island is checked on the converted
graph.

## How a second target declares its own

Nothing in shared code changes. The target adds `quantization.methodology` to its own software spec,
choosing calibration, recovery and its island, and writes `from_output_path` for the hardware values.
Its selected contract carries the output path that `output_epilogue.declare` / `derive_manifest`
derived from its facts. If that path has no preload, the bias becomes a host add
(`epilogue.bias.mode: host_float`). If it has no rectifier, the relu stays on the host. With a scale
per column, the weights take ungrouped channel scales. Another clamp becomes every activation's grid.
A rounding or format the capture graph cannot spell exactly is refused. All of these cases are covered
by `merlin/tests/targetgen/test_quant_methodology.py` (derivation) and
`test_recipe_integer_epilogue.py` (the quantized graphs, in the capture interpreter).

## How model variants are reviewed

A model's recipe differs from the shared one only through a `model_variants` row:

```yaml
- model: tiny_llama
  name: static-heads-softmax2
  status: reviewed            # the only accepted status
  reviewed_by: <who approved it, when>
  reason: <why this model needs it>
  measured_accuracy: {basis: <how measured>, <metric>: <number>, ...}
  changes: {families: [...], dynamic: {...}, calibration_count: N}   # nothing else may change
```

A row without a review, a reason or a measurement is refused, and so is a change outside `changes`.
The private gate's own `capture_recipe_variant` review (base vs selected digests) still applies to
whatever recipe a sealed capture selects.

`dynamic.planes.softmax_partner` gives the operand beside a softmax output extra static residual
planes. Each plane uses the same signed-int8 round-and-clamp producer domain; successive steps are
`quant_max + 1` times finer. This preserves the host cast proof without introducing a wider integer
clamp or floor. With two planes the context contraction takes twice as many device calls, while
the score contraction keeps its original planes. Captures record `operand_encoding: bounded_residual`.
Historical `planes.operand` recipes retain their original radix arithmetic. Selecting extra partner
planes remains a numerical recipe choice requiring measurement and review.

## Where each model's derived recipe is recorded

- Phase 0 evidence: `software/quantization-contract.json` records the methodology and the selected
  output path. `software/quantization-recipes.json` lists the shared recipe under `recipes` and one
  file per reviewed variant under `model_variants`. Phase 0 requirements accept a capture made under
  either.
- The recipe itself: its `methodology` block (`methodology_sha256`, `output_path_sha256`, `model`,
  `variants`). Its `recipe_sha256` changes with any of them.
- A capture: `meta.json` `quantization_stats.integer_epilogue`, with patterns, exits, absorbed
  stages, equalized pairs, bias corrections and kept sites with reasons; `float_island` holds the
  operations checked fp32.
- A model-specific candidate is made with
  `python -m merlin.targetgen.quant_methodology --spec S --contract C --base B --model M --out NEW`.
  It writes a NEW file and never overwrites one. Reviewed recipes stay readable and unchanged.

## Known limits

- A residual sum is not yet absorbed. Its operand would need to sit on the receiving contraction's
  accumulator grid, and the fusion pass ranges a residual over its integer format, which is
  intractable for i32. Residual-carrying contractions leave through the full-width exit with the bias
  preloaded.
- A padded max pool stays on the host until the path's pool padding value is derived.
- An activation other than a rectifier (SiLU, GELU) is not an output-path stage. A transformer's
  Linear layers therefore leave full width; the methodology changes their scale granularity and their
  bias, not their exit.
- Bias correction uses each input channel's mean over all positions, so a convolution's padded border
  is approximated.
