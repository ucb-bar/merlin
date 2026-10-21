---
title: "Design note: a target's quantization methodology, applied to every model"
kind: design
status: current
owner: targetgen
last_verified: 2026-10-21
related: [output_epilogue_capabilities, accumulator_load_order, model_check]
code_refs: [src/merlin/targetgen/quant_methodology.py, src/merlin/targetgen/quant_recipe.py, src/merlin/targetgen/_integer_epilogue.py, src/merlin/targetgen/_recipe_quantizer.py, src/merlin/targetgen/_recipe_observers.py, src/merlin/targetgen/_m2m_capture_artifacts.py, src/merlin/targetgen/quantization_spec.py, src/merlin/targetgen/output_path.py, src/merlin/xdsl_dialects/lowering/output_fusion.py, src/merlin/runtime/linalg_numpy.py, merlin/tests/runtime/test_linalg_numpy_reduction_order.py, packages/merlin-experiments/src/merlin_experiments/phase0/capture_recipe.py, packages/merlin-experiments/src/merlin_experiments/phase0/requirements.py, examples/gemmini/target/software-spec.yaml]
---

# A target's quantization methodology, applied to every model

A recipe says which number formats a unit computes. That does not quantize a model the way the unit
computes it. The output path (`output_path_v1`, see the
[output-epilogue note](output_epilogue_capabilities.md)) fixes much more: an i32 value preloaded into
the accumulator, a rectifier before or after the scale, one f32 scale per readout command covering
`columns_per_scale` columns, round half to even, a clamp, a window maximum. A model quantized without
those facts computes a different function, and the fusion pass's exactness proof then fails on
numbers rather than on hardware:

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
| `scale_granularity` | `from_output_path`, `per_tensor`, `per_channel` | retain the existing path-derived policy, select tensor scales, or explicitly select independent logical channel scales |
| `bias` | `accumulator_preload_int32`, `host_float` | a bias is the i32 code `round_half_even(b / (s_x * s_w))`, refused outside i32, where the path preloads; otherwise a host add |
| `epilogue` | `integer_domain` | the quantized graph computes the readout's own arithmetic, in the path's order |
| `rounding`, `clamp` | `from_output_path` | the requant rounding and integer range; a rounding the graph cannot spell exactly makes the recipe underivable |
| `calibration` | `activation: mse \| percentile \| minmax`, `weight: minmax` | `mse` is TorchAO's histogram observer (it minimizes the L2 quantization error); `percentile` states its percentile |
| `recovery` | `cross_layer_equalization`, `bias_correction` (both stated) | applied generically before observers and before conversion |
| `float_island` | `dtype: fp32`, `ops` from `softmax, layer_norm, rms_norm, group_norm, gelu, silu, rotary` | operations that stay fp32 on Merlin's host; a narrower value (bf16, f16) there is refused, and the model's own wider arithmetic (an f64 table) is recorded |
| `model_variants` | reviewed rows only | see below |

Every hardware value is the word `from_output_path`; no number about the hardware is written in the
spec.

`per_channel` selects source quantization granularity. A configurable per-command
readout retains its derived `columns_per_scale`, and its recipe records
`epilogue.scale.logical_granularity: per_channel`. The compiler must implement
independent output partitions where gains differ within that physical vector.
The source arithmetic proof binds the full gain vector and physical width;
packing, commands, resources and actual execution still require qualification.
An unknown physical width or a per-tensor-only scale refuses this selection.
Existing `from_output_path` recipes retain their exact bytes.

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
reader requantizes the value anyway (another quantized contraction, possibly through shape-only
views). Otherwise it leaves through the full-width accumulator exit
when the path has one, and the host applies the rest as written. A padded pool, whose padding value
the path does not state, stays on the host. When the full-width exit is available, its input is
preserved at that width and the next quantized reader supplies the grid after the host pool.

**Rewrite (after `convert_pt2e`).** Each pattern is restated as the unit computes it:

```text
acc  = int8(x) @ int8(w)            dequantize of scale 1: the integerizer emits an exact i32 contraction
acc += k                            the i32 bias code, carried exactly in f32
acc  = relu(acc)                    in the accumulator domain when the path says so
y    = f32(acc) * M                 M = f32(s_x * s_w / s_out), selected logical channel/group gains
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

**Calibration reuse.** Repeated reads of an unchanged stored weight reuse its min/max observation.
The known grouped observer is covered only while it inherits the framework's exact min/max forward;
grouping changes its qparams calculation, not the observation. Tensor identity, version, storage,
geometry and dtype must all match. Histograms and arbitrary observer subclasses keep every call.

**Agreement receipts.** Calibration comparison covers every tensor result, with
matching tree structure and per-sample shapes. Multi-result programs record both
a concatenated summary and each output's metrics: a correct mask cannot stand
in for a cache, and a large result cannot hide a smaller result's error. Empty,
non-finite or complex results refuse the comparison. Single-result receipts keep
their existing numeric behavior. These observations do not grant model numerical
admission, dataset accuracy or target execution qualification.

**Recovery.** Cross-layer equalization (Nagel et al., 2019) runs in `transform_for_annotation`, after
Conv+BN folding and before any observer. It covers contraction pairs joined only by positively
homogeneous steps: a relu, a max pool, or a product with a tensor not computed from the first
contraction. Empirical bias correction, `b -= (Q(W) - W) E[x]`, uses each site's input mean, recorded
by hooks on its activation observer during the calibration pass. It runs before conversion, so the
DerivedQuantizationSpec quantizes the corrected bias. The float island is checked on the converted
graph.

## Source reductions at quantization boundaries

Floating islands retain an arithmetic contract through source export and lowering. Under Merlin's
strict floating-reduction reference policy, a binary32 sum without permission to reassociate follows
source index order, rounding each addition at its declared type. A framework reduction tree can
produce a different result even when both implementations use binary32. For example, starting at zero and adding
`[1e8, 1, 1, 1, -1e8, 1, 1, 1]` in order gives `3`; NumPy's pairwise reduction gives `0`.
Dividing by `2` and rounding half to even then yields integer codes `2` and `0`.
The existing reduction-order regression checks this boundary and agreement with an independently
compiled strict source loop. Its example does not authorize reassociation in another program.

Small errors in a normalization reduction can similarly move a value across a quantization
half-integer. Comparing only the floating reduction's error or widening a final-output tolerance
does not establish equality of the resulting integer codes. Preserve the actual reduction order,
scale arithmetic, rounding and clamp when diagnosing such a difference.

Three comparisons answer separate questions. Capture/replay establishes the selected framework
execution and its bound inputs. Comparing the prepared source IR with an independently captured
result checks source fidelity. Comparing compiled execution with that IR's reference checks
compiler realization. Agreement with the IR reference alone cannot establish agreement with the
captured framework. A trajectory comparison also needs matching carried-state inputs and every
declared output at each step; a separately captured seed is insufficient. Missing reference
coverage or an unknown interpreter operation leaves that comparison unavailable.

Phase 2 legality preserves the selected source semantics and numerical policy. It cannot repair a
source-fidelity failure by silently changing reduction order, scales, rounding or admission thresholds.
A legal fallback preserves the original computation, including an existing numerical discrepancy.
Numerical quality against the independent floating model, complete functional equivalence,
resource legality and target execution admission remain separate obligations; see the
[model-check contract](../reference/model_check.md).

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
  changes: {families: [...], dynamic: {...}, calibration_count: N, scale_granularity: per_channel}
```

A row without a review, a reason or a measurement is refused, and so is a change outside `changes`.
An unselected variant leaves the complete shared and other
models' recipes unchanged. A new per-channel candidate still needs measured
numerical review, capture/replay and the original model admission gates.
The private gate's own `capture_recipe_variant` review (base vs selected digests) still applies to
whatever recipe a sealed capture selects.

A reviewed row may override `recovery` for its selected model by stating both
`cross_layer_equalization` and `bias_correction` as booleans. This selects the
existing generic recovery mechanisms without changing the shared policy or
another model's recipe. Later matching recovery rows replace the complete block;
partial blocks and unknown fields refuse. Unchanged recipe bytes alone do not
grant capture reuse: the ordinary source, input and issuer checks still apply.

A reviewed row may also change `bias` to either declared bias policy and add a
`static_planes` block. The latter states exactly `schema: static_planes_v1`,
`families: [contraction]`, `planes: 2` (or `3`), an `observer` (`minmax`,
`histogram` or `p<percent>`), and `owners` (null for all supported contractions,
or nonempty module-path prefixes). This selects the existing generic activation
digit decomposition: every device product remains signed int8 by int8 into int32;
the host combines the base-weighted accumulators in fp32 and then adds any bias.
The wider software activation grid does not widen the hardware operand format.
Consequently the final effective recipe must use `host_float` bias placement;
an accumulator-preload claim makes the selection underivable. All matching rows
are applied before checking compatibility, including the final contraction family.
Unsupported requests remain recorded as underivable rather than disappearing.
This choice changes device-call count and host arithmetic, so original numerical,
capture/source, host-compute and target-execution gates remain required. A review
row records a choice and its measurements; it grants none of those verdicts.

`dynamic.planes.softmax_partner` gives the operand beside a softmax output extra static residual
planes. Each plane uses the same signed-int8 round-and-clamp producer domain; successive steps are
`quant_max + 1` times finer. This preserves the host cast proof without introducing a wider integer
clamp or floor. With two planes the context contraction takes twice as many device calls, while
the score contraction keeps its original planes. Captures record `operand_encoding: bounded_residual`.
Historical `planes.operand` recipes retain their original radix arithmetic. Selecting extra partner
planes remains a numerical recipe choice requiring measurement and review.

`dynamic.planes.residual_operand` applies the same bounded residual encoding to every ordinary
operand, including both operands of the score contraction. It is mutually exclusive with the
historical `planes.operand` radix selector. With two residual planes and two softmax planes,
both score and context contractions take four device calls per head, versus one and two with
the original single-operand grid. This additional precision also requires measurement and review.

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
- Complete frontend traces use compact JSON. For sessions, the producer refreshes each trace's
  metadata pointer and capture receipt, then inventories the final session bundle after all
  sidecars exist. Every trace member and byte binding remains in the record.
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
