# Localizing a numeric mismatch, and the cost side of quantization

## 1. Read the failing row

The self-check response (`.qa_channel/resp_ID.json`, or the file given to `--out`) has one
`per_capsule` row per case; failing rows carry the full detail:

| Field | Meaning |
|---|---|
| `numeric.policy` | `exact_int`, `bounded_int` / `tolerance_float` (`atol`, `rtol` from the case), or `per_output` |
| `numeric.mismatch_count`, `max_abs_diff`, `max_rel_error` | over all outputs |
| `numeric.first_mismatch` | `{output, index, your_value}`; the expected value is withheld |
| `numeric.per_output.NAME` | `mismatch_count`, `n_elements`, `mismatch_indices` (first 64, flat row-major), `saturated` (every element wrong), `failure_class` |
| `numeric.missing_outputs` | declared outputs your program never stored |
| `tiers` | `L0`/`L1` interpret your command buffer; `L2`/`L3` execute your emitted artifact |
| `failure` | `plane`, `category`, `detail` (a decoded-artifact divergence when one is found) |
| `your_artifacts` | copies under `selfcheck_out/CASE/`: `command_buffer.json`, `instruction_trace.json`, `lowered.llvm.mlir` |

`qa/verdict.json` deliberately keeps only `status`, `policy` and `mismatch_count`.

## 2. Classify before changing code

1. `failure_class: output_never_written` (every observed value equal) or `missing_outputs`: a store
   is missing or mis-addressed, not arithmetic. `output_shape_mismatch`: the output geometry is
   wrong; the count is not a count of wrong values.
2. Which tier fails first: L0 failing means the command buffer does not compute the declared
   operation (stage list, order, operands, scale value). L0 and L1 passing with L2 or L3 failing
   means the emitted artifact differs from the buffer: a configuration it did not set (scale,
   activation, readout width), a stride or offset, or a readout exit that does not apply the
   stage (a full-width readout writes the raw accumulator).
3. The pattern of `mismatch_indices` (convert a flat index to `row = i // N`, `col = i % N` with the
   output's last extent N):

| Pattern | Usual cause |
|---|---|
| whole columns, or every column of one readout group | wrong per-column scale or bias, scale runs misaligned with `columns_per_scale`, transposed weight layout |
| whole rows or tiles; only the tail rows/columns | layout, stride, padding or tail handling; a tile edge read past its data |
| scattered elements off by exactly 1 | rounding mode (half up vs half to even, `x / s` vs `x * (1/s)`), two scales folded without `prove_scale`, bias added after scaling instead of in the accumulator |
| errors only where the true value is large or saturated | missing saturation, ReLU applied before a negative bias, a clamp to the wrong dtype range |
| errors growing with K, or only for large stimuli | accumulator overflow: compare `integer_partial_sum_bound.bound` with `signed_positive_limit`, split K or accumulate in parts |
| all values negative-free but the source has negatives (or the reverse) | ReLU applied on an exit that skips it, or applied twice |
| a whole output off by a constant factor | scale applied twice or not at all; a dequantize scale left in |

## 3. Reproduce with the library, on your own values

Feed suspect accumulators (the edges `quant_params.py case NAME --prove` prints: first/last
thresholds, +-2**24, the bound) through the reference arithmetic and the readout proofs
(`library_api.md`, first and third blocks), and through the same values in your lowering. Disagreement on a
handful of chosen values localizes rounding, order and saturation defects without any golden.
For a host readout, `integer_readout.emit_readout` is exact for every i32 input, so swapping it in
separates a readout defect from a contraction defect.

## 4. Bisect a model

- Each model has public layer cases built from its own device groups (`source_reference` says
  `isolated verification of a device group of MODEL`); run those first with
  `agent_selfcheck.py --capsules A,B`.
- `python agent_model_check.py replay-kernel --model M --program P --group SOURCE_REGION_ID` replays
  one kernel group; `quant_params.py ir` on the model's interface lists region ids
  (`prov.region_id`) with their quantization chains.
- Walk the regions in program order; the first group whose replay fails is the first diverging
  tensor. Check its chain stages against the pipeline in `semantics.md`.
- Integer index arithmetic in emitted C can overflow on large shapes even when values are right:
  `merlin/sched/ir/integer_domain.py` checks every emitted index expression for 32 or 64 bits.

## 5. Performance

- Operand bytes: int8 is 1 byte, i32/f32 are 4. A tile of K x N int8 weights needs a quarter of
  the local memory, so larger tiles fit, operands are reused longer and the array stays busy.
- Output bytes: narrowing on the device stores 1 byte per element; a raw i32 readout stores 4 and
  leaves a host pass over the output (convert, scale, round, clamp) on the critical path.
- Readout settings: each change of scale between readout commands is a configuration command.
  `partition_column_scales` counts the equal-scale runs; the fusion proof reports
  `commands` and `scale_reconfigurations` (`quant_params.py ir --path`).
- Host requantization that remains: `emit_readout(..., fixedpoint=True, saturation_first=True,
  packet=8)` is the cheapest exact form the library emits.
- Measure before and after with skill `roofline-and-utilization`; compare variants by cycles with
  unchanged numerics with skill `ab-library-pass`.
