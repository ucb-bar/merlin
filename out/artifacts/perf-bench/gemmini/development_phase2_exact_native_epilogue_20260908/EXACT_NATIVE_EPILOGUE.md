# Exact native epilogue code generation

This isolated Phase-2 candidate makes the prior analysis-only epilogue formation real for the exact
subset a target can implement: identity accumulator/output/per-channel scales, zero per-channel
bias, optional ReLU, and saturating i8 readout. It deletes the full-width intermediate and host work
instead of merely relabeling tasks. Runtime or nonidentity parameters, residual operands, and changed
floating-point order fail closed.

The generic `17x31x19` gate proves actual emitted-code change. In one same-process warm-then-measured
Spike run, the canonical path takes 13,760 proxy cycles and the selected path takes 56. Tasks fall
from 3 to 1, host segments from 2 to 0, and 1,292 bytes of i32 intermediate storage disappear. The
output DMA shrinks from 1,292 to 323 bytes. Physical fences remain 2 because a terminal output still
requires completion before readback; the compiler does not claim a fence deletion that did not occur.

The full current PT2E ResNet-50 does not enter that exact subset. Of 54 mesh producers, 32
non-residual chains have runtime/nonidentity accumulator scales, 20 terminate in a residual second
tensor, and the stem/maxpool and terminal FC lack the admitted exact terminal. Consequently 0 of 53
convolutions reach native LOOP_CONV after source-semantic formation. This is consistent with the
independent 53-convolution audit: every convolution has per-channel f32 weight scales and non-integral
f32 bias, so none can be replaced by Gemmini's single scalar store scale plus accumulator-unit bias
without changing arithmetic.

That refusal is why the ResNet target and object remain byte-identical to q535:
`c18b8671...` and `e251d0e4...`. All physical deltas are zero—tasks, scalar instructions,
intermediate bytes, fences, DMA bytes, and commands. The exact q535 result therefore transfers by
object identity (1,316,619,699 FireSim cycles, 506,265,232 instructions, 1,196,945,312 Gemmini DMA
bytes, exact 1,000-logit output), but this candidate claims no new ResNet speedup and ran no hardware.

The apparent paradox in the earlier analysis-only artifact is resolved: the canonical source path
already groups each conv-to-bias/ReLU/QDQ slice into one host segment and emits one fused host
traversal. Reassigning the same operations to a compound task changes metadata, not the emitted fused
host loop, full-width MVOUT, or synchronization. Only a target-selected narrow readout changes code.

The smallest honest next mechanism is a target-neutral ordered runtime quantized-epilogue value with
explicit accumulator encoding and a fused-host fallback. On Gemmini, first prove a compute-only
LOOP_CONV accumulator layout and follow it with an explicit full-width MVOUT into the existing exact
one-pass runtime-scale host readout. That can delete scalar im2col and its synchronization while
preserving PT2E arithmetic. Exact boundary deletion additionally needs a target capable of ordered
per-channel f32 scale, then f32 bias, then activation/rounding—or an explicitly authorized change to
the quantization contract. Precombining scales, rounding bias into accumulator units, or importing
TVM/model constants would be wrong.

Evidence is sealed in `validation/exact_native_epilogue_receipt.json`. The reviewable compiler delta
is `exact_native_epilogue_compiler.patch`; `verify_exact_native_epilogue.py` checks identity,
refusals, warm protocol, and portfolio hashes.

## Verify

```sh
artifact=out/artifacts/perf-bench/gemmini/development_phase2_exact_native_epilogue_20260908
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$artifact/compiler:merlin/python" \
  .venv/bin/pytest -q "$artifact/tests"
"$artifact/verify_exact_native_epilogue.py"
```
