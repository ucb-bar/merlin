# Independent rotary reference v3

`whole_reference.py` computes rotary preparation in its original-ATen
interpreter. `rotary_source.py` reads the dependency slice of a complete captured
source FX trace, scalar arguments, tensor metadata, and unchanged lifted
inverse-frequency words. The K=1 product uses exact integer binary32 RNE;
transpose and concatenation preserve words. `rotary_exact.py` computes sine and
cosine without libm or framework trigonometric kernels, preserves the source
binary32 scale of one, and applies integer BF16 RNE.

The proof uses exact `Fraction` alternating-arctangent enclosures and Machin's
identity for pi, outward-rounded 160-bit dyadic intervals, and 28 Taylor terms
after the seed. Range reduction propagates the pi interval, a derivative bound
of one encloses input uncertainty, and a factorial Lagrange remainder encloses
the truncated series. Both interval endpoints must round to the same binary32
word; ambiguous rounding refuses rather than selecting a precision heuristic.

Qualification covers positions 0..2047 and all 32 captured inverse frequencies:
65,536 angles and 262,144 duplicated sine/cosine BF16 words. The independent
reference agrees exactly with the selected CORE-MATH archive at binary32 and
BF16 and with torch at BF16. Binary32 framework equivalence is not claimed;
positions beyond 2047 are unqualified. This is host source arithmetic, with no
Atlas execution claim. Chipyard newlib was rejected because its sine changes
one BF16 word at position 1897.

## Reproduce the full-domain check

Run from `examples/atlas/rotary-host` after building and qualifying the selected
archive with the documented parent-example build step. Use a compatible
`$TORCH_PYTHON` with NumPy, torch, and safetensors. `$CAPTURE` must be a complete
authenticated source capture containing `frontend-trace.json` and its unchanged
source tensor archive. Select its trace explicitly; do not substitute a saved
integration fixture or a partial trace.

Set `$ARCHIVE_GATE` to the passing archive qualification receipt and
`$RV64_OUTPUT` to the raw output it authenticates. Use a fresh `$REFERENCE_OUT`.
The validator checks these hashes before comparing the independent results:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
  "$TORCH_PYTHON" reference/validate_rotary_v3.py \
  --trace "$CAPTURE/frontend-trace.json" \
  --archive-gate "$ARCHIVE_GATE" --rv64-output "$RV64_OUTPUT" \
  --out "$REFERENCE_OUT"
ROTARY_SOURCE_TRACE="$CAPTURE/frontend-trace.json" nice -n 10 \
  "$TORCH_PYTHON" -m unittest discover -s reference -p 'test_*.py'
```

`$REFERENCE_OUT/reference.json` binds the oracle, source adapter, interpreter,
actual trace, original frequency bytes, archive gate, and output archive by
SHA-256. `expected.npz` contains independently computed intermediate and output
words. The exhaustive check also replays the actual source rotary dependency
slice, expanding only its arange extent to cover the qualified domain, and
compares every word with the direct independent computation. Lifted frequency
words are authenticated against their original captured bytes.

## Check source replay and fixture refusal

The sequence smoke test supplies deliberately incorrect all-zero rotary fixtures
and verifies that production stage inputs instead come from authenticated fresh
computation:

```sh
nice -n 10 "$TORCH_PYTHON" reference/validate_rotary_sequence_v3.py \
  --source-bundle "$CAPTURE" \
  --qualification "$REFERENCE_OUT/reference.json"
```

For a stage-capsule sequence, pass `--rotary-source-bundle "$CAPTURE"` and
`--rotary-validation "$REFERENCE_OUT/reference.json"` to
`reference/whole_reference.py`, alongside its other required source and policy
arguments. Version 3 replaces the captured `to_7` cosine and `to_8` sine
preparation inputs with freshly interpreted source outputs and refuses those
inputs without authenticated replay. Other preparation and arithmetic policies
retain their behavior. Passing the rotary gate or this sequence smoke does not
establish complete whole-model compilation, decoder execution, generated cache
continuity, or quality.

## Source-order head reference

The head reference widens original BF16 words exactly, starts from positive zero,
and uses separate binary32 multiplication and ascending-K binary32 addition,
followed by one integer ties-to-even BF16 conversion. Tiny products preserve
binary32 gradual-underflow RNE. Nonfinite operands, intermediates, and final
outputs refuse. The independent oracle calls neither emitted C, compiler
numerical evaluators, nor framework linear. Its declared accumulation order is
not a claim of exact torch summation-order equivalence.

Run the directed scalar, cancellation, tie, tiny-product, finite-output, and
retained interpreter checks with the selected interpreter:

```sh
nice -n 10 "$TORCH_PYTHON" reference/test_head_v2.py
nice -n 10 "$TORCH_PYTHON" reference/test_whole_reference.py
```

A saved-head exactness check must authenticate the original BF16 weight words,
generated final-normalization inputs, independent reference, and actual compiled
output. It qualifies that head boundary; it does not rerun upstream layers or
establish a complete live whole-model result. Keep compilation, execution,
verification, and independent numerical comparison receipts separate.
