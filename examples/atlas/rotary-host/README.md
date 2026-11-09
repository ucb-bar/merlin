# Host rotary qualification

Rotary preparation uses captured binary32 source arithmetic on the host and
Merlin's existing selected unary-math route. The explicit
`host-unary-math-policy.json` selects CORE-MATH revision
`a055b9b9a86b857e49573e339cf68306ad92c831` and archive SHA-256
`0626df7b1ce6426945820fd03dbd6752931c7594676c94f2043d5673588d178b`.
Nothing in this preparation is Atlas computation. Binary32 framework
equivalence and positions beyond 2047 are not qualified.

Only the unmodified upstream binary32 `sinf.c` and `cosf.c`, MIT `LICENSE`, and
revision marker are vendored in `third_party/core-math`. No upstream Makefile or
build script runs, and no target binary is committed. Our Python build step uses
RV64GC/lp64d GCC 13.2, strict rounding flags, and deterministic archive creation.
Implicit contraction is disabled; explicit upstream `__builtin_fmaf` stays fused.
The selected archive supplies exactly `sinf`/`cosf`; `round` comes from toolchain
libm and `__ctzdi2` from libgcc. Flags and errno are unobserved by the contract.

Set `RISCV` to the pinned RV64 toolchain prefix and `TORCH_PYTHON` to the source
Torch interpreter, then run this from the example directory:

```sh
bash scripts/run_rotary_gate.sh
```

The script builds only the reviewed sources and a small symbol wrapper, checks
the archive pin, uses existing Spike/pk, and compares every one of 65,536 angles
and 262,144 BF16 sin/cos words for positions 0..2047 and all 32 frequencies.
`scripts/inv_freq.hex` contains the authenticated 128-byte read-only frequency
buffer. All outputs and receipts are generated under ignored `out/`;
`out/core-math/build-receipt.json` binds source hashes, flags, and symbols.
Chipyard newlib was rejected because its sine changes one BF16 word at position
1897/frequency 24; no tolerance or reduced domain is used.

Independent reference v3 uses exact integer/Fraction arithmetic, an outward
160-bit dyadic enclosure, explicit pi/Taylor error bounds, and unambiguous f32
RNE. It replays the original captured rotary ATen slice and computes sequence
rotary inputs instead of consuming saved rotary fixtures. Historical reference
v1/v2 arithmetic is preserved; v2's source-order host head is unchanged in v3.
After the archive gate, select a complete authenticated whole-model capture:

```sh
nice -n 10 "$TORCH_PYTHON" reference/validate_rotary_v3.py \
  --trace "$CAPTURE/frontend-trace.json" \
  --rv64-output out/rv64-rotary.bin --archive-gate out/rotary-libm-receipt.json
ROTARY_SOURCE_TRACE="$CAPTURE/frontend-trace.json" nice -n 10 "$TORCH_PYTHON" \
  -m unittest discover -s reference -p 'test_*.py'
nice -n 10 "$TORCH_PYTHON" reference/validate_rotary_sequence_v3.py \
  --source-bundle "$CAPTURE" --qualification reference/out/rotary-v3/reference.json
```

The archive and independent oracle agree at binary32 and BF16. Torch agrees at
BF16 across the qualified domain; its binary32 sin/cos can differ by one ULP.
See `reference/README.md` for the proof and sequence interface. Complete
whole-model standard capture, compilation, execution, cache continuity, and
quality remain separate gates.
