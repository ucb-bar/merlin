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
rotary inputs instead of consuming saved rotary fixtures. The source-order host head arithmetic is unchanged.
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

For functional validation on an execution host, run
`python scripts/build_execution_host_math.py` and
`python scripts/check_execution_host_math.py` after the RV64 gate. These compile
the identical source revision with native x86-64-v3/PIC flags and prove all
131,072 f32 trig words bitwise equal over the 65,536-angle domain. The execution
archive is selected separately by the linkage-contract digest in the generated
equality receipt.
The production RV64 pin and guarded emitted source are unchanged. Evidence in
out/execution-host-math records both archives; this is functional validation
only and confers no deployment qualification. The same standard example
packager selects native unary components in its separate runtime copy only with
explicit validation flags and an equality receipt; every output is computed by
the selected library, while qualified-angle membership is checked separately.

The fixed64 functional run at the branch just before final cleanup matched all 2,048,000 BF16 words of independent reference v3 with output SHA-256 `b81d4a7eb71d12ca9a1a601a1ddecb7d0354be968a1143f235acea7761e3aeba`.

For the fixed64 capture, `whole-model-policy.json` records the complete precision
policy including `host_preparation_scope`; its source digest must match the capture.
Deriving this scope automatically from the capture is a follow-up.
From this example directory, with `CAPTURE`, `STAGED`, `COMPILED`, `SNAPSHOT`,
`ATLAS_ROOT`, `BOUNDARY`, `BOUNDARY_SHA256`, `INPUTS`, `MODEL_ROOT`,
`ARITHMETIC_QUALIFICATION`, `MEMORY_QUALIFICATION`, and `NATIVE_CACHE` set:

```sh
python -m merlin.targetgen.cli stage-capture --capture "$CAPTURE" --precision-policy whole-model-policy.json --out "$STAGED"
python -m merlin.targetgen.cli native-compile --engine merlin_native --support atlas --snapshot "$SNAPSHOT" --target-source "$ATLAS_ROOT" --compile-inputs "$STAGED" --application --application-name capsule --already-quantized --application-parameters external --io-memory-bytes 67108864 --application-memory-bytes 536870912 --abi auto --mode diagnostic --out "$COMPILED/application" --status-file "$COMPILED/compile-status.json"
python ../whole-model/unchanged_fx/prepare_live_execution.py --compiled "$COMPILED" --capture "$CAPTURE" --staged "$STAGED" --snapshot "$SNAPSHOT" --atlas-root "$ATLAS_ROOT" --boundary-source "$BOUNDARY" --expected-boundary-source-sha256 "$BOUNDARY_SHA256" --inputs "$INPUTS" --functional-validation --execution-host-math-archive out/execution-host-math/libcore-math-execution-host.a --execution-host-math-equivalence out/execution-host-math/equivalence-receipt.json --execution-host-math-rv64-output out/rv64-rotary.bin
python ../whole-model/unchanged_fx/execute_graph.py --compiled "$COMPILED" --atlas-root "$ATLAS_ROOT" --model-root "$MODEL_ROOT" --backend qualified_native --native-qualification "$ARITHMETIC_QUALIFICATION" --memory-qualification "$MEMORY_QUALIFICATION" --native-cache "$NATIVE_CACHE" --compact-trace
```

Set PYTHONPATH to Merlin's src directory, the provider's Python directory and the
unchanged_fx example directory, and activate their documented compiler environment.
The existing verifier runs before packaging. The final native digest is read from
the equality receipt; the RV64 output binary supplies the authenticated angle domain.
