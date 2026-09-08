# Compiler-owned native MX operand ABI

The native Muon/Radiance MX route no longer needs a capsule golden to supply element codes and E8M0
scales. For the first supported family, `kernels/gemm_mxgemmini`, the compiler selects from semantic
command-buffer facts and attaches `params.mx_operand_abi` with this shape:

- schema and selected family;
- `M`, `N`, `K`, E4M3 element encoding, E8M0 scale encoding, and block size 32;
- explicit lhs/weight and scale-tensor names;
- row-major element codes and `[K/32, lane]` scale codes; and
- a SHA-256 binding carried by both the ABI and the emitted native program.

Scale tensors retain `scale_of` and `block` through interface parsing and emission. Selection therefore
does not infer a relationship from names such as `_scale`, and an E4M3 dtype by itself is not treated as
microscaling.

There are two admitted operand sources. Ordinary actual floating-point input and weight tensors are
block-quantized by the compiler; it selects a power-of-two E8M0 scale from each block amax and encodes
the normalized values as E4M3. Alternatively, an already-quantized model can supply explicit `mxfp8`,
block-32 element and scale codes. A call must choose exactly one source.

The native route rejects `mx_operands`, `canonical_inputs`, golden fields, and reference-kernel paths.
The harness verifies the ABI digest and source marker before compilation. Consequently a legacy answer
surface cannot silently override the selected compiler program, and an ABI or program perturbation is
a hard error.

## Qualification and controls

The focused suite uses the real R5 16x32x16 semantic interface and the declared generic kernel-family
rule. It checks native selection and full-program emission, validates the derived codes/scales with the
independently validated MX numeric model, and includes both required negative controls: changing an
actual input changes the ABI, program, and numeric result; changing the expected output fails the R5
tolerance policy. This is the strongest affordable always-on tier. It is not an RTL qualification.
When Cyclotron is available, the slow control runs the compiler-native program twice (base and perturbed
input), matches both device results exactly to the validated MX model, and rejects the perturbed expected
output. This exercises the emitted program without the legacy reference-kernel substitution.

## Deliberate gaps

- V1 covers one rank-2 mxfp8 GEMM with M/N multiples of 16, K a multiple of 32, bf16 output, and no
  epilogue. Weight-stationary shapes, batched GEMV, fused attention, fp4, and fp6 remain fail-closed.
- Actual tensors are compile-time inputs to this slice. Runtime activation quantization and pointer-based
  MX operand staging are not implemented.
- The emitted program uses the target package's MX-Gemmini co-model support library. It does not copy,
  link, or select a pinned kernel-library reference body.
- Existing public MX capsule goldens encode corpus-salted scale streams. They remain on the clearly
  stamped legacy reference route; they are not evidence for this native compiler path. Migrating those
  capsules requires workload-owned quantization metadata or a new golden defined over ordinary tensors.
- Numeric model validation is always-on and Cyclotron controls run when that model is available. A complete
  Verilator/RTL positive and both perturbation controls remain required before claiming RTL qualification.
