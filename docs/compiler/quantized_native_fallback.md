# Floating native families and the quantized contract

An exact native-dtype proof takes precedence. When it fails, a floating source
in a native arithmetic family is eligible under the declared quantized
computation. IEEE accumulation order, competing NaN word priority and reduced
storage conversion layout are not obligations of that computation. Integer
source semantics continue to require the native exact route. Movement and
alias policy remain separate.

The general rule in `numerical_semantics.quantized_native_fallback` is marked
**authored, pending operator review**. Its correctness criterion is precision
class (b): exact equality with the independent rational quantized reference.
Eager floating output is informational for that route.

The host quantizes each real input component with exact RNE and byte clamping.
The enclosing power-of-two scale comes from the declared M3 observer domain;
evaluation payloads never calibrate a scale, constrain component counts or
select a kernel. For the current range [-1,1], scale is 1/64 and the complete
code interval is [-128,127]. Every NaN sign/payload maps to storage code zero.
Negative and positive infinity map to the minimum and maximum code. Finite
rational body semantics do not license nonfinite-producing or bit-observing
operations.

Signed-byte component producers execute on the device, with full int32 readout.
Host recombination retains the exact numerator, exponent and divisor. The
exported result is quantized once by exact RNE/clamp, then dequantized to its
declared storage. Threshold parity is determined before any zero-point shift.
Representation splitting does not grant intermediate source rounding. Complete
provider coverage remains required; a partial prelude cannot place retained
host arithmetic.

Complex storage uses a final real/imaginary axis. Linear component families use
independent component quantization; runtime complex products require a
composition under strict native policy. Quantization does not grant native
status to runtime parallel products, nonlinear rounding constructions,
nonuniform result tables, index-generated payload arithmetic or unknown payload
carriers. A literal affine offset followed by native storage rounding instead
has its rounding at the quantized output boundary.

The independent oracle evaluates the original case arguments in rational
arithmetic, then applies the byte boundaries. Complex numbers are pairs of
rational numbers. It imports neither component lowering nor provider outputs.
An oracle failure is a qualification error and cannot choose host placement
based on payload data. Metadata, post-input storage, aliases and mutations
retain the existing separate whole-call checks. FireSim expectations come from
this oracle and independently recorded input storage.

A parallel product of two runtime payloads remains a composition even when an
operand depends on a reduction. Only a compile-time numeric operand removes
that objection. Quantized placement does not turn a reduction-dependent
parallel product into a primitive.

The tests cover nonfinite quantizer inputs, ties, unchanged post-input bytes,
all byte codes through a rational divisor and a layout change, negative oracle
grading, native-path priority and payload-independent admission. Source oracle
failures raise qualification errors rather than selecting a host lane.

Both operands must be runtime values for the complex-product exclusion. Frozen
numeric coefficients retain per-component linear eligibility.

The run13 qualification at base `0ae81db52` audited 2,629 native-family cases.
The affected Spike sweep passed all 480 cases: 454 moved to the device and 26
retained host semantics (22 runtime complex products and four literal-zero
divisors). Device totals increased from 1,470 to 1,924. A 200-case regression
retained device placement and identical kernel object bytes. Strict Phase 0
reached `execution_succeeded`; all 193 measured calls passed and matched their
admission lanes (46 device, 147 host).

Qualification passed 145 unit tests with three skips. Six compact moved-only
FireSim images cover all 454 moved cases and pass traced Spike, including
per-case Gemmini instruction attribution and ELF-word checks. A three-case
negative control fails exactly the case with one altered expected output byte.
Expectations use independent rational numeric bytes and source-record boundary
metadata. Detailed receipts and the complete audit are generated under
`out/codex-dev/run13q/`; generated artifacts are excluded from commits.
