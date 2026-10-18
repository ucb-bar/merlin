# EL4 public performance campaign

These are bounded, target-specific experiment drivers. Reuse the existing
Merlin compiler, reference, linker, simulator admission, archive intake and
roofline APIs. Do not implement another execution harness or relax admission.

Keep generated records and source materializations under the configured
`out/artifacts/` and `out/build/` roots. Protected model inputs and outputs stay
operator-local and never enter public calibration or tracked source files.
Compiler candidates belong in the companion gemmini-mlir task checkout.

Preserve the campaign's original start/deadline and immutable baseline. A
passing subset is exploratory evidence, not formal promotion or convergence.
