---
title: "Observable-effect kernel intake"
kind: reference
status: current
owner: ir
last_verified: 2026-11-10
related: [architecture, triton_frontend]
code_refs:
  - src/merlin/frontends/compile_inputs.py
  - src/merlin/frontends/kernel_compile_inputs.py
  - src/merlin/frontends/kernel_shape_abi.py
---

# Observable-effect kernel intake

Compiler staging accepts `t2m.effect-kernel-receipt.v1` alongside the existing
standalone tensor receipt. It projects source computation, exact weights,
provenance and the declared public signature. Outcome references, argument
values, native observations and their individual digests remain evaluator data.
No expected termination or diagnostic answer is supplied to compilation.

ABI `t2m.shared-tensor-abi.v6` explicitly selects `effect_only: true`. It permits
readonly tensors, by-value runtime scalars and an empty argument list, with no
functional tensor results. It retains named dimensions, ranges and logical
grids. Writable buffers cannot be hidden by this selection. Legacy schemas
retain their existing result requirements and cannot silently adopt ABI v6.

Observable-effect receipts may also retain an ordinary tensor-result signature
for an initially failing reference case. Compiler intake preserves those actual
declared results; it does not replace them with void or fabricate values.
Returning kernels may likewise carry both tensors and observable diagnostics.
Their independently supplied `reference.npz` tuple is evaluator data, alongside
the outcome policy in `reference.json`. Staging excludes both files and their
individual digests. It preserves every real result slot and never reads the
archive to derive compiler constants or specialize control flow. Separate
evaluation must compare the complete tuple and streams from the same call;
compiler intake itself grants no numerical or effect qualification.

The staged profile is `standalone-effect-kernel`. Existing compile-signature
v1/v2 represents the actual input/result roster, including empty lists. Its
`effect_abi` field identifies source-encoded observable effects and excludes
evaluator outcomes. Entry renaming to `forward`, source correspondence,
readonly input attributes, memory/effect evidence and unresolved-op refusals
keep their normal checks. This is byte/ABI custody, not source equivalence,
effect implementation, numeric or target admission.

Literal assertion messages containing embedded NUL may select the explicit
[length-aware native diagnostic feature](literal_diagnostics.md) after staging.
The compiler derives that selection from actual standard operations. Evaluator
streams still do not enter the compiler projection.

Real triton2MLIR execution tests compile void scalar/tensor source assertions
and zero-argument entries through the ordinary Merlin whole-body host pipeline.
Separate evaluator observations distinguish normal return from source assertion
failure. They do not grant accelerator diagnostic handling or source closure.
