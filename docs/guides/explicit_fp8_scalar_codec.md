---
title: Explicit scalar FP8 codec legalization
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related:
  - target_resolution
code_refs:
  - src/merlin/llvmlower/fp8_scalar_codec.py
  - src/merlin/llvmlower/host_llvm_helpers.py
---

`fp8_scalar_codec.legalize_file(source, output, policy)` is an explicit
prepared-model transform. With `policy=None`, it returns without reading source
files, loading Torch or searching for a codec. It selects no quantizer, device
kernel, arithmetic backend or whole-model precision policy.

The caller supplies `BoundedE4M3FNCodec` with immutable `HelperIR` bindings for
the actual helper LLVM, conversion sources and compilation record. Two distinct
C-compatible helpers use `i64(float)` for encoding and `float(i64)` for decoding;
the integer container carries the low eight storage bits. This avoids assuming
the extension ABI of a narrow C integer. Actual function signatures, extension
attributes, target triple and data layout are checked. The caller still owns
source semantics and complete compiler/header provenance.

Only scalar E4M3FN-to-F32 extension and F32-to-E4M3FN truncation are supported.
The transform inserts byte bitcasts and helper calls while retaining public
tensor and memref types, their one-byte external storage and source attributes.
Other formats and nonscalar conversions refuse. It does not widen external FP8
buffers or replace FP8 matrix arithmetic with ordinary F32 arithmetic.

A C10-compatible saturating encoder cannot implement unconstrained standard
`arith.truncf`: overflow and infinity behavior can differ. Truncation therefore
requires a current SSA proof that ordered comparison/select clamps restrict
the value to the finite E4M3FN range, or NaN. The proof also follows a closed
elementwise tensor clamp into the cast's scalar argument. Shape, example values,
metadata and source provenance tags grant no bound. Unknown bounds, fast flags,
rounding options, changed codec inputs and conflicting helper symbols refuse.

The caller must explicitly guarantee RNE and nontrapping, unobserved floating
exceptions, and that NaN payload bits are unobserved. The source codec must
retain signed zero and NaN sign/class. This effect contract is not inferred from
an eager test or from a numerical-policy annotation.

After ordinary lowering, pass the exact helper binding to
`host_llvm_helpers.link_and_inline` and require both selected symbols inlined.
That existing seam verifies modules, checks the actual scalar ABI, links them,
runs the selected compiler's always-inline pass and refuses remaining helper
call or address references. The caller supplies a matching target compiler and
tools through the existing explicit toolchain configuration. Code generation
must also refuse any remaining unsupported casts or unrealized conversions.

Qualify the complete path independently: all FP8 decode byte patterns,
rounding midpoints and neighbors, signed zeros, subnormals, overflow, infinities
and signed NaNs; compare actual framework casts separately from standard
APFloat conversion. Host agreement alone does not qualify target execution,
matrix reduction order, dynamic quantization guards or full-model accuracy.

Tensor bounds require complete output coverage: static equal extents, dense
unencoded tensor types, identity input/output maps and one parallel iterator
per dimension. A scalar rank-zero tensor has one element and an empty identity
map. A partial write, retained unproved initial element, library call, unknown
property or unknown contract grants no bound. The scalar block-argument path
requires the same parent coverage proof. Before analysis, the explicitly selected
path asks the supplied upstream compiler for generic-form MLIR and records that
invocation; it does not trust a parser that may discard custom-form attributes.

The normal outlined builder accepts `fp8_scalar_codec=None`. An explicit codec
may fuse a single-use complete pointwise clamp and encoder before outlining so
that the original bound remains in the compiled leaf. It uses the selected
upstream elementwise fusion pass, verifies the exact typed scalar expression
and scalar attributes, and reifies the original scalar operations with both
provenance dictionaries retained. It crosses no guard or effect boundary and
introduces no clamp, numerical policy or quantizer. Unsupported pairs remain
unchanged and later admission refuses unproved bounds.

For independently compiled kernels, every helper definition/global is alpha
renamed in a unique entry-derived namespace. All references change together;
unsupported assembly or special LLVM globals refuse. Actual used codec symbols
must satisfy the existing helper ABI verifier and disappear through linkage and
inlining before normal allocation, ABI and object gates. Default automatic
C-interface propagation marks public definitions only. Existing explicit
interfaces on private functions and external declarations remain intact.

This scalar representation capability does not prove source finite guards,
source effect preservation, tensor lifetimes, device reduction order or full
model execution. These require independent complete graph and target gates.
