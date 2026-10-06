# AGENT.md — merlin/runtime/c

## Purpose

The **Merlin C runtime**: a generic, data-driven driver that executes a compiled whole model (`_mlir_ciface_forward`) by building MLIR memref descriptors from a generated argument table. Target-agnostic core; the same code runs on host (verification) and bare-metal spike/Zephyr.

## What belongs here

- `merlin_model.h/.c` — the descriptor builder + `merlin_run` (arg table + weights base + input pointers + output buffer → descriptors → `merlin_invoke`).
- `merlin_host_main.c` — host verification driver (loads `weights.bin`, dumps output).
- `ordered_fma_bounds.h` — optional target-neutral certificates for zero-seeded
  increasing-K binary32 FMA reductions. Callers must prove exact reconstructed
  chunk summaries and absolute representation-error bounds. Invalid arithmetic
  eligibility selects source replay; this helper does not enable an optimization.
- `fma_product_norms.h` — outward row/column metadata and target-neutral Holder
  product bounds for the same optional certificates. Reductions stay outside
  the pair loop. The norm helper additionally requires correctly rounded sqrt.
  The explicit `merlin_fma_operand_summarize_finite` alternative preserves all
  seven fields using proved nonnegative finite binary64 adjacency. It retains
  finite-input checks, requires a valid eligibility token and a stable floating
  environment, and changes no default caller. Callers additionally declare
  nontrapping arithmetic and unobserved FP exception flags; stable rounding
  alone does not establish exception-flag equivalence.
  The separate `merlin_fma_representation_norm` type supplies only the four
  representation-error fields when an independent absolute-product bound is
  available. It avoids square/sqrt/reconstructed-max work and cannot supply the
  full Holder consumer. Explicit requirements never weaken the certificate.
- `bf16_radix_pack.h` — optional target-neutral dynamic row-scaled signed radix
  packing. Integer BF16 exponent/mantissa decoding preserves nearest-even
  coefficients without per-element division/libm. Caller strides describe
  layout; unsupported numeric/storage contracts refuse to the source path.
- `monotone_bit_polynomial.h` — optional exact source enclosures for structurally
  proved monotone real polynomial encodings. Prepared source coefficients,
  domain, outward source rounding budgets and original final FMA retain the
  source arithmetic obligation. Stable RNE, gradual underflow, nontrapping
  arithmetic and unobserved exception flags are caller requirements. Unsupported
  proofs use the checked interval implementation; no policy is enabled by default.
- `bf16_quant_frontier.h` — an explicit source-contract certificate for row
  extrema, BF16 scales and signed quantized observations. The caller proves the
  complete live consumer and supplies distinct valid buffers. A separate
  coordinator accepts caller-owned head/row/channel scratch and row-local
  refresh/refinement callbacks; refusal selects the original source path.
  Certified rows may be reused only when callbacks preserve those rows.
- `positive_scalar_interval.h` — optional original binary32 multiplication and
  FMA endpoint specializations for a finite positive scalar. Point multiplication
  preserves source signed zero; other FMA factors retain the checked corner path.
- `f32_floor_bits.h` — explicitly selected portable IEEE binary32 floor-value
  legalization. It preserves all finite source values and signed zeros, keeps
  nonfinite source calls and changes no rounding environment. Matching storage
  representation and unobserved nontrapping exception flags are caller proofs.
- The ordered-FMA header's optional outward scalar capabilities require explicit
  independent real-arithmetic enclosure, source rounding/exception and platform
  legality proofs. Defaults retain adjacency arithmetic. ISA definitions belong
  to the provider; LLVM rounding metadata alone does not request directed rounding.
  Explicit binary64-to-binary32 outward narrowing follows the same enclosure
  contract. Its default is the original RNE cast plus adjacency. A qualified
  provider may supply directed conversion without changing source rounding.

## What does not belong here

- Generated, model-specific files (`model_gen.h`, `model_io.h`, `model_call.c`, `weights.bin`) — those are emitted per model by `merlin/python/merlin/llvmlower/c_runtime.py` into a build dir.
- Target-specific harness (crt/HTIF/malloc/linker) — that is `merlin/runtime/baremetal/<env>/`.
- Per-kernel dispatch logic (the per-dispatch outliner/dispatch-table runtime, when added) layers on top but the descriptor ABI stays here.

## Interfaces

- Input: the generated `merlin_arg_t[]` table (`MERLIN_ARGS`), the weight blob, `MERLIN_INPUT_PTR`, an output buffer.
- The descriptor struct layout matches MLIR's `memref<...>` lowering exactly: `{allocated, aligned, offset, sizes[rank], strides[rank]}`. Do not change it without matching `convert-memref-to-llvm`.
- Driven end to end by `merlin/python/merlin/runtime/backends/spike_model.py` (build → run → verify).

## Invariants

- **Target-agnostic**: no ISA assumptions here; the only target-specific code is in `baremetal/`/codegen flags.
- Row-major contiguous strides; weights referenced by byte offset into the blob (never copied).
- Output emitted as exact f32 bit patterns so `spike == host` is checkable up to FP reassociation (different ISAs reassociate; gate on cos≈1 / rel<1e-4, not bit-equality).

## Testing expectations

`merlin/python/tests/test_spike_model.py` — small_llama whole-model spike == host == torch (skips without the chipyard toolchain). Verified: cos 0.9999999.

## Notes for future agents

This monolithic-`forward` path is the correctness baseline; the per-dispatch outliner + dispatch-table runtime (for multicore + bounded memory) builds on the same descriptor ABI and `spike_model` build flow.

### Optional source numeric compiler capabilities

`benchmark_buffer.h` provides exact first-difference checks and complete dirty
buffer initialization for benchmark qualification outside the timed ROI. Word
copies plus byte tails check every byte without checksums, unaligned accesses
or strict-aliasing assumptions. Keep baseline/candidate harnesses identical,
reinitialize each output before each call, and retain independent exterior
guards and immutable input checks. Compiler-derived loads/stores are portable;
target timer/command facts remain in OOT. It grants no model arithmetic policy.

`source_f32_math.h` preserves ordinary `fmaf` and `memcpy` by default. The
`source_numeric_capability` emitter can explicitly admit source FMA and fixed
object-representation copy builtins for one compiled numeric component. Its
contract requires source RNE single rounding, unobserved errno/exception flags,
nontrapping execution, and standard non-interposed copy semantics as applicable.
It does not change generic runtime copies or a model's compiler flags. Install
the generated prefix before all numeric headers and pin the actual compiler,
flags, transitive headers, LLVM and object. Source constant/FP operation order
and typed consumer proofs remain independent admission obligations.

Finite classification has its own default-off selection and obligations:
standard classification, unobserved library interposition, nontrapping execution
and unobserved exception flags. Prior FMA or copy permission does not admit it.
The default `MERLIN_SOURCE_ISFINITE` retains the ordinary C library macro;
the selected compiler builtin must classify signed zeros, subnormals, infinities
and both NaN classes correctly without changing the rounding mode. Classification
exception flags are deliberately outside this optional observation contract.

Absolute value is selected separately for binary32 and binary64 math. It requires
standard absolute-value semantics, unobserved interposition, errno, exception
flags and NaN payloads, and nontrapping execution. Existing FMA, copy or
classification permission does not admit it. Defaults retain `fabsf`/`fabs`;
an admitted component uses compiler builtins. Source arithmetic order, finite
values, signed-zero results and rounding mode remain unchanged. Preserve the
complete source-consumer proof when composing capabilities.

Min/max compiler builtins are a separate default-off capability for binary32
and binary64. They require standard min/max results, unobserved interposition,
errno and exception flags, nontrapping execution, and explicitly unobserved
min/max signed-zero and NaN-payload distinctions. The compiler's min/max
intrinsics may carry a signed-zero relaxation even without fast math; callers
must prove that this is allowed at their source-consumer frontier. No permission
is inherited from FMA, classification or absolute value. Defaults retain the
original C library operation. Independent paired tests preserve all other
returned bits, including a number paired with NaN and signed infinities, operand
evaluation and rounding mode. Original consumer/model gates remain unchanged.

Binary32 floor has its own default-off compiler choice. The contract requires
standard finite floor values, unobserved interposition and errno, nontrapping
execution and unobserved exception flags. The helper evaluates its operand once,
preserves finite values and signed zero in every rounding direction, and calls
the original `floorf` for infinities and NaNs. No NaN-payload relaxation is
introduced. The monotone polynomial's existing floor hook consumes this source
capability unless the provider explicitly selects another proved floor helper.
Default source operations and earlier capability prefixes remain unchanged.

### Optional attention endpoint row preparation

`emit_source_attention_frontier(..., prepare_endpoint_rows=True)` shares alpha and
denominator reciprocals across independent channels in the executor private
workspace. Metadata and destination arrays are disjoint, initialized, and live
for the synchronous call; no external mutable metadata is cached. Stable RNE,
nontrapping arithmetic and unobserved exception flags permit moving these
operations. Source chunk/partial accumulation order and multiply semantics are
preserved. The default checked template remains byte-identical. This optional
schedule does not enable provider routing or change the consumer certificate.

### Optional polynomial word enclosure

`emit_source_attention_frontier(..., word_interval_enclosure=True)` selects a
global integer-word enclosure using the immutable prepared monotonicity and
source rounding proof. Each endpoint retains the original source polynomial
FMAs and integer conversion. Twice the proved encoded error plus two words
encloses all interior source values; cutoff zero and finite encoding limits are
preserved. Unsupported plans retain the checked fallback. This may widen the
certificate and increase exact replay, so measured complete-group and original
consumer/model gates govern admission. Previously accepted unobserved carrier
words can change; their old comparison failure must be retained separately from
new source-consumer qualification. The default enclosure and routing stay inert.

### Optional exact zero-error block preparation

`prepare_zero_error_blocks=True` specializes a product block only after complete
immutable RHS error-norm admissions prove zero and a row's LHS error admission
proves zero with no uncertain source positions. It retains the original outward
`add(0, 0)` value, including the default adjacency epsilon, and all absolute-product
and ordered source-FMA bounds. It does not replace an enclosure by mathematical
zero or weaken the consumer certificate. Both endpoint schedules are tested;
nonzero representation errors and uncertain inputs retain the checked path.
Default emitted bytes and numeric capability hooks remain unchanged.

### Optional prepared product-row domains

`prepared_bf16_interval.h` also supplies private exact-source point results and
base/count/epoch span matching. Equal finite BF16 bins must snapshot the current
valid source interval after replay. An owned complete producer/copy proves the
immutable span's lifetime through one synchronous product call; callers cannot
infer this permission from equal sampled values. Exact source bounds retain all
radix representation-error and source-FMA admission requirements. No span may
escape or survive workspace reuse. See `prepared_probability_bins.md`.

`prepared_fma_product_bounds.h` admits concrete immutable reconstructed-column
spans and current complete source/error norms. For each row it derives a
uniform center bound from reconstructed L1 times the maximum reconstructed
column magnitude; the source absolute bound uses source L1 times maximum source
magnitude. Error bounds retain the original ordered upward sum of representation
and uncertain-position terms, using uniform column maxima. Original per-cell
Holder/Cauchy minima cannot exceed these Holder envelopes. Monotone outward
arithmetic then proves both source-prefix and final-enclosure ranges once per
row; failure retains every original per-cell check.

The caller must prove the center span contains exact real reconstructed dots,
with all weighted partials/scaling representable, before using the admitted
consumer. The existing private signed-radix producer and its bound source plan
supply this obligation; arbitrary center buffers do not. The center pointer,
width and reduction length stay bound to the same private workspace and source
norms. No writes/callbacks intervene before consumption. Original center versus
original absolute-bound comparisons remain checked even for admitted rows.
Public checked gamma APIs are unchanged. Tests cover independent rectangular
shapes, zero/error/uncertainty mixtures, source overflow fallback and invalid
admissions, plus full provider source/ownership/refusal cases.

### Optional consumer-derived norm requirements

`prepare_required_norms=True` requires producer-domain preparation. Complete
immutable RHS error admissions must prove zero before the emitted source can
bypass every reconstructed-LHS L2 consumer. A distinct `merlin_l1_norm` preserves
the identical upward L1 accumulation for the typed producer-domain proof; it
cannot be passed to a full-norm product consumer. Nonzero or unknown errors keep
full norms. Private metadata and reconstructed products remain immutable through
synchronous consumption. The existing stable RNE and unobserved, nontrapping
exception contract permits omitting unused squares and square roots. Default
emitted source and routing remain inert. The model's observed frequency of zero
errors is experiment evidence and cannot select the production policy.

### Optional certified-row endpoint retention

`retain_certified_rows=True` requires the row-independent endpoint schedule.
Private certificate state starts false on every call and becomes true only
after the complete original row consumer proves every observed value. Refinement
writes only uncertified rows. Certified rows retain their initialized endpoint
arrays while other rows continue the same refinement and source replay. No
cross-row reduction may be skipped, and the original success publication and
refusal fallback contract remain required. Unsupported source structure refuses
the transform. Default emitted source stays unchanged.

### Optional separable source FMA radius

`separable_source_radius=True` requires private exact reconstructed products,
zero admitted source representation errors and no uncertain operand positions.
The source L1/gamma product is prepared once per row and each column retains its
own admitted maximum. A uniform envelope admits every source prefix below
`FLT_MAX`; unsupported rows retain the existing checked bound. This portable
math belongs in Merlin. Target outward arithmetic remains in the OOT provider.
Wider intervals can increase exact source replay; complete consumer and source
costs determine usefulness. Composition with norm requirements and certified-row
retention must preserve their separate admissions and state lifetimes.
