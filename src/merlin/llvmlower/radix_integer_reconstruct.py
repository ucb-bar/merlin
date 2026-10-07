"""Explicit exact integer reconstruction before a single binary64 conversion.

The caller proves fully written signed-i32 radix group outputs, their canonical
digit-magnitude/range contract, at most the plan's K terms, fresh nonoverlapping
scratch/output storage, and original RNE positive-zero-seeded binary64 sums.
Every weighted term and prefix is bounded by 2**53; defined signed-i64 products
and sums therefore cannot overflow, and final conversion is exact. External
scales, certificates and source floating arithmetic remain outside this helper.
No automatic strategy or device implementation is selected.
"""

from .radix_product_groups import RadixProductPlan, plan_radix_product_groups


def c_header(plan: RadixProductPlan) -> str:
    """Emit reset, group update and final conversion under a rederived proof.

    The accumulator is a separate owned int64 buffer, never a type-punned
    binary64 destination. Restrict is justified only by the explicit complete
    nonoverlap/lifetime contract; callers account for its allocation and traffic.
    """
    canonical = plan_radix_product_groups(
        radix_bits=plan.radix_bits,
        digits=plan.digits,
        reduction_length=plan.reduction_length,
    )

    if plan != canonical:
        raise ValueError("canonical signed-i32/exact-binary64 radix plan required")
    if any(g.exponent >= 63 for g in plan.groups):
        raise ValueError("positive radix weight must fit signed i64")
    cases = "\n".join(
        f"  case {ordinal}:\n"
        f"    for (size_t t = 0; t < count; ++t)\n"
        f"      dst[t] += (int64_t)source[t] * INT64_C({1 << group.exponent});\n"
        f"    return;"
        for ordinal, group in enumerate(plan.groups)
    )
    return (
        r"""#ifndef MERLIN_RADIX_INTEGER_RECONSTRUCT_H
#define MERLIN_RADIX_INTEGER_RECONSTRUCT_H
#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <assert.h>
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53
#error exact_radix_reconstruction_requires_binary64
#endif
"""
        + f"#define MERLIN_RADIX_INTEGER_MAX_REDUCTION_LENGTH {plan.reduction_length}\n"
        + r"""
/* Preconditions: canonical group/range proof, complete immutable i32 source,
 * disjoint owned buffers, original +0/RNE reconstruction, actual K <= cap. */
static inline void merlin_radix_integer_begin_exact(int64_t *dst, size_t count) {
  for (size_t t = 0; t < count; ++t) dst[t] = INT64_C(0);
}
/* Explicit fresh full-writer alternative: the canonical first group has
 * weight one. A complete immutable first-group readout initializes every
 * scratch element directly. Accumulate only subsequent groups afterward. */
static inline void merlin_radix_integer_begin_from_first_group_exact_i64(
    int64_t *restrict dst, const int32_t *restrict source, size_t count) {
  for (size_t t = 0; t < count; ++t) dst[t] = (int64_t)source[t];
}
static inline void merlin_radix_integer_accumulate_exact_i64(
    int64_t *restrict dst, const int32_t *restrict source, size_t count,
    unsigned group) {
  /* Keep each proved constant inside its loop: selecting a variable weight
   * first can force a runtime multiply in otherwise ordinary CPU codegen.
   * Positive-constant multiplication is defined even for negative terms;
   * canonical term/prefix bounds prove multiplication and addition fit i64.
   * Do not express the source operation as a negative signed left shift. */
  switch (group) {
"""
        + cases
        + r"""
  default: assert(!"group outside emitted exact radix plan"); return;
  }
}
static inline void merlin_radix_integer_finish_exact_f64(
    double *restrict dst, const int64_t *restrict source, size_t count) {
  for (size_t t = 0; t < count; ++t) dst[t] = (double)source[t];
}
#endif
"""
    )

def c_fused_header(plan: RadixProductPlan) -> str:
    """Combine completed group planes without an intermediate integer buffer.

    This explicit storage schedule requires every canonical group output to be
    complete and immutable before entry, and the destination to be disjoint
    from all source planes and their pointer array. Read-only source planes may
    alias one another. The existing source/range/RNE proof remains mandatory;
    no partial result may escape between group callbacks. Keeping more readout
    planes live has an allocation, transfer-layout and cache cost that the
    caller must qualify independently. The original streaming emitter is
    unchanged.
    """
    # Apply the existing complete-plan admission, including all prefix bounds.
    c_header(plan)
    statements = ["    int64_t total = (int64_t)source[0][t];"]
    statements.extend(
        f"    total += (int64_t)source[{ordinal}][t] * INT64_C({1 << group.exponent});"
        for ordinal, group in enumerate(plan.groups[1:], start=1)
    )
    statements.append("    dst[t] = (double)total;")
    return (
        """#ifndef MERLIN_RADIX_FUSED_INTEGER_RECONSTRUCT_H
#define MERLIN_RADIX_FUSED_INTEGER_RECONSTRUCT_H
#include <stdint.h>
#include <stddef.h>
#include <float.h>
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53
#error exact_radix_reconstruction_requires_binary64
#endif
"""
        + f"#define MERLIN_RADIX_FUSED_INTEGER_GROUPS {len(plan.groups)}\n"
        + f"#define MERLIN_RADIX_FUSED_INTEGER_MAX_REDUCTION_LENGTH {plan.reduction_length}\n"
        + """/* Complete canonical signed-i32 planes; every weighted prefix is exact.
 * Fresh disjoint destination; stable original RNE/+0 source contract.
 * count==0 performs no memory access, including to the source pointer array. */
static inline void merlin_radix_integer_fused_exact_f64(
    double *restrict dst, const int32_t *const *restrict source, size_t count) {
  for (size_t t = 0; t < count; ++t) {
"""
        + "\n".join(statements)
        + "\n  }\n}\n#endif\n"
    )
