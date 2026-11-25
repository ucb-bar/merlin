/* Local support for builtins which otherwise introduce target libm calls.
 * Included before unmodified CORE-MATH sources. These operations are not
 * transcendentals. The policy deliberately leaves exception flags unobserved.
 */
#include <stdint.h>
#include <string.h>
static inline double merlin_fmax(double x, double y) {
    if (__builtin_isnan(x)) return y;
    if (__builtin_isnan(y)) return x;
    if (x == y) return __builtin_signbit(x) ? y : x;
    return x > y ? x : y;
}
static inline float merlin_invalid_nan(const char *payload) {
    /* CORE-MATH uses only the nonnumeric payload "<0". */
    (void)payload;
    return __builtin_nanf("");
}
static inline int merlin_unobserved_raise(int flags) {
    (void)flags;
    return 0;
}
static inline double merlin_floor(double x) {
    uint64_t u;
    memcpy(&u, &x, 8);
    unsigned exponent = (unsigned)((u >> 52) & 0x7ff);
    if (exponent >= 1075) return x; /* Already integral, infinity, or NaN. */
    if (exponent < 1023) {
        if ((u << 1) == 0) return x;
        return (u >> 63) ? -1.0 : 0.0;
    }
    uint64_t mask = (UINT64_C(1) << (1075 - exponent)) - 1;
    if ((u >> 63) && (u & mask)) u += mask;
    u &= ~mask;
    memcpy(&x, &u, 8);
    return x;
}
#define __builtin_fmax merlin_fmax
#define __builtin_floor merlin_floor
#define __builtin_nanf merlin_invalid_nan
#define feraiseexcept merlin_unobserved_raise

/* All public callers are required to pass merlin_host_check_fp_env().
 * CORE-MATH can therefore query the admitted rounding mode without libm.
 * pow's portable flag helpers only save and restore flags; flags are outside
 * this ABI's observable effects, so those saves/restores need no runtime.
 */
#include <fenv.h>
static inline int merlin_rne(void) { return FE_TONEAREST; }
#define fegetround merlin_rne
#ifndef __x86_64__
static inline int merlin_get_unobserved_flags(fexcept_t *flag, int mask) {
    (void)mask;
    *flag = 0;
    return 0;
}
static inline int merlin_set_unobserved_flags(const fexcept_t *flag, int mask) {
    (void)flag;
    (void)mask;
    return 0;
}
#define fegetexceptflag merlin_get_unobserved_flags
#define fesetexceptflag merlin_set_unobserved_flags
#endif
