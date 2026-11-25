/* IEEE-754 binary32 square root in the caller's required RNE mode.
 * No libm call and no reciprocal-then-square-root double rounding.
 */
float merlin_ieee_sqrtf(float x) {
    float result;
#if defined(__riscv) && __riscv_xlen == 64
    __asm__ ("fsqrt.s %0, %1, rne" : "=f" (result) : "f" (x));
#elif defined(__x86_64__)
    __asm__ ("sqrtss %1, %0" : "=x" (result) : "x" (x));
#else
#error Unsupported host-math hardware target
#endif
    return result;
}
