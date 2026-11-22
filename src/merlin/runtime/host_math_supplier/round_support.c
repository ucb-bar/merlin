/* Exact ties-away integer rounding support for CORE-MATH builtins which
 * GCC lowers to round()/roundf(). No target libm is linked. These are
 * internal support operations, not public transcendental entry points.
 */
#include <stdint.h>
#include <string.h>
double round(double x) {
    uint64_t u;
    memcpy(&u, &x, 8);
    unsigned exponent = (unsigned)((u >> 52) & 0x7ff);
    uint64_t sign = u & UINT64_C(0x8000000000000000);
    if (exponent < 1022) u = sign;
    else if (exponent == 1022) u = sign | UINT64_C(0x3ff0000000000000);
    else if (exponent < 1075) {
        unsigned shift = 1075 - exponent;
        uint64_t mask = (UINT64_C(1) << shift) - 1;
        u = (u + (UINT64_C(1) << (shift - 1))) & ~mask;
    }
    memcpy(&x, &u, 8);
    return x;
}

float roundf(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    unsigned exponent = (u >> 23) & 0xff;
    uint32_t sign = u & UINT32_C(0x80000000);
    if (exponent < 126) u = sign;
    else if (exponent == 126) u = sign | UINT32_C(0x3f800000);
    else if (exponent < 150) {
        unsigned shift = 150 - exponent;
        uint32_t mask = (UINT32_C(1) << shift) - 1;
        u = (u + (UINT32_C(1) << (shift - 1))) & ~mask;
    }
    memcpy(&x, &u, 4);
    return x;
}
