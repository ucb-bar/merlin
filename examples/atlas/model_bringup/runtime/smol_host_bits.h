#ifndef SMOL_HOST_BITS_H
#define SMOL_HOST_BITS_H

#include <stdint.h>

static inline float bf16_to_float(uint16_t value) {
    union { uint32_t bits; float value; } copy = {.bits = (uint32_t)value << 16};
    return copy.value;
}

static inline uint16_t float_to_bf16(float value) {
    union { float value; uint32_t bits; } copy = {.value = value};
    uint32_t bits = copy.bits;
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
        return (uint16_t)((bits >> 16) | UINT32_C(0x40));
    }
    return (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16) & 1u)) >> 16);
}

#endif
