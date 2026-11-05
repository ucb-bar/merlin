#include "smol_host_bits.h"
#include <assert.h>

static uint32_t float_bits(float value) {
    union { float value; uint32_t bits; } copy = {.value = value};
    return copy.bits;
}

static uint16_t reference(uint32_t bits) {
    uint32_t high = bits >> 16, low = bits & 0xffff;
    if ((bits & 0x7fffffff) > 0x7f800000) return (uint16_t)(high | 0x40);
    return (uint16_t)(high + (low > 0x8000 || (low == 0x8000 && (high & 1))));
}

int main(void) {
    static const uint32_t tails[] = {0, 1, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xffff};
    for (uint32_t high = 0; high <= 0xffff; ++high) {
        uint32_t bits = high << 16;
        assert(float_bits(bf16_to_float((uint16_t)high)) == bits);
        /* Signaling NaNs retain sign/payload and acquire the quiet bit on conversion back. */
        assert(float_to_bf16(bf16_to_float((uint16_t)high)) == reference(bits));
        for (unsigned i = 0; i < sizeof(tails) / sizeof(tails[0]); ++i) {
            union { uint32_t bits; float value; } copy = {.bits = bits | tails[i]};
            assert(float_to_bf16(copy.value) == reference(copy.bits));
        }
    }
    return 0;
}
