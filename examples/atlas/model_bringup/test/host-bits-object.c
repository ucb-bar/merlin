#include "smol_host_bits.h"

float widen(uint16_t value) { return bf16_to_float(value); }
uint16_t narrow(float value) { return float_to_bf16(value); }
