/* Test-only ABI declarations. Implementations do not perform BF16 math. */
#ifndef TEST_SMOL_HOST_OPS_H
#define TEST_SMOL_HOST_OPS_H
#include <stddef.h>
#include <stdint.h>
int smol_host_residual_bf16(const uint16_t *, const uint16_t *, uint16_t *, size_t);
int smol_host_silu_mul_bf16(const uint16_t *, const uint16_t *, uint16_t *, size_t);
int smol_host_rmsnorm_bf16(const uint16_t *, const uint16_t *, float, uint16_t *, size_t, size_t);
#endif
