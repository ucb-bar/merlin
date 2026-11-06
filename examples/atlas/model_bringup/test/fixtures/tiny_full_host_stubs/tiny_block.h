/* Test-only subset of the callback ABI; no block implementation. */
#ifndef TEST_TINY_BLOCK_H
#define TEST_TINY_BLOCK_H
#include <stddef.h>
#include <stdint.h>
typedef int (*tiny_projection)(void *, unsigned, const uint16_t *, size_t, size_t, size_t, uint16_t *);
#endif
