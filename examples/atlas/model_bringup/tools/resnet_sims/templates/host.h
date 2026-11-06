#ifndef RESNET_M2_HOST_H
#define RESNET_M2_HOST_H
#include <stddef.h>
#include <stdint.h>
#include "atlas_host.h"

enum { M2_FP8_NHWC, M2_F32_VECTOR, M2_BF16_TILES, M2_F32_NCHW };
enum { M2_LAUNCH, M2_MAXPOOL, M2_AVGPOOL, M2_FC_BIAS };
typedef struct {
  uint8_t *data;
  size_t bytes;
  unsigned h, w, channels, padded_channels, m;
  int dtype, scale;
} m2_tensor;
typedef struct { size_t offset; unsigned tensor; } m2_dynamic;
typedef struct {
  const uint32_t *program;
  size_t program_bytes;
  const uint8_t *initial, *expected;
  size_t arena_bytes, a_offset, a_bytes, output_offset, output_bytes;
  unsigned input, output, kernel, stride, padding, m, k, n, logical_m, logical_k, logical_n;
  int input_scale;
  int direct;
  size_t source_offset;
  const m2_dynamic *dynamic;
  size_t dynamic_count;
} m2_launch;
typedef struct { unsigned op, input, output, launch, kernel, stride, padding; } m2_node;
#ifdef M2_FAST
typedef struct { const uint8_t *weights; size_t b_offset, b_bytes; uintptr_t source_base; } m2_fast_launch;
extern const m2_fast_launch m2_fast_launches[];
#endif
#endif
