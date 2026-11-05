#include "atlas_host_layout.h"
/* Host math for the declared batch1 ResNet50 convolution geometry.
 * Compile both native checks and RV64 with -ffp-contract=off.
 */
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "atlas_numeric.h"

static int tensor_shift(uint16_t maximum, int *shift) {
  *shift = 0;
  if (!maximum) return 0;
  int exponent = (maximum >> 7) & 255, significand;
  if (exponent) {
    significand = 128 + (maximum & 127);
    exponent -= 127;
  } else {
    unsigned top = 0, fraction = maximum & 127;
    for (unsigned value = fraction; value > 1; value >>= 1) ++top;
    significand = (int)(fraction << (7 - top));
    exponent = (int)top - 133;
  }
  *shift = exponent - 8 + (significand > 224);
  return *shift < -126 || *shift > 127;
}

static int input_bits(const float *input, unsigned channels, unsigned h, unsigned w, unsigned kernel, unsigned stride,
                      unsigned padding, unsigned output_w, size_t row, size_t inner, uint16_t *bits) {
  unsigned channel = (unsigned)(inner / (kernel * kernel));
  int ih = (int)(row / output_w * stride) + (int)((inner / kernel) % kernel) - (int)padding;
  int iw = (int)(row % output_w * stride) + (int)(inner % kernel) - (int)padding;
  if (channel >= channels) return 1;
  if (ih < 0 || ih >= (int)h || iw < 0 || iw >= (int)w) {
    *bits = 0;
    return 0;
  }
  return f32_bf16(input[((size_t)channel * h + (unsigned)ih) * w + (unsigned)iw], bits);
}

/* The first pass scans exactly the im2col entries, so stride2 1x1 kernels
 * exclude unused input pixels when choosing the tensor power-of-two scale. */
int atlas_graph_pack(const float *input, unsigned channels, unsigned h, unsigned w, unsigned kernel, unsigned stride,
                     unsigned padding, uint8_t *output, size_t capacity, int *scale) {
  if (!input || !output || !scale || !channels || channels > 2048 || !h || h > 224 || !w || w > 224 ||
      (kernel != 1 && kernel != 3 && kernel != 7) || (stride != 1 && stride != 2) || padding > 3 ||
      h + 2 * padding < kernel || w + 2 * padding < kernel)
    return 1;
  unsigned oh = (h + 2 * padding - kernel) / stride + 1, ow = (w + 2 * padding - kernel) / stride + 1;
  size_t m = (size_t)oh * ow, k = (size_t)channels * kernel * kernel;
  size_t mp = (m + 31) & ~(size_t)31, kp = (k + 31) & ~(size_t)31;
  if (mp > SIZE_MAX / kp || capacity < mp * kp) return 1;
  uint16_t maximum = 0;
  for (size_t row = 0; row < m; ++row)
    for (size_t inner = 0; inner < k; ++inner) {
      uint16_t bits;
      if (input_bits(input, channels, h, w, kernel, stride, padding, ow, row, inner, &bits)) return 1;
      bits &= 0x7fff;
      if (bits > maximum) maximum = bits;
    }
  int shift;
  if (tensor_shift(maximum, &shift)) return 1;
  size_t cursor = 0;
  for (size_t mt = 0; mt < mp; mt += 32)
    for (size_t kt = 0; kt < kp; kt += 32)
      for (unsigned r = 0; r < 32; ++r)
        for (unsigned c = 0; c < 32; ++c) {
          size_t row = mt + r, inner = kt + c;
          uint8_t value = 0;
          if (row < m && inner < k) {
            uint16_t bits;
            if (input_bits(input, channels, h, w, kernel, stride, padding, ow, row, inner, &bits)) return 1;
            value = pack_bf16(bits, shift);
          }
          output[cursor++] = value;
        }
  *scale = shift;
  return 0;
}

int atlas_graph_decode(const uint8_t *physical, size_t bytes, float *output, unsigned channels, unsigned h, unsigned w) {
  if (!physical || !output || !channels || channels > 2048 || !h || h > 224 || !w || w > 224) return 1;
  size_t m = (size_t)h * w, mp = (m + 31) & ~(size_t)31, np = (channels + 31) & ~(size_t)31;
  if (mp > SIZE_MAX / np / 2 || bytes < mp * np * 2) return 1;
  for (size_t row = 0; row < m; ++row)
    for (unsigned column = 0; column < channels; ++column) {
      size_t offset = ((row / 32) * (np / 32) + column / 32) * ATLAS_BF16_TILE_BYTES +
                      ((column % 32) / 16) * ATLAS_BF16_HALF_BYTES + (row % 32) * 32 + (column % 16) * 2;
      uint32_t bits = (uint32_t)physical[offset] | ((uint32_t)physical[offset + 1] << 8);
      union {
        uint32_t u;
        float f;
      } value = {.u = bits << 16};
      output[(size_t)column * m + row] = value.f;
    }
  return 0;
}

int atlas_graph_affine(const float *input, float *output, const float *scale, const float *bias, unsigned channels,
                       size_t spatial) {
  if (!input || !output || !scale || !bias || !channels || channels > 2048 || !spatial || spatial > (size_t)224 * 224)
    return 1;
  for (unsigned c = 0; c < channels; ++c)
    for (size_t i = 0; i < spatial; ++i) {
      float product = input[(size_t)c * spatial + i] * scale[c];
      output[(size_t)c * spatial + i] = product + bias[c];
    }
  return 0;
}

int atlas_graph_relu(const float *input, float *output, size_t count) {
  if (!input || !output || !count || count > (size_t)2048 * 224 * 224) return 1;
  for (size_t i = 0; i < count; ++i) output[i] = input[i] < 0.0f ? 0.0f : input[i];
  return 0;
}

int atlas_graph_add(const float *a, const float *b, float *output, size_t count) {
  if (!a || !b || !output || !count || count > (size_t)2048 * 224 * 224) return 1;
  for (size_t i = 0; i < count; ++i) output[i] = a[i] + b[i];
  return 0;
}

int atlas_graph_maxpool(const float *input, float *output, unsigned channels, unsigned h, unsigned w, unsigned kernel,
                        unsigned stride, unsigned padding) {
  if (!input || !output || !channels || channels > 2048 || !h || h > 224 || !w || w > 224 || kernel != 3 || stride != 2 ||
      padding != 1 || h + 2 * padding < kernel || w + 2 * padding < kernel)
    return 1;
  unsigned oh = (h + 2 * padding - kernel) / stride + 1, ow = (w + 2 * padding - kernel) / stride + 1;
  for (unsigned c = 0; c < channels; ++c)
    for (unsigned row = 0; row < oh; ++row)
      for (unsigned col = 0; col < ow; ++col) {
        union {
          uint32_t u;
          float f;
        } initial = {.u = 0xff800000u};
        float maximum = initial.f;
        for (unsigned kr = 0; kr < kernel; ++kr)
          for (unsigned kc = 0; kc < kernel; ++kc) {
            int ir = (int)(row * stride + kr) - (int)padding, ic = (int)(col * stride + kc) - (int)padding;
            if (ir >= 0 && ir < (int)h && ic >= 0 && ic < (int)w) {
              float value = input[((size_t)c * h + (unsigned)ir) * w + (unsigned)ic];
              if (value > maximum) maximum = value;
            }
          }
        output[((size_t)c * oh + row) * ow + col] = maximum;
      }
  return 0;
}

/* Explicit host policy: sequential FP32 accumulation then FP32 division.
 * Equality with Torch's reduction order is a separate conformance gate. */
int atlas_graph_average(const float *input, float *output, unsigned channels, size_t spatial) {
  if (!input || !output || !channels || channels > 2048 || !spatial || spatial > (size_t)224 * 224) return 1;
  for (unsigned c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (size_t i = 0; i < spatial; ++i) sum = sum + input[(size_t)c * spatial + i];
    output[c] = sum / (float)spatial;
  }
  return 0;
}
