#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/gemmini.h"
#include "include/gemmini_testutils.h"

#define IN_ROW_DIM 17
#define IN_COL_DIM 17
#define IN_CHANNELS 18
#define OUT_CHANNELS 19
#define BATCH_SIZE 2
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 2

#define POOL_SIZE 3
#define POOL_STRIDE 2
#define POOL_PADDING 1

#define OUT_ROW_DIM ((IN_ROW_DIM + 2 * PADDING - KERNEL_DIM) / STRIDE + 1)
#define OUT_COL_DIM ((IN_COL_DIM + 2 * PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM)

#define POOL_OUT_ROW_DIM ((OUT_ROW_DIM + 2 * POOL_PADDING - POOL_SIZE) / POOL_STRIDE + 1)
#define POOL_OUT_COL_DIM ((OUT_COL_DIM + 2 * POOL_PADDING - POOL_SIZE) / POOL_STRIDE + 1)

typedef struct {
  elem_t *basePtr;
  elem_t *data;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} MemRef4D_i8;

typedef struct {
  elem_t *basePtr;
  elem_t *data;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2D_i8;

typedef struct {
  acc_t *basePtr;
  acc_t *data;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1D_i32;

extern void _mlir_ciface_conv_with_pool(MemRef4D_i8 *input, MemRef2D_i8 *weights,
                                        MemRef1D_i32 *bias, MemRef2D_i8 *output);

static MemRef4D_i8 make_memref4_i8(elem_t *base, int64_t d0, int64_t d1,
                                   int64_t d2, int64_t d3) {
  MemRef4D_i8 ref;
  ref.basePtr = base;
  ref.data = base;
  ref.offset = 0;
  ref.sizes[0] = d0;
  ref.sizes[1] = d1;
  ref.sizes[2] = d2;
  ref.sizes[3] = d3;
  ref.strides[3] = 1;
  ref.strides[2] = d3;
  ref.strides[1] = d2 * d3;
  ref.strides[0] = d1 * d2 * d3;
  return ref;
}

static MemRef2D_i8 make_memref2_i8(elem_t *base, int64_t rows, int64_t cols) {
  MemRef2D_i8 ref;
  ref.basePtr = base;
  ref.data = base;
  ref.offset = 0;
  ref.sizes[0] = rows;
  ref.sizes[1] = cols;
  ref.strides[1] = 1;
  ref.strides[0] = cols;
  return ref;
}

static MemRef1D_i32 make_memref1_i32(acc_t *base, int64_t len) {
  MemRef1D_i32 ref;
  ref.basePtr = base;
  ref.data = base;
  ref.offset = 0;
  ref.sizes[0] = len;
  ref.strides[0] = 1;
  return ref;
}

static void init_random(elem_t *buf, int len) {
  for (elem_t *ptr = buf; ptr < buf + len; ptr++) {
    *ptr = (rand() % 5) - 2;
  }
}

static void init_random_acc(acc_t *buf, int len) {
  for (acc_t *ptr = buf; ptr < buf + len; ptr++) {
    *ptr = (rand() % 5) - 2;
  }
}

static void flatten_weights(int out_channels, int kernel_dim, int in_channels,
                            int patch_size,
                            elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
                            elem_t weights_mat[patch_size][out_channels]) {
  assert(patch_size == kernel_dim * kernel_dim * in_channels);
  for (int outc = 0; outc < out_channels; outc++) {
    for (int krow = 0; krow < kernel_dim; krow++) {
      for (int kcol = 0; kcol < kernel_dim; kcol++) {
        for (int inc = 0; inc < in_channels; inc++) {
          int wmatrow = krow * kernel_dim * in_channels +
              kcol * in_channels + inc;
          weights_mat[wmatrow][outc] = weights[outc][krow][kcol][inc];
        }
      }
    }
  }
}

int main(void) {
  static elem_t input[BATCH_SIZE][IN_ROW_DIM][IN_COL_DIM][IN_CHANNELS];
  static elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
  static acc_t bias[OUT_CHANNELS];
  static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS];
  static elem_t pool_output_mat[BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM][OUT_CHANNELS];

  init_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));
  init_random(&weights[0][0][0][0], sizeof(weights) / sizeof(elem_t));
  init_random_acc(&bias[0], sizeof(bias) / sizeof(acc_t));
  flatten_weights(OUT_CHANNELS, KERNEL_DIM, IN_CHANNELS, PATCH_SIZE,
                  weights, weights_mat);

  long long input_checksum = 0;
  elem_t *input_ptr = &input[0][0][0][0];
  int input_elems = BATCH_SIZE * IN_ROW_DIM * IN_COL_DIM * IN_CHANNELS;
  for (int i = 0; i < input_elems; ++i) {
    input_checksum += input_ptr[i];
  }
  long long weight_checksum = 0;
  elem_t *weight_ptr = &weights[0][0][0][0];
  int weight_elems = OUT_CHANNELS * KERNEL_DIM * KERNEL_DIM * IN_CHANNELS;
  for (int i = 0; i < weight_elems; ++i) {
    weight_checksum += weight_ptr[i];
  }
  long long bias_checksum = 0;
  for (int i = 0; i < OUT_CHANNELS; ++i) {
    bias_checksum += bias[i];
  }
  printf("Input checksum: %lld\n", input_checksum);
  printf("Weights checksum: %lld\n", weight_checksum);
  printf("Bias checksum: %lld\n", bias_checksum);

  MemRef4D_i8 input_ref =
      make_memref4_i8(&input[0][0][0][0], BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM,
                      IN_CHANNELS);
  MemRef2D_i8 weights_ref =
      make_memref2_i8(&weights_mat[0][0], PATCH_SIZE, OUT_CHANNELS);
  MemRef1D_i32 bias_ref = make_memref1_i32(&bias[0], OUT_CHANNELS);
  MemRef2D_i8 output_ref =
      make_memref2_i8(&pool_output_mat[0][0],
                      BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM,
                      OUT_CHANNELS);

  gemmini_flush(0);
  uint64_t start = read_cycles();
  _mlir_ciface_conv_with_pool(&input_ref, &weights_ref, &bias_ref, &output_ref);
  gemmini_fence();
  uint64_t end = read_cycles();

  printf("Buddy conv_with_pool cycles: %llu\n",
         (unsigned long long)(end - start));
  long long checksum = 0;
  for (int i = 0; i < BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM; ++i) {
    for (int j = 0; j < OUT_CHANNELS; ++j) {
      checksum += pool_output_mat[i][j];
    }
  }
  printf("Buddy conv_with_pool output checksum: %lld\n", checksum);
  return 0;
}
