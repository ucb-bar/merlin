// conv1-bad-buddy.c - C harness for INTENTIONALLY WRONG Buddy-MLIR conv1
//
// This tests a version with wrong stride to verify checksum validation works

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "resnet50_params.h"
#include "images.h"

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

// External MLIR-compiled function (BAD version with wrong stride)
extern void _mlir_ciface_conv1_bad(MemRef4D_i8 *input, MemRef2D_i8 *weights,
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

#define POOL_OUT_ROW_DIM 56
#define POOL_OUT_COL_DIM 56
#define BATCH_SIZE 4
#define OUT_CHANNELS 64
#define PATCH_SIZE 147

static elem_t buddy_output[BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM][OUT_CHANNELS];

int main(int argc, char *argv[]) {
    gemmini_flush(0);

    printf("=== ResNet50 Conv1 - BAD Buddy MLIR (INTENTIONAL WRONG STRIDE) ===\n");
    printf("This should produce WRONG checksum to verify our test methodology\n\n");

    memset(buddy_output, 0, sizeof(buddy_output));

    MemRef4D_i8 input_ref = make_memref4_i8(
        (elem_t*)&images[0][0][0][0],
        BATCH_SIZE, 224, 224, 3);

    MemRef2D_i8 weights_ref = make_memref2_i8(
        (elem_t*)&conv_1_w[0][0],
        PATCH_SIZE, OUT_CHANNELS);

    MemRef1D_i32 bias_ref = make_memref1_i32(
        (acc_t*)&conv_1_b[0],
        OUT_CHANNELS);

    MemRef2D_i8 output_ref = make_memref2_i8(
        &buddy_output[0][0],
        BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM,
        OUT_CHANNELS);

    uint64_t start = read_cycles();
    _mlir_ciface_conv1_bad(&input_ref, &weights_ref, &bias_ref, &output_ref);
    gemmini_fence();
    uint64_t end = read_cycles();

    printf("BAD Buddy conv1 cycles: %llu\n", (unsigned long long)(end - start));

    long long output_checksum = 0;
    int output_elems = BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM * OUT_CHANNELS;
    const elem_t *output_ptr = &buddy_output[0][0];
    for (int i = 0; i < output_elems; i++) {
        output_checksum += output_ptr[i];
    }
    printf("Output checksum: %lld\n", output_checksum);
    printf("(This should NOT match the Gemmini C reference!)\n");

    printf("=== BAD Conv1 DONE ===\n");

    return 0;
}
