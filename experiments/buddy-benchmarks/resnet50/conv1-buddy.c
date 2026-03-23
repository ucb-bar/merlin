// conv1-buddy.c - C harness for Buddy-MLIR ResNet50 conv_1 layer
//
// This harness:
// 1. Includes the same resnet50_params.h weights as Gemmini C
// 2. Calls the Buddy-compiled conv1 function
// 3. Computes checksums for validation against Gemmini C reference

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Include the actual ResNet50 parameters (same weights as Gemmini C reference)
#include "resnet50_params.h"
#include "images.h"

// Memref descriptor types for MLIR C interface
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

// External MLIR-compiled function
extern void _mlir_ciface_conv1(MemRef4D_i8 *input, MemRef2D_i8 *weights,
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

// Output buffer - must be static to avoid stack overflow
// Shape: [batch * pool_out_row * pool_out_col][out_channels] = [12544][64]
#define POOL_OUT_ROW_DIM 56
#define POOL_OUT_COL_DIM 56
#define BATCH_SIZE 4
#define OUT_CHANNELS 64
#define PATCH_SIZE 147  // 7*7*3

static elem_t buddy_output[BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM][OUT_CHANNELS];

int main(int argc, char *argv[]) {
    gemmini_flush(0);

    printf("=== ResNet50 Conv1 - Buddy MLIR ===\n");
    printf("Input: %d x %d x %d x %d\n",
           conv_1_params.batch_size,
           conv_1_params.in_row_dim,
           conv_1_params.in_col_dim,
           conv_1_params.in_channels);
    printf("Kernel: %d x %d, stride=%d, padding=%d\n",
           conv_1_params.kernel_size, conv_1_params.kernel_size,
           conv_1_params.stride, conv_1_params.padding);
    printf("Output (after pool): %d x %d x %d x %d\n",
           conv_1_params.batch_size,
           conv_1_params.out_dim_pooled, conv_1_params.out_dim_pooled,
           conv_1_params.out_channels);

    // Compute input checksum for verification
    long long input_checksum = 0;
    const elem_t *input_ptr = &images[0][0][0][0];
    int input_elems = conv_1_params.batch_size * conv_1_params.in_row_dim *
                      conv_1_params.in_col_dim * conv_1_params.in_channels;
    for (int i = 0; i < input_elems; i++) {
        input_checksum += input_ptr[i];
    }
    printf("Input checksum: %lld\n", input_checksum);

    // Compute weight checksum
    long long weight_checksum = 0;
    const elem_t *weight_ptr = &conv_1_w[0][0];
    int weight_elems = conv_1_params.patch_size * conv_1_params.out_channels;
    for (int i = 0; i < weight_elems; i++) {
        weight_checksum += weight_ptr[i];
    }
    printf("Weight checksum: %lld\n", weight_checksum);

    // Compute bias checksum
    long long bias_checksum = 0;
    for (int i = 0; i < conv_1_params.out_channels; i++) {
        bias_checksum += conv_1_b[i];
    }
    printf("Bias checksum: %lld\n", bias_checksum);

    // Zero output buffer
    memset(buddy_output, 0, sizeof(buddy_output));

    // Create memref descriptors
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

    // Call Buddy-compiled conv1
    uint64_t start = read_cycles();
    _mlir_ciface_conv1(&input_ref, &weights_ref, &bias_ref, &output_ref);
    gemmini_fence();
    uint64_t end = read_cycles();

    printf("Buddy conv1 cycles: %llu\n", (unsigned long long)(end - start));

    // Compute output checksum
    long long output_checksum = 0;
    int output_elems = BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM * OUT_CHANNELS;
    const elem_t *output_ptr = &buddy_output[0][0];
    for (int i = 0; i < output_elems; i++) {
        output_checksum += output_ptr[i];
    }
    printf("Output checksum: %lld\n", output_checksum);
    printf("Output elements: %d\n", output_elems);

    // Print a few output values for debugging
    printf("First 10 output values: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", output_ptr[i]);
    }
    printf("\n");

    printf("=== Conv1 Buddy MLIR DONE ===\n");

    return 0;
}
