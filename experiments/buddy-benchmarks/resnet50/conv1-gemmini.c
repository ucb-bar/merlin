// conv1-gemmini.c - Standalone Gemmini C test for ResNet50 conv_1 layer
// This creates a reference checksum for validation against Buddy-MLIR
//
// Conv1 params: 7x7 conv, stride=2, padding=3, with 3x3 maxpool
// Input:  4 x 224 x 224 x 3   (batch x height x width x channels)
// Output: 4 x 56 x 56 x 64    (after conv + pool)

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Include the actual ResNet50 parameters (contains conv_1_w, conv_1_b, conv_1_params)
#include "resnet50_params.h"
#include "images.h"

int main(int argc, char *argv[]) {
    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    printf("=== ResNet50 Conv1 - Gemmini C Reference ===\n");
    printf("Input: %d x %d x %d x %d\n",
           conv_1_params.batch_size,
           conv_1_params.in_row_dim,
           conv_1_params.in_col_dim,
           conv_1_params.in_channels);
    printf("Kernel: %d x %d, stride=%d, padding=%d\n",
           conv_1_params.kernel_size, conv_1_params.kernel_size,
           conv_1_params.stride, conv_1_params.padding);
    printf("Output (before pool): %d x %d x %d x %d\n",
           conv_1_params.batch_size,
           conv_1_params.out_row_dim, conv_1_params.out_col_dim,
           conv_1_params.out_channels);
    printf("Pool: %d x %d, stride=%d, padding=%d\n",
           conv_1_params.pool_size, conv_1_params.pool_size,
           conv_1_params.pool_stride, conv_1_params.pool_padding);
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

    // Run conv_1 with tiled_conv_auto (fused conv + pool)
    uint64_t start = read_cycles();

    tiled_conv_auto(
        conv_1_params.batch_size,
        conv_1_params.in_row_dim, conv_1_params.in_col_dim,
        conv_1_params.in_channels,
        conv_1_params.out_channels,
        conv_1_params.out_row_dim, conv_1_params.out_col_dim,
        conv_1_params.stride,
        1,  // input_dilation
        1,  // kernel_dilation
        conv_1_params.padding,
        conv_1_params.kernel_size,
        false, false, false, false, false,  // transposes
        (elem_t*)images,
        (elem_t*)conv_1_w,
        (acc_t*)conv_1_b,
        (elem_t*)conv_1_out_pooled,
        RELU,
        conv_1_params.output_scale,
        conv_1_params.pool_size,
        conv_1_params.pool_stride,
        conv_1_params.pool_padding,
        tiled_matmul_type);

    gemmini_fence();
    uint64_t end = read_cycles();

    printf("Conv1 cycles: %llu\n", (unsigned long long)(end - start));

    // Compute output checksum
    long long output_checksum = 0;
    int output_elems = conv_1_params.batch_size *
                       conv_1_params.out_dim_pooled *
                       conv_1_params.out_dim_pooled *
                       conv_1_params.out_channels;
    const elem_t *output_ptr = &conv_1_out_pooled[0][0][0][0];
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

    printf("=== Conv1 Gemmini C Reference PASS ===\n");

    return 0;
}
