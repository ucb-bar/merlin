// File: riscv_qgemm_shim.c
#include <stddef.h>
#include <stdint.h>

// Include the header for your kernel
#include "bme.h" 

// This function signature matches the complex memref ABI that IREE will generate.
// Note: We only use the base pointers (the first argument for each memref).
void qgemm_i8_bias_i32_workgroup(
    // memref %at (A)
    const int8_t* restrict at_base, const int8_t* restrict at_aligned,
    size_t at_offset, size_t at_size_m, size_t at_size_k,
    size_t at_stride_m, size_t at_stride_k,
    // memref %b (B)
    const int8_t* restrict b_base, const int8_t* restrict b_aligned,
    size_t b_offset, size_t b_size_k, size_t b_size_n,
    size_t b_stride_k, size_t b_stride_n,
    // memref %bias
    const int32_t* restrict bias_base, const int32_t* restrict bias_aligned,
    size_t bias_offset, size_t bias_size_n, size_t bias_stride_n,
    // memref %out (C)
    int32_t* restrict out_base, int32_t* restrict out_aligned,
    size_t out_offset, size_t out_size_m, size_t out_size_n,
    size_t out_stride_m, size_t out_stride_n,
    // Dimensions
    size_t M, size_t N, size_t K) {
  
  // Extract the base pointers.
  // The MLIR subspan op will handle any offsets.
  int8_t* at_ptr = (int8_t*)at_base;
  int8_t* b_ptr = (int8_t*)b_base;
  int32_t* bias_ptr = (int32_t*)bias_base;
  int32_t* out_ptr = (int32_t*)out_base;

  // Call your target kernel
  i8_mm_bme_1x2(bias_ptr, out_ptr, at_ptr, b_ptr, M, N, K);
}