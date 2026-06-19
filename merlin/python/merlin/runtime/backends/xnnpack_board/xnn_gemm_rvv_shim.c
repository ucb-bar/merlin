// BOARD (RVV) XNNPACK GEMM shim: the symbol the lowered whole-model binary calls when the
// f32 linalg.matmul dispatches are routed to XNNPACK. It is the K1/RVV analogue of the host
// scalar shim (runtime/backends/xnnpack_host/xnn_gemm_shim.c): same goi weight packing, same
// "fill C, no bias, identity clamp" semantics, but driving the *RVV* microkernel
// `xnn_f32_gemm_ukernel_1x4v__rvv` (the exact kernel scripts/k1_cross_framework.py already
// cross-compiles and runs on the board) instead of the scalar 4x4 ukernel.
//
// ABI: this is invoked from the lowered model.ll as an external MLIR function
//   func.func private @merlin_xnn_gemm_f32(
//       %a : memref<MxK xf32> {bufferization.access = "read"},
//       %b : memref<KxN xf32> {bufferization.access = "read"},
//       %c : memref<MxN xf32> {bufferization.access = "write"}) -> memref<MxN xf32>
// After convert-func-to-llvm each memref is passed UNPACKED as the MLIR descriptor:
//   (float* allocated, float* aligned, intptr offset, intptr size0, size1, stride0, stride1)
// and the result is the standard 2-D memref descriptor struct, returned by value (== the %c
// descriptor in destination-passing form). We therefore declare the C entry to match that exact
// unpacked-descriptor calling convention. M/N/K and the data pointers come from the descriptors;
// row-major contiguous is assumed (stride0 == ncols, stride1 == 1 — true for these allocs).
//
// C[m,n] = sum_k A[m,k] * B[k,n]   (A row-major MxK, B row-major KxN, C row-major MxN)

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

// the expert RVV microkernel, verbatim from the vendored XNNPACK checkout.
#include "f32-gemm/gen/f32-gemm-1x4v-rvv.c"

// 2-D memref descriptor (matches MLIR's MemRefDescriptor for rank 2 / lp64d).
typedef struct {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
} merlin_memref_2d_f32;

// Pack B (K x N row-major) into the XNNPACK goi panels the 1x4v RVV kernel streams:
// per N-tile of NR lanes, [bias[NR] == 0] then for each k a panel of NR weights
// B[k, n0+lane] (lane in tile, zero-padded past N). NR = vsetvlmax_e32m4 (VLEN-dependent).
static void pack_b(size_t N, size_t K, size_t NR, const float *B, float *w) {
  size_t off = 0;
  for (size_t n0 = 0; n0 < N; n0 += NR) {
    for (size_t lane = 0; lane < NR; lane++) w[off + lane] = 0.0f;  // zero bias
    off += NR;
    for (size_t k = 0; k < K; k++) {
      for (size_t lane = 0; lane < NR; lane++) {
        size_t n = n0 + lane;
        w[off + lane] = (n < N) ? B[k * N + n] : 0.0f;
      }
      off += NR;
    }
  }
}

// The MLIR-ABI entry. Reads M/N/K + base pointers from the three unpacked descriptors, computes
// the GEMM through the RVV ukernel, and returns the (filled) C descriptor by value.
merlin_memref_2d_f32 merlin_xnn_gemm_f32(
    float *a_alloc, float *a_aligned, intptr_t a_off, intptr_t a_s0, intptr_t a_s1,
    intptr_t a_st0, intptr_t a_st1,
    float *b_alloc, float *b_aligned, intptr_t b_off, intptr_t b_s0, intptr_t b_s1,
    intptr_t b_st0, intptr_t b_st1,
    float *c_alloc, float *c_aligned, intptr_t c_off, intptr_t c_s0, intptr_t c_s1,
    intptr_t c_st0, intptr_t c_st1) {
  (void)a_alloc; (void)a_st0; (void)a_st1;
  (void)b_alloc; (void)b_s0; (void)b_st0; (void)b_st1;
  (void)c_alloc; (void)c_st0; (void)c_st1;

  const float *A = a_aligned + a_off;
  const float *B = b_aligned + b_off;
  float *C = c_aligned + c_off;
  const size_t M = (size_t)a_s0;   // A is M x K
  const size_t K = (size_t)a_s1;
  const size_t N = (size_t)c_s1;   // C is M x N

  merlin_memref_2d_f32 ret;
  ret.allocated = c_alloc; ret.aligned = c_aligned; ret.offset = c_off;
  ret.sizes[0] = c_s0; ret.sizes[1] = c_s1; ret.strides[0] = c_st0; ret.strides[1] = c_st1;

  if (M == 0 || N == 0 || K == 0) {
    if (M && N) memset(C, 0, M * N * sizeof(float));
    return ret;
  }

  const size_t NR = __riscv_vsetvlmax_e32m4();  // f32 lanes at LMUL=4 (VLEN-dependent)
  const size_t n_tiles = (N + NR - 1) / NR;
  const size_t wsize = n_tiles * (NR + (size_t)K * NR);  // bias + K panels per tile
  float *w = (float *)malloc(wsize * sizeof(float));
  if (!w) { memset(C, 0, M * N * sizeof(float)); return ret; }
  pack_b(N, K, NR, B, w);

  struct xnn_f32_default_params params;  // 1x4v takes the (empty) default params
  const size_t a_stride = K * sizeof(float);
  const size_t cm_stride = N * sizeof(float);
  const size_t cn_stride = NR * sizeof(float);

  // The 1x4v kernel does ONE activation row (mr=1) per call; call M times, weights shared.
  for (size_t m = 0; m < M; m++) {
    xnn_f32_gemm_ukernel_1x4v__rvv(
        1, N, K * sizeof(float),
        &A[m * K], a_stride,
        w,
        &C[m * N], cm_stride, cn_stride,
        &params);
  }
  free(w);
  return ret;
}
