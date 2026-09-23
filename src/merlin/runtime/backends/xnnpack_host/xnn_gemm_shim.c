// HOST XNNPACK GEMM shim: a clean C entry point that packs operands and drives the
// XNNPACK scalar f32 GEMM microkernel (xnn_f32_gemm_minmax_ukernel_4x4__scalar),
// verbatim from the vendored XNNPACK checkout.
//
// This is the COMPUTE kernel the dispatch runtime routes matmul dispatches through when
// the XNNPACK host backend is enabled (default-off). It is the host-correctness analogue
// of the K1 ceiling driver (ceiling_drivers/xnnpack_gemm_driver.c): same microkernel, same
// goi weight packing, but exposed as a plain ctypes-callable function computing a full
// M x N x K row-major GEMM (no bias, identity clamp) so it can be dropped in for a
// linalg.matmul kernel.
//
// C[m,n] = sum_k A[m,k] * B[k,n]        (A row-major MxK, B row-major KxN, C row-major MxN)
//
// The XNNPACK 4x4 scalar ukernel consumes weights in "goi" packed form: per N-tile of
// NR=4 columns, a bias[4] vector then kc/4-? panels... specifically `w` is read as
// [bias[NR]] then for each k a panel of [NR] weights W[tile*NR + lane, k]. We synthesize
// bias = 0 and transpose B (KxN) into that packed layout once per call. Tail N (< NR) is
// handled by the ukernel itself (nc & 2 / nc & 1 stores); we still pack full NR-wide tiles
// (the kernel advances w by full NR), zero-padding the short tail tile.

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

// the expert microkernel, verbatim
#include "f32-gemm/gen/f32-gemm-4x4-minmax-scalar.c"

#define NR 4
#define MR 4

// Pack B (K x N, row-major) into XNNPACK goi panels with zero bias.
// Layout: for each N-tile [n0, n0+NR): NR zeros (bias) then for k in [0,K): NR weights
// B[k, n0+lane] (lane in tile, zero-padded past N).
static void pack_b(size_t N, size_t K, const float* B, float* w) {
  size_t off = 0;
  for (size_t n0 = 0; n0 < N; n0 += NR) {
    for (size_t lane = 0; lane < NR; lane++) w[off + lane] = 0.0f;  // bias
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

// Full GEMM. Returns 0 on success, nonzero on allocation failure.
int merlin_xnn_gemm_f32(size_t M, size_t N, size_t K,
                        const float* A, const float* B, float* C) {
  if (K == 0 || N == 0 || M == 0) {
    if (C && M && N) memset(C, 0, M * N * sizeof(float));
    return 0;
  }
  const size_t n_tiles = (N + NR - 1) / NR;
  const size_t wsize = n_tiles * (NR + (size_t)K * NR);  // bias + K panels per tile
  float* w = (float*)malloc(wsize * sizeof(float));
  if (!w) return 1;
  pack_b(N, K, B, w);

  struct xnn_f32_minmax_params params;
  params.scalar.min = -INFINITY;  // identity clamp (no relu)
  params.scalar.max = INFINITY;

  const size_t a_stride = K * sizeof(float);
  const size_t cm_stride = N * sizeof(float);
  const size_t cn_stride = NR * sizeof(float);

  for (size_t m0 = 0; m0 < M; m0 += MR) {
    size_t mr = (M - m0 < MR) ? (M - m0) : MR;
    xnn_f32_gemm_minmax_ukernel_4x4__scalar(
        mr, N, K * sizeof(float),
        &A[m0 * K], a_stride,
        w,
        &C[m0 * N], cm_stride, cn_stride,
        &params);
  }
  free(w);
  return 0;
}
