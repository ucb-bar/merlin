// BOARD (RVV) OpenBLAS GEMM shim: the symbol the lowered whole-model binary calls when the
// f32 linalg.matmul dispatches are routed to OpenBLAS. It is the OpenBLAS analogue of the
// XNNPACK board shim (runtime/backends/xnnpack_board/xnn_gemm_rvv_shim.c): same unpacked
// MLIR memref-descriptor ABI, same resident-weight pack cache for B, same "plain C = A.B,
// no bias, no clamp" semantics, but driving OpenBLAS's RVV 8x8 microkernel
// (sgemm_kernel_16x8_zvl256b.c, the exact kernel the ceiling driver
// kernels/ceiling_drivers/openblas_sgemm_driver.c measures standalone) instead of the
// XNNPACK ukernel.
//
// ABI: invoked from the lowered model.ll as an external MLIR function
//   func.func private @merlin_openblas_gemm_f32(
//       %a : memref<MxK xf32> {bufferization.access = "read"},
//       %b : memref<KxN xf32> {bufferization.access = "read"},
//       %c : memref<MxN xf32> {bufferization.access = "write"}) -> memref<MxN xf32>
// After convert-func-to-llvm each memref is passed UNPACKED as the MLIR descriptor:
//   (float* allocated, float* aligned, intptr offset, intptr size0, size1, stride0, stride1)
// and the result is the standard 2-D memref descriptor struct, returned by value (== the %c
// descriptor in destination-passing form). M/N/K + data pointers come from the descriptors;
// row-major contiguous is assumed (stride0 == ncols, stride1 == 1 — true for these allocs).
//
// C[m,n] = sum_k A[m,k] * B[k,n]   (A row-major MxK, B row-major KxN, C row-major MxN)
//
// The OpenBLAS 8x8 kernel computes, with A/B PRE-PACKED, into a COL-MAJOR C (ld=ldc):
//   C[n*ldc + m] += alpha * sum_k Apack[..] * Bpack[..]
// where (MR=NR=8):
//   Apack[(mp*K + k)*8 + mr] = A[(mp*8+mr)*K + k]   (ncopy_8: per 8-row panel, col-major)
//   Bpack[(np*K + k)*8 + nr] = B[k*N + (np*8+nr)]   (tcopy_8: per 8-col panel, row-major)
//
// ARBITRARY M/N/K: we PAD M,N,K up to multiples of 8 and pack the operands into zero-padded
// buffers, so only the kernel's MR/NR=8 main loop runs (the M/N tails never execute). Zero-pad
// K => zero products; pad M/N => compute-and-discard extra rows/cols. We then copy the MxN
// submatrix of the col-major padded C back to the row-major MxN output. Correct for any shape;
// the e2e cosine gate verifies.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <riscv_vector.h>

// The OpenBLAS kernel body #includes "common.h" (BLASLONG/FLOAT/riscv_vector.h) — supplied by
// the ceiling_drivers common.h shim on the include path. CNAME names the kernel entry.
#define CNAME openblas_sgemm_kernel

// ---- the expert kernel, verbatim (pulls in common.h shim) -----------------
#include "sgemm_kernel_16x8_zvl256b.c"

// 2-D memref descriptor (matches MLIR's MemRefDescriptor for rank 2 / lp64d).
typedef struct {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
} merlin_memref_2d_f32;

#define OB_MR 16
#define OB_NR 8
static size_t round_up8(size_t x) { return (x + 7u) & ~(size_t)7u; }
static size_t round_up16(size_t x) { return (x + 15u) & ~(size_t)15u; }

// tcopy-pack B (KxN row-major) into the OpenBLAS 8-col-panel layout, zero-padded to Npad x Kpad:
//   Bpack[(np*Kpad + k)*8 + nr] = B[k*N + (np*8+nr)]   (0 outside the real K x N)
static void pack_b(size_t N, size_t K, size_t Npad, size_t Kpad, const float *B, float *Bpack) {
  const size_t n_panels = Npad / OB_NR;
  for (size_t np = 0; np < n_panels; np++) {
    for (size_t k = 0; k < Kpad; k++) {
      float *dst = &Bpack[(np * Kpad + k) * OB_NR];
      for (size_t nr = 0; nr < OB_NR; nr++) {
        size_t n = np * OB_NR + nr;
        dst[nr] = (k < K && n < N) ? B[k * N + n] : 0.0f;
      }
    }
  }
}

// ncopy-pack A (MxK row-major) into the OpenBLAS 8-row-panel layout, zero-padded to Mpad x Kpad:
//   Apack[(mp*Kpad + k)*8 + mr] = A[(mp*8+mr)*K + k]   (0 outside the real M x K)
static void pack_a(size_t M, size_t K, size_t Mpad, size_t Kpad, const float *A, float *Apack) {
  const size_t m_panels = Mpad / OB_MR;
  for (size_t mp = 0; mp < m_panels; mp++) {
    for (size_t k = 0; k < Kpad; k++) {
      float *dst = &Apack[(mp * Kpad + k) * OB_MR];
      for (size_t mr = 0; mr < OB_MR; mr++) {
        size_t m = mp * OB_MR + mr;
        dst[mr] = (m < M && k < K) ? A[m * K + k] : 0.0f;
      }
    }
  }
}

// RESIDENT-WEIGHT pack cache (fairness): B is a model weight, constant across forward passes, so
// it is packed ONCE — not on every call inside the timed region. Mirrors the XNNPACK shim's
// get_packed_b and the OpenBLAS ceiling driver's inner-compute scope (pack excluded). Keyed on
// (B,N,K): the first (cold) forward packs each weight; the timed (warm) passes reuse it, so the
// measured wall is kernel-only. Safe because routed matmuls are activation.weight (B = stable
// weight pointer); the e2e cosine gate catches any staleness. A-pack stays PER CALL below (A is
// the activation — OpenBLAS's real per-call cost, fair to include).
#define OB_PACK_CACHE_MAX 512
static struct { const float *b; size_t N, K, Npad, Kpad; float *w; } g_pack_cache[OB_PACK_CACHE_MAX];
static int g_pack_n = 0;
static float *get_packed_b(const float *B, size_t N, size_t K, size_t Npad, size_t Kpad) {
  for (int i = 0; i < g_pack_n; i++)
    if (g_pack_cache[i].b == B && g_pack_cache[i].N == N && g_pack_cache[i].K == K)
      return g_pack_cache[i].w;
  float *w = (float *)malloc(Npad * Kpad * sizeof(float));
  if (!w) return NULL;
  pack_b(N, K, Npad, Kpad, B, w);
  if (g_pack_n < OB_PACK_CACHE_MAX) {
    g_pack_cache[g_pack_n].b = B; g_pack_cache[g_pack_n].N = N; g_pack_cache[g_pack_n].K = K;
    g_pack_cache[g_pack_n].Npad = Npad; g_pack_cache[g_pack_n].Kpad = Kpad;
    g_pack_cache[g_pack_n].w = w; g_pack_n++;
  }
  return w;  // resident: never freed (weight lives for the process, like a real pre-pack)
}

// The MLIR-ABI entry. Reads M/N/K + base pointers from the three unpacked descriptors, computes
// the GEMM through the OpenBLAS RVV 8x8 ukernel, and returns the (filled) C descriptor by value.
merlin_memref_2d_f32 merlin_openblas_gemm_f32(
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

  const size_t Mpad = round_up16(M), Npad = round_up8(N), Kpad = round_up8(K);

  // Resident-weight pack (B = activation·weight's weight): packed ONCE per distinct weight,
  // excluded from the timed/warm path — fair vs ours / XNNPACK.
  float *Bpack = get_packed_b(B, N, K, Npad, Kpad);
  if (!Bpack) { memset(C, 0, M * N * sizeof(float)); return ret; }

  // A-pack is PER CALL (A is the activation; OpenBLAS's real per-call cost). C scratch is
  // col-major Mpad x Npad, zeroed (the kernel does C += alpha*A·B).
  float *Apack = (float *)malloc(Mpad * Kpad * sizeof(float));
  float *Cpad = (float *)calloc(Mpad * Npad, sizeof(float));
  if (!Apack || !Cpad) { free(Apack); free(Cpad); memset(C, 0, M * N * sizeof(float)); return ret; }
  pack_a(M, K, Mpad, Kpad, A, Apack);

  // ldc = Mpad (col-major). Only the MR/NR=8 main loop runs (Mpad,Npad multiples of 8); tails
  // never execute. K loops over the padded Kpad (the extra rows are zero -> zero products).
  openblas_sgemm_kernel((BLASLONG)Mpad, (BLASLONG)Npad, (BLASLONG)Kpad, 1.0f,
                        Apack, Bpack, Cpad, (BLASLONG)Mpad);

  // Copy back the MxN submatrix: col-major Cpad[n*Mpad + m] -> row-major C[m*N + n].
  for (size_t m = 0; m < M; m++)
    for (size_t n = 0; n < N; n++)
      C[m * N + n] = Cpad[n * Mpad + m];

  free(Apack);
  free(Cpad);
  return ret;  // Bpack is cached (resident weight) — not freed
}
