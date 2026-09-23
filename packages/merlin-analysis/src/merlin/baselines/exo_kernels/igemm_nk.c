/* Transpose-free int8 W8A8 GEMM for the K1 (rv64gcv), consuming the weight in its NATIVE [N,K]
 * layout — no transpose/repack buffer is ever materialized (the 461M-tick scatter disappears).
 *
 *   Y[m,n] = sum_k A_i16[m,k] * W_i8[n,k]           (W row n is k-contiguous)
 *
 * Two strategies are provided; the runner micro-benchmarks both on-board and keeps the faster:
 *
 *  (A) igemm_nk_strided — output-register blocked vwmacc (keeps the U-tile win). At each k, a
 *      16-wide slab of the output columns [n0:n0+16] lives across k rows -> W[n0:n0+16, k] is a
 *      STRIDED gather (stride K). We `vlse8` (strided i8 load) 16 weights, widen i8->i16, and
 *      `vwmacc.vx` with the single broadcast A[m,k]. U such 16-wide accumulators share one A-load.
 *
 *  (B) igemm_nk_dot — k-reduction dot product. Both A[m, :] and W[n, :] are k-contiguous, so we
 *      `vle16`/`vle8`+widen a VL-wide slab, `vwmul`+accumulate along k, then a `vredsum` tail to
 *      the scalar Y[m,n]. No strided access, but a reduction tail per (m,n).
 *
 * Both take W as i16 [N,K] (the glue widens i8->i16 with a cheap CONTIGUOUS streaming copy — same
 * layout, no scatter). Signature matches the EXO kernel's call site so the glue is unchanged.
 */
#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

/* (A) strided-vwmacc, output-register-blocked by U=8. RVV vector C types are SIZELESS (can't be
 * arrayed: `vint32m2_t acc[8]` is illegal), so the 8 accumulators are 8 named registers. Y[M,N],
 * X[M,K] i16, W[N,K] i16. */
#define NKU 8
void igemm_nk_strided(void *ctxt, int M, int N, int K,
                      int32_t *Y, const uint16_t *X, const uint16_t *W) {
  (void)ctxt;
  const int16_t *Xi = (const int16_t *)X;
  const int16_t *Wi = (const int16_t *)W;
  const ptrdiff_t sB = (ptrdiff_t)(K * (int)sizeof(int16_t));   /* byte stride between W rows */
  for (int m = 0; m < M; m++) {
    int n = 0;
    for (; n + 16 * NKU <= N; n += 16 * NKU) {
      vint32m2_t a0 = __riscv_vmv_v_x_i32m2(0, 16), a1 = __riscv_vmv_v_x_i32m2(0, 16);
      vint32m2_t a2 = __riscv_vmv_v_x_i32m2(0, 16), a3 = __riscv_vmv_v_x_i32m2(0, 16);
      vint32m2_t a4 = __riscv_vmv_v_x_i32m2(0, 16), a5 = __riscv_vmv_v_x_i32m2(0, 16);
      vint32m2_t a6 = __riscv_vmv_v_x_i32m2(0, 16), a7 = __riscv_vmv_v_x_i32m2(0, 16);
      for (int k = 0; k < K; k++) {
        int16_t a = Xi[(size_t)m * K + k];
        const int16_t *wk = Wi + (size_t)n * K + k;             /* &W[n, k] */
        /* each 16-wide weight slab W[n+16u .. +15, k] is strided by K rows (vlse16). */
        a0 = __riscv_vwmacc_vx_i32m2(a0, a, __riscv_vlse16_v_i16m1(wk + (size_t)0 * 16 * K, sB, 16), 16);
        a1 = __riscv_vwmacc_vx_i32m2(a1, a, __riscv_vlse16_v_i16m1(wk + (size_t)1 * 16 * K, sB, 16), 16);
        a2 = __riscv_vwmacc_vx_i32m2(a2, a, __riscv_vlse16_v_i16m1(wk + (size_t)2 * 16 * K, sB, 16), 16);
        a3 = __riscv_vwmacc_vx_i32m2(a3, a, __riscv_vlse16_v_i16m1(wk + (size_t)3 * 16 * K, sB, 16), 16);
        a4 = __riscv_vwmacc_vx_i32m2(a4, a, __riscv_vlse16_v_i16m1(wk + (size_t)4 * 16 * K, sB, 16), 16);
        a5 = __riscv_vwmacc_vx_i32m2(a5, a, __riscv_vlse16_v_i16m1(wk + (size_t)5 * 16 * K, sB, 16), 16);
        a6 = __riscv_vwmacc_vx_i32m2(a6, a, __riscv_vlse16_v_i16m1(wk + (size_t)6 * 16 * K, sB, 16), 16);
        a7 = __riscv_vwmacc_vx_i32m2(a7, a, __riscv_vlse16_v_i16m1(wk + (size_t)7 * 16 * K, sB, 16), 16);
      }
      int32_t *yr = &Y[(size_t)m * N + n];
      __riscv_vse32_v_i32m2(yr + 0 * 16, a0, 16); __riscv_vse32_v_i32m2(yr + 1 * 16, a1, 16);
      __riscv_vse32_v_i32m2(yr + 2 * 16, a2, 16); __riscv_vse32_v_i32m2(yr + 3 * 16, a3, 16);
      __riscv_vse32_v_i32m2(yr + 4 * 16, a4, 16); __riscv_vse32_v_i32m2(yr + 5 * 16, a5, 16);
      __riscv_vse32_v_i32m2(yr + 6 * 16, a6, 16); __riscv_vse32_v_i32m2(yr + 7 * 16, a7, 16);
    }
    /* tail: remaining output columns one 16-wide tile at a time */
    for (; n < N; n += 16) {
      vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, 16);
      for (int k = 0; k < K; k++) {
        int16_t a = Xi[(size_t)m * K + k];
        vint16m1_t w = __riscv_vlse16_v_i16m1(Wi + (size_t)n * K + k, sB, 16);
        acc = __riscv_vwmacc_vx_i32m2(acc, a, w, 16);
      }
      __riscv_vse32_v_i32m2(&Y[(size_t)m * N + n], acc, 16);
    }
  }
}

/* (B) k-reduction dot product. Contiguous loads of A[m,:] and W[n,:], vwmul+accumulate, vredsum. */
void igemm_nk_dot(void *ctxt, int M, int N, int K,
                  int32_t *Y, const uint16_t *X, const uint16_t *W) {
  (void)ctxt;
  const int16_t *Xi = (const int16_t *)X;
  const int16_t *Wi = (const int16_t *)W;
  for (int m = 0; m < M; m++) {
    const int16_t *xr = Xi + (size_t)m * K;
    for (int n = 0; n < N; n++) {
      const int16_t *wr = Wi + (size_t)n * K;
      vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, __riscv_vsetvlmax_e32m2());
      for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e16m1(K - k);
        vint16m1_t a = __riscv_vle16_v_i16m1(&xr[k], vl);
        vint16m1_t w = __riscv_vle16_v_i16m1(&wr[k], vl);
        acc = __riscv_vwmacc_vv_i32m2(acc, a, w, vl);           /* acc += (i32)a*w, vector-vector */
        k += vl;
      }
      vint32m1_t z = __riscv_vmv_v_x_i32m1(0, 1);
      int32_t s = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m2_i32m1(acc, z, __riscv_vsetvlmax_e32m2()));
      Y[(size_t)m * N + n] = s;
    }
  }
}
