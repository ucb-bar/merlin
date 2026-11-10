/* Transpose-free WEIGHT-ONLY int8 GEMM for the K1 (rv64gcv): the int8 weight stays in its NATIVE
 * [N,K] layout and is dequantized on the fly inside the k-reduction dot — no transpose/repack and
 * no separate dequant buffer is ever materialized (the strided vsse32 scatter disappears).
 *
 *   Y[m,n] = w_scale[n] * sum_k  X_f32[m,k] * (float)W_i8[n,k]        (W row n is k-contiguous)
 *
 * This is the f32 analogue of igemm_nk_dot: both A[m,:] and W[n,:] are k-contiguous, so we vle8 a
 * VL-wide int8 slab of W[n,:], widen i8->i16->i32->f32, vfmacc against the contiguous X[m,:] slab,
 * then a vfredusum tail to the scalar accumulator, scaled once by the per-output-channel weight
 * scale. No strided access (K1 strided loads/stores are catastrophic); M is tiny (S=8).
 *
 * Weight-only int8 (activations stay f32) is exactly the math this capture's int8 golden encodes;
 * the EXO f32 GEMM (gemm_nt_ref) is still compiled + RVV-audited for the EXO-authored-kernel story,
 * but this transpose-free dot is on the hot path so the whole-model run is fast.
 */
#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

/* Y[M,N] f32 = per-n dequant of the int8 W[N,K] dotted with X[M,K] f32. WS[N] = per-output scale. */
void fgemm_nk_dot_i8(int M, int N, int K, float *Y,
                     const float *X, const int8_t *W, const float *WS) {
  for (int m = 0; m < M; m++) {
    const float *xr = X + (size_t)m * K;
    for (int n = 0; n < N; n++) {
      const int8_t *wr = W + (size_t)n * K;
      vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, __riscv_vsetvlmax_e32m2());
      for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m2(K - k);
        vfloat32m2_t a  = __riscv_vle32_v_f32m2(&xr[k], vl);
        vint8mf2_t   w8 = __riscv_vle8_v_i8mf2(&wr[k], vl);        /* contiguous int8 weight slab */
        vint16m1_t   w16= __riscv_vwadd_vx_i16m1(w8, 0, vl);      /* i8 -> i16 (sign-extend) */
        vint32m2_t   w32= __riscv_vwadd_vx_i32m2(w16, 0, vl);     /* i16 -> i32 */
        vfloat32m2_t wf = __riscv_vfcvt_f_x_v_f32m2(w32, vl);     /* i32 -> f32 (raw int weight) */
        acc = __riscv_vfmacc_vv_f32m2(acc, a, wf, vl);           /* acc += X * (float)W_i8 */
        k += (int)vl;
      }
      vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
      float s = __riscv_vfmv_f_s_f32m1_f32(
          __riscv_vfredusum_vs_f32m2_f32m1(acc, z, __riscv_vsetvlmax_e32m2()));
      Y[(size_t)m * N + n] = WS[n] * s;                           /* dequant: * per-output scale */
    }
  }
}
