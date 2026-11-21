// Trimmed fixture: XNNPACK qs8 RVV GEMM (packed RHS, accumulator, requant epilogue, VL loop).
void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x1v__rvv(
    size_t mr, size_t nc, size_t kc, const int8_t* a, size_t a_stride,
    const void* w, int8_t* c, size_t cm_stride, size_t cn_stride)
{
  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if (nc < nr) { vl = __riscv_vsetvl_e32m4(nc); }
    nc = nc - vl;
    vint32m4_t vacc0 = __riscv_vle32_v_i32m4((const int32_t*)w, vl);
    vint32m4_t vacc1 = vacc0;
    vint32m4_t vacc2 = vacc0;
    vint32m4_t vacc3 = vacc0;
    w = (const int32_t*) w + nr;
    size_t k = kc;
    do {
      const vint8m1_t vb = __riscv_vle8_v_i8m1((const int8_t*) w, vl);
      w = (const int8_t*) w + nr;
      vacc0 = __riscv_vwmacc(vacc0, a[0], vb, vl);
      vacc1 = __riscv_vwmacc(vacc1, a[1], vb, vl);
      vacc2 = __riscv_vwmacc(vacc2, a[2], vb, vl);
      vacc3 = __riscv_vwmacc(vacc3, a[3], vb, vl);
      k -= sizeof(int8_t);
    } while (k != 0);
    vfloat32m4_t vfacc0 = __riscv_vfcvt_f(vacc0, vl);
    vfacc0 = __riscv_vfmax(vfacc0, output_min, vl);
    vfacc0 = __riscv_vfmin(vfacc0, output_max, vl);
    vint8m1_t vout0 = __riscv_vncvt_x(__riscv_vfncvt_x(vfacc0, vl), vl);
    __riscv_vse8(c, vout0, vl);
  } while (nc != 0);
}
