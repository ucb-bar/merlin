/* Trimmed fixture modeled on OpenBLAS kernel/riscv64/zaxpy_rvv.c (RVV BLAS1). */
#include "common.h"

#define VSETVL(n) __riscv_vsetvl_e64m4(n)
#define FLOAT_V_T vfloat64m4_t
#define VLSEG_FLOAT __riscv_vlseg2e64_v_f64m4
#define VSSEG_FLOAT __riscv_vsseg2e64_v_f64m4
#define VFMACCVF_FLOAT __riscv_vfmacc_vf_f64m4

int CNAME(BLASLONG n, FLOAT da_r, FLOAT da_i, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y)
{
    BLASLONG i = 0;
    size_t vl;
    FLOAT_V_T vx0, vy0;
    for (; n > 0; n -= vl, x += vl * 2, y += vl * 2) {
        vl = VSETVL(n);
        vy0 = VFMACCVF_FLOAT(vy0, da_r, vx0, vl);
        VSSEG_FLOAT(y, vy0, vl);
    }
    return 0;
}
