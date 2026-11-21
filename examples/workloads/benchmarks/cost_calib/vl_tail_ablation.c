// Stage-F L2 ablation: vector_length_polymorphic (vl_agnostic_loop_policy). RVV, no Gemmini.
//
// Saxpy over N_ELEMS floats.
//   VARIANT_FIXED: fixed-width vector body (8 floats/iter), scalar loop for the tail.
//   VARIANT_VLA  : vsetvl-driven loop, tail handled by dynamic VL (XNNPACK/OpenBLAS style).
// Aligned N -> identical work; tail-heavy N -> the fixed variant pays scalar-tail
// instructions. tail_overhead_fraction = instret(fixed)/instret(vla) - 1.
#include <stdio.h>
#include <riscv_vector.h>

#ifndef N_ELEMS
#define N_ELEMS 1032
#endif

static float x[N_ELEMS], y[N_ELEMS];

int main(void) {
  for (int i = 0; i < N_ELEMS; i++) {
    x[i] = i * 0.25f;
    y[i] = 1.0f;
  }
  const float a = 2.0f;
#if defined(VARIANT_FIXED)
  int i = 0;
  size_t vl = __riscv_vsetvl_e32m1(4);                 // fixed width: 4 lanes (VLEN=128)
  for (; i + 4 <= N_ELEMS; i += 4) {
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&x[i], vl);
    vfloat32m1_t vy = __riscv_vle32_v_f32m1(&y[i], vl);
    __riscv_vse32_v_f32m1(&y[i], __riscv_vfmacc_vf_f32m1(vy, a, vx, vl), vl);
  }
  for (; i < N_ELEMS; i++)                             // scalar tail
    y[i] += a * x[i];
#else
  for (int i = 0; i < N_ELEMS;) {                      // vector-length-agnostic loop
    size_t vl = __riscv_vsetvl_e32m1(N_ELEMS - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&x[i], vl);
    vfloat32m1_t vy = __riscv_vle32_v_f32m1(&y[i], vl);
    __riscv_vse32_v_f32m1(&y[i], __riscv_vfmacc_vf_f32m1(vy, a, vx, vl), vl);
    i += vl;
  }
#endif
  int ok = 1;
  for (int i = 0; i < N_ELEMS; i++)
    if (y[i] != 1.0f + 2.0f * (i * 0.25f))
      ok = 0;
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
