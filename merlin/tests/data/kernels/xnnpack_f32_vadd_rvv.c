// Trimmed fixture: XNNPACK f32 RVV vadd — a flat elementwise op (NEGATIVE control:
// no packed RHS, no accumulator-commit, no tiling; only a VL-agnostic loop).
void xnn_f32_vadd_ukernel__rvv_u8v(
    size_t batch, const float* input_a, const float* input_b, float* output)
{
  do {
    const size_t vl = __riscv_vsetvl_e32m8(batch);
    batch -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(input_b, vl);
    vfloat32m8_t vacc = __riscv_vfadd_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    input_a += vl; input_b += vl; output += vl;
  } while (batch != 0);
}
