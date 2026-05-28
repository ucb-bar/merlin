void kernel_add(const float *a, const float *b, float *output, int n) {
	size_t vl;
	for (int i = 0; i < n; i += vl) {
		vl = __riscv_vsetvl_e32m8(n - i);
		vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
		vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
		vfloat32m8_t vout = __riscv_vfadd_vv_f32m8(va, vb, vl);
		__riscv_vse32_v_f32m8(output + i, vout, vl);
	}
}
