// Trimmed fixture: Autocomp Gemmini matmul (weight-stationary, packed RHS via mvin,
// accumulator addressing 1<<31, preloaded systolic compute, single mvout).
void test(int8_t A[512][512], int8_t B[512][512], int8_t C[512][512]) {
  config_st(512);
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld(512, 1.0f, 16, 2);
  for (int i = 0; i < 32; i++) {
    uint32_t a_addr = 0;
    for (int ko = 0; ko < 8; ko++) {
      mvin2(&A[i * 16][ko * 64], a_addr + ko * 64, 64, 16);
    }
    for (int j = 0; j < 8; j++) {
      uint32_t res = 1u << 31;
      uint32_t b_base = 8192;
      for (int ko = 0; ko < 8; ko++) {
        mvin3(&B[ko * 64][j * 64], b_base + ko * 256, 64, 16);
        preload(b_base + ko * 256, (res) | 0x40000000, 16, 16, 16, 16);
        compute_preloaded(a_addr + ko * 64, ~((uint32_t)0), 16, 16, 16, 16);
      }
      mvout(&C[i * 16][j * 64], res, 16, 16);
    }
  }
  fence();
}
