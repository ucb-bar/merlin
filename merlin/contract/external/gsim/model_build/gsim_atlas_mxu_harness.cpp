// Minimal standalone GSIM harness for the Atlas MXU (SystolicArray) — accelerator-only M2.
// Demonstrates per-cycle register readout from the gsim-generated C++ model.
// Vector top-level IO accessors were repaired by gsim_fix_vector_io.py.
#include "SystolicArray.h"
#include <cstdio>
#include <cstring>

int main() {
  SSystolicArray* dut = new SSystolicArray();

  // Reset sequence (gsim: step() advances one full cycle).
  dut->set_reset(1);
  for (int i = 0; i < 10; i++) dut->step();
  dut->set_reset(0);

  // Drive one compute request into the 32x32 systolic array.
  uint8_t  act[32];        for (int i = 0; i < 32; i++) act[i]  = 1;
  uint16_t psum[32];       for (int i = 0; i < 32; i++) psum[i] = 0;
  uint8_t  w0[32][32];     memset(w0, 3, sizeof(w0));
  uint8_t  w1[32][32];     memset(w1, 5, sizeof(w1));

  dut->set_io$$computeReq$$valid(1);
  dut->set_io$$computeReq$$bits$$accumulate(0);
  dut->set_io$$computeReq$$bits$$weightBufSel(0);
  dut->set_io$$computeReq$$bits$$act(act);
  dut->set_io$$computeReq$$bits$$psum(psum);
  dut->set_io$$weights0(&w0[0][0]);
  dut->set_io$$weights1(&w1[0][0]);

  printf("cycle,outValid_r,outValid_r8,addend8,addend9,io_outValid,io_outBits0\n");
  for (int c = 0; c < 48; c++) {
    dut->step();
    if (c == 4) dut->set_io$$computeReq$$valid(0);  // deassert after a few cycles
    printf("%d,%u,%u,%u,%u,%u,%u\n",
      c,
      (unsigned)dut->io_outputRow_valid_r,
      (unsigned)dut->io_outputRow_valid_r_8,
      (unsigned)dut->peMesh_io_addendVec_8_r,
      (unsigned)dut->peMesh_io_addendVec_9_r,
      (unsigned)dut->get_io$$outputRow$$valid(),
      (unsigned)dut->get_io$$outputRow$$bits(0));
  }
  delete dut;
  return 0;
}
