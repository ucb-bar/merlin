// Minimal standalone GSIM harness for the Gemmini accelerator — accelerator-only M2.
// Reads the three controller FSM-state registers + io.busy each cycle.
// Vector IO accessors repaired by gsim_fix_vector_io.py; codegen defects worked
// around by gsim_fix_gemmini_codegen.py.
//
// No RoCC command is driven: a valid gemmini instruction stream is needed to make
// the controllers busy (and an invalid one trips the design's own TLB assertion).
// This harness therefore demonstrates that per-cycle register OBSERVATION works on
// the real gemmini accelerator; the controllers sit idle without a valid kernel.
#include "Gemmini.h"
#include <cstdio>

int main() {
  SGemmini* dut = new SGemmini();

  printf("phase,cycle,ex_state,load_state,store_state,io_busy,rs_busy_T4\n");

  // Observe registers THROUGH the reset window and after.
  dut->set_reset(1);
  for (int c = 0; c < 6; c++) {
    dut->step();
    printf("reset,%d,%u,%u,%u,%u,%u\n", c,
      (unsigned)dut->ex_controller$control_state,
      (unsigned)dut->load_controller$control_state,
      (unsigned)dut->store_controller$control_state,
      (unsigned)dut->get_io$$busy(),
      (unsigned)dut->reservation_station$_io_busy_T_4);
  }
  dut->set_reset(0);
  for (int c = 0; c < 12; c++) {
    dut->step();
    printf("run,%d,%u,%u,%u,%u,%u\n", c,
      (unsigned)dut->ex_controller$control_state,
      (unsigned)dut->load_controller$control_state,
      (unsigned)dut->store_controller$control_state,
      (unsigned)dut->get_io$$busy(),
      (unsigned)dut->reservation_station$_io_busy_T_4);
  }
  delete dut;
  return 0;
}
