// Minimal standalone GSIM harness for the gemmini FULL-SoC DUT (ChipTop) — M3.
//
// ChipTop is the synthesizable SoC one level below chipyard's TestHarness: it
// contains DigitalTop = Rocket core + L1/L2 + SystemBus/PeripheryBus TileLink
// interconnect + the Gemmini RoCC accelerator, with the AXI DRAM port
// (axi4_mem_0) and the TSI serial-bringup port (serial_tl_0) exposed at the chip
// boundary.  The TSI<->TileLink serial bridge / SimTSI / SimDRAM live ONLY in
// TestHarness above this and were pruned out (gsim_prune_to_dut.py), so the
// splitArray.cpp:115 harness-node blocker is gone.
//
// This harness proves per-cycle register OBSERVATION of the whole SoC datapath:
// it reads a Rocket program-counter register, the three Gemmini controller FSM
// states, and the AXI mem-read-request valid, every cycle, straight off the
// public data members of class SChipTop.  DRAM is not backed here (SimDRAM was in
// the harness), so the core stalls on its first miss — expected; the point is the
// M3 register-observation substrate, not a booted workload.
#include "ChipTop.h"
#include <cstdio>

// NOTE: '$' is part of the C identifier token here (clang extension), so the
// hierarchical member names cannot be abbreviated with macros — spelled in full.

int main() {
  SChipTop* dut = new SChipTop();

  printf("phase,cycle,wb_reg_pc,ex_reg_pc,mem_reg_pc,"
         "gem_ex_state,gem_load_state,gem_store_state,gem_rs_busy,"
         "axi_ar_valid,axi_aw_valid\n");

  auto dump = [&](const char* phase, int c) {
    printf("%s,%d,0x%llx,0x%llx,0x%llx,%u,%u,%u,%u,%u,%u\n",
      phase, c,
      (unsigned long long)dut->system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_pc,
      (unsigned long long)dut->system$tile_prci_domain$element_reset_domain$rockettile$core$ex_reg_pc,
      (unsigned long long)dut->system$tile_prci_domain$element_reset_domain$rockettile$core$mem_reg_pc,
      (unsigned)dut->system$tile_prci_domain$element_reset_domain$rockettile$gemmini$ex_controller$control_state,
      (unsigned)dut->system$tile_prci_domain$element_reset_domain$rockettile$gemmini$load_controller$control_state,
      (unsigned)dut->system$tile_prci_domain$element_reset_domain$rockettile$gemmini$store_controller$control_state,
      (unsigned)dut->system$tile_prci_domain$element_reset_domain$rockettile$gemmini$reservation_station$_io_busy_T_4,
      (unsigned)dut->get_axi4_mem_0$$bits$$ar$$valid(),
      (unsigned)dut->get_axi4_mem_0$$bits$$aw$$valid());
  };

  // Provide a running clock signal + hold reset asserted through the reset window.
  dut->set_clock_uncore(1);
  dut->set_reset_io(1);
  for (int c = 0; c < 10; c++) { dut->step(); dump("reset", c); }

  dut->set_reset_io(0);
  // Tie off AXI/serial slave-side readies so the SoC is not back-pressured.
  dut->set_axi4_mem_0$$bits$$aw$$ready(1);
  dut->set_axi4_mem_0$$bits$$w$$ready(1);
  dut->set_axi4_mem_0$$bits$$ar$$ready(1);
  dut->set_serial_tl_0$$in$$valid(0);
  dut->set_serial_tl_0$$out$$ready(1);
  dut->set_custom_boot(0);
  for (int c = 0; c < 40; c++) { dut->step(); dump("run", c); }

  printf("# total cycles stepped: %llu\n", (unsigned long long)dut->cycles);
  delete dut;
  return 0;
}
