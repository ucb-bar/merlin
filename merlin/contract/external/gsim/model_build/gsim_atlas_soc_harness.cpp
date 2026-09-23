// Standalone GSIM harness for the Atlas full-SoC DUT (ChipTop, below TestHarness) — M3.
// Demonstrates per-cycle RTL register readout from the gsim-generated C++ model of the
// whole Atlas SoC: Rocket scalar core + Atlas NPU tile + TileLink interconnect.
//
// DUT = ChipTop (extracted one level below chipyard's TestHarness via
// gsim_extract_subtree.py, dropping the TSI/SerialRAM/SimDRAM harness plumbing).
// GSIM engine relaxations required (see GSIM_ENGINE.md "atlas M3"): resort() dep-cycle
// break + genResetAll always-true reset, both confined to the debug/JTAG clock+reset
// synchronizer domain.
//
// No memory backend is attached (axi4_mem_0 / serial_tl_0 are left open — those are
// driven by the harness plumbing we intentionally removed), so the cores cannot fetch;
// this therefore proves per-cycle register OBSERVATION on the full SoC (idle), exactly
// as the gemmini M2 idle case. Registers read: Rocket core PC (wb/mem stage), Rocket
// wb valid, and the Atlas tile scalar-core fetch PC.
#include "ChipTop.h"
#include <cstdio>

#define ROCKET(x) system$tile_prci_domain$element_reset_domain$rockettile$core$##x
#define ATLAS(x)  system$domain$atlasTile$core$##x

int main() {
  SChipTop* dut = new SChipTop();

  // Reset window (gsim: step() = one full cycle; reset_io is the AsyncReset input).
  dut->set_custom_boot(0);
  dut->set_reset_io(1);
  for (int i = 0; i < 10; i++) dut->step();
  dut->set_reset_io(0);

  printf("cycle,rocket_wb_pc,rocket_mem_pc,rocket_wb_valid,atlas_fetch_pc\n");
  for (int c = 0; c < 32; c++) {
    dut->step();
    printf("%d,0x%lx,0x%lx,%u,0x%x\n", c,
      (unsigned long)dut->ROCKET(wb_reg_pc),
      (unsigned long)dut->ROCKET(mem_reg_pc),
      (unsigned)     dut->ROCKET(wb_reg_valid),
      (unsigned)     dut->ATLAS(scalar$pc_ctrl$fetch_pc_reg));
  }
  delete dut;
  return 0;
}
