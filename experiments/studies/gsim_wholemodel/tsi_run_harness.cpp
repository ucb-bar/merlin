// Run a whole-model image on the gemmini ChipTop under GSIM, driven over the chip's serial-TL port
// by fesvr's own TSI host.
//
// WHY THIS AND NOT THE DRAM BACKDOOR.  Writing the image straight into the backing store works and
// the core runs it, but it leaves two things unsolved, both at the HOST boundary rather than the
// memory one: the program's HTIF handshake (`tohost` written, `fromhost` polled) completes inside a
// single cache line and never reaches memory, so a harness watching DRAM cannot see or answer it;
// and the results are still dirty in cache when the run stops. fesvr's `tsi_t` is the thing that
// already solves both -- it loads through the SoC coherently and acts as the HTIF host -- and the
// DUT-side port it talks to is present and fully wired here. Only the host side was pruned.
//
// The tick order mirrors testchipip's SimTSI DPI exactly rather than being re-derived: a serial
// handshake that is off by a cycle still simulates and silently transfers the wrong phits.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "ChipTop.h"
#include "testchip_tsi.h"

extern "C" void gemmini_dram_init();
extern "C" void gemmini_axi_tick(SChipTop* dut, uint8_t reset);
extern "C" void gemmini_axi_stats();

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <elf> [max_cycles]\n", argv[0]); return 2; }
  const long max_cycles = (argc > 2) ? atol(argv[2]) : 200000000L;

  gemmini_dram_init();
  // has_loadmem=false: load over the serial link itself, so nothing depends on a backdoor callback
  // this harness would also have to provide.
  testchip_tsi_t* tsi = new testchip_tsi_t(argc, argv, false);

  SChipTop* dut = new SChipTop();
  dut->set_clock_uncore(1);
  dut->set_reset_io(1);
  // Clock the serial link THROUGH reset as well: a block first clocked after reset release can come
  // up with its receiver never asserting ready.
  for (int c = 0; c < 20; c++) { dut->set_serial_tl_0$$clock_in(c & 1);
                                 gemmini_axi_tick(dut, 1); dut->step(); }
  dut->set_reset_io(0);
  dut->set_custom_boot(0);          // 0 = wait for TSI, which now actually arrives

  long c = 0; int rc = -1;
  unsigned long n_in=0, n_out=0, n_inready=0, n_invalid_host=0;
  for (; c < max_cycles; ++c) {
    // DUT -> host, then host -> DUT, in SimTSI's order.
    bool     out_valid = dut->get_serial_tl_0$$out$$valid();
    uint32_t out_bits  = dut->get_serial_tl_0$$out$$bits$$phit();
    bool     in_ready  = dut->get_serial_tl_0$$in$$ready();

    tsi->tick(out_valid, out_bits, in_ready);
    tsi->switch_to_host();

    dut->set_serial_tl_0$$out$$ready(tsi->out_ready());
    dut->set_serial_tl_0$$in$$valid(tsi->in_valid());
    dut->set_serial_tl_0$$in$$bits$$phit(tsi->in_bits());

    // The serial link is clocked by its OWN pin. Left undriven it never advances, and the DUT sits in
    // the bootrom's wfi-spin waiting for phits that cannot arrive -- which looks exactly like a TSI
    // host that is not there.
    dut->set_serial_tl_0$$clock_in(c & 1);
    gemmini_axi_tick(dut, 0);
    dut->step();

    if (tsi->in_valid()) ++n_invalid_host;         // host has a phit to give
    if (tsi->in_valid() && in_ready) ++n_in;        // ... and the DUT took it
    if (out_valid) ++n_out;                         // DUT has a phit for us
    if (in_ready) ++n_inready;
    if (tsi->done()) { rc = tsi->exit_code(); break; }
    if ((c % 500000) == 0) {
      printf("# cycle %ld pc=0x%llx\n", c,
        (unsigned long long)dut->system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_pc);
      fflush(stdout);
    }
  }
  printf("# stopped at cycle %ld, tsi done=%d exit=%d\n", c, (int)tsi->done(), rc);
  printf("# link: host_has_phit=%lu accepted_by_dut=%lu dut_sent=%lu dut_ready=%lu\n",
         n_invalid_host, n_in, n_out, n_inready);
  gemmini_axi_stats();
  delete dut;
  return rc > 0 ? rc : 0;
}
