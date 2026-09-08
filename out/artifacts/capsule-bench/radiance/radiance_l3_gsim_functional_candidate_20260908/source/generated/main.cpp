// Radiance RadianceGsimConfig boot harness main loop for the GSIM-generated STestHarness.
//
// Usage: emu <soc.elf> [+plusargs...]
//   The ELF is loaded into the SimDRAM backing array by the fesvr TSI (loadmem). We drive the
//   top-level clock/reset (GSIM step() == one full clock), hold reset for a few cycles, then run
//   until fesvr reports htif done() (tohost exit) or the cycle cap is reached.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include "TestHarness.h"

extern "C" void harness_set_args(int argc, char** argv);
extern "C" bool dram_peek(uint64_t phys, void* buf, unsigned long n);
extern volatile bool g_tsi_done;
extern int g_exit_code;

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <soc.elf> [+plusargs]\n", argv[0]); return 2; }
  harness_set_args(argc, argv);

  uint64_t max_cycles = 20000000ULL;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a.rfind("+max-cycles=", 0) == 0) max_cycles = strtoull(a.c_str() + 12, nullptr, 0);
  }

  STestHarness* dut = new STestHarness();
  // Power-on zero-init: GSIM's `new` does not value-initialize members and RANDOMIZE_INIT is off,
  // so non-reset registers (e.g. icacheInFlightsReg) start as heap garbage → the Muon icache never
  // completes a fetch. Zero the whole register-storage block [_var_start,_var_end) to a clean 0 state.
  {
    uint32_t* p0 = &dut->_var_start;
    uint32_t* p1 = &dut->_var_end;
    for (uint32_t* p = p0; p < p1; ++p) *p = 0;
    fprintf(stderr, "[gsim-emu] zero-init %ld register words\n", (long)(p1 - p0));
  }

#define RPC   dut->chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_pc
#define MI00  dut->chiptop0$system$cluster_prci_domain$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$minstretReg
#define MI01  dut->chiptop0$system$cluster_prci_domain$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$be$execute$minstretReg
#define MI10  dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$minstretReg
#define MI11  dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$be$execute$minstretReg
#define MRST  dut->chiptop0$system$cluster_prci_domain$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$_be_reset_T_1
#define MPC   dut->chiptop0$system$cluster_prci_domain$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$fe$warpScheduler$currPC
#define PROBE(tag) fprintf(stderr, "[gsim-probe %s] rocket_pc=0x%llx muon_c0t0_currPC=0x%llx muon_minstret[c0t0=%llu c0t1=%llu c1t0=%llu c1t1=%llu] muon_be_reset=%u\n", \
    tag, (unsigned long long)RPC, (unsigned long long)MPC, (unsigned long long)MI00, (unsigned long long)MI01, \
    (unsigned long long)MI10, (unsigned long long)MI11, (unsigned)MRST)

// --- fetch-path localization probe (cluster0 core0) ---
#define FIC_V dut->chiptop0$system$cluster_prci_domain$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$icacheWordNodeOut$$a$$valid
#define FIF   dut->chiptop0$system$cluster_prci_domain$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$fe$warpScheduler$icacheInFlightsReg
#define FPROBE(c) fprintf(stderr, "[gsim-fetch cyc=%llu] currPC=0x%llx icacheWordNode.a.valid=%u icacheInFlights=0x%llx be_reset=%u\n", \
    (unsigned long long)(c), (unsigned long long)MPC, (unsigned)FIC_V, (unsigned long long)FIF, (unsigned)MRST)

// --- FP writeback / regfile localization probe (writeback valid, dest reg, data, regfile slot). ---
#define WBPROBE(cyc, tag, VALID, RD, DATA, TMASK, RF) do { \
  if (VALID) { unsigned rd = (unsigned)(RD); \
    fprintf(stderr, "[wb cyc=%llu %s] rd=%u data[0]=0x%08x data[1]=0x%08x tmask=0x%llx rf[rd][0]=0x%08x\n", \
      (unsigned long long)(cyc), tag, rd, (unsigned)(DATA)[0], (unsigned)(DATA)[1], \
      (unsigned long long)(TMASK), (unsigned)(RF)[rd][0]); } } while(0)
// cluster1 core0 (clid=1 cid=0, where the matmul warp issues)
#define A10V dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$respArbiter$io$$out$$bits$$reg$$valid
#define A10R dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$respArbiter$io$$out$$bits$$reg$$bits$$rd
#define A10D dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$respArbiter$io$$out$$bits$$reg$$bits$$data
#define A10T dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$respArbiter$io$$out$$bits$$reg$$bits$$tmask
#define F10  dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$collector$rfBanks_mem
// cluster1 core1
#define A11V dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$be$execute$respArbiter$io$$out$$bits$$reg$$valid
#define A11R dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$be$execute$respArbiter$io$$out$$bits$$reg$$bits$$rd
#define A11D dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$be$execute$respArbiter$io$$out$$bits$$reg$$bits$$data
#define A11T dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$be$execute$respArbiter$io$$out$$bits$$reg$$bits$$tmask
#define F11  dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$be$collector$rfBanks_mem
// c1t0 fpAddMulPipe CVFPU resp (as the core sees it): valid, tag, result lane0
#define R10V dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$fpAddMulPipe$CVFPU$resp$$valid
#define R10T dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$fpAddMulPipe$CVFPU$resp$$bits$$tag
#define R10D dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$be$execute$fpAddMulPipe$CVFPU$resp$$bits$$result
#define WB_ALL(cyc) do { \
  if (R10V) fprintf(stderr, "[fpresp cyc=%llu c1t0] valid=1 tag=%u result[0]=0x%08x\n", \
    (unsigned long long)(cyc), (unsigned)R10T, (unsigned)(uint32_t)(uint64_t)R10D); \
  WBPROBE(cyc, "c1t0", A10V, A10R, A10D, A10T, F10); \
  WBPROBE(cyc, "c1t1", A11V, A11R, A11D, A11T, F11); } while(0)

// c1t0 flush units (icache l0i + dcache l0d): flushing, counter, inFlights, dirty, and dcache wb release
#define FI_FL dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$l0i$tlnbdCache$nbdCache$flush_unit$flushing
#define FI_CT dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$l0i$tlnbdCache$nbdCache$flush_unit$flushCounter_value
#define FI_IF dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$l0i$tlnbdCache$nbdCache$flush_unit$inFlights
#define FD_FL dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$l0d$tlnbdCache$nbdCache$flush_unit$flushing
#define FD_CT dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$l0d$tlnbdCache$nbdCache$flush_unit$flushCounter_value
#define FD_IF dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$l0d$tlnbdCache$nbdCache$flush_unit$inFlights
#define FD_DR dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$l0d$tlnbdCache$nbdCache$flush_unit$isDirty
#define FD_WBV dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$l0d$tlnbdCache$nbdCache$wb$io$$release$$valid
#define FLUSH_PROBE(cyc) do { if (FI_FL || FD_FL) \
  fprintf(stderr, "[flush cyc=%llu c1t0] i{fl=%u ct=%u if=%u} d{fl=%u ct=%u if=%u dirty=%u wbrel=%u}\n", \
    (unsigned long long)(cyc), (unsigned)FI_FL, (unsigned)FI_CT, (unsigned)FI_IF, \
    (unsigned)FD_FL, (unsigned)FD_CT, (unsigned)FD_IF, (unsigned)FD_DR, (unsigned)FD_WBV); } while(0)

// per-core warpScheduler "finished" signal (allFinished = AND of all 4 => stopSim)
#define FIN00 dut->chiptop0$system$cluster_prci_domain$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$fe$warpScheduler$_io_finished_T_9
#define FIN01 dut->chiptop0$system$cluster_prci_domain$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$fe$warpScheduler$_io_finished_T_9
#define FIN10 dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain$element_reset_domain$muon_tile$core$fe$warpScheduler$_io_finished_T_9
#define FIN11 dut->chiptop0$system$cluster_prci_domain_1$element_reset_domain$element$tile_prci_domain_1$element_reset_domain$muon_tile$core$fe$warpScheduler$_io_finished_T_9
#define FIN_ALL dut->chiptop0$system$domain_1$resetAggregator$allFinished
#define FIN_PROBE(cyc) fprintf(stderr, "[fin cyc=%llu] c0t0=%u c0t1=%u c1t0=%u c1t1=%u allFinished=%u\n", \
    (unsigned long long)(cyc), (unsigned)FIN00, (unsigned)FIN01, (unsigned)FIN10, (unsigned)FIN11, (unsigned)FIN_ALL)

  // reset sequence
  dut->set_reset(1);
  for (int i = 0; i < 12; i++) dut->step();
  dut->set_reset(0);
  PROBE("post-reset");

  fprintf(stderr, "[gsim-emu] reset done, running (max_cycles=%llu)\n",
          (unsigned long long)max_cycles);

  auto start = std::chrono::steady_clock::now();
  uint64_t cycles = 0;
  extern unsigned long long g_sim_cycle;
  while (!g_tsi_done && cycles < max_cycles) {
    dut->step();
    cycles++;
    g_sim_cycle = cycles;
    if (cycles >= 1 && cycles <= 160) FPROBE(cycles);
    // The filtered certification model omits the optional response-arbiter
    // debug nets.  The completion/result contract does not depend on them;
    // keeping this probe would make an otherwise faithful model unbuildable.
    if (getenv("FLUSH_TRACE")) FLUSH_PROBE(cycles);
    if (getenv("FIN_TRACE") && (cycles % 1000ULL) == 0) FIN_PROBE(cycles);
    if ((cycles % 2000ULL) == 0) PROBE("tick");
    if ((cycles % 1000000ULL) == 0) {
      auto now = std::chrono::steady_clock::now();
      double s = std::chrono::duration<double>(now - start).count();
      fprintf(stderr, "[gsim-emu] %llu cycles, %.1fs, %.0f cyc/s\n",
              (unsigned long long)cycles, s, cycles / s);
      PROBE("progress");
    }
  }
  PROBE("final");
  auto end = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(end - start).count();

  fflush(stdout);
  fprintf(stderr, "\n[gsim-emu] FINISHED: cycles=%llu wall=%.2fs (%.0f cyc/s) done=%d exit_code=%d\n",
          (unsigned long long)cycles, secs, cycles / (secs > 0 ? secs : 1), (int)g_tsi_done, g_exit_code);

  // Optional DRAM dumps: GSIM_DUMP=addr:len[,addr:len...] (hex addr, decimal len) -> hex to stderr
  if (const char* env = getenv("GSIM_DUMP")) {
    std::string s = env; size_t pos = 0;
    while (pos < s.size()) {
      size_t comma = s.find(',', pos); if (comma == std::string::npos) comma = s.size();
      std::string tok = s.substr(pos, comma - pos); pos = comma + 1;
      size_t colon = tok.find(':'); if (colon == std::string::npos) continue;
      uint64_t addr = strtoull(tok.substr(0, colon).c_str(), nullptr, 0);
      unsigned long len = strtoul(tok.substr(colon + 1).c_str(), nullptr, 0);
      if (len == 0 || len > 4096) len = 32;
      unsigned char buf[4096];
      if (dram_peek(addr, buf, len)) {
        fprintf(stderr, "[gsim-emu] DRAM 0x%llx:", (unsigned long long)addr);
        for (unsigned long i = 0; i < len; i++) fprintf(stderr, " %02x", buf[i]);
        fprintf(stderr, "\n");
      } else fprintf(stderr, "[gsim-emu] DRAM 0x%llx: <out of range>\n", (unsigned long long)addr);
    }
  }
  return g_exit_code;
}
