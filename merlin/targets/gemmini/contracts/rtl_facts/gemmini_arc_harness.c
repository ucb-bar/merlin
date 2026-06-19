/* Drive harness for the isolated @Gemmini arcilator model (middle-tier #143-a).
 *
 * arcilator compiled @Gemmini (RoCC accelerator, no SoC/core/boot) to an arc model exposing a flat
 * state buffer; this hand-written driver replaces the SoC shell that verilator/FireSim provide for free:
 *   - 2-phase clock step (eval edge-detects the `clock` input: one cycle = clock 0->eval, 1->eval),
 *   - reset sequence,
 *   - a TileLink-UL memory slave on the scratchpad DMA port (serves mvin Gets / mvout Puts vs a DRAM
 *     byte buffer; single/multi-beat by `size`, 16B beat),
 *   - a PTW identity stub (bare-metal physical addressing),
 *   - a RoCC command feed (drive io_cmd_* from a decoded .insn stream, handshake on io_cmd_ready).
 *
 * STATUS: driver mechanics + reset + clock + RoCC handshake + a single-beat TileLink slave. Full
 * mvin/mvout multi-beat + bit-exact-vs-verilator matmul is the remaining iterative work (see
 * arc_feasibility_M1.md). Build: clang -O2 gemmini.ll gemmini_arc_harness.c -o gemmini_arc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemmini_arc_ports.h"

#define DRAM_BYTES (64u*1024u*1024u)
static unsigned char *STATE;
static unsigned char *DRAM;            /* physical memory the accelerator DMAs against */
#define GET(p)      arc_get(STATE, p##_OFF, p##_BITS)
#define SET(p,v)    arc_set(STATE, p##_OFF, p##_BITS, (uint64_t)(v))

static long g_cycles = 0;

/* one cycle: settle comb with clock low, then rising edge fires the clocked update. */
static void step(void){ SET(P_clock,0); Gemmini_eval(STATE); SET(P_clock,1); Gemmini_eval(STATE); g_cycles++; }

/* TileLink-UL slave on auto_spad_id_out: A = requests from Gemmini, D = our responses.
 * opcodes: A 4=Get,0/1=Put(Full/Partial); D 0=AccessAck,1=AccessAckData. 16B (128b) beat. */
static void tl_service(void){
  SET(P_auto_spad_id_out_a_ready, 1);        /* always accept requests */
  if (GET(P_auto_spad_id_out_a_valid)) {
    uint64_t op   = GET(P_auto_spad_id_out_a_bits_opcode);
    uint64_t addr = GET(P_auto_spad_id_out_a_bits_address);
    uint64_t src  = GET(P_auto_spad_id_out_a_bits_source);
    uint64_t sz   = GET(P_auto_spad_id_out_a_bits_size);   /* log2 bytes */
    if (op == 0 || op == 1) {                /* PutFullData / PutPartialData -> write 16B beat */
      uint64_t d[2]; arc_get128(STATE, P_auto_spad_id_out_a_bits_data_OFF, d);
      if (addr + 16 <= DRAM_BYTES) memcpy(DRAM + addr, d, 16);
      SET(P_auto_spad_id_out_d_valid, 1);
      SET(P_auto_spad_id_out_d_bits_opcode, 0);            /* AccessAck */
      SET(P_auto_spad_id_out_d_bits_size, sz);
      SET(P_auto_spad_id_out_d_bits_source, src);
    } else {                                  /* Get -> read 16B beat */
      uint64_t d[2] = {0,0};
      if (addr + 16 <= DRAM_BYTES) memcpy(d, DRAM + addr, 16);
      SET(P_auto_spad_id_out_d_valid, 1);
      SET(P_auto_spad_id_out_d_bits_opcode, 1);            /* AccessAckData */
      SET(P_auto_spad_id_out_d_bits_size, sz);
      SET(P_auto_spad_id_out_d_bits_source, src);
      arc_set128(STATE, P_auto_spad_id_out_d_bits_data_OFF, d);
    }
  } else {
    SET(P_auto_spad_id_out_d_valid, 0);
  }
}

/* PTW identity stub: ack requests, never fault (bare-metal physical addressing). */
static void ptw_service(void){
  SET(P_io_ptw_0_req_ready, 1);
  if (GET(P_io_ptw_0_req_valid)) {
    SET(P_io_ptw_0_resp_valid, 1);
    SET(P_io_ptw_0_resp_bits_pte_ppn, GET(P_io_ptw_0_req_bits_bits_addr));
    SET(P_io_ptw_0_resp_bits_pte_v,1); SET(P_io_ptw_0_resp_bits_pte_r,1);
    SET(P_io_ptw_0_resp_bits_pte_w,1); SET(P_io_ptw_0_resp_bits_pte_x,1);
    SET(P_io_ptw_0_resp_bits_pte_a,1); SET(P_io_ptw_0_resp_bits_pte_d,1);
  } else SET(P_io_ptw_0_resp_valid, 0);
}

static void servicing_step(void){ tl_service(); ptw_service(); SET(P_io_resp_ready,1); step(); }

static void reset_dut(int n){ SET(P_reset,1); for(int i=0;i<n;i++) servicing_step(); SET(P_reset,0); }

/* feed one RoCC custom-3 instruction; spin (servicing mem) until io_cmd_ready handshake. */
#define ROCC_OPCODE 0x0b   /* custom-3 opcode field (7b); the .insn used 0x7b full byte */
static int rocc_cmd(int funct,int xd,int xs1,int xs2,int rd,uint64_t rs1,uint64_t rs2,long budget){
  SET(P_io_cmd_bits_inst_opcode, ROCC_OPCODE); SET(P_io_cmd_bits_inst_funct, funct);
  SET(P_io_cmd_bits_inst_xd,xd); SET(P_io_cmd_bits_inst_xs1,xs1); SET(P_io_cmd_bits_inst_xs2,xs2);
  SET(P_io_cmd_bits_inst_rd,rd); SET(P_io_cmd_bits_rs1,rs1); SET(P_io_cmd_bits_rs2,rs2);
  SET(P_io_cmd_valid,1);
  long t=0; while(t++<budget){ servicing_step(); if(GET(P_io_cmd_ready)){ SET(P_io_cmd_valid,0); servicing_step(); return 0; } }
  SET(P_io_cmd_valid,0); return -1; /* no handshake within budget */
}

int main(int argc,char**argv){
  STATE = calloc(ARC_STATE_BYTES + 64, 1);
  DRAM  = calloc(DRAM_BYTES, 1);
  Gemmini_passthrough(STATE);
  reset_dut(20);
  /* probe: after reset the accelerator should be idle + ready for a command */
  printf("post-reset: io_cmd_ready=%llu io_busy=%llu cycles=%ld\n",
         (unsigned long long)GET(P_io_cmd_ready), (unsigned long long)GET(P_io_busy), g_cycles);
  /* smoke: issue a CONFIG_EX (funct 0) and confirm the handshake + the model clocks live */
  int rc = rocc_cmd(/*funct CONFIG*/0, 0,1,1, 0, 0, 0, 5000);
  printf("CONFIG cmd: handshake=%s  io_busy=%llu  total_cycles=%ld\n",
         rc==0?"OK":"TIMEOUT", (unsigned long long)GET(P_io_busy), g_cycles);
  /* free-run a few cycles, measure cycle rate of the full @Gemmini arc model */
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  long N = (argc>1)?atol(argv[1]):200000; for(long i=0;i<N;i++) servicing_step();
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
  printf("full @Gemmini arc model: %ld cycles in %.3fs = %.3f M-cycle/s\n", N, s, N/s/1e6);
  return 0;
}
