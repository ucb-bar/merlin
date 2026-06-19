/* Full matmul replay on the isolated @Gemmini arc model (middle-tier #143-a, bit-exact attempt).
 * Replays a capsule's exact RoCC stream (a2_replay.h) into the arc model, serving the scratchpad DMA
 * over a multi-beat TileLink-UH slave, then reads the result from DRAM and checks it vs the golden.
 * Build: clang -O2 -I. gemmini.ll gemmini_arc_replay.c -o gemmini_arc_replay
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gemmini_arc_ports.h"
#include "replay_active.h"

#define DRAM_BYTES (64u*1024u*1024u)
static unsigned char *STATE, *DRAM;
#define GET(p)   arc_get(STATE, p##_OFF, p##_BITS)
#define SET(p,v) arc_set(STATE, p##_OFF, p##_BITS, (uint64_t)(v))
static long g_cycles=0;
/* host<->accelerator communication telemetry (the SoC shell normally hides this) */
static long t_cmds=0, t_resp=0, t_get=0, t_put=0, t_getB=0, t_putB=0, t_ptw=0, t_busy=0;

/* configurable memory read latency (cycles) — models DRAM/bus delay the ideal slave otherwise hides.
 * ARC_MEM_LATENCY=0 (default) = ideal memory; >0 makes cycle counts reflect realistic memory timing. */
static long MEM_LAT=0;
/* pending D-response FIFO (each = one 16B beat), with a ready-cycle gate for latency */
typedef struct { int opcode; uint64_t src, size; uint8_t data[16]; long ready; } dresp_t;
static dresp_t DQ[4096]; static int dq_head=0, dq_tail=0;
static void dq_push(int op,uint64_t src,uint64_t sz,const uint8_t*d){
  DQ[dq_tail].opcode=op; DQ[dq_tail].src=src; DQ[dq_tail].size=sz; DQ[dq_tail].ready=g_cycles+MEM_LAT;
  if(d) memcpy(DQ[dq_tail].data,d,16); else memset(DQ[dq_tail].data,0,16);
  dq_tail=(dq_tail+1)%4096;
}
static int dq_empty(void){return dq_head==dq_tail;}

/* Put-burst accumulation state */
static int put_rem=0; static uint64_t put_addr=0, put_src=0, put_size=0;

static void tl_service(void){
  SET(P_auto_spad_id_out_a_ready,1);                 /* always accept A */
  if(GET(P_auto_spad_id_out_a_valid)){
    uint64_t op=GET(P_auto_spad_id_out_a_bits_opcode);
    uint64_t addr=GET(P_auto_spad_id_out_a_bits_address);
    uint64_t src=GET(P_auto_spad_id_out_a_bits_source);
    uint64_t sz=GET(P_auto_spad_id_out_a_bits_size);
    uint64_t nbeats=(1ULL<<sz)/16; if(nbeats==0) nbeats=1;
    if(op==0||op==1){                                /* PutFull/Partial: write each beat (mvout DMA) */
      uint64_t d[2]; arc_get128(STATE,P_auto_spad_id_out_a_bits_data_OFF,d);
      if(put_rem==0){ put_addr=addr; put_src=src; put_size=sz; put_rem=(int)nbeats; t_put++; }
      uint64_t off=put_addr + (uint64_t)(((int)nbeats-put_rem))*16;
      if(off+16<=DRAM_BYTES) memcpy(DRAM+off,d,16);
      t_putB+=16;
      if(--put_rem==0) dq_push(0,put_src,put_size,0);    /* AccessAck after last beat */
    } else {                                          /* Get: queue nbeats AckData from DRAM (mvin DMA) */
      t_get++; t_getB+=nbeats*16;
      for(uint64_t k=0;k<nbeats;k++){
        uint8_t buf[16]={0}; uint64_t off=addr+k*16;
        if(off+16<=DRAM_BYTES) memcpy(buf,DRAM+off,16);
        dq_push(1,src,sz,buf);
      }
    }
  }
  /* drive D: present front beat once its latency has elapsed; pop on handshake (d_valid & d_ready) */
  if(!dq_empty() && DQ[dq_head].ready<=g_cycles){
    dresp_t*r=&DQ[dq_head];
    SET(P_auto_spad_id_out_d_valid,1);
    SET(P_auto_spad_id_out_d_bits_opcode,r->opcode);
    SET(P_auto_spad_id_out_d_bits_size,r->size);
    SET(P_auto_spad_id_out_d_bits_source,r->src);
    SET(P_auto_spad_id_out_d_bits_param,0); SET(P_auto_spad_id_out_d_bits_denied,0);
    SET(P_auto_spad_id_out_d_bits_corrupt,0); SET(P_auto_spad_id_out_d_bits_sink,0);
    arc_set128(STATE,P_auto_spad_id_out_d_bits_data_OFF,(uint64_t*)r->data);
    if(GET(P_auto_spad_id_out_d_ready)) dq_head=(dq_head+1)%4096;
  } else SET(P_auto_spad_id_out_d_valid,0);
}
static void ptw_service(void){
  SET(P_io_ptw_0_req_ready,1);
  if(GET(P_io_ptw_0_req_valid)){
    t_ptw++;
    SET(P_io_ptw_0_resp_valid,1);
    SET(P_io_ptw_0_resp_bits_pte_ppn,GET(P_io_ptw_0_req_bits_bits_addr));
    SET(P_io_ptw_0_resp_bits_pte_v,1);SET(P_io_ptw_0_resp_bits_pte_r,1);
    SET(P_io_ptw_0_resp_bits_pte_w,1);SET(P_io_ptw_0_resp_bits_pte_x,1);
    SET(P_io_ptw_0_resp_bits_pte_a,1);SET(P_io_ptw_0_resp_bits_pte_d,1);
  } else SET(P_io_ptw_0_resp_valid,0);
}
static void step(void){
  tl_service(); ptw_service(); SET(P_io_resp_ready,1);
  SET(P_clock,0); Gemmini_eval(STATE); SET(P_clock,1); Gemmini_eval(STATE); g_cycles++;
  if(GET(P_io_resp_valid)) t_resp++;          /* result returned to host (xd responses) */
  if(GET(P_io_busy)) t_busy++;                /* accelerator-active cycles */
}
static void drain(int n){ for(int i=0;i<n && (GET(P_io_busy)||!dq_empty());i++) step(); }

static int rocc(int funct,uint64_t rs1,uint64_t rs2,long budget){
  SET(P_io_cmd_bits_inst_opcode,0x0b); SET(P_io_cmd_bits_inst_funct,funct);
  SET(P_io_cmd_bits_inst_xd,0); SET(P_io_cmd_bits_inst_xs1,1); SET(P_io_cmd_bits_inst_xs2,1);
  SET(P_io_cmd_bits_inst_rd,0); SET(P_io_cmd_bits_rs1,rs1); SET(P_io_cmd_bits_rs2,rs2);
  SET(P_io_cmd_valid,1);
  for(long t=0;t<budget;t++){ if(GET(P_io_cmd_ready)){ t_cmds++; step(); SET(P_io_cmd_valid,0); step(); return 0; } step(); }
  SET(P_io_cmd_valid,0); return -1;
}

int main(void){
  { const char*e=getenv("ARC_MEM_LATENCY"); if(e) MEM_LAT=atol(e); }
  STATE=calloc(ARC_STATE_BYTES+64,1); DRAM=calloc(DRAM_BYTES,1);
  for(int i=0;i<N_PLACE;i++) memcpy(DRAM+PLACE[i].addr,PLACE[i].data,PLACE[i].len);
  Gemmini_passthrough(STATE);
  SET(P_reset,1); for(int i=0;i<20;i++) step(); SET(P_reset,0); for(int i=0;i<5;i++) step();
  int rc=0;
  for(int i=0;i<N_INSN;i++){
    if(INSN[i].is_fence){ drain(2000000); continue; }
    if(rocc(INSN[i].funct,INSN[i].rs1,INSN[i].rs2,2000000)<0){ printf("insn %d funct %d: NO HANDSHAKE\n",i,INSN[i].funct); rc=1; }
  }
  drain(2000000);
  /* generic readback: OUT_ROWS x OUT_COLS, row stride OUT_STRIDE bytes, element OUT_ELEM bytes (i32/i8) */
  long mism=0; int shown=0; printf("Y0 first row: ");
  for(int r=0;r<OUT_ROWS;r++) for(int c=0;c<OUT_COLS;c++){
    long off=OUT_ADDR+(long)r*OUT_STRIDE+(long)c*OUT_ELEM, v;
    if(OUT_ELEM==1){ int8_t x; memcpy(&x,DRAM+off,1); v=x; }
    else { int32_t x; memcpy(&x,DRAM+off,4); v=x; }
    long g=OUT_GOLDEN[r*OUT_COLS+c]; if(v!=g) mism++;
    if(r==0 && shown++<16) printf("%ld ",v);
  }
  printf("\ncycles=%ld  mismatches=%ld/%d  -> %s\n",g_cycles,mism,OUT_N,mism==0?"BIT-EXACT PASS":"MISMATCH");
  /* host<->accelerator communication report (control + DMA traffic the SoC shell normally hides) */
  printf("HOST-COMM: rocc_cmds=%ld (host->accel control)  resp_to_host=%ld  busy_cyc=%ld (%.0f%%)\n"
         "  DMA mvin: %ld Get xacts, %ld B read   |  mvout: %ld Put xacts, %ld B written\n"
         "  PTW(addr-translation) reqs=%ld\n",
         t_cmds, t_resp, t_busy, g_cycles?100.0*t_busy/g_cycles:0.0,
         t_get, t_getB, t_put, t_putB, t_ptw);
  return rc|(mism!=0);
}
