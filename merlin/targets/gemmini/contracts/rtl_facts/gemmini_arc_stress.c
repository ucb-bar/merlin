/* Stress / validation harness for the @Gemmini arc middle-tier (does the tool actually work + is it good?)
 *
 * Three tests on the A2 16x16x16 matmul (replay_active.h must be the A2 spec):
 *   (1) RANDOM DIFFERENTIAL: N trials of random int8 W/A0; arc output must match an INDEPENDENT C
 *       reference matmul (A0 @ W, i8*i8->i32) bit-exact every time. Faithful => matches across the
 *       whole input space, not just the canned vector.
 *   (2) DETERMINISM: same inputs run twice => identical output + cycles.
 *   (3) NEGATIVE CONTROL: corrupt one input byte => the arc output must CHANGE (and the C ref predicts
 *       exactly how). Proves the tool genuinely simulates the compute, not echoes/trivially passes.
 * Build: clang -O2 -I. gemmini_arc_stress.c gemmini.o -o gemmini_arc_stress
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gemmini_arc_ports.h"
#include "replay_active.h"

#define DRAM_BYTES (64u*1024u*1024u)
#define MDIM 16
#define KDIM 16
#define NDIM 16
static unsigned char *STATE, *DRAM;
#define GET(p)   arc_get(STATE, p##_OFF, p##_BITS)
#define SET(p,v) arc_set(STATE, p##_OFF, p##_BITS, (uint64_t)(v))
static long g_cycles;
typedef struct { int opcode; uint64_t src,size; uint8_t data[16]; } dresp_t;
static dresp_t DQ[8192]; static int dq_h,dq_t;
static void dq_push(int op,uint64_t s,uint64_t z,const uint8_t*d){DQ[dq_t].opcode=op;DQ[dq_t].src=s;DQ[dq_t].size=z;if(d)memcpy(DQ[dq_t].data,d,16);else memset(DQ[dq_t].data,0,16);dq_t=(dq_t+1)%8192;}
static int dq_empty(void){return dq_h==dq_t;}
static int put_rem; static uint64_t put_addr,put_src,put_size;
static void tl(void){
  SET(P_auto_spad_id_out_a_ready,1);
  if(GET(P_auto_spad_id_out_a_valid)){
    uint64_t op=GET(P_auto_spad_id_out_a_bits_opcode),addr=GET(P_auto_spad_id_out_a_bits_address),
             src=GET(P_auto_spad_id_out_a_bits_source),sz=GET(P_auto_spad_id_out_a_bits_size);
    uint64_t nb=(1ULL<<sz)/16; if(!nb)nb=1;
    if(op==0||op==1){uint64_t d[2];arc_get128(STATE,P_auto_spad_id_out_a_bits_data_OFF,d);
      if(!put_rem){put_addr=addr;put_src=src;put_size=sz;put_rem=(int)nb;}
      uint64_t off=put_addr+(uint64_t)(((int)nb-put_rem))*16; if(off+16<=DRAM_BYTES)memcpy(DRAM+off,d,16);
      if(--put_rem==0)dq_push(0,put_src,put_size,0);
    } else for(uint64_t k=0;k<nb;k++){uint8_t b[16]={0};uint64_t off=addr+k*16;if(off+16<=DRAM_BYTES)memcpy(b,DRAM+off,16);dq_push(1,src,sz,b);}
  }
  if(!dq_empty()){dresp_t*r=&DQ[dq_h];SET(P_auto_spad_id_out_d_valid,1);SET(P_auto_spad_id_out_d_bits_opcode,r->opcode);
    SET(P_auto_spad_id_out_d_bits_size,r->size);SET(P_auto_spad_id_out_d_bits_source,r->src);
    SET(P_auto_spad_id_out_d_bits_param,0);SET(P_auto_spad_id_out_d_bits_denied,0);SET(P_auto_spad_id_out_d_bits_corrupt,0);SET(P_auto_spad_id_out_d_bits_sink,0);
    arc_set128(STATE,P_auto_spad_id_out_d_bits_data_OFF,(uint64_t*)r->data);
    if(GET(P_auto_spad_id_out_d_ready))dq_h=(dq_h+1)%8192;
  } else SET(P_auto_spad_id_out_d_valid,0);
}
static void ptw(void){SET(P_io_ptw_0_req_ready,1);
  if(GET(P_io_ptw_0_req_valid)){SET(P_io_ptw_0_resp_valid,1);SET(P_io_ptw_0_resp_bits_pte_ppn,GET(P_io_ptw_0_req_bits_bits_addr));
    SET(P_io_ptw_0_resp_bits_pte_v,1);SET(P_io_ptw_0_resp_bits_pte_r,1);SET(P_io_ptw_0_resp_bits_pte_w,1);
    SET(P_io_ptw_0_resp_bits_pte_x,1);SET(P_io_ptw_0_resp_bits_pte_a,1);SET(P_io_ptw_0_resp_bits_pte_d,1);}
  else SET(P_io_ptw_0_resp_valid,0);}
static void step(void){tl();ptw();SET(P_io_resp_ready,1);SET(P_clock,0);Gemmini_eval(STATE);SET(P_clock,1);Gemmini_eval(STATE);g_cycles++;}
static void drain(int n){for(int i=0;i<n&&(GET(P_io_busy)||!dq_empty());i++)step();}
static int rocc(int f,uint64_t r1,uint64_t r2,long b){SET(P_io_cmd_bits_inst_opcode,0x0b);SET(P_io_cmd_bits_inst_funct,f);
  SET(P_io_cmd_bits_inst_xd,0);SET(P_io_cmd_bits_inst_xs1,1);SET(P_io_cmd_bits_inst_xs2,1);SET(P_io_cmd_bits_inst_rd,0);
  SET(P_io_cmd_bits_rs1,r1);SET(P_io_cmd_bits_rs2,r2);SET(P_io_cmd_valid,1);
  for(long t=0;t<b;t++){if(GET(P_io_cmd_ready)){step();SET(P_io_cmd_valid,0);step();return 0;}step();}SET(P_io_cmd_valid,0);return -1;}

/* one full matmul: reset, place given W/A0 bytes, replay the A2 stream, read MxN i32 result into out[] */
static long run_once(const int8_t*W,const int8_t*A0,int32_t*out){
  memset(STATE,0,ARC_STATE_BYTES); dq_h=dq_t=0; put_rem=0; g_cycles=0;
  Gemmini_passthrough(STATE);
  SET(P_reset,1); for(int i=0;i<20;i++)step(); SET(P_reset,0); for(int i=0;i<5;i++)step();
  memcpy(DRAM+PLACE[0].addr,W,KDIM*NDIM);          /* PLACE[0]=W (arg0) */
  memcpy(DRAM+PLACE[1].addr,A0,MDIM*KDIM);         /* PLACE[1]=A0(arg1) */
  memset(DRAM+OUT_ADDR,0,MDIM*OUT_STRIDE);
  for(int i=0;i<N_INSN;i++){ if(INSN[i].is_fence){drain(2000000);continue;}
    rocc(INSN[i].funct,INSN[i].rs1,INSN[i].rs2,2000000); }
  drain(2000000);
  for(int r=0;r<MDIM;r++)for(int c=0;c<NDIM;c++){int32_t v;memcpy(&v,DRAM+OUT_ADDR+(long)r*OUT_STRIDE+(long)c*4,4);out[r*NDIM+c]=v;}
  return g_cycles;
}
/* independent reference: out = A0(MxK) @ W(KxN), i8*i8 -> i32 */
static void ref_mm(const int8_t*W,const int8_t*A0,int32_t*out){
  for(int i=0;i<MDIM;i++)for(int j=0;j<NDIM;j++){int32_t a=0;for(int k=0;k<KDIM;k++)a+=(int)A0[i*KDIM+k]*(int)W[k*NDIM+j];out[i*NDIM+j]=a;}
}
static int cmp(const int32_t*a,const int32_t*b){int n=0;for(int i=0;i<MDIM*NDIM;i++)if(a[i]!=b[i])n++;return n;}

int main(int argc,char**argv){
  STATE=calloc(ARC_STATE_BYTES+64,1); DRAM=calloc(DRAM_BYTES,1);
  long N=(argc>1)?atol(argv[1]):500; unsigned seed=(argc>2)?atoi(argv[2]):1;
  int8_t W[KDIM*NDIM],A0[MDIM*KDIM]; int32_t arc[MDIM*NDIM],ref[MDIM*NDIM];
  long fails=0, cyc0=-1; int cyc_varies=0;
  /* (1) random differential */
  for(long t=0;t<N;t++){
    for(int i=0;i<KDIM*NDIM;i++)W[i]=(int8_t)(rand_r(&seed)%256-128);
    for(int i=0;i<MDIM*KDIM;i++)A0[i]=(int8_t)(rand_r(&seed)%256-128);
    long cy=run_once(W,A0,arc); ref_mm(W,A0,ref);
    int m=cmp(arc,ref); if(m){fails++; if(fails<=3)printf("  trial %ld: %d/%d mismatches (cyc=%ld)\n",t,m,MDIM*NDIM,cy);}
    if(cyc0<0)cyc0=cy; else if(cy!=cyc0)cyc_varies=1;
  }
  printf("(1) RANDOM DIFFERENTIAL: %ld trials, %ld FAILED  -> %s   (cycles const=%s @ %ld)\n",
         N,fails,fails?"BAD":"all bit-exact vs C reference",cyc_varies?"NO":"yes",cyc0);
  /* (2) determinism: same input twice */
  for(int i=0;i<KDIM*NDIM;i++)W[i]=(int8_t)(rand_r(&seed)%256-128);
  for(int i=0;i<MDIM*KDIM;i++)A0[i]=(int8_t)(rand_r(&seed)%256-128);
  int32_t o1[MDIM*NDIM],o2[MDIM*NDIM]; long c1=run_once(W,A0,o1),c2=run_once(W,A0,o2);
  printf("(2) DETERMINISM: out identical=%s  cycles identical=%s (%ld,%ld)\n",
         cmp(o1,o2)?"NO":"yes", c1==c2?"yes":"NO",c1,c2);
  /* (3) negative control: flip one A0 byte -> output MUST change exactly as the reference predicts */
  ref_mm(W,A0,ref); A0[0]=(int8_t)(A0[0]^0x5A);
  int32_t o3[MDIM*NDIM],r3[MDIM*NDIM]; run_once(W,A0,o3); ref_mm(W,A0,r3);
  int changed=cmp(o3,o1), tracks_ref=(cmp(o3,r3)==0);
  printf("(3) NEGATIVE CONTROL: 1 input byte flipped -> arc output changed=%s  still matches reference=%s\n",
         changed?"yes":"NO(!)", tracks_ref?"yes":"NO(!)");
  printf("\nVERDICT: %s\n",(fails==0&&cmp(o1,o2)==0&&c1==c2&&changed&&tracks_ref)?
         "PASS — faithful (random-exact), deterministic, and genuinely computing":"FAIL — see above");
  return fails!=0;
}
