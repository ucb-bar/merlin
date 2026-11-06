/* One-token TinyLlama block with LOADMEM weights and activations packed at runtime. */
#include "tiny_block.h"
#include "tiny_projection.h"
#include "tiny_bundle_assets.h"
#include "atlas_host.h"
#include "atlas_host_layout.h"
#include "atlas_host_check.h"
#include "host_matrix_fp8pack.h"
#include <stdint.h>
#include <stddef.h>
__attribute__((section(".tohost"),aligned(64))) volatile uint64_t tohost;
__attribute__((aligned(64))) volatile uint64_t atlas_test_mailbox[16];
static uint64_t workspace[131072/8];
uint16_t block_output[2048], block_keys[256], block_values[256];
static atlas_host device;
static size_t cursor;
static int launch(void *ctx,unsigned logical,const uint16_t *input,size_t rows,size_t k,size_t width,uint16_t *output) {
  (void)ctx;
  if(rows!=1) return -1;
  int scale;
  if(host_matrix_scale_bf16(input,rows,k,&scale)) return -2;
  for(size_t col=0;col<width;) {
    size_t i=cursor;
    if(i>=12) return -3;
    const tiny_panel *p=&tiny_panels[i];
    if(p->logical!=logical || p->k!=k || p->width!=width || p->column!=col || p->a_scale!=scale) return -4;
    uint8_t *dram=(uint8_t*)p->weights-32*k;
    atlas_test_mailbox[3]=i;
    if(host_matrix_pack_bf16_at_scale(input,rows,k,32,k,32,32,scale,dram,32*k)) return -5;
    if(atlas_host_load_raw(&device,p->program,p->program_words)!=ATLAS_HOST_OK) return -6;
    atlas_host_buffer buffer={(uintptr_t)dram,p->total_bytes,ATLAS_HOST_DEVICE_READ_WRITE};
    if(atlas_host_start(&device,&buffer,1)!=ATLAS_HOST_OK
       || atlas_host_wait(&device,200000000)!=ATLAS_HOST_OK) return -7;
    if(device.active || device.invocation_sequence!=i+1) return -8;
    __asm__ volatile("fence iorw,iorw" ::: "memory");
    /* Check the full physical output and both guards. */
    if(atlas_host_check_exact(dram+p->output_offset,tiny_panel_expected[i],p->store_bytes).mismatch_bytes) return -9;
    for(size_t g=0;g<256;g++)
      if(dram[p->output_offset-256+g]!=0xa5 || dram[p->output_offset+p->store_bytes+g]!=0x5a) return -10;
    for(size_t j=0;j<p->n;j++) {
      size_t at=p->output_offset+(j/32)*2048+(j%32)/16*1024+(j%16)*2;
      output[col+j]=(uint16_t)dram[at]|((uint16_t)dram[at+1]<<8);
    }
    col+=p->n; cursor++; atlas_test_mailbox[2]=cursor;
  }
  return 0;
}
__attribute__((noinline,noreturn)) static void complete(uint64_t state,uint64_t code) {
  atlas_test_mailbox[1]=state; atlas_test_mailbox[4]=code;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  tohost=state==2?1:3;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  for(;;) __asm__ volatile("wfi");
}
int main(void) {
  atlas_test_mailbox[0]=UINT64_C(0x54494e59424c4b31); atlas_test_mailbox[1]=1;
  if(tiny_block_workspace_bytes()>sizeof(workspace)) complete(3,1);
  atlas_host_map map={ATLAS_IMEM_BASE,ATLAS_CSR_BASE,ATLAS_IMEM_BYTES};
  atlas_host_mmio mmio={atlas_host_direct_read32,atlas_host_direct_write32,0};
  atlas_host_memory memory={1,0,0};
  if(atlas_host_init(&device,map,mmio,memory)!=ATLAS_HOST_OK) complete(3,2);
  int code=tiny_block_run(1,tiny_entry[0],tiny_input_norm,tiny_post_norm,tiny_epsilon,tiny_cosine,tiny_sine,
                         0,0,launch,0,workspace,sizeof(workspace),block_keys,block_values,block_output);
  if(code) complete(3,(uint64_t)(100-code));
  if(cursor!=12 || atlas_test_mailbox[2]!=12) complete(3,200);
  if(atlas_host_check_exact(block_output,tiny_expected_output[0],sizeof(block_output)).mismatch_bytes) complete(3,201);
  if(atlas_host_check_exact(block_keys,tiny_expected_keys[0],sizeof(block_keys)).mismatch_bytes) complete(3,202);
  if(atlas_host_check_exact(block_values,tiny_expected_values[0],sizeof(block_values)).mismatch_bytes) complete(3,203);
  complete(2,0);
}
